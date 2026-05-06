from __future__ import annotations

import json
import os
import sys
import tempfile
import tomllib
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nvidia_converge
from nvidia_converge.cli import _commands_succeeded, _run_plan_actions, main
from nvidia_converge.audit import _parse_dpkg_packages
from nvidia_converge.desired import load_desired
from nvidia_converge.doctor import diagnose
from nvidia_converge.planner import build_plan, lock_actions
from nvidia_converge.models import CommandResult, DesiredState, PackageInfo, PlanAction, RollbackSnapshot
from nvidia_converge.rollback import _rollback_commands, apply_rollback
from nvidia_converge.runner import CommandRunner
from nvidia_converge.verify import verify_stack
from test_planner import _audit


def main_tests() -> int:
    test_default_desired()
    test_yaml_desired()
    test_driver_version_branch()
    test_invalid_desired_file()
    test_apply_requires_root()
    test_read_only_commands_reject_apply()
    test_bad_rollback_snapshot()
    test_version_flag()
    test_package_version_matches_cli_version()
    test_validate_command()
    test_schema_command()
    test_validation_schema_command()
    test_integration_results_schema_command()
    test_support_command()
    test_report_has_schema_required_keys()
    test_integration_results_example_has_required_keys()
    test_desired_schema_mentions_bare_object()
    test_plan()
    test_secure_boot_disabled_finding()
    test_secure_boot_verify_policy()
    test_cli_plan_report()
    test_install_dry_run()
    test_install_dry_run_does_not_write_rollback()
    test_snapshot_dry_run_does_not_write_rollback()
    test_failed_command_results_are_not_safe_for_post_install_verify()
    test_plan_execution_stops_after_failed_command()
    test_package_parser_deduplicates()
    test_rollback_filters_unrelated_packages()
    test_zypper_rollback_commands()
    test_apply_rollback_stops_after_failed_command()
    test_zypper_lock_plan()
    test_gpu_integration_uses_virtualenv_for_python_tooling()
    test_gpu_integration_validates_every_generated_report()
    print("all tests passed")
    return 0


def test_default_desired() -> None:
    desired = load_desired(None)
    assert desired.driver == "580-open"
    assert desired.cuda_compat == "13.0"
    assert desired.fabric_manager is True


def test_yaml_desired() -> None:
    desired = load_desired("examples/compute-580-open.yaml")
    assert desired.driver_major == "580"
    assert desired.open_kernel_module is True


def test_driver_version_branch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "desired.yaml"
        path.write_text(
            """
---
desired:
  driver: 595.71.05
...
""",
            encoding="utf-8",
        )
        desired = load_desired(str(path))
        assert desired.driver_major == "595"


def test_invalid_desired_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "desired.yaml"
        path.write_text("not yaml\n", encoding="utf-8")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            assert main(["plan", "--desired", str(path)]) == 2


def test_apply_requires_root() -> None:
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        return
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        assert main(["lock", "--apply"]) == 2


def test_read_only_commands_reject_apply() -> None:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            main(["plan", "--apply"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("plan --apply should be rejected")


def test_bad_rollback_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snapshot.json"
        path.write_text("{}", encoding="utf-8")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            assert main(["rollback", "--snapshot", str(path)]) == 2


def test_version_flag() -> None:
    try:
        with redirect_stdout(StringIO()):
            main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0


def test_package_version_matches_cli_version() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert nvidia_converge.__version__ == pyproject["project"]["version"]


def test_validate_command() -> None:
    out = StringIO()
    with redirect_stdout(out):
        assert main(["validate", "--desired", "examples/compute-580-open.yaml"]) == 0
    assert "Desired state: valid" in out.getvalue()
    out = StringIO()
    with redirect_stdout(out):
        assert main(["validate", "--desired", "examples/compute-580-open.yaml", "--json"]) == 0
    validation = json.loads(out.getvalue())
    assert validation["valid"] is True
    assert validation["desired"]["driver"] == "580-open"


def test_schema_command() -> None:
    out = StringIO()
    with redirect_stdout(out):
        assert main(["schema", "report"]) == 0
    schema = json.loads(out.getvalue())
    assert schema["title"] == "nvidia-converge report"


def test_validation_schema_command() -> None:
    out = StringIO()
    with redirect_stdout(out):
        assert main(["schema", "validation"]) == 0
    schema = json.loads(out.getvalue())
    assert schema["title"] == "nvidia-converge validation result"


def test_integration_results_schema_command() -> None:
    out = StringIO()
    with redirect_stdout(out):
        assert main(["schema", "integration-results"]) == 0
    schema = json.loads(out.getvalue())
    assert schema["title"] == "nvidia-converge integration results"


def test_support_command() -> None:
    out = StringIO()
    with redirect_stdout(out):
        assert main(["support", "--json"]) == 0
    matrix = json.loads(out.getvalue())
    assert matrix["package_managers"]["apt-get"]["audit"] is True


def test_report_has_schema_required_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "plan.json"
        with redirect_stdout(StringIO()):
            assert main(["plan", "--out", str(out)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        schema = json.loads(Path("schemas/report.schema.json").read_text(encoding="utf-8"))
        assert set(schema["required"]).issubset(report)


def test_integration_results_example_has_required_keys() -> None:
    example = json.loads(Path("integrations/results.example.json").read_text(encoding="utf-8"))
    schema = json.loads(Path("schemas/integration-results.schema.json").read_text(encoding="utf-8"))
    assert set(schema["required"]).issubset(example)
    assert example["overall_status"] == "blocked"


def test_desired_schema_mentions_bare_object() -> None:
    schema = json.loads(Path("schemas/desired.schema.json").read_text(encoding="utf-8"))
    assert "oneOf" in schema
    assert any(option.get("$ref") == "#/$defs/desired" for option in schema["oneOf"])


def test_plan() -> None:
    desired = load_desired(None)
    audit = _audit()
    plan = build_plan(desired, audit, diagnose(desired, audit))
    ids = [action.id for action in plan]
    assert "install.packages" in ids
    assert "lock.apt" in ids
    locks = lock_actions(desired, audit)
    assert "nvidia-driver-580-open" in locks[0].commands[0]


def test_secure_boot_disabled_finding() -> None:
    findings = diagnose(DesiredState(secure_boot="disabled"), _audit())
    assert any(finding.id == "secure-boot.enabled" for finding in findings)


def test_secure_boot_verify_policy() -> None:
    checks = verify_stack(DesiredState(secure_boot="disabled"), CommandRunner(), _audit())
    assert any(check.name == "secure-boot.policy" and not check.ok for check in checks)


def test_cli_plan_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "plan.json"
        with redirect_stdout(StringIO()):
            assert main(["plan", "--out", str(out)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["desired"]["driver"] == "580-open"
        assert report["plan"]
        assert isinstance(report["sbom"], list)


def test_install_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "install.json"
        with redirect_stdout(StringIO()):
            rc = main(["install", "--out", str(out)])
        assert rc in {0, 2}
        report = json.loads(out.read_text(encoding="utf-8"))
        assert any(result.get("skipped") for result in report["command_results"])


def test_install_dry_run_does_not_write_rollback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path.cwd()
        try:
            os.chdir(tmp)
            out = Path(tmp) / "install.json"
            with redirect_stdout(StringIO()):
                rc = main(["install", "--out", str(out)])
            assert rc in {0, 2}
            report = json.loads(out.read_text(encoding="utf-8"))
            assert report["rollback"]["path"] is None
            assert not Path("nvidia-converge-rollback.json").exists()
        finally:
            os.chdir(cwd)


def test_snapshot_dry_run_does_not_write_rollback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path.cwd()
        try:
            os.chdir(tmp)
            out = Path(tmp) / "snapshot.json"
            with redirect_stdout(StringIO()):
                assert main(["snapshot", "--out", str(out)]) == 0
            report = json.loads(out.read_text(encoding="utf-8"))
            assert report["rollback"]["path"] is None
            assert not Path("nvidia-converge-rollback.json").exists()
        finally:
            os.chdir(cwd)


def test_failed_command_results_are_not_safe_for_post_install_verify() -> None:
    assert _commands_succeeded([CommandResult(["apt-get", "install"], 100)]) is False
    assert _commands_succeeded([CommandResult(["apt-get", "install"], None, skipped=True, reason="dry-run")]) is True


def test_plan_execution_stops_after_failed_command() -> None:
    runner = _FakeRunner([100, 0])
    actions = [
        PlanAction("install.packages", "Install packages.", [["apt-get", "install"], ["systemctl", "restart", "docker"]]),
        PlanAction("lock.apt", "Lock packages.", [["apt-mark", "hold", "nvidia-driver-580-open"]]),
    ]
    results = _run_plan_actions(actions, runner)
    assert [result.command for result in results] == [["apt-get", "install"]]


def test_package_parser_deduplicates() -> None:
    packages = _parse_dpkg_packages("libnvidia-gl\t1\nlibnvidia-gl\t1\nzlib1g\t1\n")
    assert len(packages) == 1
    assert packages[0].name == "libnvidia-gl"


def test_rollback_filters_unrelated_packages() -> None:
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-driver-580-open", "580.126.16-1", "apt", True),
            PackageInfo("bash", "5.2", "apt", True),
        ],
        "apt-get",
    )
    assert commands == [["apt-get", "install", "-y", "nvidia-driver-580-open=580.126.16-1"]]


def test_zypper_rollback_commands() -> None:
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-open-595", "595.71.05-1", "rpm", True),
            PackageInfo("bash", "5.2-1", "rpm", True),
        ],
        "zypper",
    )
    assert commands == [["zypper", "--non-interactive", "install", "--oldpackage", "nvidia-open-595=595.71.05-1"]]


def test_apply_rollback_stops_after_failed_command() -> None:
    snapshot = RollbackSnapshot(
        path=None,
        packages=[],
        kernel="6.8.0-test",
        module_version=None,
        commands=[
            ["apt-get", "install", "-y", "nvidia-driver-580=580.1"],
            ["apt-get", "install", "-y", "cuda-compat-13-0=13.0"],
        ],
    )
    runner = _FakeRunner([100, 0])
    results = apply_rollback(snapshot, runner)
    assert [result.command for result in results] == [["apt-get", "install", "-y", "nvidia-driver-580=580.1"]]


def test_zypper_lock_plan() -> None:
    audit = _audit()
    audit.package_manager = "zypper"
    locks = lock_actions(load_desired(None), audit)
    assert locks[0].id == "lock.zypper"


def test_gpu_integration_uses_virtualenv_for_python_tooling() -> None:
    workflow = Path(".github/workflows/gpu-integration.yml").read_text(encoding="utf-8")
    assert "python3 -m pip install --user" not in workflow
    assert "python3 -m venv .venv" in workflow
    assert ".venv/bin/python -m nvidia_converge plan" in workflow
    assert '"$PWD/.venv/bin/python" -m nvidia_converge install' in workflow


def test_gpu_integration_validates_every_generated_report() -> None:
    workflow = Path(".github/workflows/gpu-integration.yml").read_text(encoding="utf-8")
    assert "Validate generated report schemas" in workflow
    assert 'Path("artifacts/reports").glob("*.json")' in workflow
    assert "jsonschema.validate(json.load(handle), schema)" in workflow


class _FakeRunner:
    def __init__(self, returncodes: list[int | None]):
        self.returncodes = list(returncodes)
        self.results: list[CommandResult] = []

    def run(self, command: list[str], *, mutate: bool = False, allow_fail: bool = True, input_text: str | None = None) -> CommandResult:
        del mutate, allow_fail, input_text
        result = CommandResult(command, self.returncodes.pop(0))
        self.results.append(result)
        return result


if __name__ == "__main__":
    raise SystemExit(main_tests())
