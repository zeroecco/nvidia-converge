from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TESTS_ROOT.parent
sys.path[:0] = [str(REPOSITORY_ROOT), str(TESTS_ROOT)]

from planner_fixtures import _audit, _stage_policy, _suse_audit

import nvidia_converge
from nvidia_converge.audit import _parse_dpkg_packages
from nvidia_converge.cli import _commands_succeeded, _run_plan_actions, main
from nvidia_converge.desired import load_desired
from nvidia_converge.doctor import diagnose
from nvidia_converge.models import (
    CommandResult,
    DesiredState,
    PackageInfo,
    PlanAction,
    RollbackSnapshot,
)
from nvidia_converge.planner import build_plan, lock_actions
from nvidia_converge.rollback import _rollback_commands, apply_rollback
from nvidia_converge.runner import CommandRunner
from nvidia_converge.verify import verify_stack


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
    test_broken_stdout_exits_without_traceback()
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
    test_human_output_marks_skipped_verification_as_skip()
    test_package_parser_deduplicates()
    test_rollback_filters_unrelated_packages()
    test_zypper_rollback_commands()
    test_apply_rollback_stops_after_package_failure_without_module_or_services()
    test_zypper_lock_plan()
    test_production_workflows_bind_dispatch_to_current_main()
    test_gpu_integration_uses_virtualenv_for_python_tooling()
    test_gpu_integration_validates_every_generated_report()
    test_gpu_integration_exports_only_sanitized_attestation()
    print("all tests passed")
    return 0


def test_default_desired() -> None:
    desired = load_desired(None)
    assert desired.driver == "580-open"
    assert desired.cuda_compat == "none"
    assert desired.fabric_manager is False


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
    assert nvidia_converge.__version__ == _project_version()


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


def test_broken_stdout_exits_without_traceback() -> None:
    stdout = sys.stdout
    try:
        sys.stdout = _BrokenStdout()
        assert main(["schema", "report"]) == 1
    finally:
        sys.stdout = stdout


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
    assert matrix["python_runtime"]["minimum_version"] == "3.10"
    assert matrix["python_runtime"]["root_controlled_for_applied_execution"] is True


def test_report_has_schema_required_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "plan.json"
        with redirect_stdout(StringIO()):
            assert main(["plan", "--out", str(out)]) in {0, 2}
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
    assert [action.id for action in plan] == ["unsupported.package-policy-staging"]
    locks = lock_actions(desired, audit)
    assert locks[0].commands[-1][-1] == "nvidia-driver-pinning-580"
    _stage_policy(audit, desired)
    staged_plan = build_plan(desired, audit, diagnose(desired, audit))
    assert "install.packages" in [action.id for action in staged_plan]


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
            assert main(["plan", "--out", str(out)]) in {0, 2}
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
        if report["command_results"]:
            assert any(result.get("skipped") for result in report["command_results"])
        else:
            assert [action["id"] for action in report["plan"]] == ["unsupported.package-manager"]


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
            if report["rollback"] is not None:
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


def test_human_output_marks_skipped_verification_as_skip() -> None:
    from nvidia_converge.human import render_human
    from nvidia_converge.models import Report, Verification

    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        verification=[Verification("module.load", False, CommandResult(["modprobe", "nvidia"], None, skipped=True, reason="dry-run"))],
    )
    output = render_human("verify", report, apply=False)
    assert "- skip: module.load" in output
    assert "- fail: module.load" not in output


def test_package_parser_deduplicates() -> None:
    packages = _parse_dpkg_packages(
        "ii \tlibnvidia-gl\t1\tamd64\n"
        "ii \tlibnvidia-gl\t1\tamd64\n"
        "ii \tzlib1g\t1\tamd64\n"
    )
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
    assert commands == [
        [
            "apt-get",
            "install",
            "-y",
            "--allow-change-held-packages",
            "--allow-downgrades",
            "--no-download",
            "--no-install-recommends",
            "--purge",
            "nvidia-driver-580-open=580.126.16-1",
        ]
    ]


def test_zypper_rollback_commands() -> None:
    commands = _rollback_commands(
        [
            PackageInfo("nvidia-open-595", "595.71.05-1", "rpm", True),
            PackageInfo("bash", "5.2-1", "rpm", True),
        ],
        "zypper",
    )
    assert commands == [
        [
            "zypper",
            "--non-interactive",
            "--disable-repositories",
            "--no-refresh",
            "install",
            "--oldpackage",
            "--no-recommends",
            "--no-force-resolution",
            "--",
            "nvidia-open-595=595.71.05-1",
        ]
    ]


def test_apply_rollback_stops_after_package_failure_without_module_or_services() -> None:
    snapshot = RollbackSnapshot(
        path=None,
        packages=[PackageInfo("nvidia-driver-570", "570.1", "apt", True)],
        kernel="6.8.0-test",
        module_version=None,
        commands=[],
        package_manager="apt-get",
        introduced_packages=["nvidia-open"],
    )
    runner = _FakeRunner([100])
    results = apply_rollback(snapshot, runner)
    commands = [result.command for result in results]
    assert commands[:4] == [
        ["systemctl", "mask", "--now", "docker.socket"],
        ["systemctl", "mask", "--now", "docker.service"],
        ["systemctl", "mask", "--now", "nvidia-persistenced.service"],
        ["systemctl", "mask", "--now", "nvidia-fabricmanager.service"],
    ]
    assert commands[4] == [
        "apt-get",
        "install",
        "-y",
        "--allow-change-held-packages",
        "--allow-downgrades",
        "--no-download",
        "--no-install-recommends",
        "--purge",
        "nvidia-driver-570=570.1",
        "nvidia-open-",
    ]
    assert len(commands) == 5


def test_zypper_lock_plan() -> None:
    audit = _suse_audit()
    locks = lock_actions(load_desired(None), audit)
    assert locks[0].id == "lock.zypper"


def test_production_workflows_bind_dispatch_to_current_main() -> None:
    cases = (
        (
            Path(".github/workflows/production-gpu-qualification.yml"),
            Path(".github/workflows/gpu-integration.yml"),
            "production-gpu-qualification",
            4,
        ),
        (
            Path(".github/workflows/production-release.yml"),
            Path(".github/workflows/release.yml"),
            "production-release",
            3,
        ),
    )
    for path, old_path, action, checkout_count in cases:
        assert path.is_file()
        assert not old_path.exists()
        workflow = path.read_text(encoding="utf-8")
        assert "repository_dispatch:" in workflow
        assert f"types: [{action}]" in workflow
        assert "workflow_dispatch:" not in workflow
        assert "client_payload" not in workflow
        assert 'DISPATCH_ACTION: ${{ github.event.action }}' in workflow
        assert '"$GITHUB_EVENT_NAME" != repository_dispatch' in workflow
        assert f'"$DISPATCH_ACTION" != {action}' in workflow
        assert '"$GITHUB_REF" != refs/heads/main' in workflow
        assert '.default_branch | select(. == "main")' in workflow
        assert '"$live_main_sha" != "$GITHUB_SHA"' in workflow
        controls = workflow[
            workflow.index("  repository-controls:") :
            workflow.index("  qualification-build:")
            if action == "production-gpu-qualification"
            else workflow.index("  gates:")
        ]
        assert controls.index(
            "Bind the dispatch to the live default-branch head"
        ) < controls.index("Check out repository-control checker")
        assert workflow.count("uses: actions/checkout@") == checkout_count
        assert workflow.count("ref: ${{ github.sha }}") == checkout_count
        assert "ref: ${{ github.ref }}" not in workflow

    gpu = cases[0][0].read_text(encoding="utf-8")
    assert 'QUALIFICATION_APPLY: "true"' in gpu
    assert "inputs.apply" not in gpu

    release = cases[1][0].read_text(encoding="utf-8")
    assert 'release_tag="v${release_version}"' in release
    assert "GITHUB_REF_NAME" not in release
    assert release.index(
        "Recheck current main, release controls, and immutable configuration"
    ) < release.index("Mint a short-lived Release Creator token")
    publish = release[release.index("  publish:") :]
    assert "contents: read" in publish
    assert "\n      contents: write\n" not in publish
    assert "permission-contents: write" in publish


def test_gpu_integration_uses_virtualenv_for_python_tooling() -> None:
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "python3 -m pip install --user" not in workflow
    assert "for candidate in python3.12 python3.11 python3.10 python3" in workflow
    assert "sys.version_info >= (3, 10)" in workflow
    assert "import ensurepip" in workflow
    assert "validate_trusted_path()" in workflow
    assert "PATH: /usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin" in workflow
    assert "sys.base_prefix" in workflow
    assert "sys.exec_prefix" in workflow
    assert 'sysconfig.get_path("stdlib")' in workflow
    assert 'for path in "${python_trust_paths[@]:4}"' in workflow
    assert 'getattr(module, "__file__", None)' in workflow
    assert 'for package in (ensurepip, venv)' in workflow
    assert "Python trust path is empty, relative, or malformed" in workflow
    assert 'sudo "$PYTHON_BIN" -I -S -m venv' in workflow
    assert '"$PYTHON_BIN" -m venv .venv' in workflow
    assert "no Python >=3.10 interpreter with venv/ensurepip support found" in workflow
    assert "python3 -m venv .venv" not in workflow
    assert '"$QUALIFICATION_PYTHON" -I -m nvidia_converge plan' in workflow
    assert '"$QUALIFICATION_PYTHON" -I -m nvidia_converge install' in workflow
    assert "Build clean-source qualification wheel" in workflow
    assert "python -m build --wheel --no-isolation" in workflow
    gpu_job = workflow[
        workflow.index("  gpu:") : workflow.index("  validate-attestations:")
    ]
    assert "python -m build" not in gpu_job
    assert "--upgrade pip" not in gpu_job
    assert 'git archive --format=tar "$GITHUB_SHA"' in workflow
    assert 'archived_desired="$desired_source/$DESIRED_FILE"' in workflow
    assert "validate_trusted_directory /opt" in workflow
    assert 'PYTHON_BIN="$PWD/.venv/bin/python"' not in workflow
    assert 'PYTHONPATH="$PWD"' not in workflow


def test_gpu_integration_validates_every_generated_report() -> None:
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    gpu_job = workflow[
        workflow.index("  gpu:") : workflow.index("  validate-attestations:")
    ]
    assert "jsonschema" not in gpu_job
    assert "Validate retained report schemas" in workflow
    assert 'Path("attestation-artifacts")' in workflow
    assert "jsonschema.validate(" in workflow
    assert "strict_format_checker()" in workflow


def test_gpu_integration_exports_only_sanitized_attestation() -> None:
    workflow = Path(".github/workflows/production-gpu-qualification.yml").read_text(encoding="utf-8")
    assert "scripts/export_integration_attestation.py" in workflow
    assert "path: artifacts/export/" in workflow
    assert "id: export-attestation" in workflow
    assert "id: verify-attestation" in workflow
    assert "--verify-only" in workflow
    assert 'if [[ ! -d artifacts || -L artifacts ]]; then' in workflow
    assert "chmod 0700 artifacts" in workflow
    assert (
        "if: ${{ always() && steps.export-attestation.outcome == 'success' }}"
        in workflow
    )
    assert (
        "if: ${{ always() && steps.verify-attestation.outcome == 'success' }}"
        in workflow
    )
    export_step = workflow.index(
        "- name: Export privacy-minimized integration attestation"
    )
    verify_step = workflow.index(
        "- name: Verify exact sanitized upload handoff"
    )
    upload_step = workflow.index("- name: Upload integration artifacts")
    assert export_step < verify_step < upload_step
    assert "--verify-only" not in workflow[export_step:verify_step]
    assert "--verify-only" in workflow[verify_step:upload_step]
    assert "cli.audit_host =" not in workflow
    assert "Independently restore controlled-fault pre-state" in workflow


class _FakeRunner:
    def __init__(self, returncodes: list[int | None]):
        self.returncodes = list(returncodes)
        self.results: list[CommandResult] = []

    def run(self, command: list[str], *, mutate: bool = False, allow_fail: bool = True, input_text: str | None = None) -> CommandResult:
        del mutate, allow_fail, input_text
        if command[:3] == ["systemctl", "mask", "--now"]:
            return CommandResult(command, 0)
        result = CommandResult(command, self.returncodes.pop(0))
        self.results.append(result)
        return result


class _BrokenStdout:
    def write(self, text: str) -> None:
        del text
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def _project_version() -> str:
    in_project = False
    for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.strip() == "[project]":
            in_project = True
            continue
        if in_project and line.startswith("["):
            break
        if in_project and line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("pyproject.toml is missing [project] version")


if __name__ == "__main__":
    raise SystemExit(main_tests())
