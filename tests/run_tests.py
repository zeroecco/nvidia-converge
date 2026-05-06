from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvidia_converge.cli import main
from nvidia_converge.audit import _parse_dpkg_packages
from nvidia_converge.desired import load_desired
from nvidia_converge.doctor import diagnose
from nvidia_converge.planner import build_plan, lock_actions
from nvidia_converge.models import PackageInfo
from nvidia_converge.rollback import _rollback_commands
from test_planner import _audit


def main_tests() -> int:
    test_default_desired()
    test_yaml_desired()
    test_driver_version_branch()
    test_invalid_desired_file()
    test_apply_requires_root()
    test_version_flag()
    test_schema_command()
    test_report_has_schema_required_keys()
    test_desired_schema_mentions_bare_object()
    test_plan()
    test_cli_plan_report()
    test_install_dry_run()
    test_install_dry_run_does_not_write_rollback()
    test_package_parser_deduplicates()
    test_rollback_filters_unrelated_packages()
    test_zypper_lock_plan()
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


def test_version_flag() -> None:
    try:
        with redirect_stdout(StringIO()):
            main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0


def test_schema_command() -> None:
    out = StringIO()
    with redirect_stdout(out):
        assert main(["schema", "report"]) == 0
    schema = json.loads(out.getvalue())
    assert schema["title"] == "nvidia-converge report"


def test_report_has_schema_required_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "plan.json"
        with redirect_stdout(StringIO()):
            assert main(["plan", "--out", str(out)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        schema = json.loads(Path("schemas/report.schema.json").read_text(encoding="utf-8"))
        assert set(schema["required"]).issubset(report)


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


def test_zypper_lock_plan() -> None:
    audit = _audit()
    audit.package_manager = "zypper"
    locks = lock_actions(load_desired(None), audit)
    assert locks[0].id == "lock.zypper"


if __name__ == "__main__":
    raise SystemExit(main_tests())
