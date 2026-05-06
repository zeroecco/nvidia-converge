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
from nvidia_converge.desired import load_desired
from nvidia_converge.doctor import diagnose
from nvidia_converge.planner import build_plan, lock_actions
from test_planner import _audit


def main_tests() -> int:
    test_default_desired()
    test_yaml_desired()
    test_driver_version_branch()
    test_invalid_desired_file()
    test_apply_requires_root()
    test_version_flag()
    test_plan()
    test_cli_plan_report()
    test_install_dry_run()
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


if __name__ == "__main__":
    raise SystemExit(main_tests())
