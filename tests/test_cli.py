import json
import os

from nvidia_converge.cli import main


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("nvidia-converge ")


def test_lock_defaults_to_human_output(capsys, tmp_path):
    desired = tmp_path / "desired.yaml"
    desired.write_text(
        """
---
desired:
  driver: 595.71.05
  cuda_compat: 13.0
  container_runtime: docker
  fabric_manager: true
  kernel_policy: pin-compatible
""",
        encoding="utf-8",
    )
    rc = main(["lock", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("nvidia-converge lock dry-run")
    assert '"audit"' not in captured.out
    assert "nvidia-driver-595 " in captured.out


def test_plan_writes_machine_readable_report(tmp_path):
    out = tmp_path / "plan.json"
    rc = main(["plan", "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert report["desired"]["driver"] == "580-open"
    assert "audit" in report
    assert "findings" in report
    assert "plan" in report
    assert "sbom" in report


def test_json_flag_prints_machine_readable_report(capsys):
    rc = main(["plan", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    report = json.loads(captured.out)
    assert report["schema_version"] == "1.0"


def test_bad_desired_file_is_clean_error(capsys, tmp_path):
    desired = tmp_path / "desired.yaml"
    desired.write_text("not yaml\n", encoding="utf-8")
    rc = main(["plan", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_apply_requires_root_when_not_root(capsys):
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        return
    rc = main(["lock", "--apply"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "must be run as root" in captured.err


def test_install_is_dry_run_without_apply(tmp_path):
    out = tmp_path / "install.json"
    rc = main(["install", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc in {0, 2}
    skipped = [result for result in report["command_results"] if result.get("skipped")]
    assert skipped
    assert all(result.get("reason") == "dry-run" for result in skipped)
