import json
import os

from nvidia_converge.cli import _install_status, main
from nvidia_converge.human import render_human
from nvidia_converge.models import CommandResult, DesiredState, Report, Verification


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("nvidia-converge ")


def test_schema_command_outputs_json(capsys):
    rc = main(["schema", "desired"])
    captured = capsys.readouterr()
    assert rc == 0
    schema = json.loads(captured.out)
    assert schema["title"] == "nvidia-converge desired state"


def test_support_command_outputs_human_summary(capsys):
    rc = main(["support"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("nvidia-converge support matrix")
    assert "apt-get" in captured.out
    assert "Known limits:" in captured.out


def test_support_json_outputs_machine_readable_matrix(capsys):
    rc = main(["support", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    support = json.loads(captured.out)
    assert support["package_managers"]["apt-get"]["install"] is True
    assert support["package_managers"]["zypper"]["lock"] is True


def test_validate_command_outputs_human_summary(capsys, tmp_path):
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
    rc = main(["validate", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("nvidia-converge validate")
    assert "Desired state: valid" in captured.out
    assert "595.71.05" in captured.out


def test_validate_json_outputs_machine_readable_payload(capsys):
    rc = main(["validate", "--desired", "examples/compute-580-open.yaml", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0"
    assert "generated_at" in payload
    assert payload["valid"] is True
    assert payload["desired"]["driver"] == "580-open"


def test_validate_writes_machine_readable_payload(capsys, tmp_path):
    out = tmp_path / "validation.json"
    rc = main(["validate", "--desired", "examples/compute-580-open.yaml", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Desired state: valid" in captured.out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert "generated_at" in payload
    assert payload["valid"] is True
    assert payload["desired"]["driver"] == "580-open"


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


def test_bad_json_desired_shape_is_clean_error(capsys, tmp_path):
    desired = tmp_path / "desired.json"
    desired.write_text("[]", encoding="utf-8")
    rc = main(["validate", "--desired", str(desired)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "JSON must be an object" in captured.err
    assert "Traceback" not in captured.err


def test_apply_requires_root_when_not_root(capsys):
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        return
    rc = main(["lock", "--apply"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "must be run as root" in captured.err


def test_read_only_commands_reject_apply(capsys):
    try:
        main(["plan", "--apply"])
    except SystemExit as exc:
        assert exc.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: --apply" in captured.err


def test_bad_rollback_snapshot_is_clean_error(capsys, tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    rc = main(["rollback", "--snapshot", str(snapshot)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_install_is_dry_run_without_apply(tmp_path):
    out = tmp_path / "install.json"
    rc = main(["install", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc in {0, 2}
    skipped = [result for result in report["command_results"] if result.get("skipped")]
    assert skipped
    assert all(result.get("reason") == "dry-run" for result in skipped)
    assert report["rollback"]["path"] is None
    assert not (tmp_path / "nvidia-converge-rollback.json").exists()


def test_install_status_fails_on_command_failure():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[CommandResult(["apt-get", "install"], 100, stderr="failed")],
        verification=[Verification("nvidia-smi", True)],
    )
    assert _install_status(report) == 2


def test_install_status_passes_when_commands_and_checks_pass():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[CommandResult(["apt-get", "install"], 0)],
        verification=[Verification("nvidia-smi", True)],
    )
    assert _install_status(report) == 0


def test_human_output_includes_failed_command_stderr():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[CommandResult(["apt-get", "install"], 100, stderr="package not found\nmore detail")],
    )
    output = render_human("install", report, apply=True)
    assert "- fail: apt-get install" in output
    assert "  package not found" in output
    assert "more detail" not in output


def test_human_output_includes_failed_command_stdout_fallback():
    report = Report(
        "1.0",
        "2026-05-06T00:00:00+00:00",
        DesiredState(),
        command_results=[CommandResult(["zypper", "install"], 4, stdout="solver failed")],
    )
    output = render_human("install", report, apply=True)
    assert "- fail: zypper install" in output
    assert "  solver failed" in output
