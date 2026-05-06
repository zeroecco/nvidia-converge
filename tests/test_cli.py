import json

from nvidia_converge.cli import main


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


def test_install_is_dry_run_without_apply(tmp_path):
    out = tmp_path / "install.json"
    rc = main(["install", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc in {0, 2}
    skipped = [result for result in report["command_results"] if result.get("skipped")]
    assert skipped
    assert all(result.get("reason") == "dry-run" for result in skipped)
