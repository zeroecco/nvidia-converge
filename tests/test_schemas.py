import json
from importlib import resources

import jsonschema
import pytest

import nvidia_converge
from nvidia_converge.cli import main
from nvidia_converge.schemas import load_schema


def test_report_schema_validates_plan_report(tmp_path):
    out = tmp_path / "plan.json"
    assert main(["plan", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(report, load_schema("report"))


def test_report_schema_validates_all_command_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("nvidia_converge.rollback.SNAPSHOT_DIR", tmp_path / "snapshots")
    schema = load_schema("report")
    commands = {
        "doctor": ["doctor"],
        "plan": ["plan"],
        "install": ["install"],
        "verify": ["verify"],
        "lock": ["lock"],
        "snapshot": ["snapshot"],
    }
    for name, command in commands.items():
        out = tmp_path / f"{name}.json"
        main([*command, "--out", str(out)])
        jsonschema.validate(json.loads(out.read_text(encoding="utf-8")), schema)

    snapshot = {
        "path": str(tmp_path / "snapshot.json"),
        "packages": [{"name": "nvidia-driver-595", "version": "595.71.05-1", "manager": "apt", "installed": True}],
        "kernel": "6.8.0-111-generic",
        "module_version": "595.71.05",
        "commands": [["apt-get", "install", "-y", "nvidia-driver-595=595.71.05-1"]],
    }
    snapshot_path = tmp_path / "rollback-snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    rollback_out = tmp_path / "rollback.json"
    main(["rollback", "--snapshot", str(snapshot_path), "--out", str(rollback_out)])
    jsonschema.validate(json.loads(rollback_out.read_text(encoding="utf-8")), schema)


def test_validation_schema_validates_validate_output(tmp_path):
    out = tmp_path / "validation.json"
    assert main(["validate", "--desired", "examples/compute-580-open.yaml", "--out", str(out)]) == 0
    validation = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(validation, load_schema("validation"))


def test_desired_schema_accepts_example_config():
    desired = json.loads(
        """
{
  "desired": {
    "role": "compute",
    "driver": "580-open",
    "cuda_compat": "13.0",
    "secure_boot": "signed",
    "container_runtime": "docker",
    "fabric_manager": true,
    "mig": "disabled",
    "kernel_policy": "pin-compatible"
  }
}
"""
    )
    jsonschema.validate(desired, load_schema("desired"))


def test_desired_schema_accepts_bare_desired_object():
    desired = {
        "role": "compute",
        "driver": "595.71.05",
        "cuda_compat": "13.0",
        "secure_boot": "signed",
        "container_runtime": "docker",
        "fabric_manager": True,
        "mig": "disabled",
        "kernel_policy": "pin-compatible",
    }
    jsonschema.validate(desired, load_schema("desired"))


def test_desired_schema_rejects_unsupported_values():
    desired = {
        "role": "compute",
        "driver": "latest",
        "cuda_compat": "13",
        "secure_boot": "signed",
        "container_runtime": "dockre",
        "fabric_manager": True,
        "mig": "disabledd",
        "kernel_policy": "pin-compatible",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(desired, load_schema("desired"))


def test_integration_results_example_validates():
    with open("integrations/results.example.json", encoding="utf-8") as handle:
        results = json.load(handle)
    jsonschema.validate(results, load_schema("integration-results"))


def test_package_includes_pep561_marker():
    assert resources.files(nvidia_converge).joinpath("py.typed").is_file()
