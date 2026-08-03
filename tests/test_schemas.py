import json
from hashlib import sha256
from importlib import resources
from pathlib import Path

import jsonschema
import pytest

import nvidia_converge
from nvidia_converge.cli import main
from nvidia_converge.models import FileSnapshot
from nvidia_converge.schemas import load_schema, strict_format_checker
from nvidia_converge.verify import _CUDA_DRIVER_PROBE_SHA256


def _validate(instance, schema):
    jsonschema.validate(
        instance,
        schema,
        format_checker=strict_format_checker(),
    )


def test_report_schema_validates_plan_report(tmp_path):
    out = tmp_path / "plan.json"
    assert main(["plan", "--out", str(out)]) in {0, 2}
    report = json.loads(out.read_text(encoding="utf-8"))
    _validate(report, load_schema("report"))


def test_report_schema_validates_all_command_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("nvidia_converge.rollback.SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr("nvidia_converge.rollback.nvidia_module_unload_order", list)
    monkeypatch.setattr(
        "nvidia_converge.rollback._capture_managed_files",
        lambda _package_manager: [
            FileSnapshot("/etc/docker/daemon.json", False, None, None)
        ],
    )
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
        _validate(json.loads(out.read_text(encoding="utf-8")), schema)

    snapshot_path = tmp_path / "rollback-snapshot.json"
    from test_rollback import _snapshot_document

    snapshot_path.write_text(
        json.dumps(_snapshot_document(snapshot_path)),
        encoding="utf-8",
    )
    rollback_out = tmp_path / "rollback.json"
    main(["rollback", "--snapshot", str(snapshot_path), "--out", str(rollback_out)])
    _validate(json.loads(rollback_out.read_text(encoding="utf-8")), schema)


def test_validation_schema_validates_validate_output(tmp_path):
    out = tmp_path / "validation.json"
    assert (
        main(
            [
                "validate",
                "--desired",
                "examples/compute-580-open.yaml",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    validation = json.loads(out.read_text(encoding="utf-8"))
    _validate(validation, load_schema("validation"))


def test_desired_schema_accepts_example_config():
    desired = json.loads(
        """
{
  "desired": {
    "role": "compute",
    "driver": "580-open",
    "cuda_compat": "none",
    "secure_boot": "signed",
    "container_runtime": "docker",
    "fabric_manager": true,
    "mig": "disabled",
    "kernel_policy": "pin-compatible"
  }
}
"""
    )
    _validate(desired, load_schema("desired"))


def test_desired_schema_accepts_bare_desired_object():
    desired = {
        "role": "compute",
        "driver": "595.71.05",
        "cuda_compat": "none",
        "secure_boot": "signed",
        "container_runtime": "docker",
        "fabric_manager": True,
        "mig": "disabled",
        "kernel_policy": "pin-compatible",
    }
    _validate(desired, load_schema("desired"))


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
        _validate(desired, load_schema("desired"))


def test_desired_schema_rejects_non_devel_container_image():
    desired = {
        "container_test_image": (
            "nvidia/cuda:13.1.2-base-ubuntu22.04@sha256:" + "a" * 64
        )
    }

    with pytest.raises(jsonschema.ValidationError):
        _validate(desired, load_schema("desired"))


def test_integration_results_example_validates():
    with open("integrations/results.example.json", encoding="utf-8") as handle:
        results = json.load(handle)
    _validate(results, load_schema("integration-results"))


@pytest.mark.parametrize(
    "value",
    (
        "not-a-date",
        "2026-02-30T12:00:00Z",
        "2026-08-02 12:00:00Z",
        "2026-08-02T12:00:00",
    ),
)
def test_strict_format_checker_rejects_invalid_date_time(value):
    with pytest.raises(jsonschema.ValidationError):
        _validate(
            value,
            {"type": "string", "format": "date-time"},
        )


@pytest.mark.parametrize(
    "value",
    (
        "https://github.com/has space",
        "https://github.com/%zz",
        "https://[invalid",
        "//github.com/no-scheme",
    ),
)
def test_strict_format_checker_rejects_invalid_uri(value):
    with pytest.raises(jsonschema.ValidationError):
        _validate(value, {"type": "string", "format": "uri"})


def test_strict_format_checker_accepts_formats_used_by_evidence():
    _validate(
        "2026-08-02T12:34:56.123456+00:00",
        {"type": "string", "format": "date-time"},
    )
    _validate(
        "https://api.github.com/repos/zeroecco/nvidia-converge/actions/artifacts/1/zip",
        {"type": "string", "format": "uri"},
    )


@pytest.mark.parametrize(
    "name",
    ("desired", "integration-results", "report", "validation"),
)
def test_packaged_schemas_match_authoritative_schemas(name):
    assert Path(f"schemas/{name}.schema.json").read_bytes() == Path(
        f"nvidia_converge/schemas/{name}.schema.json"
    ).read_bytes()


def test_integration_schema_does_not_accept_family_support_booleans():
    with open("integrations/results.example.json", encoding="utf-8") as handle:
        results = json.load(handle)
    results["required_coverage"]["rhel_family"] = False

    with pytest.raises(jsonschema.ValidationError):
        _validate(results, load_schema("integration-results"))


def test_package_includes_pep561_marker():
    assert resources.files(nvidia_converge).joinpath("py.typed").is_file()


def test_package_includes_integrity_bound_cuda_driver_probe_source():
    probe = resources.files(nvidia_converge).joinpath("probes/cuda_driver_probe.c")

    assert probe.is_file()
    assert sha256(probe.read_bytes()).hexdigest() == _CUDA_DRIVER_PROBE_SHA256
