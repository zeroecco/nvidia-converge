import json
from importlib import resources

import jsonschema

import nvidia_converge
from nvidia_converge.cli import main
from nvidia_converge.schemas import load_schema


def test_report_schema_validates_plan_report(tmp_path):
    out = tmp_path / "plan.json"
    assert main(["plan", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(report, load_schema("report"))


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


def test_integration_results_example_validates():
    with open("integrations/results.example.json", encoding="utf-8") as handle:
        results = json.load(handle)
    jsonschema.validate(results, load_schema("integration-results"))


def test_package_includes_pep561_marker():
    assert resources.files(nvidia_converge).joinpath("py.typed").is_file()
