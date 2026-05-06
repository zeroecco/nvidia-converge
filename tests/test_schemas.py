import json

import jsonschema

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
