from pathlib import Path


def test_gpu_integration_uses_virtualenv_for_python_tooling():
    workflow = Path(".github/workflows/gpu-integration.yml").read_text(encoding="utf-8")
    assert "python3 -m pip install --user" not in workflow
    assert "python3 -m venv .venv" in workflow
    assert ".venv/bin/python -m nvidia_converge plan" in workflow
    assert '"$PWD/.venv/bin/python" -m nvidia_converge install' in workflow


def test_gpu_integration_validates_every_generated_report():
    workflow = Path(".github/workflows/gpu-integration.yml").read_text(encoding="utf-8")
    assert "Validate generated report schemas" in workflow
    assert 'Path("artifacts/reports").glob("*.json")' in workflow
    assert "jsonschema.validate(json.load(handle), schema)" in workflow
