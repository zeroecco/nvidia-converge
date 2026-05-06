from nvidia_converge.desired import load_desired


def test_loads_default_desired_state():
    desired = load_desired(None)
    assert desired.role == "compute"
    assert desired.driver == "580-open"
    assert desired.cuda_compat == "13.0"
    assert desired.secure_boot == "signed"
    assert desired.container_runtime == "docker"
    assert desired.fabric_manager is True
    assert desired.mig == "disabled"
    assert desired.kernel_policy == "pin-compatible"


def test_loads_simple_yaml(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
---
desired:
  role: compute
  driver: 580-open
  cuda_compat: 13.0
  secure_boot: signed
  container_runtime: docker
  fabric_manager: true
  mig: disabled
  kernel_policy: pin-compatible
""",
        encoding="utf-8",
    )
    desired = load_desired(str(path))
    assert desired.driver_major == "580"
    assert desired.open_kernel_module is True
    assert desired.fabric_manager is True


def test_loads_yaml_with_document_end_marker(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
---
desired:
  driver: 595.71.05
  fabric_manager: true
...
""",
        encoding="utf-8",
    )
    desired = load_desired(str(path))
    assert desired.driver == "595.71.05"
    assert desired.driver_major == "595"
    assert desired.fabric_manager is True
