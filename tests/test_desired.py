import pytest

from nvidia_converge.desired import DesiredConfigError, load_desired
from nvidia_converge.models import DesiredState


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
    assert desired.exact_driver_version is True
    assert desired.matches_driver_version("595.71.05") is True
    assert desired.matches_driver_version("595.60.01") is False
    assert desired.fabric_manager is True


def test_driver_branch_matches_major_version():
    desired = DesiredState(driver="580-open")
    assert desired.exact_driver_version is False
    assert desired.matches_driver_version("580.126.16") is True
    assert desired.matches_driver_version("595.71.05") is False


def test_rejects_unknown_desired_field(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  driver: 580-open
  typo: value
""",
        encoding="utf-8",
    )
    with pytest.raises(DesiredConfigError, match="unknown desired-state field"):
        load_desired(str(path))


def test_rejects_json_array_desired_file(tmp_path):
    path = tmp_path / "desired.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DesiredConfigError, match="JSON must be an object"):
        load_desired(str(path))


def test_rejects_json_desired_array_value(tmp_path):
    path = tmp_path / "desired.json"
    path.write_text('{"desired": []}', encoding="utf-8")
    with pytest.raises(DesiredConfigError, match="desired state must be an object"):
        load_desired(str(path))


def test_rejects_non_boolean_fabric_manager(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  fabric_manager: yes
""",
        encoding="utf-8",
    )
    with pytest.raises(DesiredConfigError, match="fabric_manager"):
        load_desired(str(path))


def test_rejects_unsupported_container_runtime(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  container_runtime: dockre
""",
        encoding="utf-8",
    )
    with pytest.raises(DesiredConfigError, match="container_runtime"):
        load_desired(str(path))


def test_rejects_unsupported_mig_mode(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  mig: disabledd
""",
        encoding="utf-8",
    )
    with pytest.raises(DesiredConfigError, match="mig"):
        load_desired(str(path))


def test_rejects_invalid_driver_format(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  driver: latest
""",
        encoding="utf-8",
    )
    with pytest.raises(DesiredConfigError, match="driver"):
        load_desired(str(path))


def test_rejects_invalid_cuda_compat_format(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  cuda_compat: thirteen
""",
        encoding="utf-8",
    )
    with pytest.raises(DesiredConfigError, match="cuda_compat"):
        load_desired(str(path))
