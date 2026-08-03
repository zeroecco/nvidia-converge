import pytest

from nvidia_converge.cuda_compat import cuda_minor_compatibility_status
from nvidia_converge.desired import (
    DesiredConfigError,
    container_cuda_full_version,
    container_cuda_version,
    load_desired,
)
from nvidia_converge.models import DesiredState

_DIGEST = "a" * 64


def _cuda_image(version: str, *, flavor: str = "devel") -> str:
    return f"nvidia/cuda:{version}-{flavor}-ubuntu22.04@sha256:{_DIGEST}"


def test_loads_default_desired_state():
    desired = load_desired(None)
    assert desired.role == "compute"
    assert desired.driver == "580-open"
    assert desired.cuda_compat == "none"
    assert desired.secure_boot == "signed"
    assert desired.container_runtime == "docker"
    assert desired.container_test_image.startswith(
        "nvidia/cuda:13.1.2-devel-ubuntu22.04@sha256:"
    )
    assert desired.fabric_manager is False
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
  cuda_compat: none
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


@pytest.mark.parametrize("quote", ["'", '"'])
def test_loads_yaml_scalar_with_matching_outer_quotes(tmp_path, quote):
    path = tmp_path / "desired.yaml"
    path.write_text(
        f"desired:\n  driver: {quote}595.71.05{quote} # exact version\n",
        encoding="utf-8",
    )

    desired = load_desired(str(path))

    assert desired.driver == "595.71.05"


@pytest.mark.parametrize(
    "scalar",
    [
        "'580-open",
        '580-open"',
        "'580-open\"",
        '\"580-open\'',
        "580-'open'",
        "'\"580-open\"'",
        '\"\'580-open\'\"',
    ],
)
def test_rejects_unbalanced_or_mixed_yaml_scalar_quoting(tmp_path, scalar):
    path = tmp_path / "desired.yaml"
    path.write_text(f"desired:\n  driver: {scalar}\n", encoding="utf-8")

    with pytest.raises(DesiredConfigError, match=r"YAML scalar.*quot"):
        load_desired(str(path))


def test_rejects_non_none_cuda_compat_until_loader_deployment_is_modeled(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        "desired:\n  cuda_compat: 13.1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DesiredConfigError,
        match="reversible CUDA compatibility-library deployment mode",
    ):
        load_desired(str(path))


@pytest.mark.parametrize(
    ("driver", "cuda_version", "expected"),
    [
        ("535", "12.2", "compatible"),
        ("570", "12.9", "compatible"),
        ("580", "13.1", "compatible"),
        ("535", "13.3", "unsupported"),
        ("580", "99.9", "unknown"),
    ],
)
def test_native_cuda_minor_compatibility_matrix(driver, cuda_version, expected):
    assert cuda_minor_compatibility_status(driver, cuda_version) == expected


def test_accepts_native_r580_cuda_13_1_pairing(tmp_path):
    path = tmp_path / "desired.yaml"
    image = _cuda_image("13.1.2")
    path.write_text(
        f"desired:\n  driver: 580-open\n  container_test_image: {image}\n",
        encoding="utf-8",
    )

    desired = load_desired(str(path))

    assert desired.container_test_image == image
    assert container_cuda_version(image) == "13.1"
    assert container_cuda_full_version(image) == "13.1.2"


def test_rejects_r535_cuda_13_3_without_forward_compatibility(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        "desired:\n"
        "  driver: 535-open\n"
        f"  container_test_image: {_cuda_image('13.3.0')}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DesiredConfigError,
        match=r"branch 535 cannot run CUDA 13\.3 without forward compatibility",
    ):
        load_desired(str(path))


def test_rejects_unknown_cuda_image_minor(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        "desired:\n"
        "  driver: 580-open\n"
        f"  container_test_image: {_cuda_image('99.9.0')}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DesiredConfigError,
        match=r"CUDA 99\.9 is not in the qualified NVIDIA compatibility matrix",
    ):
        load_desired(str(path))


def test_rejects_container_image_without_audited_probe_toolchain(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        f"desired:\n  container_test_image: {_cuda_image('13.1.2', flavor='base')}\n",
        encoding="utf-8",
    )

    with pytest.raises(DesiredConfigError, match="CUDA devel image"):
        load_desired(str(path))


def test_rejects_container_image_without_full_cuda_version(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        f"desired:\n  container_test_image: {_cuda_image('13.1')}\n",
        encoding="utf-8",
    )

    with pytest.raises(DesiredConfigError, match="major.minor.patch"):
        load_desired(str(path))


def test_loads_yaml_with_document_end_marker(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
---
desired:
  driver: 595.71.05
  cuda_compat: none
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


def test_driver_branch_rejects_longer_numeric_prefix():
    desired = DesiredState(driver="580-open")
    assert desired.matches_driver_version("5800.1") is False


def test_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "desired.json"
    path.write_text(
        '{"desired":{"driver":"580-open","driver":"595.71.05"}}', encoding="utf-8"
    )

    with pytest.raises(DesiredConfigError, match="duplicate JSON key: 'driver'"):
        load_desired(str(path))


def test_rejects_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text(
        """
desired:
  driver: 580-open
  driver: 595.71.05
""",
        encoding="utf-8",
    )

    with pytest.raises(
        DesiredConfigError, match="duplicate or empty YAML key: 'driver'"
    ):
        load_desired(str(path))


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        (
            "desired.json",
            '{"desired":{"driver":"580-open"},"metadata":{"owner":"ops"}}',
        ),
        ("desired.yaml", "desired:\n  driver: 580-open\nmetadata:\n  owner: ops\n"),
    ],
)
def test_rejects_siblings_of_desired_wrapper(tmp_path, filename, contents):
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(
        DesiredConfigError, match="wrapper contains unexpected field.*metadata"
    ):
        load_desired(str(path))


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


@pytest.mark.parametrize(
    "content",
    [
        "desired:\n  mig: enabled\n",
        "desired:\n  mig: disabled\n  mig_profile: full\n",
        "desired:\n  mig: enabled\n  mig_profile: none\n",
    ],
)
def test_mig_mode_requires_its_matching_lifecycle_profile(tmp_path, content):
    path = tmp_path / "desired.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DesiredConfigError, match="mig_profile"):
        load_desired(str(path))


def test_rejects_ambiguous_unsigned_secure_boot_policy(tmp_path):
    path = tmp_path / "desired.yaml"
    path.write_text("desired:\n  secure_boot: unsigned\n", encoding="utf-8")

    with pytest.raises(DesiredConfigError, match="secure_boot"):
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
