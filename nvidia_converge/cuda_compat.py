from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from .models import CommandResult, CudaCompatibilityInfo, PackageInfo
from .runner import CommandRunner

_CUDA_COMPAT_PACKAGE_PATTERN = re.compile(r"^cuda-compat-(\d+)-(\d+)$")
CudaCompatStatus = Literal["compatible", "not-applicable", "unsupported", "unknown"]
CudaMinorCompatibilityStatus = Literal["compatible", "unsupported", "unknown"]
_CUDA_FORWARD_COMPATIBILITY: dict[str, dict[str, CudaCompatStatus]] = {
    "12.2": {
        "535": "not-applicable",
        "570": "unsupported",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "12.3": {
        "535": "compatible",
        "570": "unsupported",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "12.4": {
        "535": "compatible",
        "570": "unsupported",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "12.5": {
        "535": "compatible",
        "570": "unsupported",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "12.6": {
        "535": "compatible",
        "570": "unsupported",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "12.8": {
        "535": "compatible",
        "570": "not-applicable",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "12.9": {
        "535": "compatible",
        "570": "compatible",
        "580": "unsupported",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "13.0": {
        "535": "compatible",
        "570": "compatible",
        "580": "not-applicable",
        "590": "unsupported",
        "595": "unsupported",
        "610": "unsupported",
    },
    "13.1": {
        "535": "compatible",
        "570": "compatible",
        "580": "compatible",
        "590": "not-applicable",
        "595": "unsupported",
        "610": "unsupported",
    },
    "13.2": {
        "535": "compatible",
        "570": "compatible",
        "580": "compatible",
        "590": "compatible",
        "595": "not-applicable",
        "610": "unsupported",
    },
    "13.3": {
        "535": "compatible",
        "570": "compatible",
        "580": "compatible",
        "590": "compatible",
        "595": "compatible",
        "610": "not-applicable",
    },
}
# NVIDIA's minor-version compatibility policy applies within a CUDA major
# family.  Keep the qualified minor set tied to the more granular vendor
# forward-compatibility matrix above so a future/unknown image tag cannot be
# accepted merely because its major number happens to be recognized.
_CUDA_MINOR_COMPATIBILITY_MINIMUM_DRIVER: dict[str, int] = {
    "12": 525,
    "13": 580,
}
_CUDA_COMPAT_PROBE = """\
import ctypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"CUDA compatibility library is missing: {path}")
library = ctypes.CDLL(str(path))
library.cuInit.argtypes = [ctypes.c_uint]
library.cuInit.restype = ctypes.c_int
status = int(library.cuInit(0))
if status != 0:
    raise SystemExit(f"CUDA compatibility library cuInit failed with status {status}")
driver_version = ctypes.c_int()
library.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
library.cuDriverGetVersion.restype = ctypes.c_int
status = int(library.cuDriverGetVersion(ctypes.byref(driver_version)))
if status != 0:
    raise SystemExit(
        f"CUDA compatibility library cuDriverGetVersion failed with status {status}"
    )
print(f"CUDA compatibility driver API {driver_version.value}")
"""


def cuda_compat_package_name(version: str) -> str:
    return f"cuda-compat-{version.replace('.', '-')}"


def cuda_forward_compatibility_status(
    driver_major: str, version: str
) -> CudaCompatStatus:
    """Return NVIDIA's published forward-compatibility matrix status."""
    return _CUDA_FORWARD_COMPATIBILITY.get(version, {}).get(driver_major, "unknown")


def cuda_minor_compatibility_minimum_driver(version: str) -> int | None:
    """Return the native minor-compatibility driver floor for a known CUDA minor."""
    if version not in _CUDA_FORWARD_COMPATIBILITY:
        return None
    major, separator, _minor = version.partition(".")
    if not separator:
        return None
    return _CUDA_MINOR_COMPATIBILITY_MINIMUM_DRIVER.get(major)


def cuda_minor_compatibility_status(
    driver_major: str, version: str
) -> CudaMinorCompatibilityStatus:
    """Classify native CUDA minor compatibility without forward-compat libraries."""
    minimum = cuda_minor_compatibility_minimum_driver(version)
    if minimum is None or not driver_major.isdecimal():
        return "unknown"
    return "compatible" if int(driver_major) >= minimum else "unsupported"


def cuda_compat_library_path(version: str) -> Path:
    return Path(f"/usr/local/cuda-{version}/compat/libcuda.so.1")


def cuda_compat_version_from_package(name: str) -> str | None:
    match = _CUDA_COMPAT_PACKAGE_PATTERN.fullmatch(name)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def find_cuda_compat_package(
    version: str, packages: list[PackageInfo]
) -> PackageInfo | None:
    expected_name = cuda_compat_package_name(version)
    return next(
        (
            package
            for package in packages
            if package.installed and package.name == expected_name
        ),
        None,
    )


def find_cuda_compatibility(
    version: str, observations: list[CudaCompatibilityInfo]
) -> CudaCompatibilityInfo | None:
    return next(
        (observation for observation in observations if observation.version == version),
        None,
    )


def probe_cuda_compat_library(version: str, runner: CommandRunner) -> CommandResult:
    path = cuda_compat_library_path(version)
    return runner.run(
        ["python3", "-I", "-c", _CUDA_COMPAT_PROBE, str(path)],
        allow_fail=True,
    )
