from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .cuda_compat import (
    CudaMinorCompatibilityStatus,
    cuda_minor_compatibility_minimum_driver,
    cuda_minor_compatibility_status,
)
from .files import BoundedFileError, read_bounded_utf8
from .models import DesiredState

DEFAULT_DESIRED = DesiredState()
_DESIRED_FIELDS = set(DesiredState.__dataclass_fields__)
_ALLOWED_VALUES = {
    "role": {"compute"},
    "secure_boot": {"signed", "disabled"},
    "container_runtime": {"docker"},
    "mig": {"disabled", "enabled"},
    "mig_profile": {"none", "full"},
    "kernel_policy": {"pin-compatible"},
}
_DRIVER_PATTERN = re.compile(r"^\d+(?:-open|\.\d+(?:\.\d+)?)?$")
_CUDA_COMPAT_PATTERN = re.compile(r"^(?:none|\d+\.\d+)$")
_CONTAINER_IMAGE_PATTERN = re.compile(
    r"^nvidia/cuda:[A-Za-z0-9][A-Za-z0-9._-]{0,127}@sha256:[a-f0-9]{64}$"
)
_CONTAINER_CUDA_TAG_PATTERN = re.compile(r"^(?P<version>\d+\.\d+\.\d+)(?:-|$)")
_CONTAINER_DEVEL_TAG_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:-[A-Za-z0-9._]+)*-devel(?:-[A-Za-z0-9._]+)*$"
)
MAX_DESIRED_BYTES = 64 * 1024


class DesiredConfigError(ValueError):
    pass


def load_desired(
    path: str | None, *, require_root_controlled: bool = False
) -> DesiredState:
    if not path:
        return DEFAULT_DESIRED
    input_path = Path(os.path.abspath(path)) if require_root_controlled else Path(path)
    try:
        text = read_bounded_utf8(
            input_path,
            max_bytes=MAX_DESIRED_BYTES,
            require_root_controlled=require_root_controlled,
            require_trusted_ancestors=require_root_controlled,
        )
    except OSError as exc:
        raise DesiredConfigError(
            f"cannot read desired-state file {path!r}: {exc.strerror}"
        ) from exc
    except BoundedFileError as exc:
        raise DesiredConfigError(
            f"cannot read desired-state file {path!r}: {exc}"
        ) from exc
    try:
        data = _parse_structured(text)
    except json.JSONDecodeError as exc:
        raise DesiredConfigError(
            f"invalid JSON in {path!r}: line {exc.lineno}: {exc.msg}"
        ) from exc
    except RecursionError as exc:
        raise DesiredConfigError(
            f"invalid desired-state file {path!r}: structure is too deeply nested"
        ) from exc
    except ValueError as exc:
        raise DesiredConfigError(f"invalid desired-state file {path!r}: {exc}") from exc
    if "desired" in data and set(data) != {"desired"}:
        extras = sorted(set(data) - {"desired"})
        raise DesiredConfigError(
            f"desired-state wrapper contains unexpected field(s): {', '.join(extras)}"
        )
    desired = data.get("desired", data)
    if not isinstance(desired, dict):
        raise DesiredConfigError("desired state must be an object")
    unknown = sorted(set(desired) - _DESIRED_FIELDS)
    if unknown:
        raise DesiredConfigError(
            f"unknown desired-state field(s): {', '.join(unknown)}"
        )
    values = {field: desired[field] for field in _DESIRED_FIELDS if field in desired}
    return _build_desired(values)


def _parse_structured(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        parsed = json.loads(text, object_pairs_hook=_object_without_duplicates)
        if not isinstance(parsed, dict):
            raise ValueError("desired-state JSON must be an object")
        return parsed
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by desired-state files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped in {"---", "..."}:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            raise ValueError(f"unsupported YAML line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not key or key in parent:
            label = key or "<empty>"
            raise ValueError(f"duplicate or empty YAML key: {label!r}")
        value = _strip_inline_comment(value.strip())
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _coerce_scalar(value: str) -> Any:
    starts_quoted = value.startswith(("'", '"'))
    ends_quoted = value.endswith(("'", '"'))
    if starts_quoted or ends_quoted:
        if not starts_quoted or not ends_quoted or value[0] != value[-1]:
            raise ValueError(
                f"YAML scalar has unbalanced or mixed quoting: {value!r}"
            )
        if "'" in value[1:-1] or '"' in value[1:-1]:
            raise ValueError(
                "YAML scalar uses unsupported nested or escaped quoting: "
                f"{value!r}"
            )
        return value[1:-1]
    if "'" in value or '"' in value:
        raise ValueError(
            f"YAML scalar quotes must form one matching outer pair: {value!r}"
        )
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    if quote is not None:
        raise ValueError(f"YAML scalar has unbalanced quoting: {value!r}")
    return value


def _build_desired(values: dict[str, Any]) -> DesiredState:
    if "fabric_manager" in values and not isinstance(values["fabric_manager"], bool):
        raise DesiredConfigError("desired.fabric_manager must be true or false")
    for field, value in values.items():
        if field != "fabric_manager" and not isinstance(value, str):
            raise DesiredConfigError(f"desired.{field} must be a string")
    desired = DesiredState(**values)
    _validate_desired(desired)
    return desired


def _validate_desired(desired: DesiredState) -> None:
    for field, allowed in _ALLOWED_VALUES.items():
        value = getattr(desired, field)
        if value not in allowed:
            raise DesiredConfigError(
                f"desired.{field} must be one of: {', '.join(sorted(allowed))}"
            )
    if not _DRIVER_PATTERN.match(desired.driver):
        raise DesiredConfigError(
            "desired.driver must be a driver branch like 580-open or a version like 595.71.05"
        )
    if not _CUDA_COMPAT_PATTERN.match(desired.cuda_compat):
        raise DesiredConfigError(
            "desired.cuda_compat must be 'none' or a major.minor version like 13.1"
        )
    if desired.cuda_compat != "none":
        raise DesiredConfigError(
            "desired.cuda_compat must be 'none' in this release because a reversible "
            "CUDA compatibility-library deployment mode is not yet modeled"
        )
    expected_mig_profile = "full" if desired.mig == "enabled" else "none"
    if desired.mig_profile != expected_mig_profile:
        raise DesiredConfigError(
            f"desired.mig_profile must be {expected_mig_profile!r} when "
            f"desired.mig is {desired.mig!r}"
        )
    if not _CONTAINER_IMAGE_PATTERN.match(desired.container_test_image):
        raise DesiredConfigError(
            "desired.container_test_image must pin an nvidia/cuda tag to a lowercase sha256 digest"
        )
    image_tag = desired.container_test_image.removeprefix("nvidia/cuda:").split("@", 1)[
        0
    ]
    cuda_version = container_cuda_version(desired.container_test_image)
    if cuda_version is None:
        raise DesiredConfigError(
            "desired.container_test_image tag must start with a CUDA "
            "major.minor.patch version"
        )
    if _CONTAINER_DEVEL_TAG_PATTERN.fullmatch(image_tag) is None:
        raise DesiredConfigError(
            "desired.container_test_image must be a digest-pinned CUDA devel image "
            "so the audited Driver API probe can be built inside the isolated container"
        )
    compatibility = container_cuda_minor_compatibility_status(desired)
    if compatibility == "unknown":
        raise DesiredConfigError(
            f"desired.container_test_image CUDA {cuda_version} is not in the "
            "qualified NVIDIA compatibility matrix"
        )
    if compatibility == "unsupported":
        minimum = cuda_minor_compatibility_minimum_driver(cuda_version)
        raise DesiredConfigError(
            f"desired.driver branch {desired.driver_major} cannot run CUDA "
            f"{cuda_version} without forward compatibility; native minor-version "
            f"compatibility requires branch {minimum} or newer while "
            "desired.cuda_compat is 'none'"
        )


def container_cuda_version(image: str) -> str | None:
    """Extract the CUDA major.minor encoded in a pinned nvidia/cuda image tag."""
    full_version = container_cuda_full_version(image)
    return full_version.rsplit(".", 1)[0] if full_version else None


def container_cuda_full_version(image: str) -> str | None:
    """Extract the complete CUDA version encoded in a pinned image tag."""
    prefix = "nvidia/cuda:"
    if not image.startswith(prefix):
        return None
    image_tag = image.removeprefix(prefix).split("@", 1)[0]
    match = _CONTAINER_CUDA_TAG_PATTERN.match(image_tag)
    return match.group("version") if match else None


def container_cuda_minor_compatibility_status(
    desired: DesiredState, *, driver_major: str | None = None
) -> CudaMinorCompatibilityStatus:
    """Classify native compatibility for the configured image and driver branch."""
    version = container_cuda_version(desired.container_test_image)
    if version is None:
        return "unknown"
    return cuda_minor_compatibility_status(
        driver_major or desired.driver_major,
        version,
    )
