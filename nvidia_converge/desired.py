from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from .models import DesiredState


DEFAULT_DESIRED = DesiredState()
_DESIRED_FIELDS = set(DesiredState.__dataclass_fields__)
_ALLOWED_VALUES = {
    "role": {"compute"},
    "secure_boot": {"signed", "unsigned", "disabled"},
    "container_runtime": {"docker"},
    "mig": {"disabled", "enabled"},
    "kernel_policy": {"pin-compatible"},
}
_DRIVER_PATTERN = re.compile(r"^\d+(?:-open|\.\d+(?:\.\d+)?)?$")
_CUDA_COMPAT_PATTERN = re.compile(r"^\d+\.\d+$")


class DesiredConfigError(ValueError):
    pass


def load_desired(path: str | None) -> DesiredState:
    if not path:
        return DEFAULT_DESIRED
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DesiredConfigError(f"cannot read desired-state file {path!r}: {exc.strerror}") from exc
    try:
        data = _parse_structured(text)
    except json.JSONDecodeError as exc:
        raise DesiredConfigError(f"invalid JSON in {path!r}: line {exc.lineno}: {exc.msg}") from exc
    except ValueError as exc:
        raise DesiredConfigError(f"invalid desired-state file {path!r}: {exc}") from exc
    desired = data.get("desired", data)
    if not isinstance(desired, dict):
        raise DesiredConfigError("desired state must be an object")
    unknown = sorted(set(desired) - _DESIRED_FIELDS)
    if unknown:
        raise DesiredConfigError(f"unknown desired-state field(s): {', '.join(unknown)}")
    values = {field: desired[field] for field in _DESIRED_FIELDS if field in desired}
    return _build_desired(values)


def _parse_structured(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return cast(dict[str, Any], json.loads(text))
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
        value = _strip_inline_comment(value.strip())
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _coerce_scalar(value: str) -> Any:
    value = value.strip("'\"")
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
            raise DesiredConfigError(f"desired.{field} must be one of: {', '.join(sorted(allowed))}")
    if not _DRIVER_PATTERN.match(desired.driver):
        raise DesiredConfigError("desired.driver must be a driver branch like 580-open or a version like 595.71.05")
    if not _CUDA_COMPAT_PATTERN.match(desired.cuda_compat):
        raise DesiredConfigError("desired.cuda_compat must be a major.minor version like 13.0")
