from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DesiredState


DEFAULT_DESIRED = DesiredState()


def load_desired(path: str | None) -> DesiredState:
    if not path:
        return DEFAULT_DESIRED
    text = Path(path).read_text(encoding="utf-8")
    data = _parse_structured(text)
    desired = data.get("desired", data)
    if not isinstance(desired, dict):
        raise ValueError("desired state must be an object")
    values = {field: desired[field] for field in DesiredState.__dataclass_fields__ if field in desired}
    return DesiredState(**values)


def _parse_structured(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by desired-state files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            raise ValueError(f"unsupported YAML line: {raw_line!r}")
        key, value = raw_line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
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
