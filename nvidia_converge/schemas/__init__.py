from __future__ import annotations

import json
import re
from datetime import datetime
from importlib import resources
from typing import Any, Literal, cast
from urllib.parse import urlsplit

SchemaName = Literal["desired", "integration-results", "report", "validation"]
_RFC3339_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)
_URI = re.compile(
    r"^[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*$"
)
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![A-Fa-f0-9]{2})")


def load_schema(name: SchemaName) -> dict[str, Any]:
    with resources.files(__name__).joinpath(f"{name}.schema.json").open(encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def schema_json(name: SchemaName) -> str:
    return json.dumps(load_schema(name), indent=2, sort_keys=True)


def strict_format_checker() -> Any:
    """Return dependency-free validators for every format used by our schemas."""

    import jsonschema  # type: ignore[import-untyped]

    checker = jsonschema.FormatChecker()
    checker.checks("date-time")(_is_rfc3339_date_time)
    checker.checks("uri")(_is_uri)
    return checker


def _is_rfc3339_date_time(value: object) -> bool:
    if not isinstance(value, str):
        return True
    if _RFC3339_DATE_TIME.fullmatch(value) is None:
        return False
    normalized = value[:10] + "T" + value[11:]
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def _is_uri(value: object) -> bool:
    if not isinstance(value, str):
        return True
    if (
        _URI.fullmatch(value) is None
        or _INVALID_PERCENT_ESCAPE.search(value) is not None
    ):
        return False
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() in {"http", "https"}:
            if not parsed.netloc or parsed.hostname is None:
                return False
            _ = parsed.port
        return bool(parsed.scheme)
    except ValueError:
        return False
