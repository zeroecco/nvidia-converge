from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import cast

MAX_XML_BYTES = 1024 * 1024
MAX_XML_ELEMENTS = 50_000
MAX_XML_DEPTH = 256
_FORBIDDEN_DECLARATIONS = (b"<!DOCTYPE", b"<!ENTITY")


class SafeXmlError(ValueError):
    pass


def parse_bounded_xml(
    text: str,
    *,
    max_bytes: int = MAX_XML_BYTES,
    max_elements: int = MAX_XML_ELEMENTS,
    max_depth: int = MAX_XML_DEPTH,
) -> ET.Element[str]:
    """Parse bounded command XML without DTDs or entity declarations."""

    if max_bytes <= 0 or max_elements <= 0 or max_depth <= 0:
        raise ValueError("XML safety limits must be positive")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SafeXmlError("XML contains invalid Unicode") from exc
    if len(encoded) > max_bytes:
        raise SafeXmlError("XML exceeds the safety size limit")
    upper = encoded.upper()
    if any(declaration in upper for declaration in _FORBIDDEN_DECLARATIONS):
        raise SafeXmlError("XML DTDs and entity declarations are forbidden")

    parser: ET.XMLPullParser[ET.Element[str]] = ET.XMLPullParser(
        events=("start", "end")
    )
    root: ET.Element[str] | None = None
    element_count = 0
    depth = 0

    def consume_events() -> None:
        nonlocal root, element_count, depth
        events = cast(
            "Iterator[tuple[str, ET.Element[str]]]",
            parser.read_events(),
        )
        for event, element in events:
            if event == "start":
                element_count += 1
                depth += 1
                if element_count > max_elements:
                    raise SafeXmlError("XML exceeds the element safety limit")
                if depth > max_depth:
                    raise SafeXmlError("XML exceeds the depth safety limit")
                if root is None:
                    root = element
            else:
                depth -= 1
                if depth < 0:
                    raise SafeXmlError("XML element nesting is invalid")

    try:
        for offset in range(0, len(text), 64 * 1024):
            parser.feed(text[offset : offset + 64 * 1024])
            consume_events()
        parser.close()
        consume_events()
    except (ET.ParseError, RecursionError) as exc:
        raise SafeXmlError("XML is malformed") from exc
    if root is None or depth != 0:
        raise SafeXmlError("XML has no complete root element")
    return root
