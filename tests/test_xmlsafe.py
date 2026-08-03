import pytest

from nvidia_converge.xmlsafe import SafeXmlError, parse_bounded_xml


def test_bounded_xml_accepts_a_small_document():
    root = parse_bounded_xml("<stream><item name='ok'/></stream>")

    assert root.tag == "stream"
    assert root.find("item") is not None


@pytest.mark.parametrize(
    "payload",
    [
        "<!DOCTYPE stream [<!ENTITY x 'expanded'>]><stream>&x;</stream>",
        "<!entity x 'expanded'><stream/>",
    ],
)
def test_bounded_xml_rejects_dtds_and_entity_declarations(payload):
    with pytest.raises(SafeXmlError, match="forbidden"):
        parse_bounded_xml(payload)


def test_bounded_xml_rejects_size_element_and_depth_exhaustion():
    with pytest.raises(SafeXmlError, match="size"):
        parse_bounded_xml("<root>oversized</root>", max_bytes=8)
    with pytest.raises(SafeXmlError, match="element"):
        parse_bounded_xml("<root><a/><b/></root>", max_elements=2)
    with pytest.raises(SafeXmlError, match="depth"):
        parse_bounded_xml("<root><a><b/></a></root>", max_depth=2)


@pytest.mark.parametrize("payload", ["", "<root>", "text", "<a/><b/>"])
def test_bounded_xml_rejects_malformed_documents(payload):
    with pytest.raises(SafeXmlError):
        parse_bounded_xml(payload)


def test_bounded_xml_rejects_nonpositive_limits():
    with pytest.raises(ValueError, match="positive"):
        parse_bounded_xml("<root/>", max_depth=0)
