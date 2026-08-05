import xml.etree.ElementTree as ET

_BLOCK_TAGS = {"para", "headnote"}


def parse_fullcontent(xml: str) -> list[dict]:
    """Parse the ES `fullcontent` field (case document XML) into an ordered
    list of typed content blocks, safe to render directly without ever
    passing raw markup to the browser."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed document XML: {exc}") from exc

    blocks = []
    for element in root.iter():
        if element.tag not in _BLOCK_TAGS:
            continue
        text = "".join(element.itertext()).strip()
        if not text:
            continue
        links = [
            {"text": "".join(link.itertext()).strip(), "doc_id": link.attrib["href"]}
            for link in element.iter("link")
            if "href" in link.attrib
        ]
        blocks.append({"type": "paragraph", "text": text, "links": links})
    return blocks
