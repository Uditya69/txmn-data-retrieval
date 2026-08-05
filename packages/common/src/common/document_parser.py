import re
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser

_BLOCK_TAGS = {"para", "headnote"}
_TAG_RE = re.compile(r"<[^>]+>")
_HTML_BLOCK_TAGS = {"p", "div", "li"}


def _is_legacy_html(content: str) -> bool:
    """Older indexed documents store `fullcontent` as a full XHTML page
    (`<!DOCTYPE html>...<p>...`) instead of the newer custom XML schema
    (`<document><body><para>...`). XHTML's unescaped void elements
    (`<meta>`, `<br>`, ...) aren't well-formed XML, so it must be routed to
    the lenient HTML parser rather than the strict one."""
    head = content.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


class _HTMLBlockExtractor(HTMLParser):
    """Collects the text inside top-level block tags (a case's legacy HTML
    has no `<para>`/`<link>` citation markup - just plain formatting tags
    like `<font>`/`<span>`/`<b>` around paragraph text)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self._buffer: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _HTML_BLOCK_TAGS:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in _HTML_BLOCK_TAGS and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                self._flush()

    def handle_data(self, data):
        if self._depth > 0:
            self._buffer.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        if text:
            self.blocks.append({"type": "paragraph", "text": text, "links": []})
        self._buffer = []


def _parse_legacy_html(html: str) -> list[dict]:
    parser = _HTMLBlockExtractor()
    parser.feed(html)
    parser.close()
    return parser.blocks


def strip_tags_fallback(xml: str) -> list[dict]:
    """Best-effort text extraction for fullcontent that fails both the
    strict XML and lenient HTML parsers (a genuinely mismatched tag at the
    source) - strips all markup and returns it as a single block so the
    document is still readable, just without paragraph/link structure."""
    text = re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", xml))).strip()
    if not text:
        return []
    return [{"type": "paragraph", "text": text, "links": []}]


def parse_fullcontent(xml: str) -> list[dict]:
    """Parse the ES `fullcontent` field (case document XML, or legacy XHTML
    for older documents) into an ordered list of typed content blocks, safe
    to render directly without ever passing raw markup to the browser."""
    if _is_legacy_html(xml):
        return _parse_legacy_html(xml)

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
