import re
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser

# Tags that become their own block. "para" covers plain paragraphs AND fact
# labels (class="fact") - the class attribute is what distinguishes them,
# not the tag name. "headnote" and "dbs_members" (counsel) render as their
# own block types so the frontend can style them distinctly (italic
# headnote, bold fact label) instead of one flat paragraph stream.
_BLOCK_TAGS = {"para", "headnote", "dbs_members"}
_TAG_RE = re.compile(r"<[^>]+>")
_HTML_BLOCK_TAGS = {"p", "div", "li"}
_ITALIC_TAGS = {"i", "em"}
# db_heading/db_subheading/dbs_act/dbs_section label the legal topic and
# statute a headnote concerns; db_counsela/db_counselr name counsel. None
# carry their own <b> in the source XML, but they read as case-report
# structure rather than body text, so they're bolded like poc's XmlNode does.
_BOLD_TAGS = {"b", "strong", "db_heading", "db_subheading", "dbs_act", "dbs_section", "db_counsela", "db_counselr"}


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
    like `<font>`/`<span>`/`<b>` around paragraph text). Legacy documents
    lose inline bold/italic granularity - they're flattened to one text
    span, same as before this module tracked formatting at all."""

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
            self.blocks.append({"type": "paragraph", "spans": [_text_span(text)]})
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
    document is still readable, just without paragraph/span/link structure."""
    text = re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", xml))).strip()
    if not text:
        return []
    return [{"type": "paragraph", "spans": [_text_span(text)]}]


def _text_span(text: str, bold: bool = False, italic: bool = False) -> dict:
    return {"type": "text", "text": text, "bold": bold, "italic": italic}


def _walk_spans(element: ET.Element, bold: bool, italic: bool) -> list[dict]:
    """Recursively builds an ordered list of spans from an element's mixed
    content, tracking which <i>/<b> ancestors are currently active so
    formatting survives arbitrary nesting (e.g. bold text inside an
    italicized citation). `<link href="...">` becomes its own span type
    regardless of surrounding formatting - poc's corpus never combines
    link with bold/italic, so this doesn't need to track both at once."""
    spans: list[dict] = []
    if element.text:
        spans.append(_text_span(element.text, bold, italic))
    for child in element:
        tag = child.tag
        if tag == "link" and "href" in child.attrib:
            text = "".join(child.itertext())
            if text:
                spans.append({"type": "link", "text": text, "doc_id": child.attrib["href"]})
        elif tag in _ITALIC_TAGS:
            spans.extend(_walk_spans(child, bold, True))
        elif tag in _BOLD_TAGS:
            spans.extend(_walk_spans(child, True, italic))
        else:
            spans.extend(_walk_spans(child, bold, italic))
        if child.tail:
            spans.append(_text_span(child.tail, bold, italic))
    return spans


def _block_type(element: ET.Element) -> str:
    if element.tag == "headnote":
        return "headnote"
    if element.tag == "dbs_members":
        return "counsel"
    if element.tag == "para" and element.attrib.get("class") == "fact":
        return "fact_label"
    return "paragraph"


def parse_fullcontent(xml: str) -> list[dict]:
    """Parse the ES `fullcontent` field (case document XML, or legacy XHTML
    for older documents) into an ordered list of typed content blocks, safe
    to render directly without ever passing raw markup to the browser.
    Each block is `{type, spans}` where every span is either
    `{type: "text", text, bold, italic}` or `{type: "link", text, doc_id}`."""
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
        spans = _walk_spans(element, False, False)
        if not any(span.get("text", "").strip() for span in spans if span["type"] == "text") and not any(
            span["type"] == "link" for span in spans
        ):
            continue
        blocks.append({"type": _block_type(element), "spans": spans})
    return blocks
