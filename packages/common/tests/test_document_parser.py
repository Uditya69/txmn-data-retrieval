import pytest

from common.document_parser import parse_fullcontent


def test_parse_fullcontent_returns_empty_list_for_no_content_blocks():
    xml = "<document><body></body></document>"
    assert parse_fullcontent(xml) == []


def test_parse_fullcontent_extracts_plain_paragraph_text():
    xml = "<document><body><para>Plain paragraph text.</para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{"type": "paragraph", "text": "Plain paragraph text.", "links": []}]


def test_parse_fullcontent_flattens_inline_emphasis_tags_into_paragraph_text():
    xml = "<document><body><para><b>Pankaj Jain, J.</b> - This is a review.</para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{"type": "paragraph", "text": "Pankaj Jain, J. - This is a review.", "links": []}]


def test_parse_fullcontent_extracts_case_citation_links_from_paragraph():
    xml = (
        "<document><body><para>See "
        '<link id="1" href="101010000000055057" type="case">[1957] 32 ITR 592 (Raj.)</link>'
        ".</para></body></document>"
    )

    blocks = parse_fullcontent(xml)

    assert blocks == [{
        "type": "paragraph",
        "text": "See [1957] 32 ITR 592 (Raj.).",
        "links": [{"text": "[1957] 32 ITR 592 (Raj.)", "doc_id": "101010000000055057"}],
    }]


def test_parse_fullcontent_includes_headnotes_in_document_order():
    xml = (
        "<document><body><digest><headnotes>"
        '<headnote id="">Section 148A of the Income-tax Act.</headnote>'
        "</headnotes></digest>"
        "<caseOrder><order><para>ORDER</para></order></caseOrder>"
        "</body></document>"
    )

    blocks = parse_fullcontent(xml)

    assert [b["text"] for b in blocks] == ["Section 148A of the Income-tax Act.", "ORDER"]


def test_parse_fullcontent_skips_blank_paragraphs():
    xml = "<document><body><para>   </para><para>Real text.</para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{"type": "paragraph", "text": "Real text.", "links": []}]


def test_parse_fullcontent_raises_value_error_on_malformed_xml():
    with pytest.raises(ValueError):
        parse_fullcontent("<document><body><para>unclosed</body></document>")


def test_parse_fullcontent_extracts_paragraphs_from_legacy_html_documents():
    """Older indexed documents store fullcontent as a full XHTML page, not
    the newer <document><body><para> schema - its unescaped void elements
    (<meta>, <br>) aren't well-formed XML, so it needs the lenient path."""
    html = (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">\r\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>\r\n'
        '<meta http-equiv="content-type" content="text/html;charset=utf-8">\r\n'
        "</head><body>\r\n"
        '<p align="right"><font face="Times New Roman"><span style="font-size:9pt">'
        "<b>[1991] 58 TAXMAN 216 (CAL)</b></span></font></p>\r\n"
        "<p>Commissioner of Income-tax v. Arvind Investments Ltd.</p>\r\n"
        "</body></html>"
    )

    blocks = parse_fullcontent(html)

    assert blocks == [
        {"type": "paragraph", "text": "[1991] 58 TAXMAN 216 (CAL)", "links": []},
        {"type": "paragraph", "text": "Commissioner of Income-tax v. Arvind Investments Ltd.", "links": []},
    ]
