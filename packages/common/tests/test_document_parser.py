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
