import pytest

from common.document_parser import parse_fullcontent


def _text(s, bold=False, italic=False):
    return {"type": "text", "text": s, "bold": bold, "italic": italic}


def test_parse_fullcontent_returns_empty_list_for_no_content_blocks():
    xml = "<document><body></body></document>"
    assert parse_fullcontent(xml) == []


def test_parse_fullcontent_extracts_plain_paragraph_text():
    xml = "<document><body><para>Plain paragraph text.</para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{"type": "paragraph", "spans": [_text("Plain paragraph text.")]}]


def test_parse_fullcontent_preserves_inline_bold_as_a_separate_span():
    xml = "<document><body><para><b>Pankaj Jain, J.</b> - This is a review.</para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{
        "type": "paragraph",
        "spans": [_text("Pankaj Jain, J.", bold=True), _text(" - This is a review.")],
    }]


def test_parse_fullcontent_preserves_inline_italic_as_a_separate_span():
    xml = "<document><body><para><i>Ramana Dayaram Shetty</i> v. <i>International Airport Authority</i></para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{
        "type": "paragraph",
        "spans": [
            _text("Ramana Dayaram Shetty", italic=True),
            _text(" v. "),
            _text("International Airport Authority", italic=True),
        ],
    }]


def test_parse_fullcontent_extracts_case_citation_links_from_paragraph():
    xml = (
        "<document><body><para>See "
        '<link id="1" href="101010000000055057" type="case">[1957] 32 ITR 592 (Raj.)</link>'
        ".</para></body></document>"
    )

    blocks = parse_fullcontent(xml)

    assert blocks == [{
        "type": "paragraph",
        "spans": [
            _text("See "),
            {"type": "link", "text": "[1957] 32 ITR 592 (Raj.)", "doc_id": "101010000000055057"},
            _text("."),
        ],
    }]


def test_parse_fullcontent_bolds_topic_and_statute_labels_within_a_headnote():
    xml = (
        "<document><body><headnote>"
        '<db_heading id="1">Classification of services</db_heading> - '
        '<dbs_act id="2">Karnataka GST Act, 2017</dbs_act>'
        "</headnote></body></document>"
    )

    blocks = parse_fullcontent(xml)

    assert blocks == [{
        "type": "headnote",
        "spans": [
            _text("Classification of services", bold=True),
            _text(" - "),
            _text("Karnataka GST Act, 2017", bold=True),
        ],
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

    assert [b["type"] for b in blocks] == ["headnote", "paragraph"]
    assert [b["spans"][0]["text"] for b in blocks] == ["Section 148A of the Income-tax Act.", "ORDER"]


def test_parse_fullcontent_marks_fact_class_paragraphs_as_fact_label_blocks():
    xml = '<document><body><para class="fact">CASES REFERRED TO</para></body></document>'

    blocks = parse_fullcontent(xml)

    assert blocks == [{"type": "fact_label", "spans": [_text("CASES REFERRED TO")]}]


def test_parse_fullcontent_extracts_counsel_block_from_dbs_members():
    xml = (
        "<document><body><digest><dbs_members>"
        '<db_counsela id="1">B. Mohan Babu</db_counsela>, Sr. Manager <i> for the Applicant. </i>'
        "</dbs_members></digest></body></document>"
    )

    blocks = parse_fullcontent(xml)

    assert blocks == [{
        "type": "counsel",
        "spans": [
            _text("B. Mohan Babu", bold=True),
            _text(", Sr. Manager "),
            _text(" for the Applicant. ", italic=True),
        ],
    }]


def test_parse_fullcontent_skips_blank_paragraphs():
    xml = "<document><body><para>   </para><para>Real text.</para></body></document>"

    blocks = parse_fullcontent(xml)

    assert blocks == [{"type": "paragraph", "spans": [_text("Real text.")]}]


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
        {"type": "paragraph", "spans": [_text("[1991] 58 TAXMAN 216 (CAL)")]},
        {"type": "paragraph", "spans": [_text("Commissioner of Income-tax v. Arvind Investments Ltd.")]},
    ]
