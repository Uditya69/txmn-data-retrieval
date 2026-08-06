from agents.citations import extract_cited_doc_ids, validate_citations


def test_extract_cited_doc_ids_finds_all_bracketed_ids():
    answer = "The rate is 10% per [12345] and confirmed in [67890]."
    assert extract_cited_doc_ids(answer) == {"12345", "67890"}


def test_extract_cited_doc_ids_returns_empty_set_with_no_citations():
    assert extract_cited_doc_ids("No citations here.") == set()


def test_validate_citations_returns_empty_list_when_all_cited_ids_were_seen():
    answer = "See [12345] and [67890]."
    assert validate_citations(answer, {"12345", "67890", "99999"}) == []


def test_validate_citations_returns_sorted_invalid_ids():
    answer = "See [999] and [111] and [222]."
    assert validate_citations(answer, {"222"}) == ["111", "999"]


def test_extract_cited_doc_ids_ignores_prose_bracket_with_spaces():
    answer = "This is important [emphasis added] and cited in [d1]."
    assert extract_cited_doc_ids(answer) == {"d1"}


def test_extract_cited_doc_ids_ignores_markdown_link_bracket():
    answer = "See [Section 80C](https://example.com/80c) and [d1]."
    # "Section 80C" contains a space so the prose/markdown-label bracket is
    # not treated as a citation; only [d1] is.
    assert extract_cited_doc_ids(answer) == {"d1"}


def test_extract_cited_doc_ids_splits_comma_separated_ids_in_one_bracket():
    answer = "Confirmed in [d1, d2]."
    assert extract_cited_doc_ids(answer) == {"d1", "d2"}


def test_validate_citations_does_not_falsely_reject_answer_with_prose_bracket():
    answer = "The rate is 10% [emphasis added], see [12345]."
    assert validate_citations(answer, {"12345"}) == []


def test_extract_cited_doc_ids_ignores_single_alphabetic_word_bracket():
    answer = "The court noted [sic] in the ruling, cited in [12345]."
    assert extract_cited_doc_ids(answer) == {"12345"}
    assert "sic" not in extract_cited_doc_ids(answer)


def test_extract_cited_doc_ids_ignores_lone_digit_footnote_marker():
    answer = "This is disputed [1], see [12345]."
    result = extract_cited_doc_ids(answer)
    assert "1" not in result
    assert result == {"12345"}


def test_extract_cited_doc_ids_still_extracts_realistic_short_doc_ids():
    answer = "Confirmed in [d1] and also [999]."
    assert extract_cited_doc_ids(answer) == {"d1", "999"}
