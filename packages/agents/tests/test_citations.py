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
