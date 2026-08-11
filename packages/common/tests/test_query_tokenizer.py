from common.query_tokenizer import (
    classify_query_shape, normalize_citation_spacing, merge_keyword_number,
    merge_court_city, strip_stopwords, extract_quoted_phrases,
)


def test_classify_query_shape_detects_citation():
    assert classify_query_shape("2024 ITR 123 exemption") == "citation"


def test_classify_query_shape_detects_party_citation():
    assert classify_query_shape("Krishana Goel vs. Principal Commissioner of Income-tax") == "citation"


def test_classify_query_shape_detects_provision():
    assert classify_query_shape("Section 54F exemption eligibility") == "provision"


def test_classify_query_shape_defaults_to_plain():
    assert classify_query_shape("can a company claim depreciation on goodwill") == "plain"


def test_classify_query_shape_prefers_citation_when_both_patterns_present():
    # a citation with a section number embedded is still primarily a citation lookup
    assert classify_query_shape("2024 ITR 123 on Section 54F") == "citation"


def test_normalize_citation_spacing_splits_year_from_source():
    assert normalize_citation_spacing("2024taxman.com 123") == "2024 taxman.com 123"


def test_normalize_citation_spacing_leaves_normal_text_unchanged():
    assert normalize_citation_spacing("Section 54F exemption") == "Section 54F exemption"


def test_merge_keyword_number_merges_section_and_number():
    assert merge_keyword_number(["Section", "6", "Income"]) == ["Section 6", "Income"]


def test_merge_keyword_number_backtracks_when_no_number_follows():
    assert merge_keyword_number(["Section", "Income"]) == ["Section", "Income"]


def test_merge_court_city_merges_delhi_high_court():
    assert merge_court_city(["Delhi", "High", "Court", "ruling"]) == ["Delhi High Court", "ruling"]


def test_merge_court_city_backtracks_when_no_court_token_follows():
    assert merge_court_city(["Delhi", "weather"]) == ["Delhi", "weather"]


def test_strip_stopwords_removes_recognized_stopword():
    assert "ABLE" not in strip_stopwords(["ABLE", "Section", "6"])


def test_strip_stopwords_never_drops_a_known_journal():
    # ITR is both plausible-stopword-adjacent and a real journal - must survive
    assert "ITR" in strip_stopwords(["citation", "in", "ITR"])


def test_extract_quoted_phrases_keeps_quoted_text_as_one_token():
    assert '"Income India"' not in extract_quoted_phrases('Section 6 of "Income India" in case of Supreme court')
    assert "Income India" in extract_quoted_phrases('Section 6 of "Income India" in case of Supreme court')
