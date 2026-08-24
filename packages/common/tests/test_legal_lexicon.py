import pytest
from common.legal_lexicon import (
    is_known_court, is_known_journal, is_stopword, expand_synonyms, normalize,
    SECTION_PATTERN, CITATION_PATTERN, PARTY_PATTERN,
)


def test_is_known_court_recognizes_high_court_and_bench_zonetypes():
    assert is_known_court("HIGHCOURT") is True
    assert is_known_court("AAR") is True  # BENCH-zoned in token.js
    assert is_known_court("NOTACOURT") is False


def test_is_known_journal_recognizes_itr():
    assert is_known_journal("ITR") is True
    assert is_known_journal("RANDOMWORD") is False


def test_is_known_journal_normalizes_punctuation_variants():
    # the lexicon stores space-separated multi-word abbreviations ("TAXMANN COM"); a query
    # token spells the same abbreviation with punctuation instead of spaces.
    assert is_known_journal("taxmann.com") is True
    assert is_known_journal("S.T.R.") is True


def test_is_stopword_recognizes_able():
    assert is_stopword("ABLE") is True
    assert is_stopword("SECTION") is False


def test_is_stopword_recognizes_common_connectives_missing_before_2026_08_24():
    """AND/FOR/A/AS were absent from the lexicon - harmless while _PHRASE_BOOSTS was gated to
    section-type chunks only (a stray "and" chunk got a small weight), but once that gate was
    removed (es_client.py's _build_field_query applies the same massive tier to every chunk
    type), an un-stripped connective surviving as its own chunk would get a 100000-scale
    heading boost matching virtually every document in the corpus."""
    for word in ("AND", "FOR", "A", "AS"):
        assert is_stopword(word) is True


def test_expand_synonyms_returns_alternatives_for_acit():
    result = expand_synonyms("ACIT")
    assert "ASSISTANT COMMISSIONER INCOME TAX" in result
    assert "ACIT" in result


def test_expand_synonyms_returns_input_unchanged_when_no_entry():
    assert expand_synonyms("RANDOMWORD") == ["RANDOMWORD"]


def test_normalize_applies_section_number_dash_fix():
    assert normalize("115I") == "115-I"


def test_normalize_returns_input_unchanged_when_no_entry():
    assert normalize("RANDOMWORD") == "RANDOMWORD"


@pytest.mark.parametrize("text", ["Section 54F", "Sec. 12", "u/s 80C", "Rule 6(3)", "Article 226"])
def test_section_pattern_matches_known_provision_formats(text):
    assert SECTION_PATTERN.search(text) is not None


def test_section_pattern_does_not_match_plain_number():
    assert SECTION_PATTERN.search("the number 80C alone") is None


@pytest.mark.parametrize("text", ["2024 ITR 123", "(2023) 5 SCC 100", "AIR 2022 SC 456"])
def test_citation_pattern_matches_known_citation_formats(text):
    assert CITATION_PATTERN.search(text) is not None


@pytest.mark.parametrize("text", ["Krishana Goel vs. Principal Commissioner", "State v. Doe", "X versus Y"])
def test_party_pattern_matches_vs_variants(text):
    assert PARTY_PATTERN.search(text) is not None


from common.legal_lexicon import KNOWN_ACT_NAMES, KNOWN_COURT_FULL_NAMES


def test_known_court_full_names_includes_original_nine():
    for name in [
        "Supreme Court", "Delhi High Court", "Bombay High Court", "Madras High Court",
        "Calcutta High Court", "Karnataka High Court", "Gujarat High Court",
        "Income Tax Appellate Tribunal", "Customs Excise and Service Tax Appellate Tribunal",
    ]:
        assert name in KNOWN_COURT_FULL_NAMES


def test_known_act_names_includes_original_ten():
    for name in [
        "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
        "bharatiya sakshya adhiniyam", "indian penal code", "income-tax act",
        "income tax act", "cgst act", "igst act", "customs act",
        "code of criminal procedure", "indian evidence act",
    ]:
        assert name in KNOWN_ACT_NAMES
