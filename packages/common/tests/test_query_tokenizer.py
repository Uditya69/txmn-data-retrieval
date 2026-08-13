from common.query_tokenizer import (
    classify_query_shape, normalize_citation_spacing, merge_keyword_number,
    merge_court_city, merge_citation_span, strip_stopwords, extract_quoted_phrases,
    expand_query_synonyms, extract_boost_phrases, analyze_query, chunk_query,
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


def test_merge_keyword_number_merges_subsection_with_letter_and_parens():
    assert merge_keyword_number(["Section", "5(8)", "Income"]) == ["Section 5(8)", "Income"]
    assert merge_keyword_number(["Section", "69C", "cash"]) == ["Section 69C", "cash"]


def test_merge_keyword_number_does_not_merge_non_keyword_word_before_number():
    # a bare word immediately before a number (a party name next to a citation number,
    # e.g. "Spa 175") is not a keyword+number pair and must not be fused into one token -
    # doing so used to shred real citations like "175 taxmann.com 251" apart.
    assert merge_keyword_number(["Spa", "175", "taxmann.com", "251"]) == ["Spa", "175", "taxmann.com", "251"]


def test_merge_citation_span_merges_number_journal_number():
    assert merge_citation_span(["133", "taxmann.com", "196", "article"]) == ["133 taxmann.com 196", "article"]
    assert merge_citation_span(["97", "ITR", "660", "section"]) == ["97 ITR 660", "section"]


def test_merge_citation_span_backtracks_when_middle_token_is_not_a_journal():
    assert merge_citation_span(["money", "17", "years", "limitation"]) == ["money", "17", "years", "limitation"]
    assert merge_citation_span(["Hotels", "Spa", "251"]) == ["Hotels", "Spa", "251"]


def test_merge_citation_span_finds_the_real_span_even_with_a_leading_bare_number():
    # "Spa" + "175" is not a citation ("175" isn't followed by a journal token) - the span
    # one position later ("175 taxmann.com 251") is the real citation and must still merge.
    assert merge_citation_span(["Spa", "175", "taxmann.com", "251"]) == ["Spa", "175 taxmann.com 251"]


def test_extract_boost_phrases_boosts_full_citation_and_section_together():
    query = "Sunil Chopra v CAPL Hotels Spa 175 taxmann.com 251 section 5(8) time value money 17 years limitation"
    assert extract_boost_phrases(query) == ["175 taxmann.com 251", "section 5(8)"]


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


def test_expand_query_synonyms_appends_known_abbreviation_expansion():
    expanded = expand_query_synonyms("ACIT order on depreciation")
    assert "ASSISTANT COMMISSIONER INCOME TAX" in expanded
    assert expanded.startswith("ACIT order on depreciation")


def test_expand_query_synonyms_leaves_query_unchanged_when_no_lexicon_entry():
    assert expand_query_synonyms("can a company claim depreciation") == "can a company claim depreciation"


def test_expand_query_synonyms_does_not_duplicate_expansion_for_repeated_token():
    expanded = expand_query_synonyms("ACIT order ACIT appeal")
    assert expanded.count("ASSISTANT COMMISSIONER INCOME TAX") == 1


def test_extract_boost_phrases_finds_merged_keyword_number():
    assert "Section 6" in extract_boost_phrases("Section 6 of Income Tax Act")


def test_extract_boost_phrases_finds_merged_court_city():
    assert "Delhi High Court" in extract_boost_phrases("Delhi High Court ruling on depreciation")


def test_extract_boost_phrases_finds_quoted_phrase():
    assert "Income India" in extract_boost_phrases('Section 6 of "Income India" in case of Supreme court')


def test_extract_boost_phrases_excludes_single_word_tokens():
    assert extract_boost_phrases("can a company claim depreciation on goodwill") == []


def test_extract_boost_phrases_excludes_stripped_stopwords():
    # "in case of" is stopped away; only the merged court phrase survives as multi-word
    assert extract_boost_phrases("ruling in case of Delhi High Court") == ["Delhi High Court"]


def test_analyze_query_reports_shape_and_final_boost_phrases():
    result = analyze_query("Section 6 of Income Tax Act")

    assert result["query"] == "Section 6 of Income Tax Act"
    assert result["shape"] == "provision"
    assert result["boost_phrases"] == ["Section 6"]


def test_analyze_query_matches_extract_boost_phrases_and_classify_query_shape():
    """analyze_query re-derives the same pipeline the actual search path runs
    independently (es_client.raw_search) - its output must never diverge from
    calling those functions directly, or the trace would be lying about what
    the search actually did."""
    query = "Reliance Industries 1995 taxmann.com 569 Rule 57A section 5(8)"

    result = analyze_query(query)

    assert result["shape"] == classify_query_shape(query)
    assert set(result["boost_phrases"]) == set(extract_boost_phrases(query))


def test_analyze_query_reports_pipeline_stage_provenance():
    result = analyze_query("Husco International 133 taxmann.com 196")

    stage_names = [s["stage"] for s in result["pipeline_stages"]]
    assert stage_names == [
        "quoted_phrase_extraction", "citation_span_merge",
        "keyword_number_merge", "court_city_merge", "stopword_strip",
    ]
    citation_stage = next(s for s in result["pipeline_stages"] if s["stage"] == "citation_span_merge")
    assert "133 taxmann.com 196" in citation_stage["tokens"]
    assert "this codebase (new" in citation_stage["source"]
    quoted_stage = result["pipeline_stages"][0]
    assert "centax-node" in quoted_stage["source"]


def test_analyze_query_omits_expanded_query_when_no_synonym_applies():
    result = analyze_query("can a company claim depreciation")
    assert result["expanded_query"] is None


def test_analyze_query_includes_expanded_query_when_synonym_applies():
    result = analyze_query("ACIT order on depreciation")
    assert result["expanded_query"] is not None
    assert "ASSISTANT COMMISSIONER INCOME TAX" in result["expanded_query"]


def test_chunk_query_groups_unrecognized_word_run_into_one_text_chunk():
    """The gap this closes: a party name like "Dimension Data India" has no Section/
    Court/citation keyword anchor, so extract_boost_phrases left it as three
    independent words. chunk_query must group it into one phrase chunk instead,
    matching centax-node's own getLowLevelQuery output for this exact query
    (QueryText: "Dimension Data India", Type: "TX", Proximity: 5)."""
    chunks = chunk_query("Dimension Data India section 92C ITES comparables")

    text_chunks = [c for c in chunks if c["type"] == "text"]
    assert any(c["text"] == "Dimension Data India" for c in text_chunks)
    for c in text_chunks:
        assert c["proximity"] == 5


def test_chunk_query_splits_text_runs_at_keyword_boundaries():
    chunks = chunk_query("Dimension Data India section 92C ITES comparables unreliable segmental results outsourcing")

    texts = [(c["type"], c["text"]) for c in chunks]
    assert ("text", "Dimension Data India") in texts
    assert ("section", "section 92C") in texts
    assert ("text", "ITES comparables unreliable segmental results outsourcing") in texts


def test_chunk_query_section_chunk_is_exact_phrase_not_fuzzy_proximity():
    chunks = chunk_query("Section 6 of Income Tax Act")
    section_chunk = next(c for c in chunks if c["type"] == "section")
    assert section_chunk["text"] == "Section 6"
    assert section_chunk["proximity"] == 0


def test_chunk_query_generates_zero_padded_section_number_alternative():
    """Matches centax-node's production getLowLevelQuery response for this exact
    query: QueryText "SECTION 92C | SECTION 092C"."""
    chunks = chunk_query("section 92C ITES")
    section_chunk = next(c for c in chunks if c["type"] == "section")
    assert section_chunk["alt_text"] == "section 092C"


def test_chunk_query_omits_zero_pad_alternative_when_number_already_three_digits():
    chunks = chunk_query("section 271 penalty")
    section_chunk = next(c for c in chunks if c["type"] == "section")
    assert section_chunk["alt_text"] is None


def test_chunk_query_citation_triple_gets_citation_proximity():
    chunks = chunk_query("Husco International 133 taxmann.com 196 royalty")
    citation_chunk = next(c for c in chunks if c["type"] == "citation")
    assert citation_chunk["text"] == "133 taxmann.com 196"
    assert citation_chunk["proximity"] == 2


def test_chunk_query_court_city_gets_exact_proximity():
    chunks = chunk_query("Delhi High Court ruling on depreciation")
    court_chunk = next(c for c in chunks if c["type"] == "court_city")
    assert court_chunk["text"] == "Delhi High Court"
    assert court_chunk["proximity"] == 0


def test_chunk_query_quoted_phrase_gets_exact_proximity():
    chunks = chunk_query('Section 6 of "Income India" in case of Supreme court')
    quoted_chunk = next(c for c in chunks if c["type"] == "quoted")
    assert quoted_chunk["text"] == "Income India"
    assert quoted_chunk["proximity"] == 0


def test_chunk_query_single_leftover_word_still_becomes_its_own_text_chunk():
    chunks = chunk_query("goodwill")
    assert chunks == [{"text": "goodwill", "proximity": 5, "type": "text", "alt_text": None}]


def test_chunk_query_empty_query_returns_no_chunks():
    assert chunk_query("") == []
