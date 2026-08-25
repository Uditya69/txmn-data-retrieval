from common.query_tokenizer import (
    classify_query_shape, normalize_citation_spacing, merge_keyword_number,
    merge_court_city, merge_citation_span, strip_stopwords, extract_quoted_phrases,
    expand_query_synonyms, extract_boost_phrases, chunk_query, build_dense_sparse_query,
    classify_intent_mode, detect_group_signals, expand_query_normalizations,
)


def test_normalize_citation_spacing_splits_year_from_source():
    assert normalize_citation_spacing("2024taxman.com 123") == "2024 taxman.com 123"


def test_normalize_citation_spacing_leaves_normal_text_unchanged():
    assert normalize_citation_spacing("Section 54F exemption") == "Section 54F exemption"


def test_classify_query_shape_still_works_for_intent_py():
    # classify_query_shape is retired from ES-boost-profile/routing duty (es_client.py now
    # uses common.instant_classifier.effective_label instead) but stays live for AI Mode's
    # anchor-detection floor (retrieval_api.ai_mode.intent._has_legal_anchor/
    # build_lexicon_check) - this pins its original citation/provision/plain behavior so a
    # regression there doesn't go untested now that the ES-boost tests no longer cover it.
    assert classify_query_shape("2024 ITR 123 exemption") == "citation"
    assert classify_query_shape("Section 54F exemption eligibility") == "provision"
    assert classify_query_shape("can a company claim depreciation on goodwill") == "plain"


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


def test_merge_keyword_number_merges_reversed_number_then_keyword_order():
    # a user typing the number before the keyword ("55 section" instead of "section 55")
    # must still be recognized as a section reference, canonicalized to "Section 55" so
    # downstream chunk-type detection (which checks the merged phrase's first word) works.
    assert merge_keyword_number(["Income", "55", "Section"]) == ["Income", "Section 55"]
    assert merge_keyword_number(["5(8)", "Section", "Income"]) == ["Section 5(8)", "Income"]


def test_merge_keyword_number_normalizes_sec_abbreviation_to_section():
    # "sec" is a recognized _SECTION_KEYWORDS variant so merging/routing already worked, but
    # the merged phrase text used to stay literally "sec 55" - a real doc heading reads
    # "Section 55", so that phrase-boost match_phrase never hit it. legal_lexicon.json's
    # normalizations dict already has "SEC" -> "SECTION"; this wires it into the merge so
    # "sec 55" and "section 55" produce the same phrase-boost text.
    assert merge_keyword_number(["sec", "55", "Income"]) == ["SECTION 55", "Income"]


def test_merge_keyword_number_normalizes_sec_abbreviation_in_reversed_order():
    assert merge_keyword_number(["Income", "55", "sec"]) == ["Income", "SECTION 55"]


def test_merge_keyword_number_leaves_unmapped_keyword_variants_unchanged():
    # "u/s" and "sec." are recognized _SECTION_KEYWORDS (routing/classification still works)
    # but have no legal_lexicon.json normalizations entry at all - nothing to canonicalize to,
    # so the merge leaves them exactly as typed rather than guessing an unverified expansion.
    assert merge_keyword_number(["u/s", "55", "Income"]) == ["u/s 55", "Income"]


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


def test_build_dense_sparse_query_strips_question_scaffolding():
    """The exact bug this fixes: Instant mode has no LLM rewrite step, so without this,
    "section 55" and "what is section 55" would send different, noise-diluted text to Milvus
    dense/sparse (they already send identical text to ES - chunk_query already cleans the
    phrase-boost clauses)."""
    assert build_dense_sparse_query(chunk_query("what is section 55"), fallback="what is section 55") == "section 55"
    assert build_dense_sparse_query(chunk_query("section 55"), fallback="section 55") == "section 55"


def test_build_dense_sparse_query_keeps_real_content_words_for_conceptual_queries():
    cleaned = build_dense_sparse_query(
        chunk_query("how is depreciation computed under the income tax act"),
        fallback="how is depreciation computed under the income tax act",
    )
    assert cleaned == "depreciation computed under income tax act"


def test_build_dense_sparse_query_falls_back_to_original_when_chunks_are_empty():
    assert build_dense_sparse_query([], fallback="original text") == "original text"


def test_chunk_query_does_not_emit_a_standalone_and_chunk():
    """Regression test: "and" surviving as its own chunk used to only cost a small
    _BOOST_PROFILES weight (harmless), but es_client.py's _build_field_query now applies the
    same massive phrase-boost tier to every chunk type - a stray connective chunk at that
    scale would match virtually every document in the corpus."""
    chunks = chunk_query("section 14 and section 151A")
    assert not any(c["text"].strip().upper() == "AND" for c in chunks)


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


def test_expand_query_normalizations_appends_department_abbreviation_expansion():
    # legal_lexicon.json's normalizations dict (1299 entries, zero key overlap with the
    # separate synonyms dict) was sitting completely unwired - normalize() was never called
    # in the real pipeline before detect_group_signals/merge_keyword_number. This wires the
    # plain abbreviation/department-name entries (ASST -> ASSISTANT, DBODCIRCULAR -> DBOD,
    # CHALLANNO -> CHALLAN, ...) into the same query-broadening role expand_query_synonyms
    # already plays for the `synonyms` dict.
    expanded = expand_query_normalizations("ASST commissioner order")
    assert "ASSISTANT" in expanded
    assert expanded.startswith("ASST commissioner order")


def test_expand_query_normalizations_leaves_query_unchanged_when_no_lexicon_entry():
    assert expand_query_normalizations("can a company claim depreciation") == "can a company claim depreciation"


def test_expand_query_normalizations_does_not_duplicate_expansion_for_repeated_token():
    expanded = expand_query_normalizations("ASST order ASST appeal")
    assert expanded.count("ASSISTANT") == 1


def test_expand_query_normalizations_skips_multiword_act_name_expansions():
    # ~1200 of the 1299 normalizations entries expand a glued-together acronym into a
    # multi-word Act name (RESTRICTIVETRADEPRACTICESACT -> "RESTRICTIVE TRADE PRACTICES
    # ACT") - deliberately left unwired here (unverified whether real queries ever contain
    # that glued form at all), unlike the single-word abbreviation entries this targets.
    assert expand_query_normalizations("RESTRICTIVETRADEPRACTICESACT") == "RESTRICTIVETRADEPRACTICESACT"


def test_expand_query_normalizations_skips_group_signal_and_section_targets():
    # CASELAWS/RULE/ARTICLE are already wired via detect_group_signals' should-clause boost,
    # and SEC's SECTION target via merge_keyword_number's phrase-text canonicalization -
    # expanding the raw query text with these too would just be redundant/noisy.
    assert expand_query_normalizations("landmark ruling on GST") == "landmark ruling on GST"
    assert expand_query_normalizations("sec 55 exemption") == "sec 55 exemption"


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


def test_detect_group_signals_true_for_ruling_inside_merged_text_chunk():
    # "ruling" ends up merged into the trailing "ruling GST" text chunk (see chunk_query),
    # not its own standalone chunk - the detector must scan words inside a chunk's text,
    # not just compare each chunk's whole text against the lexicon.
    chunks = chunk_query("landmark Supreme Court ruling on GST")
    assert detect_group_signals(chunks) == {"CASELAWS"}


def test_detect_group_signals_true_for_judgement_synonym():
    chunks = chunk_query("Bombay High Court judgement on capital gains")
    assert detect_group_signals(chunks) == {"CASELAWS"}


def test_detect_group_signals_detects_bare_rule_word():
    # legal_lexicon.json's normalizations dict has no identity entry for the bare canonical
    # word ("RULE" -> "RULE" would be a no-op normalization, so the data curator skipped
    # it - only suffixed/abbreviated forms like RULENO/RULES are listed) - centax-node's own
    # token table (constants/token.js) has a dedicated entry for the bare word itself, so
    # this must recognize it too, not just the normalize()-mapped variants.
    chunks = chunk_query("landmark rule laid down by the tribunal")
    assert detect_group_signals(chunks) == {"RULE"}


def test_detect_group_signals_detects_bare_article_word_as_experts_opinion():
    # ARTICLE's real ES groups.group.name is "Experts Opinion", not "ARTICLE" - verified
    # against a real doc pulled into this investigation (heading "[2019] 106 taxmann.com 47
    # (Article)", groups.group.name == "Experts Opinion", id 111050000000000051).
    chunks = chunk_query("landmark article on GST reforms")
    assert detect_group_signals(chunks) == {"Experts Opinion"}


def test_detect_group_signals_false_when_no_signal_word_present():
    chunks = chunk_query("Section 54F exemption eligibility")
    assert detect_group_signals(chunks) == set()


def test_detect_group_signals_false_for_empty_chunks():
    assert detect_group_signals([]) == set()


def test_chunk_query_empty_query_returns_no_chunks():
    assert chunk_query("") == []


def test_classify_intent_mode_tags_bare_section_reference_as_keyword():
    assert classify_intent_mode("what is section 55") == "keyword"


def test_classify_intent_mode_tags_bare_citation_as_keyword():
    assert classify_intent_mode("133 taxmann.com 196") == "keyword"


def test_classify_intent_mode_tags_bare_known_court_as_keyword():
    assert classify_intent_mode("what is ITAT") == "keyword"


def test_classify_intent_mode_tags_bare_act_name_as_keyword():
    assert classify_intent_mode("what is the income tax act") == "keyword"


def test_classify_intent_mode_tags_act_name_with_year_as_keyword():
    assert classify_intent_mode("income tax act 1961") == "keyword"


def test_classify_intent_mode_tags_framing_verb_plus_section_as_keyword():
    # "explain"/"define"/etc. aren't general-English connectives (so weren't in the
    # original stopword list), but they carry no lexical search value either - a query
    # that's otherwise a pure anchor lookup shouldn't fall into the expensive hybrid
    # path just because the user wrapped it in a framing verb instead of "what is".
    assert classify_intent_mode("explain section 55") == "keyword"
    assert classify_intent_mode("define section 55") == "keyword"
    assert classify_intent_mode("definition of section 55") == "keyword"


def test_classify_intent_mode_does_not_tag_synonym_lexicon_term_as_keyword():
    # "PE" is a recognized abbreviation (legal_lexicon.json synonyms -> Permanent
    # Establishment) but names a broad legal concept, not a precise lookup - synonym
    # entries are deliberately excluded from the keyword gate (courts/journals/act
    # names/structural refs only), so this must stay hybrid.
    assert classify_intent_mode("what is PE") == "hybrid"


def test_classify_intent_mode_tags_conceptual_query_as_hybrid():
    assert classify_intent_mode("how is depreciation computed under the income tax act") == "hybrid"


def test_classify_intent_mode_tags_party_name_plus_section_as_hybrid():
    # a mix of an unanchored text run (party name) alongside a structural chunk still
    # needs semantic search for the unanchored part - only an all-anchor query qualifies.
    assert classify_intent_mode("Dimension Data India section 92C") == "hybrid"


def test_classify_intent_mode_tags_empty_query_as_hybrid():
    assert classify_intent_mode("") == "hybrid"
