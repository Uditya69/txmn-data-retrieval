from common.legal_lexicon import fuzzy_correct_query


def test_fuzzy_correct_query_fixes_misspelled_court_name():
    corrected, corrections = fuzzy_correct_query("case from AHMDABAD tribunal")
    assert "AHMEDABAD" in corrected.split()
    assert corrections == [{"original": "AHMDABAD", "corrected": "AHMEDABAD", "score": corrections[0]["score"]}]
    assert corrections[0]["score"] >= 80.0


def test_fuzzy_correct_query_leaves_exact_known_term_unchanged():
    corrected, corrections = fuzzy_correct_query("ruling from ITR")
    assert corrected == "ruling from ITR"
    assert corrections == []


def test_fuzzy_correct_query_leaves_short_tokens_unchanged():
    # "GST" (3 chars) fuzzy-matches "GSTL" at 85.7 - below the min token length floor
    # keeps common short legal abbreviations from being silently mangled.
    corrected, corrections = fuzzy_correct_query("GST refund claim")
    assert corrected == "GST refund claim"
    assert corrections == []


def test_fuzzy_correct_query_leaves_numeric_and_section_tokens_unchanged():
    corrected, corrections = fuzzy_correct_query("Section 80C exemption")
    assert corrected == "Section 80C exemption"
    assert corrections == []


def test_fuzzy_correct_query_leaves_unrelated_words_unchanged():
    corrected, corrections = fuzzy_correct_query("what is the penalty for late filing")
    assert corrected == "what is the penalty for late filing"
    assert corrections == []


def test_fuzzy_correct_query_handles_empty_string():
    corrected, corrections = fuzzy_correct_query("")
    assert corrected == ""
    assert corrections == []


def test_fuzzy_correct_query_fixes_misspelled_general_legal_term():
    corrected, corrections = fuzzy_correct_query("sectionnn 54")
    assert corrected.split()[0] == "SECTION"
    assert corrections == [{"original": "sectionnn", "corrected": "SECTION", "score": corrections[0]["score"]}]


def test_fuzzy_correct_query_fixes_misspelled_assessee_and_tribunal():
    corrected, corrections = fuzzy_correct_query("asessee filed appeal before tribunl")
    assert "ASSESSEE" in corrected.split()
    assert "TRIBUNAL" in corrected.split()
    originals = {c["original"] for c in corrections}
    assert originals == {"asessee", "tribunl"}


def test_fuzzy_correct_query_does_not_collapse_reassessment_into_assessment():
    # "reassessment" is a distinct legal concept (reopening an assessment), not a
    # misspelling of "assessment" - it scores 90.9 against ASSESSMENT via edit-distance,
    # above the fuzzy threshold, so it must be an exact-known term of its own or fuzzy
    # matching would silently corrupt its meaning.
    corrected, corrections = fuzzy_correct_query("notice for reassessment")
    assert corrected == "notice for reassessment"
    assert corrections == []


def test_fuzzy_correct_query_does_not_catch_heavily_garbled_spelling():
    # "sexan" for "section" only scores 50 via edit-distance ratio - too far below
    # the safe threshold to correct without risking false positives elsewhere.
    corrected, corrections = fuzzy_correct_query("sexan 55")
    assert corrected == "sexan 55"
    assert corrections == []
