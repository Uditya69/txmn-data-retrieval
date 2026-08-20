import numpy as np

from common.instant_classifier.features import (
    GazetteerFeaturizer, IntentLanguageFeaturizer, RegexFeaturizer, StructuralFeaturizer, build_feature_union,
)


def test_regex_featurizer_detects_section_citation_party():
    rows = RegexFeaturizer().transform(["Section 52", "2024 ITR 123", "CIT vs Infosys", "plain text query"])
    assert rows[0].tolist() == [1.0, 0.0, 0.0]
    assert rows[1].tolist() == [0.0, 1.0, 0.0]
    assert rows[2][2] == 1.0
    assert rows[3].tolist() == [0.0, 0.0, 0.0]


def test_gazetteer_featurizer_detects_court_journal_legal_term():
    rows = GazetteerFeaturizer().transform(["ITAT order", "TAXMAN reference", "CBDT circular", "random words here"])
    assert rows[0][0] == 1.0  # ITAT is a known court
    assert rows[1][1] == 1.0  # TAXMAN is a known journal
    assert rows[2][2] == 1.0  # CBDT has a synonym expansion
    assert rows[3].tolist() == [0.0, 0.0, 0.0]


def test_structural_featurizer_counts_tokens_and_trailing_question_mark():
    rows = StructuralFeaturizer().transform(["Section 52", "How do I file tax returns?", '"quoted phrase" query'])
    assert rows[0][0] == 2.0 and rows[0][1] == 0.0 and rows[0][2] == 0.0
    assert rows[1][1] == 1.0
    assert rows[2][2] == 1.0


def test_intent_language_featurizer_detects_question_words_and_first_person():
    rows = IntentLanguageFeaturizer().transform(["How do I evade tax", "Section 52", "We should appeal if possible"])
    assert rows[0][0] == 1.0 and rows[0][1] == 1.0  # "how", "i"
    assert rows[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert rows[2][2] == 1.0 and rows[2][3] == 1.0  # "should", "if"


def test_build_feature_union_transforms_a_batch_of_queries_to_a_2d_array():
    union = build_feature_union()
    matrix = union.fit_transform(["Section 52", "How do I evade tax", "Where is Section 52 applicable"])
    assert matrix.shape[0] == 3
    assert isinstance(matrix, np.ndarray) or hasattr(matrix, "toarray")
