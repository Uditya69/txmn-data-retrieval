from common.instant_classifier import classify, confidence_threshold, effective_label
from common.instant_classifier.labels import FALLBACK, HYBRID, INTENT, KEYWORD


def test_classify_returns_confident_keyword_for_bare_section_ref():
    result = classify("Section 52")
    assert result.label == KEYWORD
    assert result.confidence > 0.5


def test_classify_returns_confident_intent_for_pure_question():
    result = classify("How do I evade tax")
    assert result.label == INTENT
    assert result.confidence > 0.5


def test_classify_returns_hybrid_for_anchor_plus_question():
    result = classify("Where is Section 52 applicable")
    assert result.label == HYBRID


def test_confidence_threshold_is_a_float_between_zero_and_one():
    threshold = confidence_threshold()
    assert 0.0 <= threshold <= 1.0


def test_effective_label_matches_classify_when_confident():
    assert effective_label("Section 52") == KEYWORD


def test_effective_label_falls_back_below_threshold(monkeypatch):
    import common.instant_classifier as module

    monkeypatch.setattr(module, "classify", lambda query: module.ClassifierResult(label=KEYWORD, confidence=0.0))
    assert module.effective_label("anything") == FALLBACK


def test_effective_label_short_circuits_on_empty_query():
    assert effective_label("") == FALLBACK
    assert effective_label("   ") == FALLBACK
