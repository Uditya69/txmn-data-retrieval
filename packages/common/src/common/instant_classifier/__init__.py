from functools import lru_cache

from common.instant_classifier.labels import ClassifierResult, HYBRID, resolve_routing
from common.instant_classifier.pipeline import load_artifact

__all__ = [
    "classify", "confidence_threshold", "effective_label", "effective_label_with_confidence", "ClassifierResult",
]


@lru_cache
def _load():
    return load_artifact()


def classify(query: str) -> ClassifierResult:
    pipeline, _ = _load()
    proba = pipeline.predict_proba([query])[0]
    label = pipeline.classes_[proba.argmax()]
    confidence = float(proba.max())
    return ClassifierResult(label=label, confidence=confidence)


def confidence_threshold() -> float:
    _, meta = _load()
    return meta["confidence_threshold"]


def effective_label_with_confidence(query: str) -> tuple[str, float]:
    """Like effective_label(), but also surfaces the raw model confidence for
    observability (Langfuse spans, WS trace steps) - effective_label() alone can't
    distinguish "model was confident" from "model was unsure and this defaulted to HYBRID"."""
    if not query.strip():
        return HYBRID, 0.0
    result = classify(query)
    return resolve_routing(result, confidence_threshold()), result.confidence


def effective_label(query: str) -> str:
    label, _ = effective_label_with_confidence(query)
    return label
