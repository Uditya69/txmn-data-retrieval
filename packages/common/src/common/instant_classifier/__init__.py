from functools import lru_cache

from common.instant_classifier.labels import ClassifierResult, FALLBACK, resolve_routing
from common.instant_classifier.pipeline import load_artifact

__all__ = ["classify", "confidence_threshold", "effective_label", "ClassifierResult"]


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


def effective_label(query: str) -> str:
    if not query.strip():
        return FALLBACK
    return resolve_routing(classify(query), confidence_threshold())
