from dataclasses import dataclass

KEYWORD = "KEYWORD"
HYBRID = "HYBRID"
INTENT = "INTENT"

LABELS = (KEYWORD, HYBRID, INTENT)

_BOOST_PROFILE_KEY = {KEYWORD: KEYWORD, HYBRID: HYBRID, INTENT: INTENT}

_ROUTING = {
    KEYWORD: {"es": True, "milvus": False, "fuse": False},
    HYBRID: {"es": True, "milvus": True, "fuse": True},
    INTENT: {"es": False, "milvus": True, "fuse": False},
}


@dataclass(frozen=True)
class ClassifierResult:
    label: str
    confidence: float


def boost_profile_key(label: str) -> str:
    return _BOOST_PROFILE_KEY[label]


def resolve_routing(result: ClassifierResult, threshold: float) -> str:
    """Below threshold: trust neither the label nor its confidence for routing
    purposes, fall back to HYBRID (query both backends and fuse) rather than
    risk skipping a backend the model was unsure about."""
    if result.confidence < threshold:
        return HYBRID
    return result.label


def routing_plan(effective_label: str) -> dict:
    return _ROUTING[effective_label]
