from fastapi import APIRouter
from pydantic import BaseModel

from common.instant_classifier import classify, confidence_threshold
from common.instant_classifier.labels import resolve_routing, routing_plan

router = APIRouter()


class ClassifierAnalysisRequest(BaseModel):
    query: str


@router.post("/v1/classifier-analysis")
async def get_classifier_analysis(req: ClassifierAnalysisRequest):
    """Runs the real Instant-mode ML classifier (common.instant_classifier - a trained
    sklearn LogisticRegression pipeline, not a rule/regex heuristic) standalone against a
    query, for measuring its own performance in isolation from any actual retrieval call.
    Surfaces both the raw model output (label + confidence straight from predict_proba)
    and the post-threshold effective label resolve_routing() falls back to HYBRID for when
    confidence is too low to trust - the two can differ, and that difference is exactly
    what a classifier eval needs to see. routing_plan shows which backends the effective
    label would turn on/off, matching the forced /v1/search/{keyword,intent,hybrid}
    endpoints 1:1 (KEYWORD -> es-only, INTENT -> milvus-only, HYBRID -> both + fuse)."""
    if not req.query.strip():
        return {
            "query": req.query, "raw_label": None, "raw_confidence": 0.0,
            "confidence_threshold": confidence_threshold(), "effective_label": "HYBRID",
            "routing_plan": routing_plan("HYBRID"),
        }

    raw = classify(req.query)
    threshold = confidence_threshold()
    effective_label = resolve_routing(raw, threshold)

    return {
        "query": req.query,
        "raw_label": raw.label,
        "raw_confidence": raw.confidence,
        "confidence_threshold": threshold,
        "effective_label": effective_label,
        "below_threshold": raw.confidence < threshold,
        "routing_plan": routing_plan(effective_label),
    }
