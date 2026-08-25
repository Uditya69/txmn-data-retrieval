import logging

from fastapi import APIRouter
from pydantic import BaseModel

from common.config import get_settings
from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_persona_topics_collection
from persona.prompt import render_persona_context
from persona.repository import get_current_snapshot
from retrieval_api.ai_mode.intent import build_lexicon_check, extract_intent
from retrieval_api.gateway_client import GatewayClient

router = APIRouter()

logger = logging.getLogger(__name__)


class IntentAnalysisRequest(BaseModel):
    query: str
    user_id: str | None = None


@router.post("/v1/intent-analysis")
async def get_intent_analysis(req: IntentAnalysisRequest):
    """Runs the real extract_intent SLM call (search_query rewrite, intent category
    tagging, filters) against a query, standalone - no retrieval/rerank/synthesis, just
    the classification step /ws/search's AI Mode path also runs. When user_id is given,
    reuses /ws/search's own persona lookup + trust-gate path (persona.prompt.
    render_persona_context) so persona's effect on classification can be tested against
    a real seeded user, with the gate/lookup outcome surfaced in the response instead of
    only being visible inside a full AI Mode trace."""
    settings = get_settings()
    gateway = GatewayClient(base_url=settings.gateway_url)

    persona_found = False
    persona_context = ""
    topic_count = None

    if req.user_id is not None:
        try:
            persona_settings = get_persona_settings()
            mongo_client = get_mongo_client(persona_settings)
            persona_topics_collection = get_persona_topics_collection(mongo_client, persona_settings)
            snapshot = await get_current_snapshot(persona_topics_collection, req.user_id)
            if snapshot:
                persona_found = True
                topic_count = len(snapshot)
            persona_context = render_persona_context(snapshot, persona_settings)
        except Exception:
            # A down/unreachable persona store must never crash this request - mirrors
            # the same degrade-to-guest pattern ws.py's /ws/search uses.
            logger.exception("Persona lookup failed for user %r; proceeding without persona context", req.user_id)
            persona_found = False
            persona_context = ""
            topic_count = None

    result = await extract_intent(gateway, req.query, persona_context=persona_context)
    if not settings.expose_reasoning:
        result.pop("reasoning", None)

    return {
        **result,
        "persona_found": persona_found,
        "persona_context_used": persona_context,
        "topic_count": topic_count,
        "lexicon_check": build_lexicon_check(req.query),
    }
