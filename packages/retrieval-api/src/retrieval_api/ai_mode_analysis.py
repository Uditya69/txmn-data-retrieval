import logging

from fastapi import APIRouter
from pydantic import BaseModel

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_personas_collection
from persona.prompt import render_persona_context
from persona.repository import get_persona
from retrieval_api.ai_mode.intent import build_lexicon_check
from retrieval_api.ai_mode.pipeline import run_ai_mode
from retrieval_api.gateway_client import GatewayClient

router = APIRouter()

logger = logging.getLogger(__name__)


class AiModeAnalysisRequest(BaseModel):
    query: str
    user_id: str | None = None


@router.post("/v1/ai-mode-analysis")
async def get_ai_mode_analysis(req: AiModeAnalysisRequest):
    """Runs the real AI Mode pipeline (run_ai_mode - intent classification, retrieval,
    rerank, synthesis) standalone over plain HTTP, with the same persona lookup +
    trust-gate path /ws/search's AI Mode branch uses when a user_id is given. Lets a
    query's full final answer be curled directly for comparison, instead of only being
    visible through /ws/search's websocket + UI. Slower per call than
    /v1/intent-analysis - this runs the whole pipeline (real ES/Milvus retrieval + LLM
    synthesis), not just the one classification call."""
    settings = get_settings()
    gateway = GatewayClient(base_url=settings.gateway_url)
    es_client = get_es_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        logger.exception("Milvus connection failed; proceeding without Milvus for this request")
        milvus_client = None

    persona_found = False
    persona_context = ""
    query_count = None

    if req.user_id is not None:
        try:
            persona_settings = get_persona_settings()
            mongo_client = get_mongo_client(persona_settings)
            personas_collection = get_personas_collection(mongo_client, persona_settings)
            persona = await get_persona(personas_collection, req.user_id)
            if persona is not None:
                persona_found = True
                query_count = persona.get("query_count", 0)
            persona_context = render_persona_context(persona)
        except Exception:
            # A down/unreachable persona store must never crash this request - mirrors
            # the same degrade-to-guest pattern ws.py's /ws/search uses.
            logger.exception("Persona lookup failed for user %r; proceeding without persona context", req.user_id)
            persona_found = False
            persona_context = ""
            query_count = None

    try:
        result = await run_ai_mode(gateway, es_client, milvus_client, req.query, persona_context=persona_context)
        return {
            **result,
            "persona_found": persona_found,
            "persona_context_used": persona_context,
            "query_count": query_count,
            "lexicon_check": build_lexicon_check(req.query),
        }
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
