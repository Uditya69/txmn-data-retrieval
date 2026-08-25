import json
import logging
from datetime import datetime

from persona.repository import record_query_event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Given a single legal-research query, extract a structured research signal.
Respond with JSON only:
{"concepts": ["..."], "legal_entities": ["..."], "research_objective": ["..."],
 "specificity": 0.0-1.0, "confidence": 0.0-1.0}
Omit a list field (empty list) or score (0.0) if you are not reasonably confident about it."""

_RESPONSE_FORMAT = {"type": "json_object"}


async def extract_query_understanding(gateway, query: str) -> dict:
    # Only a malformed/non-JSON model response is swallowed here (mirrors
    # ai_mode/intent.py::extract_intent's json.loads handling) - a gateway
    # transport failure (e.g. RuntimeError from a dead connection) is
    # deliberately allowed to propagate to the caller, record_persona_signal,
    # whose own broader try/except decides whether to skip the persona write
    # entirely for that case.
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format=_RESPONSE_FORMAT,
    )
    try:
        result = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        logger.debug("query understanding extraction returned malformed JSON for query %r: %r", query, response)
        return {}
    return result if isinstance(result, dict) else {}


async def record_persona_signal(
    events, topics, gateway, user_id: str, query: str, categories: list[str],
    timestamp: datetime, settings, interaction_signals: dict | None = None,
) -> None:
    try:
        understanding_raw = await extract_query_understanding(gateway, query)
        # query_embed is the same role/provider Instant mode's semantic cache
        # uses (Voyage) - reused here purely as an embedding function for
        # persona-internal topic-clustering similarity, a separate space from
        # the Milvus corpus cosine comparison hard rule 1 governs. See
        # design.md decision #2.
        embedding = await gateway.embed(role="query_embed", text=query)
        await record_query_event(
            events, topics, user_id, query, understanding_raw, embedding, categories,
            interaction_signals or {"submitted": True}, timestamp, settings,
        )
    except Exception:
        logger.warning("persona signal recording failed for user %r", user_id, exc_info=True)
