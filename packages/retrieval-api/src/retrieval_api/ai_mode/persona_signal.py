import json
import logging

from persona.repository import record_signal

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Given a single legal-research query, classify the asker's likely
expertise and phrasing style. Respond with JSON only:
{"expertise_level": "student" | "practitioner" | "expert", "query_style": "broad" | "precise-citation"}
Omit a key if you are not reasonably confident about it."""

_RESPONSE_FORMAT = {"type": "json_object"}


async def extract_expertise_signal(gateway, query: str) -> dict:
    # Only a malformed/non-JSON model response is swallowed here (mirrors
    # ai_mode/intent.py::extract_intent's json.loads handling) - a gateway
    # transport failure (e.g. RuntimeError from a dead connection) is
    # deliberately allowed to propagate to the caller, record_persona_signal,
    # whose own broader try/except decides whether to skip the persona write
    # entirely for that case (see test_record_persona_signal_swallows_gateway_errors:
    # a gateway outage must not still write a persona doc with an empty
    # expertise patch).
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
        logger.debug("expertise signal extraction returned malformed JSON for query %r: %r", query, response)
        return {}
    return result if isinstance(result, dict) else {}


async def record_persona_signal(personas, gateway, user_id: str, query: str, categories: list[str]) -> None:
    try:
        expertise_patch = await extract_expertise_signal(gateway, query)
        await record_signal(personas, user_id, categories, expertise_patch)
    except Exception:
        logger.warning("persona signal recording failed for user %r", user_id, exc_info=True)
