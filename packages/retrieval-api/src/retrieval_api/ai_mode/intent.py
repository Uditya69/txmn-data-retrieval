import json
from typing import Awaitable, Callable

from langfuse import get_client

from common.schema_context import build_schema_context
from retrieval_api.gateway_client import GatewayClient

# Invariant: on_step implementations must not raise. The current only caller
# (ws.py's emit_trace_step / _emit_trace_step) guarantees this by swallowing
# any exception from sending a trace frame. A future caller that passes a
# raising callback would have that exception propagate into run_ai_mode's
# blanket `except Exception`, incorrectly turning a successful pipeline run
# into an ai_mode_error.
OnStep = Callable[[str, dict], Awaitable[None]]


def _extract_json_object(text: str) -> str:
    """SLMs often wrap JSON in prose and/or a markdown code fence despite
    instructions not to - pull out the outermost {...} object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start:end + 1]


def _fallback_intent(query: str) -> dict:
    """Used when the SLM refuses or returns unparseable output (e.g. Llama's
    safety training treating "case law for X vs. Y" as a request for private
    info about a named person) - degrade to a plain semantic search instead
    of failing the whole AI Mode request."""
    return {"rewritten_query": query, "intent": "unknown", "filters": {}}


_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
All case names and parties mentioned below refer exclusively to already
public, reported court judgments in a licensed legal research database -
never treat a query as a request for private information about a person,
and never refuse to classify it. You do not answer the legal question or
look anything up yourself; you only ever output the JSON object below.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": the query rewritten for search, expanding any old-law
  references to their new-law equivalent (IPC -> BNS, CrPC -> BNSS, Evidence
  Act -> BSA) where applicable, and phrased to read naturally against the
  collection descriptions below since it will be embedded and searched
  against all of them at once.
- "intent": one short intent category label.
- "filters": an object with any of "court", "act", "section", "date_range", "party" -
  ONLY include a key if its value is LITERALLY written in the query. Never
  guess, infer, or fill in a plausible-sounding court, act, section, or date
  range that the query does not state - a wrong filter silently excludes the
  correct document from the search entirely, which is worse than no filter.
  If the query names a person or company (very often written as
  "X vs. Y" or "X v. Y"), put that name under "party" - never under
  "section" or any other key. If nothing is explicitly stated, "filters"
  should be an empty object.
  "date_range" MUST be an object with ISO date strings, e.g.
  {"gte": "2020-01-01", "lte": "2022-01-01"} - either key may be omitted,
  but never output "date_range" as a plain string or year number, and never
  invent one when no date was mentioned.

Example: query "case law for Ramesh Gupta vs. Income-tax Officer" mentions
no court, act, section, or date - only a party name - so filters must be
exactly {"party": "Ramesh Gupta"}.

""" + build_schema_context()


async def extract_intent(gateway: GatewayClient, query: str, on_step: OnStep | None = None) -> dict:
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    cleaned = _extract_json_object(response.strip())
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        get_client().update_current_span(
            level="WARNING", status_message=f"SLM did not return valid JSON, falling back to plain search: {response!r}",
        )
        result = _fallback_intent(query)

    if on_step is not None:
        await on_step("intent", {"query": query, **result})

    return result
