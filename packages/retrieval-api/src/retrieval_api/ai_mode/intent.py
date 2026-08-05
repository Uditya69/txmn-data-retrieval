import json

from common.schema_context import build_schema_context
from retrieval_api.gateway_client import GatewayClient


def _extract_json_object(text: str) -> str:
    """SLMs often wrap JSON in prose and/or a markdown code fence despite
    instructions not to - pull out the outermost {...} object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start:end + 1]

_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": the query rewritten for search, expanding any old-law
  references to their new-law equivalent (IPC -> BNS, CrPC -> BNSS, Evidence
  Act -> BSA) where applicable, and phrased to read naturally against the
  collection descriptions below since it will be embedded and searched
  against all of them at once.
- "intent": one short intent category label.
- "filters": an object with any of "court", "act", "section", "date_range", "party"
  the query explicitly mentions; omit keys that aren't mentioned.
  "date_range" MUST be an object with ISO date strings, e.g.
  {"gte": "2020-01-01", "lte": "2022-01-01"} - either key may be omitted,
  but never output "date_range" as a plain string or year number.

""" + build_schema_context()


async def extract_intent(gateway: GatewayClient, query: str) -> dict:
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    cleaned = _extract_json_object(response.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SLM did not return valid JSON: {response!r}") from exc
