import json
import re

from retrieval_api.gateway_client import GatewayClient

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": the query rewritten for search, expanding any old-law
  references to their new-law equivalent (IPC -> BNS, CrPC -> BNSS, Evidence
  Act -> BSA) where applicable.
- "intent": one short intent category label.
- "filters": an object with any of "court", "act", "date_range", "party"
  the query explicitly mentions; omit keys that aren't mentioned.
"""


async def extract_intent(gateway: GatewayClient, query: str) -> dict:
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    cleaned = _CODE_FENCE_RE.sub("", response.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SLM did not return valid JSON: {response!r}") from exc
