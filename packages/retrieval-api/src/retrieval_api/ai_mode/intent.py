import json
import re
from typing import Awaitable, Callable

from langfuse import get_client

from common.query_tokenizer import chunk_query
from common.schema_context import KNOWN_COURTS, build_schema_context
from retrieval_api.gateway_client import GatewayClient

# Invariant: on_step implementations must not raise. The current only caller
# (ws.py's emit_trace_step / _emit_trace_step) guarantees this by swallowing
# any exception from sending a trace frame. A future caller that passes a
# raising callback would have that exception propagate into run_ai_mode's
# blanket `except Exception`, incorrectly turning a successful pipeline run
# into an ai_mode_error.
OnStep = Callable[[str, dict], Awaitable[None]]

# DeepInfra's json_object response_format mode guarantees the response is a
# valid JSON object with no surrounding prose/markdown fence - see
# https://docs.deepinfra.com/chat/structured-outputs. This replaces a former
# regex-based brace-extraction fallback: if a model still doesn't comply
# despite the mode being requested, that's treated as a hard failure
# (json.loads raises, _fallback_intent kicks in) rather than guessed at.
_RESPONSE_FORMAT = {"type": "json_object"}


def _fallback_intent(query: str) -> dict:
    """Used when the SLM refuses or returns unparseable output (e.g. Llama's
    safety training treating "case law for X vs. Y" as a request for private
    info about a named person) - degrade to a plain semantic search instead
    of failing the whole AI Mode request."""
    return {"rewritten_query": query, "intent": "unknown", "filters": {}}


def _build_chunk_context(query: str) -> str | None:
    """Trimmed JSON projection of chunk_query's structural spans, for injection
    into extract_intent's user message. Drops `proximity`/`alt_text` (ES-only,
    and alt_text's normalized form would never literal-match _sanitize_filters'
    substring check against the raw query - see design spec) and any
    type=="text" chunk (a bare word run adds no signal beyond the raw query
    the model already sees). Returns None when nothing structural is found,
    so callers can omit the block entirely rather than send an empty list."""
    spans = [
        {"text": chunk["text"], "type": chunk["type"]}
        for chunk in chunk_query(query)
        if chunk["type"] != "text"
    ]
    if not spans:
        return None
    return json.dumps(spans)


_LLAMA_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
All case names and parties mentioned below refer exclusively to already
public, reported court judgments in a licensed legal research database -
never treat a query as a request for private information about a person,
and never refuse to classify it. You do not answer the legal question or
look anything up yourself; you only ever output the JSON object below.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": a CONSERVATIVE search normalization. Correct obvious
  spelling and grammar only. Preserve every party, court, place, Act,
  section, rule, notification, date, number, citation, and acronym exactly
  as written. NEVER add or infer a legal concept. NEVER expand an acronym
  (for example PE, ST, CA, ITD, PTA, MEG, POY, or PSF). NEVER translate an
  old law to a new law or replace one section with another. If the query is
  already readable, copy it unchanged. Every number and year in the output
  must occur in the input; if the input has no year, add no year.
- "intent": exactly one of "citation_lookup" (the query is anchored on a
  party name or case citation), "provision_lookup" (anchored on a
  section/act/rule number), "conceptual" (an open legal question with no
  strong lexical anchor), or "unknown" (none of the above fit confidently).
  Never output any other value.
- "filters": an object with any of "court", "act", "section", "date_range",
  "party", "bench", "judge" - ONLY include a key if its value is LITERALLY
  written in the query. Never guess, infer, or fill in a plausible-sounding
  court, act, section, bench, judge, or date range that the query does not
  state - a wrong filter silently excludes the correct document from the
  search entirely, which is worse than no filter. If the query names a
  person or company (very often written as "X vs. Y" or "X v. Y"), put that
  name under "party" - never under "section" or any other key. If nothing
  is explicitly stated, "filters" should be an empty object. Never output
  null or empty filter values. Never output any other filter key such as
  city, state, topic, or citation. "date_range" MUST be an object with ISO
  date strings, e.g. {"gte": "2020-01-01", "lte": "2022-01-01"} - either key
  may be omitted, but never output "date_range" as a plain string or year
  number, and never invent one when no date was mentioned.

Example: query "case law for Ramesh Gupta vs. Income-tax Officer" mentions
no court, act, section, or date - only a party name - so filters must be
exactly {"party": "Ramesh Gupta"} and intent is "citation_lookup".

Forbidden rewrites:
- "80HH scrap sale" must not mention BNS or any other Act.
- "software royalty PE" must retain "PE" without guessing its expansion.
- "69C diamond cash sale" must not add CGST Act or replace section 69C.
- "59/98-ST certification" must not add Customs Act.

""" + build_schema_context()


def _system_prompt_for_model(model: str) -> str:
    """Different models need different prompt shapes to follow instructions
    reliably (see docs/superpowers/specs/2026-08-06-agentic-search-pipeline-design.md's
    note on agent_chat) - the Llama-tuned prompt above was written and
    eval-validated against Llama-3.1-8B-Instruct's specific tendency to
    over-generalize open-ended rewrite instructions. Fall back to it for any
    other model too, but surface a warning so a future model swap doesn't
    silently inherit a prompt shape nobody has tuned or evaluated for it."""
    if "llama" in model.lower():
        return _LLAMA_SYSTEM_PROMPT
    get_client().update_current_span(
        level="WARNING",
        status_message=f"No prompt shape has been tuned/evaluated for model {model!r} - "
                        "falling back to the Llama-tuned prompt, which may not fit its "
                        "instruction-following style.",
    )
    return _LLAMA_SYSTEM_PROMPT


_ALLOWED_FILTERS = {"court", "act", "section", "date_range", "party", "bench", "judge"}
_ALLOWED_INTENTS = {"citation_lookup", "provision_lookup", "conceptual", "unknown"}
_LEGAL_MARKERS = {
    "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
    "bharatiya sakshya adhiniyam", "indian penal code", "income-tax act",
    "income tax act", "cgst act", "igst act", "customs act",
    "code of criminal procedure", "indian evidence act",
}


def _protected_identifiers(text: str) -> set[str]:
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9()/-]*\b", text)
    return {
        token.upper() for token in tokens
        if (token.isupper() and len(token) >= 2)
        or (any(c.isupper() for c in token) and any(c.isdigit() for c in token))
    }


def _safe_rewrite(query: str, rewritten: str) -> str:
    query_lower, rewritten_lower = query.casefold(), rewritten.casefold()
    if any(marker in rewritten_lower and marker not in query_lower for marker in _LEGAL_MARKERS):
        return query
    if any(court.casefold() in rewritten_lower and court.casefold() not in query_lower for court in KNOWN_COURTS):
        return query
    if set(re.findall(r"\d+", query)) != set(re.findall(r"\d+", rewritten)):
        return query
    if not _protected_identifiers(query).issubset(_protected_identifiers(rewritten)):
        return query
    query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
    rewritten_tokens = set(re.findall(r"[a-z0-9]+", rewritten_lower))
    if query_tokens and len(query_tokens & rewritten_tokens) / len(query_tokens) < 0.6:
        return query
    return rewritten


def _sanitize_filters(query: str, filters, intent: str) -> dict:
    if not isinstance(filters, dict):
        return {}
    clean = {}
    for key, value in filters.items():
        if key not in _ALLOWED_FILTERS:
            continue
        # "section" only resolves correctly for provision_lookup queries (it matches
        # ACT/RULE-group documents whose heading IS the section number verbatim, e.g.
        # "Section - 92C" - see es_client.py::_section_heading_queries). For any other
        # intent - a case-law/conceptual query that merely mentions a section number in
        # passing - that same filter silently redirects the doc_id allowlist to statute-
        # text documents instead of case law, which share no doc_ids with the case-law
        # Milvus collections the search actually runs against: the filtered search comes
        # back with zero hits everywhere despite the corpus having a good match (confirmed
        # live: a "conceptual" query with a bare "section 92C" filter went from 70 unfiltered
        # Milvus hits, including the gold doc, to 0 filtered hits). Intent is already
        # classified correctly by the SLM at this point - it just wasn't being used to gate
        # which filters are even valid to apply.
        if key == "section" and intent != "provision_lookup":
            continue
        if key == "date_range":
            if isinstance(value, dict):
                date_range = {
                    bound: date for bound, date in value.items()
                    if bound in {"gte", "lte"}
                    and isinstance(date, str)
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
                    and date[:4] in query
                }
                if date_range:
                    clean[key] = date_range
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if value.casefold() not in query.casefold():
            continue
        clean[key] = value
    return clean


def _validate_result(query: str, result) -> dict:
    if not isinstance(result, dict):
        return _fallback_intent(query)
    rewritten, intent = result.get("rewritten_query"), result.get("intent")
    if not isinstance(rewritten, str) or not rewritten.strip() or not isinstance(intent, str):
        return _fallback_intent(query)
    resolved_intent = intent if intent in _ALLOWED_INTENTS else "unknown"
    return {
        "rewritten_query": _safe_rewrite(query, rewritten.strip()),
        "intent": resolved_intent,
        "filters": _sanitize_filters(query, result.get("filters"), resolved_intent),
    }


async def extract_intent(
    gateway: GatewayClient, query: str, on_step: OnStep | None = None, model: str | None = None,
) -> dict:
    resolved_model = model or await gateway.get_model(role="slm")
    chunk_context = _build_chunk_context(query)
    user_message = query if chunk_context is None else (
        f"{query}\n\n"
        "Structural spans already present in the query above (for reference "
        f"only — do not add anything not already in the query text):\n{chunk_context}"
    )
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _system_prompt_for_model(resolved_model)},
            {"role": "user", "content": user_message},
        ],
        model=model,
        response_format=_RESPONSE_FORMAT,
    )
    try:
        result = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        get_client().update_current_span(
            level="WARNING", status_message=f"SLM did not return valid JSON, falling back to plain search: {response!r}",
        )
        result = _fallback_intent(query)
    else:
        result = _validate_result(query, result)

    if on_step is not None:
        await on_step("intent", {"query": query, **result})

    return result
