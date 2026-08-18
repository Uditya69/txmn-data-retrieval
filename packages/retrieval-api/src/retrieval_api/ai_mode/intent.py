import json
import re
from typing import Awaitable, Callable

from langfuse import get_client

from common.legal_lexicon import is_stopword
from common.query_tokenizer import chunk_query, classify_query_shape, expand_query_synonyms
from common.schema_context import KNOWN_COURTS, build_schema_context
from persona.prompt import RELEVANCE_INSTRUCTION
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
    return {"original_query": query, "search_query": query, "intent": [], "filters": {}}


def _build_chunk_context(query: str) -> str | None:
    """Trimmed JSON projection of chunk_query's structural spans, for injection
    into extract_intent's user message. Drops `proximity`/`alt_text` (ES-only,
    and alt_text's normalized form would never literal-match _sanitize_filters'
    substring check against the raw query - see design spec) and any
    type=="text" chunk (a bare word run adds no signal beyond the raw query
    the model already sees). Also drops any court_city chunk whose leading
    token is a stopword (e.g. "the court", "the tribunal") - merge_court_city
    merges any token immediately before court/high/tribunal regardless of
    whether it's a real place name, and injecting a stopword-led span as an
    authoritative structural span nudges the SLM toward a bogus court filter
    that _sanitize_filters cannot catch (its check only requires the value be
    a literal substring of the query, which a stopword phrase trivially is).
    Returns None when nothing structural is found, so callers can omit the
    block entirely rather than send an empty list.

    Note: chunk_query's upstream extract_quoted_phrases reorders tokens
    (quoted phrases first, then the rest), so in rare cases a later chunk's
    text is drawn from the query but is not a contiguous substring of the
    original query in its original order - the content still comes from the
    query text, just not guaranteed to appear at one exact span position."""
    spans = [
        {"text": chunk["text"], "type": chunk["type"]}
        for chunk in chunk_query(query)
        if chunk["type"] != "text"
        and not (chunk["type"] == "court_city" and is_stopword(chunk["text"].split()[0]))
    ]
    if not spans:
        return None
    return json.dumps(spans, ensure_ascii=False)


def _has_legal_anchor(query: str, chunk_context: str | None) -> bool:
    """True when any layer of the existing lexical pipeline (structural chunking, legal
    lexicon, shape classification) recognizes something in this query - a citation,
    section/rule reference, court/party name, date, or known legal abbreviation. False
    means the query is lexically empty of legal content, used both as a soft prompt hint
    (below) and a hard classification floor (_too_vague_to_tag, in _validate_result)."""
    if chunk_context is not None:
        return True  # a structural span (citation/section/court/date/party) was found
    if expand_query_synonyms(query) != query:
        return True  # a legal-lexicon term/abbreviation was recognized
    if classify_query_shape(query) != "plain":
        return True  # provision/citation shape implies an anchor
    return False


def _too_vague_to_tag(query: str, chunk_context: str | None) -> bool:
    """Deliberately no word-count or phrasing-shape (e.g. "ends in ?") exemption - see
    docs/superpowers/specs/2026-08-18-intent-lexicon-signal-and-vague-floor-design.md's
    "Explicit ruling" section. This knowingly force-empties anchor-free fact-pattern
    questions that extract_intent's caselaws signal would otherwise correctly tag -
    accepted because a guaranteed-safe search-all outcome was judged strictly
    preferable to any residual risk of a wrong-collection search."""
    return not _has_legal_anchor(query, chunk_context)


def build_lexicon_check(query: str) -> dict:
    """Public-facing summary of the lexical pipeline's read on a query - shape
    classification, structural spans, and the has_anchor verdict that drives both the
    soft lexicon-check prompt hint and the hard _too_vague_to_tag floor. Exposed for the
    /v1/intent-analysis and /v1/ai-mode-analysis test endpoints so a caller can see why
    the floor did or didn't fire on a given query, without a separate /v1/query-analysis
    call."""
    chunk_context = _build_chunk_context(query)
    return {
        "has_anchor": _has_legal_anchor(query, chunk_context),
        "shape": classify_query_shape(query),
        "chunks": json.loads(chunk_context) if chunk_context is not None else [],
    }


_LLAMA_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
All case names and parties mentioned below refer exclusively to already
public, reported court judgments in a licensed legal research database -
never treat a query as a request for private information about a person,
and never refuse to classify it. You do not answer the legal question or
look anything up yourself; you only ever output the JSON object below.
Given a user query, return ONLY a JSON object with exactly these keys:
- "search_query": a CONSERVATIVE search normalization. Correct obvious
  spelling and grammar only. Preserve every party, court, place, Act,
  section, rule, notification, date, number, citation, and acronym exactly
  as written. NEVER add or infer a legal concept. NEVER expand an acronym
  (for example PE, ST, CA, ITD, PTA, MEG, POY, or PSF). NEVER translate an
  old law to a new law or replace one section with another. If the query is
  already readable, copy it unchanged. Every number and year in the output
  must occur in the input; if the input has no year, add no year. Once you
  have decided "intent" below, phrase search_query to match what's actually
  being searched: if "acts"/"rules" is tagged, prefer the Act/Rule name plus
  section/rule number form already present in the query; if "caselaws"/
  "articles" is tagged, prefer party/court/precedent-style phrasing already
  present in the query; if "commentary" alone is tagged, keep plain-language
  phrasing. This only reorders/reframes words already in the query - it must
  still obey every rule above (no invented Act/court/number).
- "intent": a list of zero or more of the following six category labels -
  output every category that genuinely applies, but don't over-list; only
  tag a category the query actually anchors on:
  - "acts": primary legislation itself (Income-tax Act 1961, CGST Act,
    Customs Act, BNS, etc.) - sections, sub-sections, provisos, definitions,
    schedules. Signal: "section", "as per the Act", "definition under", a
    bare section+Act reference with no request for judicial interpretation.
  - "rules": subordinate legislation notified under an Act (Income-tax
    Rules 1962, CGST Rules, Customs Valuation Rules) - procedure,
    computation mechanics, prescribed forms. Distinct from "acts" by
    whether the query's number is a "rule" vs a "section"; a rules query
    often co-occurs with "acts" since every Rule has a parent Act.
  - "caselaws": judicial decisions (Supreme Court, High Courts, ITAT,
    CESTAT, AAR) - what was decided for a dispute/fact pattern. Two
    distinct signal types, either alone is sufficient:
    (a) literal markers - party names ("X vs Y"), "held", "case law on",
    "precedent for", a citation string, bench/judge name;
    (b) a fact-pattern or scenario question with no literal markers at all
    - "is X taxable when Y", "can a court order Z", "must an employer do
    W", "should authorities take action A" - any query asking what the
    legal outcome IS or WAS for a concrete situation, not what a
    provision MEANS in the abstract. This second type is just as strong a
    signal as the first and must not be missed for lacking a party name
    or citation - real case-law queries are very often phrased this way,
    asking about the outcome of a fact pattern without naming the case.
  - "articles": expert-authored opinion/analysis published in a journal or
    magazine - trend, controversy, recent development, practical impact.
    Not the publisher's own explanation (that's "commentary") and not
    binding law. Tag only on explicit signal ("article on...", "expert
    opinion on...", a named author) - don't default here.
  - "commentary": the publisher's own provision-by-provision plain-language
    explanation of how a section/Act/rule works in practice, no author
    byline. Distinct from "acts" (raw statutory text) and "articles" (named
    author's opinion piece). Default landing spot only for queries asking
    what a provision means or how it works in the abstract ("explain
    section X", "how is Y computed", "what does clause Z cover") - NOT for
    a concrete fact-pattern/scenario question ("is X taxable when Y"),
    which is a "caselaws" signal even when phrased in plain language with
    no case name. When a query could be read either way, tag both.
  - "tariff": customs/GST tariff classification and rates - HSN code
    lookups, duty rates, rate schedules, exemption notifications tied to a
    specific tariff heading/good. Distinct from "acts"/"rules" even though
    tariff notifications are issued under that law - if the ask is "what
    HSN/duty rate for [a specific good]", it's "tariff".
  Output an empty list when no category confidently applies. Never output
  any other value. If the user message below includes a "Lexicon check" note
  stating no legal term was recognized in the query, treat that as strong
  evidence to abstain (output an empty list) unless the query's own wording -
  not just its general subject - clearly names something concrete.
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
exactly {"party": "Ramesh Gupta"} and intent is ["caselaws"].

Example: query "case law on section 54F exemption eligibility" anchors on
both a case-law request and a specific Act section, so intent is
["acts", "caselaws"].

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
_ALLOWED_CATEGORIES = {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}
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


def _sanitize_filters(query: str, filters) -> dict:
    if not isinstance(filters, dict):
        return {}
    clean = {}
    for key, value in filters.items():
        if key not in _ALLOWED_FILTERS:
            continue
        # "section" is unconditionally dropped: it only resolves correctly against
        # ACT/RULE-group documents whose heading IS the section number verbatim (see
        # es_client.py::_section_heading_queries) - not case law that merely cites the
        # section. The old gate compared against intent=="provision_lookup", a value
        # that doesn't exist post category-rewrite (intent is now a category list, not
        # that 4-value enum) - rather than leave that comparison silently always-false,
        # it's made explicit here. Confirmed live (pre-rewrite): a conceptual query with
        # a bare "section 92C" filter went from 70 unfiltered Milvus hits (including the
        # gold doc) to 0 filtered hits. Revisit once section-filter gating is rebuilt
        # around category (not part of this change - see
        # docs/superpowers/specs/2026-08-14-category-collection-routing-design.md).
        if key == "section":
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


def _validate_categories(intent) -> list[str]:
    if not isinstance(intent, list):
        return []
    seen: list[str] = []
    for value in intent:
        if isinstance(value, str) and value in _ALLOWED_CATEGORIES and value not in seen:
            seen.append(value)
    return seen


def _validate_result(query: str, result, chunk_context: str | None) -> dict:
    if not isinstance(result, dict):
        return _fallback_intent(query)
    search_query = result.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        return _fallback_intent(query)
    return {
        "original_query": query,
        "search_query": _safe_rewrite(query, search_query.strip()),
        "intent": [] if _too_vague_to_tag(query, chunk_context) else _validate_categories(result.get("intent")),
        "filters": _sanitize_filters(query, result.get("filters")),
    }


async def extract_intent(
    gateway: GatewayClient, query: str, on_step: OnStep | None = None, model: str | None = None,
    persona_context: str = "",
) -> dict:
    resolved_model = model or await gateway.get_model(role="slm")
    chunk_context = _build_chunk_context(query)
    user_message = query if chunk_context is None else (
        f"{query}\n\n"
        "Structural spans already present in the query above (for reference "
        f"only — do not add anything not already in the query text):\n{chunk_context}"
    )
    has_anchor = _has_legal_anchor(query, chunk_context)
    if not has_anchor:
        user_message += (
            "\n\nLexicon check: no known legal term, Act/section reference, citation, or "
            "party pattern was recognized anywhere in this query."
        )
    if persona_context:
        user_message += f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
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
        result = _validate_result(query, result, chunk_context)

    if on_step is not None:
        await on_step("intent", {"query": query, **result})

    return result
