import json
import logging

from langfuse import get_client

from persona.prompt import RELEVANCE_INSTRUCTION
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.gateway_client import GatewayClient

logger = logging.getLogger(__name__)

_RESPONSE_FORMAT = {"type": "json_object"}
_MAX_KEYWORDS = 2

_SYSTEM_PROMPT = """You are assisting a lexical (BM25/Elasticsearch) search over Indian tax
and statutory documents such as Acts , Rules , Articles , Tariff , Commentaries , etc.
You are given a query that is already a precise anchor lookup (a bare
section/rule/citation reference, a court name, an Act name) - it does not need rewriting or
reinterpreting. Your only job: suggest at most 2 additional real legal keywords or synonyms
that would help the lexical search surface more relevant documents, if any genuinely apply.

Rules:
- Only suggest a term if it is a real, established legal/tax term directly tied to what is
  already in the query (a well-known synonym, an alternate name for the same Act/concept, a
  closely related statutory term) - never a vague or generic word, never something you are
  not confident is correct.
- Never invent a party, court, date, section number, or citation that isn't already implied
  by the query. Never change what the query is about.
- A bare section/rule number with no Act, court, or subject named (e.g. "section 52",
  "rule 8") is ambiguous - the same number means a completely different provision in each Act
  it appears in (tax, company, procedural, or otherwise), and you have no way to know which one
  the query means. Never guess or assert what such a bare number "is about" or add a keyword
  based on that guess - a wrong guess actively misdirects the lexical search toward the wrong
  Act instead of leaving it to search on the number alone. Only add a keyword for a bare
  section/rule number if the query itself also names the Act, court, or subject.
- If you are not confident any term genuinely helps, output an empty list - this is the
  common, expected case, not a fallback to avoid.

Return ONLY a JSON object: {"keywords": [...]} - a list of 0 to 2 short strings, nothing
else."""


def _validate_keywords(existing_query: str, raw) -> list[str]:

    if not isinstance(raw, list):
        return []
    existing_lower = existing_query.lower()
    seen: set[str] = set()
    keywords: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        term = value.strip()
        if not term or term.lower() in existing_lower or term.lower() in seen:
            continue
        seen.add(term.lower())
        keywords.append(term)
        if len(keywords) == _MAX_KEYWORDS:
            break
    return keywords


async def expand_keyword_terms(
    gateway: GatewayClient, query: str, on_step: OnStep | None = None, persona_context: str = "",
) -> list[str]:
    """Opt-in SLM pass for AI Mode's keyword path (common.config.Settings.
    keyword_mode_expansion_enabled, env-only, default OFF) - the keyword path otherwise
    skips the SLM entirely (see pipeline.py). When enabled, asks the model for at most 2
    genuinely-confident legal keyword additions to broaden the ES lexical search, never a
    query rewrite. Degrades to no expansion (empty list) on any failure - an experimental
    recall booster must never turn into a hard failure for a path that worked fine without
    it a moment ago."""
    user_message = query
    if persona_context:
        # The base system prompt's "never guess on a bare section/rule number" rule
        # exists because a bare number is genuinely ambiguous across Acts with no way
        # for the model to know which one - but a persona note naming the user's usual
        # Act/subject is exactly the kind of context that can responsibly resolve that
        # ambiguity, same additive-only role persona plays in intent.py's search_query
        # expansion. Gated on non-empty persona_context so guest traffic and this
        # probe's own "before" baseline run are unaffected.
        user_message += (
            f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}\n"
            "If the query above is a bare section/rule number with no Act/court/subject "
            "of its own, you may use the note above to resolve which Act/subject it "
            "belongs to and add that Act name as a keyword - still capped at 2 keywords "
            "total. A keyword drawn from the note must be a term the note ITSELF names "
            "(the Act, the section/rule number, or a subject phrase literally present in "
            "it) - never your own independent guess at what that section covers beyond "
            "what the note actually says, even if you are confident of the general legal "
            "subject matter from training knowledge; that guess is exactly what produces a "
            "wrong-section keyword (e.g. inventing an unrelated deduction topic for a bare "
            "section number just because the note names that Act). If you are not "
            "genuinely confident the note is about this specific query - not just plausible, "
            "genuinely confident - output an empty list rather than using it. If the note "
            "conflicts with or is unrelated to the query, ignore it and follow the base "
            "rules above."
        )
    reasoning = None
    try:
        response, reasoning = await gateway.chat_with_reasoning(
            role="slm",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=_RESPONSE_FORMAT,
            temperature=0.6,
        )
        result = json.loads(response)
        keywords = _validate_keywords(query, result.get("keywords"))
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the keyword path over this
        get_client().update_current_span(
            level="WARNING", status_message=f"keyword expansion failed, continuing without it: {exc}",
        )
        keywords = []

    if on_step is not None:
        # reasoning is included unconditionally here (unlike the final ai_mode_done
        # message, which strips it behind common.config.Settings.expose_reasoning) -
        # same pattern intent.py's "intent" trace step already uses: the trace is a
        # dev/debug view (gated by the request's own "trace" flag), not the
        # user-facing answer, so it always shows why the model decided what it did -
        # including a genuinely-empty added_keywords, which is the common case and
        # exactly where "why didn't it add anything" is worth seeing.
        await on_step("keyword_expansion", {"query": query, "added_keywords": keywords, "reasoning": reasoning})

    return keywords
