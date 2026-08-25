import json
import re
from typing import Awaitable, Callable

from langfuse import get_client

from common.legal_lexicon import is_stopword
from common.query_tokenizer import chunk_query, classify_query_shape, expand_query_synonyms
from common.schema_context import build_schema_context
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

- "search_query": the text that will actually be run against the search
  backend.
  - Preserve every party, court, place, Act, section, rule, notification,
    date, number, citation, and acronym the user actually typed - never
    drop, rename, or silently swap one for a different one. NEVER
    translate an old law to a new law or replace one section with
    another.
  - Correct obvious spelling/grammar. Casing, spacing, punctuation,
    hyphenation, and exact phrasing used anywhere in this prompt's own
    text or examples (e.g. "Income-tax Act") are illustrative only, not a
    template - real user input arrives in every casing, spacing,
    abbreviation, and misspelling imaginable, and must never be expected
    to match this prompt's own formatting choices.
  - If the query is short, bare, or just a keyword/citation with little
    surrounding context (e.g. "section 55", "cost of acquisition", "PE")
    and you are CONFIDENT what it refers to, you may add closely related
    supporting context to search_query - the Act/Rule name it belongs to,
    the section/rule number, or the concept a bare term names - to make
    the search more effective. Only add what you are genuinely confident
    is correct and directly related to what's already there; when unsure,
    leave the query as-is rather than guess. Never change the query's
    meaning, never substitute a different legal concept for the one
    asked about, never invent a party/court/date that isn't implied by
    what the user wrote.
  - Default assumption for a bare section/rule number with no Act/Rule
    named anywhere in the query: treat it as the Income-tax Act, 1961 -
    this system's overwhelming default domain - and add "Income-tax Act
    1961" to search_query. This applies no matter which intent category
    below ends up tagged (acts, rules, commentary, or caselaws alike) -
    the category only changes how search_query is phrased afterward
    (see the "intent" rules below), not whether the Act name gets added.
    Only skip this default when the query itself names a different Act,
    or a different Act is unambiguously implied by other content in the
    query (e.g. a court/case context that only makes sense under a
    different Act).
  - If the query is already a clear, complete sentence, keep changes
    minimal - reordering/reframing what's already present is usually
    enough; only add something new when an obvious anchor is still
    missing (e.g. a bare section number with "acts"/"rules" tagged, and
    no Act/Rule name yet in the query).
  - Once you have decided "intent" below, phrase search_query to match
    what's actually being searched: if "acts"/"rules" is tagged, prefer
    the Act/Rule name plus section/rule number form; if "caselaws"/
    "articles" is tagged, prefer party/court/precedent-style phrasing; if
    "commentary" alone is tagged, keep plain-language phrasing.
  Example: query "section 55" with intent ["acts"] -> search_query
  "Income-tax Act 1961 Section 55 cost of acquisition" (confident: section
  55 of the Income-tax Act deals with cost of acquisition/improvement -
  the Act name and year were added, nothing already in the query was
  changed or removed).

- "intent": Return one or more of the following categories only. Tag a
  category only when the query genuinely anchors on it - don't over-list.

  - "acts"
      The law itself, as enacted by Parliament - the section, sub-section,
      proviso, definition, or schedule text of an Act.
      Select when:
        - Query cites a section number alongside an Act name (Income-tax
          Act, CGST Act, Customs Act, BNS, etc.)
        - Query uses "section", "as per the Act", "definition under"
      Example: "What does section 80C of the Income-tax Act cover?"

  - "rules"
      Subordinate rules issued under an Act - procedure, computation
      mechanics, or prescribed forms that operationalize the Act.
      Select when:
        - Query cites a "rule" number
        - Query asks about a prescribed form, procedure, or Rule-level
          mechanics (Income-tax Rules, CGST Rules, Customs Valuation Rules)
      Example: "Which form is prescribed under Rule 12 of the Income-tax
      Rules?"

  - "caselaws"
      A judicial decision - what a court actually decided for a real
      dispute or fact pattern.
      Select when:
        - Query names something concrete: parties ("X vs Y"), "held",
          "case law on", "precedent for", a citation string, a bench/judge
        - OR query describes a real-world situation and asks what would
          legally happen ("is X taxable when Y", "can a court order Z")
      Example: "Is compensation received for compulsory land acquisition
      taxable?"

  - "articles"
      A named expert's own published opinion or analysis - not binding
      law, not the publisher's own explanatory writing.
      Select when:
        - Query explicitly asks for "article on...", "expert opinion
          on...", or names an author
      Example: "Any recent articles on the impact of the new TDS rules on
      freelancers?"

  - "commentary"
      The publisher's own plain-language, provision-by-provision
      explanation of how a section/Act/rule works - unauthored.
      Select when:
        - Query asks what a provision means or how it works in the
          abstract: "explain section X", "how is Y computed"
      Example: "How is depreciation computed under the Income-tax Act?"

  - "tariff"
      Customs/GST tariff classification and rates for one specific good.
      Select when:
        - Query asks for an HSN code, duty rate, rate schedule, or
          exemption notification tied to a specific good or tariff heading
      Example: "What is the customs duty rate for imported solar panels?"

  Boundary cases - when a query could match more than one category, use
  these to decide:
    - rules vs acts: if the query cites a Rule, also tag acts alongside
      it - every Rule has a parent Act.
    - commentary vs caselaws: if the query describes a real-world
      situation and asks what would legally happen, tag caselaws, not
      commentary - even in plain language with no case name.
    - commentary vs articles: commentary has no named author; if the
      query names an author or asks for a published opinion/analysis,
      tag articles instead.
    - commentary vs acts: commentary is the explanation of a provision;
      acts is the provision's own statutory text. When the query names a
      specific section/rule number and asks to "explain"/"what is"/"tell
      me about" it, tag BOTH acts and commentary together, not one
      exclusively - the section's own text is exactly what the
      commentary is explaining, so retrieving only the explanation and
      never the provision itself would leave out the thing being
      explained. Tag commentary alone only when the query is about a
      broader mechanism/topic with no single section/rule anchor (e.g.
      "how is depreciation computed").
    - tariff vs acts/rules: if the actual ask is an HSN code or duty
      rate for a specific good, tag tariff, even though the notification
      is technically issued under an Act or Rules.

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

  "party" and "X vs. Y" case names: every case has two sides, but "party"
  takes ONLY the specific, named side - never the whole "X vs. Y" string,
  and never a bare government/office designation on its own. Concretely:
  - "Priya Sharma vs. Commissioner of Income Tax" -> party: "Priya Sharma"
    (the named individual/company), not "Priya Sharma vs. Commissioner of
    Income Tax" and not "Commissioner of Income Tax" alone.
  - The other side (Commissioner of Income Tax, Income-tax Officer, ACIT,
    Union of India, State of X, etc.) is a generic office name that recurs
    across thousands of unrelated cases - it does not usefully narrow a
    search, so never output it as "party" by itself.
  - If BOTH sides are specific named entities (e.g. two companies, or two
    individuals in a partition/joint case) with no generic office on
    either side, use the first-named side only - "party" holds one string,
    not a list; do not concatenate both names into it.

Example: query "case law for Ramesh Gupta vs. Income-tax Officer" mentions
no court, act, section, or date - only a party name - so filters must be
exactly {"party": "Ramesh Gupta"} and intent is ["caselaws"].

""" + build_schema_context()


# Qwen3-4B-Thinking-2507's own reasoning trace (visible via reasoning_content, plumbed
# through gateway_client.chat_with_reasoning) showed it reading the Llama prompt's nested
# IF/AND/OR bullet checklists as literal boolean logic to satisfy,
# then talking itself out of an obviously-correct tag (a bare "Section 52" query) because
# one sub-bullet ("cites a section number alongside an Act name") wasn't met, even though
# a second, independently-sufficient sub-bullet ("query uses 'section'") was. Confirmed:
# this exact model has a documented instruction-following gap on rigid structured-output
# checklists - https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/discussions/2. Two
# changes from the Llama prompt, deliberately: (1) categories are described as flowing
# prose ("what this category IS"), not nested trigger bullets ("IF X AND/OR Y") - a
# reasoning model already reasons about definitions in prose in its own <think> block, so
# meeting it in that shape avoids the AND/OR-parsing failure mode entirely; (2) the
# Llama prompt's separate "Boundary cases" bullet list is dropped rather than restated -
# each definition below is written to already carry its own boundary (e.g. Rule's
# definition names its relationship to the parent Act inline) rather than layering a
# second, separately-parseable rule list on top that can disagree with the first. Backed
# by worked examples instead, since few-shot anchors this model to the intended
# read reliably where abstract restated rules did not.
_QWEN3_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
All case names and parties mentioned below refer exclusively to already
public, reported court judgments in a licensed legal research database -
never treat a query as a request for private information about a person,
and never refuse to classify it. You do not answer the legal question or
look anything up yourself; you only ever output the JSON object below.
Given a user query, return ONLY a JSON object with exactly these keys:

- "search_query": the text that will actually be run against the search
  backend.
  - Preserve every party, court, place, Act, section, rule, notification,
    date, number, citation, and acronym the user actually typed - never
    drop, rename, or silently swap one for a different one. NEVER
    translate an old law to a new law or replace one section with
    another.
  - Correct obvious spelling/grammar. Casing, spacing, punctuation,
    hyphenation, and exact phrasing used anywhere in this prompt's own
    text or examples (e.g. "Income-tax Act") are illustrative only, not a
    template - real user input arrives in every casing, spacing,
    abbreviation, and misspelling imaginable, and must never be expected
    to match this prompt's own formatting choices.
  - If the query is short, bare, or just a keyword/citation with little
    surrounding context (e.g. "section 55", "cost of acquisition", "PE")
    and you are CONFIDENT what it refers to, you may add closely related
    supporting context to search_query - the Act/Rule name it belongs to,
    the section/rule number, or the concept a bare term names - to make
    the search more effective. Only add what you are genuinely confident
    is correct and directly related to what's already there; when unsure,
    leave the query as-is rather than guess. Never change the query's
    meaning, never substitute a different legal concept for the one
    asked about, never invent a party/court/date that isn't implied by
    what the user wrote.
  - Default assumption for a bare section/rule number with no Act/Rule
    named anywhere in the query: treat it as the Income-tax Act, 1961 -
    this system's overwhelming default domain - and add "Income-tax Act
    1961" to search_query. This applies no matter which intent category
    below ends up tagged (acts, rules, commentary, or caselaws alike) -
    the category only changes how search_query is phrased afterward
    (see the "intent" rules below), not whether the Act name gets added.
    Only skip this default when the query itself names a different Act,
    or a different Act is unambiguously implied by other content in the
    query (e.g. a court/case context that only makes sense under a
    different Act).
  - If the query is already a clear, complete sentence, keep changes
    minimal - reordering/reframing what's already present is usually
    enough; only add something new when an obvious anchor is still
    missing (e.g. a bare section number with "acts"/"rules" tagged, and
    no Act/Rule name yet in the query).
  - Once you have decided "intent" below, phrase search_query to match
    what's actually being searched: if "acts"/"rules" is tagged, prefer
    the Act/Rule name plus section/rule number form; if "caselaws"/
    "articles" is tagged, prefer party/court/precedent-style phrasing; if
    "commentary" alone is tagged, keep plain-language phrasing.
  Example: query "section 55" with intent ["acts"] -> search_query
  "Income-tax Act 1961 Section 55 cost of acquisition" (confident: section
  55 of the Income-tax Act deals with cost of acquisition/improvement -
  the Act name and year were added, nothing already in the query was
  changed or removed).

- "intent": Return one or more of the categories below - never more than
  genuinely applies, never fewer. Judge each query against what the
  category actually IS, not a checklist of trigger phrases; the definitions
  below are written to be sufficient on their own.

  - "acts": A primary legislation enacted by Parliament or a State
    Legislature. It contains the main substantive law, including
    definitions, rights, obligations, powers, procedures and penalties.
    Queries asking about a section, statutory provision, legal requirement,
    eligibility, liability or interpretation of the Act itself relate to
    an Act.

  - "rules": A subordinate/delegated legislation made under the authority
    of an Act by the Government or another competent authority. Rules
    generally prescribe the detailed procedure, conditions, forms, manner,
    timelines or implementation mechanism for provisions of the Act.
    Queries referring to a rule, prescribed procedure, form, manner,
    condition or compliance requirement relate to Rules.

  - "caselaws": The law and legal principles emerging from judgments,
    orders or decisions of courts, tribunals or other judicial/quasi-
    judicial authorities. Case law is relevant where a customer seeks
    judicial interpretation, legal precedent, applicability of a judgment,
    treatment of a factual situation by courts, or the current judicial
    position on an issue.

  - "articles": An explanatory or analytical publication written by a
    subject-matter expert discussing a legal, tax, regulatory or practical
    issue. It may analyse legislation, rules, case law and recent
    developments and provide interpretation or practical guidance. Queries
    seeking explanation, analysis, practical understanding, overview,
    implications or expert discussion of a topic - especially one naming
    an author or asking for a published opinion - may relate to an Article.

  - "commentary": A detailed expert explanation and interpretation of a
    specific Act, provision, rule or legal subject, usually organised
    provision-wise or topic-wise. Commentary explains the meaning, scope,
    background, interpretation and practical application of the law and
    may refer to relevant case law and other authorities. Queries
    requiring in-depth interpretation or comprehensive understanding of a
    provision or subject - with no named author and no real-world fact
    pattern - relate to Commentary. When the query names a specific
    section/rule number and asks to "explain"/"what is"/"tell me about"
    it, tag "acts" alongside "commentary", not commentary alone - the
    section's own statutory text is exactly what the commentary explains,
    so leaving it out would retrieve the explanation without the thing
    being explained. Tag commentary alone only for a broader mechanism/
    topic query with no single section/rule anchor (e.g. "how is
    depreciation computed").

  - "tariff": Customs/GST tariff classification, HSN code, duty rate or
    exemption applicable to a specific good, product or service under the
    Customs Tariff Act, GST law or related notifications. Tariff is
    relevant where a customer seeks HSN classification, applicable
    duty/tax rate, exemption notification, or the correct tariff heading
    for a particular good or import/export transaction.

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

  "party" and "X vs. Y" case names: every case has two sides, but "party"
  takes ONLY the specific, named side - never the whole "X vs. Y" string,
  and never a bare government/office designation on its own. Concretely:
  - "Priya Sharma vs. Commissioner of Income Tax" -> party: "Priya Sharma"
    (the named individual/company), not "Priya Sharma vs. Commissioner of
    Income Tax" and not "Commissioner of Income Tax" alone.
  - The other side (Commissioner of Income Tax, Income-tax Officer, ACIT,
    Union of India, State of X, etc.) is a generic office name that recurs
    across thousands of unrelated cases - it does not usefully narrow a
    search, so never output it as "party" by itself.
  - If BOTH sides are specific named entities (e.g. two companies, or two
    individuals in a partition/joint case) with no generic office on
    either side, use the first-named side only - "party" holds one string,
    not a list; do not concatenate both names into it.

Worked examples (query -> output), covering cases that are easy to
misjudge:

1. Query: "Section 52"
   A bare section reference, no Act named. Still squarely "asking about a
   section" per the acts definition - the Act's name being unstated doesn't
   make it any less a question about statutory text.
   -> {"search_query": "Section 52", "intent": ["acts"], "filters": {}}

2. Query: "prescribed form and procedure under Rule 6 of the Income-tax
   Rules 1962 for TDS returns"
   A Rule citation describing its own prescribed procedure - "rules" per
   definition; no separate case, author, or fact pattern is asked about.
   -> {"search_query": "prescribed form and procedure under Rule 6 of the
   Income-tax Rules 1962 for TDS returns", "intent": ["rules"],
   "filters": {}}

3. Query: "case law for Ramesh Gupta vs. Income-tax Officer"
   Names a specific party and a generic office; no Act, section, or date.
   -> {"search_query": "case law for Ramesh Gupta vs. Income-tax Officer",
   "intent": ["caselaws"], "filters": {"party": "Ramesh Gupta"}}

4. Query: "How is depreciation computed under the Income-tax Act?"
   Asks how a provision works in the abstract, no named author, no real
   dispute - commentary, not acts (the question is about the mechanism,
   not the statutory text itself) and not caselaws (no fact pattern).
   -> {"search_query": "How is depreciation computed under the Income-tax
   Act?", "intent": ["commentary"], "filters": {}}

5. Query: "Any recent articles on the impact of the new TDS rules on
   freelancers?"
   Explicitly asks for articles/expert analysis, not the rule text itself.
   -> {"search_query": "Any recent articles on the impact of the new TDS
   rules on freelancers?", "intent": ["articles"], "filters": {}}

6. Query: "What is the customs duty rate for imported solar panels?"
   Asks for a duty rate on one specific good.
   -> {"search_query": "What is the customs duty rate for imported solar
   panels?", "intent": ["tariff"], "filters": {}}

""" + build_schema_context()


def _system_prompt_for_model(model: str) -> str:
    """Different models need different prompt shapes to follow instructions
    reliably - the Llama-tuned prompt above was written and eval-validated
    against Llama-3.1-8B-Instruct's specific tendency to over-generalize
    open-ended rewrite instructions. Qwen3-4B-Thinking-2507 gets its
    own prompt (see _QWEN3_SYSTEM_PROMPT's docstring for why the shape differs) rather
    than silently inheriting the Llama prompt. Fall back to the Llama prompt for any
    other/unrecognized model too, but surface a warning so a future model swap doesn't
    silently inherit a prompt shape nobody has tuned or evaluated for it."""
    model_lower = model.lower()
    if "llama" in model_lower:
        return _LLAMA_SYSTEM_PROMPT
    if "qwen3" in model_lower:
        return _QWEN3_SYSTEM_PROMPT
    get_client().update_current_span(
        level="WARNING",
        status_message=f"No prompt shape has been tuned/evaluated for model {model!r} - "
                        "falling back to the Llama-tuned prompt, which may not fit its "
                        "instruction-following style.",
    )
    return _LLAMA_SYSTEM_PROMPT


_ALLOWED_FILTERS = {"court", "act", "section", "date_range", "party", "bench", "judge"}
_ALLOWED_CATEGORIES = {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}


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
        # Trusted verbatim from the SLM - no post-hoc rejection/rewrite-to-original.
        # See _QWEN3_SYSTEM_PROMPT/_LLAMA_SYSTEM_PROMPT's search_query instructions for
        # the guardrails now enforced at the prompt level instead (confidence-gated
        # expansion of bare/keyword queries, never inventing an unrelated concept).
        "search_query": search_query.strip(),
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
        "Structural spans already present in the query above (for reference, to help "
        "judge what's already there and how confident an expansion would be - "
        f"search_query may still expand per the system instructions above):\n{chunk_context}"
    )
    has_anchor = _has_legal_anchor(query, chunk_context)
    if not has_anchor:
        user_message += (
            "\n\nLexicon check: no known legal term, Act/section reference, citation, or "
            "party pattern was recognized anywhere in this query."
        )
    if persona_context:
        user_message += f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
    response, reasoning = await gateway.chat_with_reasoning(
        role="slm",
        messages=[
            {"role": "system", "content": _system_prompt_for_model(resolved_model)},
            {"role": "user", "content": user_message},
        ],
        model=model,
        response_format=_RESPONSE_FORMAT,
        # Was pinned to near-zero (0.01) for determinism against a plain-completion model;
        # that fought Qwen3-4B-Thinking-2507's tuned decoding distribution once the slm
        # role moved to it (CHAT_PROVIDER=local) - near-greedy decoding on a Thinking model
        # is a plausible contributor to the self-contradicting reasoning loops observed in
        # its reasoning_content (e.g. re-deriving the same "Section 52" verdict five times
        # before landing on the wrong one). 0.6/top_p=0.95/top_k=20/min_p=0 is Qwen's own
        # recommended sampling config for the -Thinking variant - only temperature is
        # plumbed through this call today, so only that moves here. Determinism across
        # identical calls (the original reason for pinning near-zero - collections_for_
        # intent() routing depends on a stable "intent" list) is no longer guaranteed at
        # this setting; re-evaluate against evals/intent_filter_cases.json before relying
        # on this for routing-sensitive comparisons. https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
        temperature=0.6,
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

    result["reasoning"] = reasoning

    if on_step is not None:
        await on_step("intent", {"query": query, **result})

    return result
