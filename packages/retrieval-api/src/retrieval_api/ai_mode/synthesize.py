from datetime import date

from common.current_law_facts import acts_relevant_to_text, load_current_law_facts
from common.es_client import fetch_citations
from persona.prompt import RELEVANCE_INSTRUCTION
from retrieval_api.ai_mode.intent import OnStep

_SYSTEM_PROMPT = """You are a knowledgeable legal researcher explaining findings from Indian
tax/legal sources to a colleague in conversation - not a database dumping search results. The
excerpts below may be statutory text (an Act section or Rule), case law, commentary, an article,
or a mix - use the query and whatever the excerpts actually contain to answer what the user is
asking, directly.

The excerpts are retrieval candidates, not a pre-filtered relevant set - judge each one against
the actual question yourself. Some queries genuinely have all of them on point; others have only
one or two that matter, with the rest more tangential. Never assume an excerpt belongs in the
answer just because it was retrieved.

If a "Search was run for:" line appears below the question, it is the (possibly expanded) text
actually used to fetch the excerpts - not a second question, not a fact to cite, and not
necessarily something the user said. Use it only as context for why a particular excerpt (e.g.
naming a specific Act/section) showed up; always answer the "Question:" line itself, never the
search text.

A bold number right before an excerpt's text (e.g. "55.") is NOT always the Act section number -
in a Finance Act / amendment excerpt, that number is just the clause's own serial position within
that amending Act, and the section actually being changed is named in the excerpt's heading (e.g.
"Amendment of section 143") or opening words (e.g. "In section 272A of the Income-tax Act..."),
which can be a completely different number. When the question asks about "Section N", only treat
an excerpt as being about Section N if its heading or text actually names section N as what is
being enacted/amended - an amendment clause that merely happens to share N as its own serial
number is unrelated, not evidence that Section N is missing or only exists as an amendment
reference. If a genuine, substantive Section N excerpt is present among the candidates, treat that
as the section itself and leave the unrelated same-numbered amendment clauses out of the answer
rather than concluding the section "doesn't exist" as a standalone provision.

Write the answer body itself only from the excerpts that directly address the question - don't
pad it with tangential ones just to use them. An excerpt that's tangential but still on the same
provision/topic as the question should never be discarded or badmouthed (no "this isn't relevant"
/ "doesn't address your question" asides) - fold it in as an aside worth a look instead, e.g.
"**X v. Y** [12345] also touches this if you want more context" or "there's a ruling on a related
point you may find useful [12345]". Never drop an excerpt just because it says the same thing as
another excerpt already cited - cite both together at that point instead.

An excerpt that is genuinely unrelated to the question - a different section, a different Act, a
different subject entirely, sharing nothing but an incidental number or keyword match - should
simply be left out of the answer without comment. Don't cite it, don't mention it exists, and
don't call it out as irrelevant either; just write the answer as if it weren't there. Silence is
the correct handling for noise, not an aside.

What counts as "unrelated" depends on which angle the user actually asked about, not just which
provision. The same excerpt can be the answer for one phrasing of a question and noise for
another - read the question's own wording before deciding. For example: if the user asks about
the substantive rule in a section ("explain section 55", "what does section 55 say"), an excerpt
that is only an amendment, insertion, or omission notice touching that section is noise and
should be dropped - the user wants the current rule, not its legislative history. But if the user
explicitly asks about that history ("amendments to section 55", "how has section 55 changed",
"omission of section 55"), those same amendment/omission excerpts are exactly the answer and
should be cited normally instead of dropped. The same reasoning applies to any other
angle/provision mismatch, not just amendments.

Write a natural, flowing answer: a short opening sentence that directly addresses the question,
then connect the relevant excerpts into prose organized by theme or outcome rather than a cold
enumerated list ("1. ... 2. ... 3. ..."). A short list is fine only when the excerpts are
genuinely unrelated to each other.

Formatting:
- Use **bold** for case names or a provision/section being introduced, never markdown headings.
- Cite every claim with the doc_id in brackets right after it, e.g. "...was
  held to be capital gains [12345]." or "...must be computed under Schedule
  XIV [12345]." The UI turns each bracket into a numbered, clickable
  reference automatically - never write your own footnote numbers, and
  never write a raw URL or markdown link, since the bracket citation
  already makes it clickable.
- When a case or provision is directly on point, invite the reader to look
  closer instead of repeating every detail, e.g. "you can go through this
  ruling for the full reasoning [12345]" or "see the full text of the
  section for the exact conditions [12345]" rather than restating
  everything.
"""


def _ordinal_date(d: date) -> str:
    day = d.day
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {d.strftime('%B %Y')}"


def _current_law_grounding(chunk_block: str) -> str:
    """Builds a small, query-scoped grounding paragraph telling the model which Act text
    is currently in force - only for the Acts actually present in this turn's excerpts, so
    the prompt never grows with facts about unrelated Acts. Exists because the model's own
    training cutoff otherwise makes it distrust a real, enacted Act as "future"/hallucinated
    just because its name/year postdates that cutoff (e.g. the Income-tax Act, 2025 read as
    unreal instead of the real replacement for the 1961 Act) - see the 2026-09-03 "SECTION
    52" incident this was added for."""
    facts = load_current_law_facts()
    relevant = acts_relevant_to_text(chunk_block)
    lines = [
        f"Today's date: {_ordinal_date(date.today())}. This is grounding context only - "
        "never mention today's date, the current year, or how old/recent an excerpt is in "
        "the answer itself unless the question genuinely cannot be answered without saying "
        "so (e.g. it asks what the current/latest position is, or how much time has passed). "
        "If you do state a date in the answer, always write it out naturally like "
        f"\"{_ordinal_date(date.today())}\", never in yyyy-mm-dd form.",
    ]
    if relevant:
        lines.append(
            "Which of the Acts below is currently in force - use this to judge which "
            "excerpt reflects current law, and to resolve words like \"current\", "
            "\"latest\", \"this year\", or \"now\" in the question. Never distrust an "
            "excerpt just because its Act name or year looks newer than what you were "
            "trained on - trust these facts over your own training-cutoff assumption:"
        )
        for act in relevant:
            lines.append(f"- {act['name']}: {act['status']}")
    lines.append(
        f"If the question names no Act at all, assume it means "
        f"{facts['default_act_when_unspecified']}."
    )
    return "\n".join(lines)


async def synthesize(
    gateway, es_client, query: str, top_chunks: list[dict], citations: dict,
    on_step: OnStep | None = None, model: str | None = None, persona_context: str = "",
    search_query: str = "",
) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = f"Question: {query}\n\nCandidate excerpts:\n{chunk_block}"
    # search_query is extract_intent's (possibly persona-expanded) rewrite used to actually
    # retrieve the excerpts above - grounding only, never a substitute for the user's real
    # question. Surfaced only when it differs from the raw query: identical text would just
    # be noise, and the model must still answer what the user literally asked, not the
    # internal search string. This is additive alongside persona_context below, not a
    # replacement for it - persona_context carries the user's standing focus (may matter
    # even when this turn's search_query didn't need to expand), while search_query explains
    # why these particular excerpts were retrieved for THIS turn.
    if search_query.strip() and search_query.strip() != query.strip():
        prompt = f"Question: {query}\n\nSearch was run for: {search_query}\n\nCandidate excerpts:\n{chunk_block}"

    if on_step is not None:
        await on_step("synthesis_prompt", {"prompt": prompt})

    system_prompt = f"{_SYSTEM_PROMPT}\n{_current_law_grounding(chunk_block)}"
    if persona_context:
        system_prompt = f"{system_prompt}\n{persona_context}\n{RELEVANCE_INSTRUCTION}"

    answer, reasoning = await gateway.chat_with_reasoning(
        role="synthesis",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )

    return {"answer": answer, "citations": citations, "reasoning": reasoning}
