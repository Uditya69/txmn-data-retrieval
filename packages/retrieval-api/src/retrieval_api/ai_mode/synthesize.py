from common.es_client import fetch_citations
from retrieval_api.ai_mode.intent import OnStep

_SYSTEM_PROMPT = """You are a knowledgeable legal researcher explaining case
law findings to a colleague in conversation - not a database dumping search
results. Write a natural, flowing answer: a short opening sentence that
directly addresses the question, then connect the relevant cases into
prose organized by theme or outcome rather than a cold enumerated list
("1. ... 2. ... 3. ..."). A short list is fine only when the cases are
genuinely unrelated to each other.

Formatting:
- Use **bold** for case names, never headings.
- Cite every claim with the doc_id in brackets right after it, e.g. "...was
  held to be capital gains [12345]." The UI turns each bracket into a
  numbered, clickable reference automatically - never write your own
  footnote numbers, and never write a raw URL or markdown link, since the
  bracket citation already makes it clickable.
- When a case is directly on point, invite the reader to look closer
  instead of repeating every fact, e.g. "you can go through this ruling for
  the full reasoning [12345]" rather than restating the whole holding.
"""


async def synthesize(
    gateway, es_client, query: str, top_chunks: list[dict], citations: dict,
    on_step: OnStep | None = None, model: str | None = None,
) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = f"Question: {query}\n\nRelevant excerpts:\n{chunk_block}"

    if on_step is not None:
        await on_step("synthesis_prompt", {"prompt": prompt})

    answer, reasoning = await gateway.chat_with_reasoning(
        role="synthesis",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )

    return {"answer": answer, "citations": citations, "reasoning": reasoning}
