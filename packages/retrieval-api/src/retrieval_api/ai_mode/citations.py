import asyncio

import retrieval_api.ai_mode.rerank as rerank_module
from common.es_client import fetch_citations
from retrieval_api.ai_mode.intent import OnStep


async def prefetch_citations(es_client, candidates: list[dict], top_n_docs: int = 20) -> dict[str, dict]:
    ordered_by_score = sorted(candidates, key=lambda row: row["rrf_score"], reverse=True)
    seen: list[str] = []
    for row in ordered_by_score:
        doc_id = row["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) == top_n_docs:
            break
    return await fetch_citations(es_client, seen)


async def rerank_and_prefetch(
    gateway, es_client, query: str, candidates: list[dict], on_step: OnStep | None = None
) -> tuple[list[dict], dict[str, dict]]:
    top_chunks, citations = await asyncio.gather(
        rerank_module.rerank_top_chunks(gateway, query, candidates),
        prefetch_citations(es_client, candidates),
    )

    if on_step is not None:
        trace_chunks = [
            {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "rerank_score": c["rerank_score"], "text": c["text"]}
            for c in top_chunks
        ]
        await on_step("rerank", {"considered_count": len(candidates), "top_chunks": trace_chunks})

    return top_chunks, citations
