import asyncio

import retrieval_api.ai_mode.rerank as rerank_module
from common.es_client import fetch_citations


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
    gateway, es_client, query: str, candidates: list[dict]
) -> tuple[list[dict], dict[str, dict]]:
    top_chunks, citations = await asyncio.gather(
        rerank_module.rerank_top_chunks(gateway, query, candidates),
        prefetch_citations(es_client, candidates),
    )
    return top_chunks, citations
