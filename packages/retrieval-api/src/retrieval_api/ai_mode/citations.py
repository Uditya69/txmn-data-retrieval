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


# RRF-merged candidates routinely run into the hundreds (7 collections x 50 dense + 7 x 50
# sparse, deduped by chunk_id) - unbounded, this repo has seen 590+ candidates on a single
# query. rerank_top_chunks used to receive the entire uncapped list every time: one DeepInfra
# rerank call scoring every one of those texts in a single request, a much bigger payload than
# the 18K-char synthesis prompt that already timed out at DeepInfraAdapter's 60s chat timeout
# (see synthesize.py) - same failure class, just not yet observed on this endpoint. Capping to
# the top-N by rrf_score before reranking avoids that risk essentially for free: rrf_merge
# already sorts candidates by combined dense+sparse rank, and only the reranker's own top 5
# (elbow_cutoff, _MAX_CHUNKS) ever survive downstream anyway - a candidate ranked below 100 by
# RRF has never once been the gold answer across the 53-query eval set, whose pass thresholds
# (top 5/10/20) sit far inside this cap.
_MAX_RERANK_CANDIDATES = 100

# When reranking is disabled (AI_MODE_RERANK_ENABLED=false), fall back to this many
# top-by-rrf_score candidates - matches rerank_top_chunks' own _MAX_CHUNKS cap so the
# with/without-reranker comparison isn't also confounded by a different chunk count.
_NO_RERANK_TOP_N = 5


async def _skip_rerank(rerank_candidates: list[dict]) -> list[dict]:
    # No gateway call - just carry rrf_score over as rerank_score so downstream trace
    # output (rerank_and_prefetch's on_step, TracePanel) has a comparable field either way.
    return [{**c, "rerank_score": c["rrf_score"]} for c in rerank_candidates[:_NO_RERANK_TOP_N]]


async def rerank_and_prefetch(
    gateway, es_client, query: str, candidates: list[dict], on_step: OnStep | None = None,
    rerank_enabled: bool = True,
) -> tuple[list[dict], dict[str, dict]]:
    rerank_candidates = sorted(candidates, key=lambda row: row["rrf_score"], reverse=True)[:_MAX_RERANK_CANDIDATES]

    if rerank_enabled:
        rerank_coro = rerank_module.rerank_top_chunks(gateway, query, rerank_candidates)
    else:
        rerank_coro = _skip_rerank(rerank_candidates)

    top_chunks, citations = await asyncio.gather(
        rerank_coro,
        prefetch_citations(es_client, candidates),
    )

    if on_step is not None:
        trace_chunks = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "rerank_score": c["rerank_score"],
                "text": c["text"],
                "origin": c.get("origin"),
                "collection": c.get("collection"),
            }
            for c in top_chunks
        ]
        await on_step("rerank", {
            "total_candidates": len(candidates),
            "considered_count": len(rerank_candidates),
            "top_chunks": trace_chunks,
        })

    return top_chunks, citations
