from common.config import get_settings
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.trace_utils import collection_trace

_INTENT_RRF_WEIGHTS: dict[str, tuple[float, float]] = {
    "citation_lookup": (0.5, 1.5),
    "provision_lookup": (0.5, 1.5),
    "conceptual": (1.5, 0.5),
    "unknown": (1.0, 1.0),
}


def rrf_merge(
    dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60,
    dense_weight: float = 1.0, sparse_weight: float = 1.0,
) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked_list, weight in ((dense_ranked, dense_weight), (sparse_ranked, sparse_weight)):
        for rank, row in enumerate(ranked_list, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            rows.setdefault(chunk_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[chunk_id], "rrf_score": score} for chunk_id, score in ordered]


def _flatten(by_collection: dict[str, list[dict]]) -> list[dict]:
    flattened = [row for rows in by_collection.values() for row in rows]
    return sorted(flattened, key=lambda row: row["score"], reverse=True)


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    rewritten_query: str,
    doc_id_allowlist: list[str] | None,
    intent: str = "unknown",
    on_step: OnStep | None = None,
) -> list[dict]:
    dense_weight, sparse_weight = (
        _INTENT_RRF_WEIGHTS.get(intent, (1.0, 1.0)) if get_settings().intent_rrf_weighting_enabled else (1.0, 1.0)
    )

    dense_vector = await gateway.embed(role="query_embed", text=rewritten_query)

    dense_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    sparse_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )

    # Circuit breaker: a resolved doc_id_allowlist that's non-empty but the wrong kind of
    # document for these collections (e.g. the section-filter bug intent.py's section/intent
    # gate exists to prevent - a filter meant for ACT/RULE statute-text documents applied to
    # a case-law query) silently zeroes every collection even though an unfiltered search
    # would find real matches. If the allowlist was non-empty but produced zero hits
    # everywhere, retry once unfiltered rather than returning nothing - the embedding is
    # already computed, so this only costs the two Milvus round-trips, and only in the case
    # that's already about to return zero results anyway.
    if doc_id_allowlist and not any(dense_by_collection.values()) and not any(sparse_by_collection.values()):
        if on_step is not None:
            await on_step("filter_fallback", {
                "reason": "doc_id_allowlist matched zero Milvus results across every collection; retrying unfiltered",
                "doc_id_allowlist_count": len(doc_id_allowlist),
            })
        dense_by_collection = await hybrid_search(
            milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector,
            sparse_query_text=rewritten_query, doc_id_allowlist=None, limit=50,
        )
        sparse_by_collection = await hybrid_search(
            milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
            sparse_query_text=rewritten_query, doc_id_allowlist=None, limit=50,
        )

    if on_step is not None:
        await on_step("milvus_dense", collection_trace(dense_by_collection))
        await on_step("milvus_sparse", collection_trace(sparse_by_collection))

    merged = rrf_merge(
        _flatten(dense_by_collection), _flatten(sparse_by_collection),
        dense_weight=dense_weight, sparse_weight=sparse_weight,
    )

    if on_step is not None:
        top_candidates = [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "rrf_score": row["rrf_score"],
                "text_preview": row["text"][:200],
            }
            for row in merged[:15]
        ]
        await on_step("rrf_merge", {
            "candidate_count": len(merged), "top_candidates": top_candidates,
            "dense_weight": dense_weight, "sparse_weight": sparse_weight,
        })

    return merged
