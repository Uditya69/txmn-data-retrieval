from common.config import get_settings
from common.es_client import raw_search
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

_ES_BOOST_LIMIT = 10
_ES_BOOST_K = 60
_ES_BOOST_WEIGHT = 1.0


def _apply_es_doc_boost(
    merged: list[dict], es_hits: list[dict], k: int = _ES_BOOST_K, weight: float = _ES_BOOST_WEIGHT,
) -> list[dict]:
    """ES's top-N doc_ids for the same query are a rank-based confirming signal, not a raw
    score, so this stays inside CLAUDE.md rule 3's rank-position-only fusion rule. Boosted
    once per doc_id (the single best-scoring chunk already in `merged`), not fanned out to
    every chunk sharing that doc_id - a document with many surviving chunks shouldn't get
    the same ES signal counted multiple times."""
    es_rank_by_doc_id = {row["doc_id"]: rank for rank, row in enumerate(es_hits, start=1)}
    if not es_rank_by_doc_id:
        return merged

    best_row_by_doc_id: dict[str, dict] = {}
    for row in merged:
        doc_id = row["doc_id"]
        if doc_id not in es_rank_by_doc_id:
            continue
        current_best = best_row_by_doc_id.get(doc_id)
        if current_best is None or row["rrf_score"] > current_best["rrf_score"]:
            best_row_by_doc_id[doc_id] = row

    for doc_id, row in best_row_by_doc_id.items():
        row["rrf_score"] += weight / (k + es_rank_by_doc_id[doc_id])

    return sorted(merged, key=lambda row: row["rrf_score"], reverse=True)


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
    es_client,
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
    if on_step is not None:
        await on_step("milvus_dense", collection_trace(dense_by_collection))

    sparse_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_sparse", collection_trace(sparse_by_collection))

    merged = rrf_merge(
        _flatten(dense_by_collection), _flatten(sparse_by_collection),
        dense_weight=dense_weight, sparse_weight=sparse_weight,
    )

    if get_settings().ai_mode_es_boost_enabled:
        es_hits = await raw_search(es_client, rewritten_query, limit=_ES_BOOST_LIMIT)
        merged = _apply_es_doc_boost(merged, es_hits)
        if on_step is not None:
            await on_step("es_boost", {
                "es_doc_ids": [row["doc_id"] for row in es_hits],
                "boosted_doc_ids": sorted(set(row["doc_id"] for row in es_hits) & {row["doc_id"] for row in merged}),
            })

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
