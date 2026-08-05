from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.gateway_client import GatewayClient


def rrf_merge(dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked_list in (dense_ranked, sparse_ranked):
        for rank, row in enumerate(ranked_list, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            rows.setdefault(chunk_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[chunk_id], "rrf_score": score} for chunk_id, score in ordered]


def _flatten(by_collection: dict[str, list[dict]]) -> list[dict]:
    flattened = [row for rows in by_collection.values() for row in rows]
    return sorted(flattened, key=lambda row: row["score"], reverse=True)


def _collection_trace(by_collection: dict[str, list[dict]]) -> dict:
    return {
        "collections": [
            {
                "name": name,
                "hit_count": len(rows),
                "top_hits": [
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "score": row["score"],
                        "text_preview": row["text"][:200],
                    }
                    for row in rows[:5]
                ],
            }
            for name, rows in by_collection.items()
        ]
    }


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    rewritten_query: str,
    doc_id_allowlist: list[str] | None,
    on_step: OnStep | None = None,
) -> list[dict]:
    dense_vector = await gateway.embed(role="query_embed", text=rewritten_query)

    dense_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_dense", _collection_trace(dense_by_collection))

    sparse_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_sparse", _collection_trace(sparse_by_collection))

    merged = rrf_merge(_flatten(dense_by_collection), _flatten(sparse_by_collection))

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
        await on_step("rrf_merge", {"candidate_count": len(merged), "top_candidates": top_candidates})

    return merged
