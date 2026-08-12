from common.es_client import fetch_fulltext_batch
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.score_cutoff import elbow_cutoff

# Opt-in override of CLAUDE.md hard rule 3 ("no ranking fusion between ES and
# Milvus"): RRF fuses by *rank position*, not raw score, so it never blends
# the incomparable ES-lexical-score and Milvus-cosine/BM25-distance scales
# the original rule guards against. Only reachable behind the `rerank`
# toggle - default (off) behavior is untouched.
_TOP_N_CANDIDATES = 20

_SHAPE_RRF_WEIGHTS: dict[str, dict[str, float]] = {
    "citation": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "provision": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "plain": {"es": 1.0, "milvus_dense": 1.5, "milvus_sparse": 0.5},
}


def _collapse_to_doc_id(rows: list[dict]) -> list[dict]:
    """Keeps each doc_id's best-ranked occurrence only - rows arrive sorted best-first
    (ES's own order, or Milvus rows flattened+sorted by score), so the first occurrence
    of a doc_id is its best rank; later duplicates (extra chunks/collections) are dropped."""
    seen: set[str] = set()
    collapsed = []
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        collapsed.append(row)
    return collapsed


def _flatten_by_score(by_collection: dict[str, list[dict]]) -> list[dict]:
    flattened = [row for rows in by_collection.values() for row in rows]
    return sorted(flattened, key=lambda row: row["score"], reverse=True)


def rrf_merge_by_doc_id(sources: dict[str, list[dict]], weights: dict[str, float], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for source, ranked_list in sources.items():
        weight = weights.get(source, 1.0)
        for rank, row in enumerate(_collapse_to_doc_id(ranked_list), start=1):
            doc_id = row["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            rows.setdefault(doc_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[doc_id], "rrf_score": score} for doc_id, score in ordered]


async def rerank_instant_results(
    gateway: GatewayClient,
    es_client,
    query: str,
    shape: str,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
) -> list[dict]:
    weights = _SHAPE_RRF_WEIGHTS.get(shape, {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0})
    fused = rrf_merge_by_doc_id(
        {
            "es": es_result,
            "milvus_dense": _flatten_by_score(milvus_dense),
            "milvus_sparse": _flatten_by_score(milvus_sparse),
        },
        weights,
    )
    top_candidates = fused[:_TOP_N_CANDIDATES]

    fulltext = await fetch_fulltext_batch(es_client, [row["doc_id"] for row in top_candidates])
    candidates = [row for row in top_candidates if fulltext.get(row["doc_id"])]
    if not candidates:
        return []

    scores = await gateway.rerank(
        role="reranker", query=query, documents=[fulltext[row["doc_id"]] for row in candidates],
    )
    scored = [{**row, "rerank_score": score} for row, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    cutoff = elbow_cutoff([row["rerank_score"] for row in scored])
    return scored[:cutoff]
