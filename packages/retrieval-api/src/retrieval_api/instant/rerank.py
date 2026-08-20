from common.es_client import fetch_fulltext_batch, trim_to_token_budget
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.score_cutoff import elbow_cutoff

# Opt-in override of CLAUDE.md hard rule 3 ("no ranking fusion between ES and
# Milvus"): RRF fuses by *rank position*, not raw score, so it never blends
# the incomparable ES-lexical-score and Milvus-cosine/BM25-distance scales
# the original rule guards against. Only reachable behind the `rerank`
# toggle - default (off) behavior is untouched.
_TOP_N_CANDIDATES = 20

_LABEL_RRF_WEIGHTS: dict[str, dict[str, float]] = {
    "KEYWORD": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "HYBRID": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "INTENT": {"es": 1.0, "milvus_dense": 1.5, "milvus_sparse": 0.5},
    "FALLBACK": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
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
    label: str,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
    rrf: bool = True,
    rerank: bool = True,
    on_step: OnStep | None = None,
) -> list[dict]:
    """rrf and rerank are independent toggles, both defaulting to on (today's combined
    "rerank" toggle behavior) but each callable alone:
    - rrf=False: skip fusion entirely rather than inventing a non-rank-based way to mix
      ES and Milvus scores (CLAUDE.md hard rule 3) - candidates are ES's own top ranking,
      Milvus isn't consulted at all.
    - rerank=False: skip the DeepInfra cross-encoder call - candidates keep whichever
      ranking (RRF or plain ES) selected them, just capped/exposed as "reranked" for the
      UI's single-ranked-list display. Rows keep their own `rrf_score`/`score` field
      rather than being relabeled as `rerank_score` - that field only appears once the
      cross-encoder actually ran, so the UI/trace can tell which stage produced a score."""
    if rrf:
        weights = _LABEL_RRF_WEIGHTS.get(label, {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0})
        fused = rrf_merge_by_doc_id(
            {
                "es": es_result,
                "milvus_dense": _flatten_by_score(milvus_dense),
                "milvus_sparse": _flatten_by_score(milvus_sparse),
            },
            weights,
        )
    else:
        fused = _collapse_to_doc_id(es_result)

    top_candidates = fused[:_TOP_N_CANDIDATES]
    if on_step is not None and rrf:
        await on_step("rrf_merge", {"candidate_count": len(top_candidates), "top_candidates": top_candidates})
    if not top_candidates:
        return []

    if not rerank:
        return top_candidates

    fulltext = await fetch_fulltext_batch(es_client, [row["doc_id"] for row in top_candidates])
    candidates = [row for row in top_candidates if fulltext.get(row["doc_id"])]
    if not candidates:
        return []

    # center=False: full document text, not a highlighted snippet - see
    # trim_to_token_budget's docstring for why the head (not the middle) is kept.
    scores = await gateway.rerank(
        role="reranker", query=query,
        documents=[trim_to_token_budget(fulltext[row["doc_id"]], center=False) for row in candidates],
    )
    scored = [{**row, "rerank_score": score} for row, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    cutoff = elbow_cutoff([row["rerank_score"] for row in scored])
    top_chunks = scored[:cutoff]
    if on_step is not None:
        await on_step("rerank", {
            "total_candidates": len(top_candidates),
            "considered_count": len(candidates),
            "top_chunks": top_chunks,
        })
    return top_chunks
