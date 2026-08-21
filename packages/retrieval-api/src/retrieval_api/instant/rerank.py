from common.instant_classifier.labels import boost_profile_key
from retrieval_api.ai_mode.intent import OnStep

# Opt-in override of CLAUDE.md hard rule 3 ("no ranking fusion between ES and
# Milvus"): RRF fuses by *rank position*, not raw score, so it never blends
# the incomparable ES-lexical-score and Milvus-cosine/BM25-distance scales
# the original rule guards against. Only reachable behind the `rrf`
# toggle - default (off) behavior is untouched.
_TOP_N_CANDIDATES = 20

_LABEL_RRF_WEIGHTS: dict[str, dict[str, float]] = {
    "KEYWORD": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "HYBRID": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "INTENT": {"es": 1.0, "milvus_dense": 1.5, "milvus_sparse": 0.5},
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


def _fallback_fused(
    plan: dict | None,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
) -> list[dict]:
    """Single-source ranking used when rrf is off. Manual mode (plan=None, or any plan that
    searched ES) keeps the long-standing ES-only fallback. A plan that skipped ES entirely
    ({"es": False, "milvus": True}, e.g. the INTENT label) would always fall back to an
    empty list there even though Milvus found matches - instead rank-fuse Milvus
    dense+sparse, the same sanctioned rank-based fusion the rrf=True path already performs
    between those two sources."""
    if plan is not None and not plan.get("es", True) and plan.get("milvus", False):
        return rrf_merge_by_doc_id(
            {"milvus_dense": _flatten_by_score(milvus_dense), "milvus_sparse": _flatten_by_score(milvus_sparse)},
            {"milvus_dense": 1.0, "milvus_sparse": 1.0},
        )
    return _collapse_to_doc_id(es_result)


async def rerank_instant_results(
    label: str,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
    rrf: bool = True,
    plan: dict | None = None,
    on_step: OnStep | None = None,
) -> list[dict]:
    """Instant mode's only fusion stage - rank-based RRF (or, with rrf=False, a plain
    single-source ranking via _fallback_fused). No AI/cross-encoder call is involved."""
    if rrf:
        weights = _LABEL_RRF_WEIGHTS.get(
            boost_profile_key(label), {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0},
        )
        fused = rrf_merge_by_doc_id(
            {
                "es": es_result,
                "milvus_dense": _flatten_by_score(milvus_dense),
                "milvus_sparse": _flatten_by_score(milvus_sparse),
            },
            weights,
        )
    else:
        fused = _fallback_fused(plan, es_result, milvus_dense, milvus_sparse)

    top_candidates = fused[:_TOP_N_CANDIDATES]
    if on_step is not None:
        await on_step("rrf_merge", {"candidate_count": len(top_candidates), "top_candidates": top_candidates})
    return top_candidates
