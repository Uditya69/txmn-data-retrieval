from common.document_parser import strip_tags_to_text
from common.es_client import fetch_fulltext_batch, trim_to_token_budget
from common.instant_classifier.labels import boost_profile_key
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.score_cutoff import elbow_cutoff

# Opt-in override of CLAUDE.md hard rule 3 ("no ranking fusion between ES and
# Milvus"): RRF fuses by *rank position*, not raw score, so it never blends
# the incomparable ES-lexical-score and Milvus-cosine/BM25-distance scales
# the original rule guards against. Only reachable behind the `rrf`
# toggle - default (off) behavior is untouched.
_TOP_N_CANDIDATES = 20

# milvus_sparse carries no weight here - common.config.Settings.milvus_sparse_enabled is off
# by default app-wide (CLAUDE.md hard rule 3), so the milvus_sparse arg these weights would
# apply to is always an empty dict in practice; dropped from the weight tables so it can't
# accidentally out-rank es/milvus_dense on a query where it happens to be non-empty.
_LABEL_RRF_WEIGHTS: dict[str, dict[str, float]] = {
    "KEYWORD": {"es": 1.5, "milvus_dense": 0.5},
    "HYBRID": {"es": 1.5, "milvus_dense": 0.5},
    "INTENT": {"es": 1.0, "milvus_dense": 1.5},
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


def _union_by_doc_id(*sources: list[dict]) -> list[dict]:
    """Candidate pool for the cross-encoder rerank step: a plain set union across sources,
    deduped by doc_id, no scoring or rank math at all - not even RRF's rank-position fusion.
    Order doesn't matter here, unlike rrf_merge_by_doc_id's output: the reranker re-sorts
    every candidate by actual relevance, so any ranking this function guessed would just be
    thrown away. This also means a plan that skipped ES entirely needs no special-casing
    (contrast _fallback_fused below) - an empty es_result just contributes nothing to the
    union, and milvus_dense's hits pass through untouched."""
    seen: set[str] = set()
    union: list[dict] = []
    for rows in sources:
        for row in _collapse_to_doc_id(rows):
            doc_id = row["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            union.append(row)
    return union


def _fallback_fused(
    plan: dict | None,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
) -> list[dict]:
    """Single-source ranking used when neither the reranker nor rrf is on. Manual mode
    (plan=None, or any plan that searched ES) keeps the long-standing ES-only fallback. A
    plan that skipped ES entirely ({"es": False, "milvus": True}, e.g. the INTENT label)
    would always fall back to an empty list there even though Milvus found matches - instead
    rank-fuse Milvus dense+sparse, the same sanctioned rank-based fusion the rrf=True path
    already performs between those two sources."""
    if plan is not None and not plan.get("es", True) and plan.get("milvus", False):
        return rrf_merge_by_doc_id(
            {"milvus_dense": _flatten_by_score(milvus_dense)},
            {"milvus_dense": 1.0},
        )
    return _collapse_to_doc_id(es_result)


async def rerank_instant_results(
    gateway,
    es_client,
    query: str,
    label: str,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
    rrf: bool = False,
    rerank: bool = False,
    plan: dict | None = None,
    on_step: OnStep | None = None,
) -> list[dict]:
    """rrf and rerank are independent toggles:
    - rerank=True: candidate gathering is pure code (see _union_by_doc_id) - a cross-encoder
      is about to score every candidate for actual relevance, so guessing a rank via RRF
      first would just get overwritten. gateway.rerank(role="reranker", ...) then produces
      the real ordering (rows keyed by `rerank_score`), trimmed by elbow_cutoff.
    - rerank=False, rrf=True: rank-based RRF fusion (rows keyed by `rrf_score`) - the
      fallback ranking mechanism for when there's no cross-encoder in the loop.
    - both False: plain single-source ranking via _fallback_fused (today's simplest
      behavior, rows keep whichever `score` ES/Milvus gave them)."""
    if rerank:
        candidates_pool = _union_by_doc_id(es_result, _flatten_by_score(milvus_dense))
        step_name = "rerank_candidates"
    elif rrf:
        weights = _LABEL_RRF_WEIGHTS.get(
            boost_profile_key(label), {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0},
        )
        candidates_pool = rrf_merge_by_doc_id(
            {
                "es": es_result,
                "milvus_dense": _flatten_by_score(milvus_dense),
            },
            weights,
        )
        step_name = "rrf_merge"
    else:
        candidates_pool = _fallback_fused(plan, es_result, milvus_dense, milvus_sparse)
        step_name = "rrf_merge"

    top_candidates = candidates_pool[:_TOP_N_CANDIDATES]
    if on_step is not None:
        await on_step(step_name, {"candidate_count": len(top_candidates), "top_candidates": top_candidates})
    if not top_candidates:
        return []

    if not rerank:
        return top_candidates

    fulltext = await fetch_fulltext_batch(es_client, [row["doc_id"] for row in top_candidates])
    candidates = [row for row in top_candidates if fulltext.get(row["doc_id"])]
    if not candidates:
        return []

    # strip_tags_to_text first: fullcontent is stored as raw XML/HTML markup
    # (<document><body><para>... or legacy <!DOCTYPE html>...) - fetch_fulltext_batch
    # returns it as-is, unlike every other reranker call site in this repo (ES
    # sparse-fallback snippets, Instant's own highlight-fragment paths), which strip tags
    # before ever calling trim_to_token_budget. Stripping first also means the token
    # budget is spent on real text, not markup. center=False: full document text, not a
    # highlighted snippet - see trim_to_token_budget's docstring for why the head (not
    # the middle) is kept.
    scores = await gateway.rerank(
        role="reranker", query=query,
        documents=[
            trim_to_token_budget(strip_tags_to_text(fulltext[row["doc_id"]]), center=False) for row in candidates
        ],
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
