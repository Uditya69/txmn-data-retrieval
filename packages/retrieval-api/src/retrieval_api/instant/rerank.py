from common.instant_classifier.labels import boost_profile_key
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.score_cutoff import elbow_cutoff

# Opt-in override of CLAUDE.md hard rule 3 ("no ranking fusion between ES and
# Milvus"): RRF fuses by *rank position*, not raw score, so it never blends
# the incomparable ES-lexical-score and Milvus-cosine/BM25-distance scales
# the original rule guards against. Only reachable behind the `rrf`
# toggle - default (off) behavior is untouched.
_TOP_N_CANDIDATES = 20

# Qwen3-Reranker (this repo's cross-encoder model, both DeepInfra and the self-hosted
# fallback) is instruction-tuned - passed through to gateway.rerank() below. Without this,
# DeepInfra silently falls back to its generic default ("Given a web search query, retrieve
# relevant passages that answer the query"), which has no notion of legal-research relevance
# vs. generic keyword/topic similarity - the model's own docs cite a 1-5% relevance drop from
# omitting a task-specific instruction. Candidates span every content type Instant mode
# searches (MILVUS_COLLECTIONS above: case law - case_summary/digest/headnotes/facts/held/
# ruling - as well as statutory text - act_section/rule_section - and article_section/
# commentary_section), same "statutory text, case law, commentary, an article, or a mix"
# framing ai_mode/synthesize.py's own system prompt uses - not case-law-specific. Kept as a
# plain constant, not a per-label variant: every Instant mode query is the same underlying
# task, unlike _LABEL_RRF_WEIGHTS above, which varies by query shape for a different reason
# (lexical vs. semantic retrieval balance).
_RERANK_INSTRUCTION = (
    "Given a legal research query about Indian income-tax and allied law, rank the following "
    "passages by how directly they help answer the query. Passages may be statutory text (an "
    "Act section or Rule), case law, commentary, or an expert article - judge each on its own "
    "terms: for case law, prioritize the relevant holding, ratio, or facts; for statutory text, "
    "prioritize the operative provision itself; for commentary/articles, prioritize direct "
    "analysis of the query's issue. In every case, prefer passages that substantively address "
    "the query over ones that merely share keywords or topic with it."
)

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


async def _enrich_es_only_candidates_with_milvus_text(
    candidates: list[dict], query: str, milvus_client, dense_vector: list[float] | None,
) -> list[dict]:
    """A candidate whose surviving union row came from ES (no "chunk_id") may still have real
    pipeline-chunked text sitting in Milvus for that same doc_id - it just didn't rank in this
    query's own top-N-per-collection dense search window. Fetches it directly by doc_id filter
    (dense_vector is never None here, so hybrid_search's dense-only branch runs -
    common.milvus_client.hybrid_search never reaches the sparse_vector field in that branch,
    consistent with milvus_sparse_enabled's default-off everywhere else in this repo) rather
    than relying on ES's own doc-level highlight, which has no notion of "the meaningful
    passage" the way a real chunk does. Docs with nothing in Milvus at all (a content-type gap)
    keep whatever text they already had."""
    es_only_doc_ids = [row["doc_id"] for row in candidates if "chunk_id" not in row]
    if not es_only_doc_ids or dense_vector is None or milvus_client is None:
        return candidates
    by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector, sparse_query_text=query,
        doc_id_allowlist=es_only_doc_ids, limit=len(es_only_doc_ids),
    )
    best_chunk_by_doc_id: dict[str, dict] = {}
    for rows in by_collection.values():
        for row in rows:
            existing = best_chunk_by_doc_id.get(row["doc_id"])
            if existing is None or row["score"] > existing["score"]:
                best_chunk_by_doc_id[row["doc_id"]] = row
    return [
        {**row, "text": best_chunk_by_doc_id[row["doc_id"]]["text"]}
        if "chunk_id" not in row and row["doc_id"] in best_chunk_by_doc_id
        else row
        for row in candidates
    ]


def _rrf_candidates(es_result: list[dict], milvus_dense: dict[str, list[dict]], label: str) -> list[dict]:
    weights = _LABEL_RRF_WEIGHTS.get(
        boost_profile_key(label), {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0},
    )
    return rrf_merge_by_doc_id(
        {"es": es_result, "milvus_dense": _flatten_by_score(milvus_dense)},
        weights,
    )


async def rerank_instant_results(
    gateway,
    query: str,
    label: str,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
    rrf: bool = False,
    rerank: bool = False,
    plan: dict | None = None,
    on_step: OnStep | None = None,
    milvus_client=None,
    dense_vector: list[float] | None = None,
    reranker_model: str | None = None,
) -> list[dict]:
    """rrf and rerank compose:
    - rerank=True, rrf=True: the candidate pool is RRF-fused first (_rrf_candidates, same
      rank-based fusion as the rrf-only path) and *then* capped/reranked - RRF picks which
      20 candidates the cross-encoder ever sees, instead of a plain union's source-order
      concatenation. The RRF ordering itself is still discarded once picked (rows keep
      `rerank_score`, not `rrf_score`) - the cross-encoder re-sorts whatever RRF selected.
    - rerank=True, rrf=False: candidate gathering is pure code (see _union_by_doc_id) - a
      cross-encoder is about to score every candidate for actual relevance, so guessing a
      rank via RRF first would just get overwritten. gateway.rerank(role="reranker", ...)
      then produces the real ordering (rows keyed by `rerank_score`), trimmed by
      elbow_cutoff.
    - rerank=False, rrf=True: rank-based RRF fusion (rows keyed by `rrf_score`) - the
      fallback ranking mechanism for when there's no cross-encoder in the loop.
    - both False: plain single-source ranking via _fallback_fused (today's simplest
      behavior, rows keep whichever `score` ES/Milvus gave them)."""
    if rerank and rrf:
        candidates_pool = _rrf_candidates(es_result, milvus_dense, label)
        step_name = "rerank_candidates"
    elif rerank:
        candidates_pool = _union_by_doc_id(es_result, _flatten_by_score(milvus_dense))
        step_name = "rerank_candidates"
    elif rrf:
        candidates_pool = _rrf_candidates(es_result, milvus_dense, label)
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

    # Prefer real Milvus chunk text over ES's own doc-level highlight wherever it exists for
    # a candidate's doc_id, even if this query's own Milvus dense search didn't surface it -
    # see _enrich_es_only_candidates_with_milvus_text's docstring.
    top_candidates = await _enrich_es_only_candidates_with_milvus_text(top_candidates, query, milvus_client, dense_vector)

    # Each candidate's own row["text"] is used directly - the ES highlight-derived snippet
    # (raw_search) or the real Milvus chunk (hybrid_search/the enrichment above), same as
    # every other reranker call site in this repo (ai_mode/rerank.py, ES sparse-fallback). A
    # doc_id whose surviving row still has no text (nothing highlightable and nothing in
    # Milvus either) is dropped rather than fed to the reranker with empty input.
    candidates = [row for row in top_candidates if row.get("text")]
    if not candidates:
        return []

    scores = await gateway.rerank(
        role="reranker", query=query, documents=[row["text"] for row in candidates],
        model=reranker_model, instruction=_RERANK_INSTRUCTION,
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
