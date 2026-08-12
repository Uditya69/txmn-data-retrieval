DEFAULT_ES_BOOST_K = 60
DEFAULT_ES_BOOST_WEIGHT = 1.0


def apply_es_doc_boost(
    merged: list[dict], es_hits: list[dict], k: int = DEFAULT_ES_BOOST_K, weight: float = DEFAULT_ES_BOOST_WEIGHT,
) -> list[dict]:
    """ES's top-N doc_ids for the same query are a rank-based confirming signal, not a raw
    score, so this stays inside CLAUDE.md rule 3's rank-position-only fusion rule. Boosted
    once per doc_id (the single best-scoring row already in `merged`), not fanned out to
    every row sharing that doc_id - a document with many surviving rows shouldn't get the
    same ES signal counted multiple times. Never injects an ES-only doc_id that isn't
    already in `merged`, sidestepping the ES-is-doc-level/Milvus-is-chunk-level granularity
    mismatch (see ai_mode/retrieve.py and instant/rerank.py, the two callers of this)."""
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
