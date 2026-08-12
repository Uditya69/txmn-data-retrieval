import pytest

from retrieval_api.rrf_utils import apply_es_doc_boost


def test_apply_es_doc_boost_boosts_only_the_best_scoring_row_per_doc_id():
    merged = [
        {"chunk_id": "a", "doc_id": "d1", "rrf_score": 0.02},
        {"chunk_id": "b", "doc_id": "d1", "rrf_score": 0.01},
        {"chunk_id": "c", "doc_id": "d2", "rrf_score": 0.015},
    ]
    es_hits = [{"doc_id": "d1"}]

    boosted = apply_es_doc_boost(merged, es_hits, k=60, weight=1.0)

    boosted_by_chunk = {row["chunk_id"]: row["rrf_score"] for row in boosted}
    assert boosted_by_chunk["a"] == pytest.approx(0.02 + 1.0 / 61)
    assert boosted_by_chunk["b"] == pytest.approx(0.01)  # not the best row for d1 - untouched
    assert boosted_by_chunk["c"] == pytest.approx(0.015)  # d2 never appeared in es_hits


def test_apply_es_doc_boost_reorders_by_boosted_score():
    merged = [
        {"chunk_id": "a", "doc_id": "d1", "rrf_score": 0.01},
        {"chunk_id": "b", "doc_id": "d2", "rrf_score": 0.012},
    ]
    es_hits = [{"doc_id": "d1"}]  # rank-1 ES hit for d1 should push it above d2

    boosted = apply_es_doc_boost(merged, es_hits, k=60, weight=1.0)

    assert boosted[0]["chunk_id"] == "a"


def test_apply_es_doc_boost_returns_unchanged_when_no_es_hits():
    merged = [{"chunk_id": "a", "doc_id": "d1", "rrf_score": 0.01}]

    boosted = apply_es_doc_boost(merged, [])

    assert boosted == merged


def test_apply_es_doc_boost_never_injects_an_es_only_doc_id():
    merged = [{"chunk_id": "a", "doc_id": "d1", "rrf_score": 0.01}]
    es_hits = [{"doc_id": "d1"}, {"doc_id": "d2-es-only"}]

    boosted = apply_es_doc_boost(merged, es_hits)

    assert {row["doc_id"] for row in boosted} == {"d1"}
