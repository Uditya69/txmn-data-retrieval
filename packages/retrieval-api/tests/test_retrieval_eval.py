import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from retrieval_api.retrieval_eval import _run_paths, doc_rank, evaluate_case, load_cases


def test_default_run_paths_are_timestamped_and_keep_latest_aliases():
    created_at = datetime(2026, 8, 6, 12, 34, 56, tzinfo=timezone.utc)
    result, snapshot, latest, latest_snapshot = _run_paths(None, "BM25 before fix", created_at)
    assert result == Path(".eval-results/20260806T123456000000Z-BM25-before-fix.json")
    assert snapshot == Path(".eval-results/20260806T123456000000Z-BM25-before-fix.dataset.json")
    assert latest == Path(".eval-results/latest.json")
    assert latest_snapshot == Path(".eval-results/latest.dataset.json")


def test_explicit_output_keeps_snapshot_beside_result():
    result, snapshot, latest, latest_snapshot = _run_paths(
        Path("/tmp/result.json"), "ignored", datetime.now(timezone.utc),
    )
    assert result == Path("/tmp/result.json")
    assert snapshot == Path("/tmp/result.dataset.json")
    assert latest is None
    assert latest_snapshot is None


def test_repository_eval_dataset_spans_1936_to_2026_and_has_stress_cases():
    root = Path(__file__).parents[3]
    cases = load_cases(root / "evals" / "retrieval_cases.json")
    assert len(cases) == 53
    assert {case["class"] for case in cases} == {"direct", "indirect", "adversarial"}
    assert cases[20]["id"] == "Q21"
    assert cases[-1]["id"] == "Q53"


def test_doc_rank_dedupes_chunks_and_returns_first_gold_document_rank():
    rows = [
        {"doc_id": "d1", "chunk_id": "d1-a"},
        {"doc_id": "d1", "chunk_id": "d1-b"},
        {"doc_id": "gold", "chunk_id": "gold-a"},
    ]
    assert doc_rank(rows, {"gold"}) == 2
    assert doc_rank(rows, {"missing"}) is None


def test_load_cases_validates_unique_ids_and_known_collections(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "Q1", "class": "direct", "query": "q", "gold_doc_ids": ["d1"],
         "expected_collections": ["facts"], "pass_at": 5},
    ]))
    assert load_cases(path)[0]["id"] == "Q1"

    path.write_text(json.dumps([
        {"id": "Q1", "class": "direct", "query": "q", "gold_doc_ids": ["d1"],
         "expected_collections": ["unknown"], "pass_at": 5},
    ]))
    with pytest.raises(ValueError, match="unknown collection"):
        load_cases(path)


@pytest.mark.asyncio
async def test_evaluate_case_reports_each_retrieval_stage(monkeypatch):
    import retrieval_api.retrieval_eval as module

    async def fake_raw_search(client, query, limit=50):
        return [{"doc_id": "gold", "score": 1.0}]

    async def fake_hybrid(client, collections, dense_vector, sparse_query_text,
                          doc_id_allowlist=None, limit=50):
        suffix = "dense" if dense_vector is not None else "sparse"
        return {name: [{"doc_id": "gold", "chunk_id": f"gold-{name}-{suffix}",
                        "text": "gold text", "score": 1.0}] for name in collections}

    async def fake_intent(gateway, query, model=None):
        return {"rewritten_query": "rewritten", "filters": {}, "intent": "test"}

    async def fake_allowlist(es_client, filters):
        return None

    async def fake_rerank(gateway, query, candidates, top_n=None, model=None):
        return [{**row, "rerank_score": 1.0} for row in candidates]

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, model=None):
        return {"answer": "See [gold1].", "citations": {}, "reasoning": None}

    async def fake_agentic(gateway, es_client, milvus_client, query):
        return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": []}

    monkeypatch.setattr(module, "raw_search", fake_raw_search)
    monkeypatch.setattr(module, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(module, "extract_intent", fake_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_allowlist)
    monkeypatch.setattr(module, "rerank_top_chunks", fake_rerank)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)
    monkeypatch.setattr(module, "run_agentic_search", fake_agentic)

    class Gateway:
        async def embed(self, role, text):
            assert role == "query_embed"
            return [0.1]

    result = await evaluate_case(
        {"id": "Q1", "class": "direct", "query": "raw", "gold_doc_ids": ["gold"],
         "expected_collections": ["facts"], "pass_at": 5},
        Gateway(), object(), object(), langfuse_enabled=False,
    )

    assert result["ranks"] == {
        "es": 1, "raw_dense": 1, "raw_sparse": 1,
        "rewritten_dense": 1, "rewritten_sparse": 1, "rrf": 1, "reranker": 1,
        "agentic": None,
    }
    assert result["collection_ranks"]["raw_dense"]["facts"] == 1
    assert result["rewritten_query"] == "rewritten"


@pytest.mark.asyncio
async def test_evaluate_case_records_agentic_hit_when_gold_doc_cited(monkeypatch):
    import retrieval_api.retrieval_eval as eval_module

    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": True, "answer": "See [gold-doc].", "doc_ids": ["gold-doc", "other-doc"]}

    monkeypatch.setattr(eval_module, "run_agentic_search", fake_run_agentic_search)

    class Gateway:
        async def embed(self, role, text):
            return [0.1]

    case = {
        "id": "Q1", "class": "direct", "query": "q", "gold_doc_ids": ["gold-doc"],
        "expected_collections": ["metadata"], "pass_at": 5,
    }
    result = await eval_module.evaluate_case(
        case, Gateway(), object(), object(), langfuse_enabled=False,
    )

    assert result["ranks"]["agentic"] == 1
    assert "agentic" not in result["errors"]


@pytest.mark.asyncio
async def test_evaluate_case_records_agentic_miss_when_unverifiable(monkeypatch):
    import retrieval_api.retrieval_eval as eval_module

    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": ["bad-doc"]}

    monkeypatch.setattr(eval_module, "run_agentic_search", fake_run_agentic_search)

    class Gateway:
        async def embed(self, role, text):
            return [0.1]

    case = {
        "id": "Q2", "class": "direct", "query": "q", "gold_doc_ids": ["gold-doc"],
        "expected_collections": ["metadata"], "pass_at": 5,
    }
    result = await eval_module.evaluate_case(
        case, Gateway(), object(), object(), langfuse_enabled=False,
    )

    assert result["ranks"]["agentic"] is None
    assert "unverifiable_answer" in result["errors"]["agentic"]


def test_sample_12_query_ids_are_a_subset_of_the_full_dataset():
    from retrieval_api.retrieval_eval import SAMPLE_12_QUERY_IDS

    root = Path(__file__).parents[3]
    cases = load_cases(root / "evals" / "retrieval_cases.json")
    all_ids = {case["id"] for case in cases}
    assert len(SAMPLE_12_QUERY_IDS) == 12
    assert len(set(SAMPLE_12_QUERY_IDS)) == 12  # no duplicates
    assert set(SAMPLE_12_QUERY_IDS).issubset(all_ids)
    sampled_classes = {case["class"] for case in cases if case["id"] in SAMPLE_12_QUERY_IDS}
    assert sampled_classes == {"direct", "indirect", "adversarial"}  # every class represented


@pytest.mark.asyncio
async def test_evaluate_case_records_citation_validity_against_reranked_chunks(monkeypatch):
    import retrieval_api.retrieval_eval as module

    async def fake_raw_search(client, query, limit=50):
        return []

    async def fake_hybrid(client, collections, dense_vector, sparse_query_text,
                          doc_id_allowlist=None, limit=50):
        suffix = "dense" if dense_vector is not None else "sparse"
        # doc_id "gold1" (not "gold") because agents.citations.extract_cited_doc_ids
        # only recognizes tokens containing at least one digit as doc_ids.
        return {name: [{"doc_id": "gold1", "chunk_id": f"gold1-{name}-{suffix}",
                        "text": "gold text", "score": 1.0}] for name in collections}

    async def fake_intent(gateway, query, model=None):
        return {"rewritten_query": query, "filters": {}, "intent": "test"}

    async def fake_allowlist(es_client, filters):
        return None

    async def fake_rerank(gateway, query, candidates, top_n=None, model=None):
        return [{**candidates[0], "rerank_score": 1.0}]  # only "gold1" survives reranking

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, model=None):
        return {"answer": "The point is settled [gold1] and also [not-retrieved2].", "citations": {}, "reasoning": None}

    async def fake_agentic(gateway, es_client, milvus_client, query):
        return {"ok": True, "doc_ids": ["gold1"]}

    monkeypatch.setattr(module, "raw_search", fake_raw_search)
    monkeypatch.setattr(module, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(module, "extract_intent", fake_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_allowlist)
    monkeypatch.setattr(module, "rerank_top_chunks", fake_rerank)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)
    monkeypatch.setattr(module, "run_agentic_search", fake_agentic)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    case = {"id": "Q01", "class": "direct", "query": "q", "gold_doc_ids": ["gold1"],
            "expected_collections": ["facts"], "pass_at": 5}

    result = await evaluate_case(case, gateway, es_client=object(), milvus_client=object(), langfuse_enabled=False)

    assert result["citation_count"] == 2
    assert result["citation_invalid_ids"] == ["not-retrieved2"]
    assert result["citation_valid"] is False
    assert result["gold_cited"] is True
    assert result["synthesis_answer"] == "The point is settled [gold1] and also [not-retrieved2]."
