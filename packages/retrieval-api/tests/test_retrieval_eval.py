import json

import pytest

from retrieval_api.retrieval_eval import doc_rank, evaluate_case, load_cases


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

    async def fake_intent(gateway, query):
        return {"rewritten_query": "rewritten", "filters": {}, "intent": "test"}

    async def fake_allowlist(es_client, filters):
        return None

    async def fake_rerank(gateway, query, candidates, top_n=None):
        return [{**row, "rerank_score": 1.0} for row in candidates]

    monkeypatch.setattr(module, "raw_search", fake_raw_search)
    monkeypatch.setattr(module, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(module, "extract_intent", fake_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_allowlist)
    monkeypatch.setattr(module, "rerank_top_chunks", fake_rerank)

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
    }
    assert result["collection_ranks"]["raw_dense"]["facts"] == 1
    assert result["rewritten_query"] == "rewritten"
