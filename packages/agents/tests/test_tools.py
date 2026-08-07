import pytest

from agents.tools import TOOL_SCHEMAS, dispatch_tool_call
from common.schemas import MILVUS_COLLECTIONS


def test_tool_schemas_cover_all_four_tools_and_milvus_tools_have_no_collection_param():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {"search_es", "search_milvus_dense", "search_milvus_sparse", "lookup_doc"}
    # The model must never pick a single Milvus collection to search - guessing wrong
    # silently misses gold hits (the same failure CLAUDE.md hard rule 4 documents for
    # AI Mode's collection routing). Every Milvus tool call searches all 7 collections.
    for name in ("search_milvus_dense", "search_milvus_sparse"):
        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == name)
        assert "collection" not in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["query"]


@pytest.mark.asyncio
async def test_dispatch_search_es_calls_raw_search(monkeypatch):
    import agents.tools as tools_module

    async def fake_raw_search(client, query, limit=20):
        assert query == "gst exemption"
        return [{"doc_id": "d1", "score": 1.0}]

    monkeypatch.setattr(tools_module, "raw_search", fake_raw_search)

    result = await dispatch_tool_call(
        "search_es", {"query": "gst exemption"}, gateway=None, es_client=object(), milvus_client=None,
    )

    assert result == {"rows": [{"doc_id": "d1", "score": 1.0}]}


@pytest.mark.asyncio
async def test_dispatch_search_milvus_dense_embeds_then_searches_all_collections(monkeypatch):
    import agents.tools as tools_module

    class FakeGateway:
        async def embed(self, role, text):
            assert role == "query_embed"
            return [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert collections == MILVUS_COLLECTIONS
        assert dense_vector == [0.1, 0.2]
        return {
            "held": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9}],
            "facts": [{"chunk_id": "c2", "doc_id": "d2", "score": 0.95}],
        }

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_dense", {"query": "gst"},
        gateway=FakeGateway(), es_client=None, milvus_client=object(),
    )

    # Merged across collections, tagged with source collection, sorted by score desc.
    assert result == {"rows": [
        {"chunk_id": "c2", "doc_id": "d2", "score": 0.95, "collection": "facts"},
        {"chunk_id": "c1", "doc_id": "d1", "score": 0.9, "collection": "held"},
    ]}


@pytest.mark.asyncio
async def test_dispatch_search_milvus_sparse_passes_none_dense_vector_and_all_collections(monkeypatch):
    import agents.tools as tools_module

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert dense_vector is None
        assert sparse_query_text == "gst"
        assert collections == MILVUS_COLLECTIONS
        return {"digest": [{"chunk_id": "c2", "doc_id": "d2", "score": 3.1}]}

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_sparse", {"query": "gst"},
        gateway=object(), es_client=None, milvus_client=object(),
    )

    assert result == {"rows": [{"chunk_id": "c2", "doc_id": "d2", "score": 3.1, "collection": "digest"}]}


@pytest.mark.asyncio
async def test_dispatch_search_milvus_merges_and_truncates_to_top_n(monkeypatch):
    import agents.tools as tools_module

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {
            collection: [{"chunk_id": f"{collection}-{i}", "doc_id": f"d{i}", "score": i / 100}
                         for i in range(10)]
            for collection in collections
        }

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_sparse", {"query": "gst"},
        gateway=object(), es_client=None, milvus_client=object(),
    )

    assert len(result["rows"]) == tools_module.MILVUS_MERGE_LIMIT
    scores = [row["score"] for row in result["rows"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_dispatch_lookup_doc_returns_citation_or_none(monkeypatch):
    import agents.tools as tools_module

    async def fake_fetch_citations(client, doc_ids):
        result = {}
        if "d1" in doc_ids:
            result["d1"] = {"court": "SC"}
        return result

    monkeypatch.setattr(tools_module, "fetch_citations", fake_fetch_citations)

    found = await dispatch_tool_call("lookup_doc", {"doc_id": "d1"}, gateway=None, es_client=object(), milvus_client=None)
    assert found == {"citation": {"court": "SC"}}

    missing = await dispatch_tool_call("lookup_doc", {"doc_id": "d999"}, gateway=None, es_client=object(), milvus_client=None)
    assert missing == {"citation": None}


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises_value_error():
    with pytest.raises(ValueError, match="unknown tool"):
        await dispatch_tool_call("not_a_tool", {}, gateway=None, es_client=None, milvus_client=None)


@pytest.mark.asyncio
async def test_dispatch_search_es_truncates_long_text_field(monkeypatch):
    import agents.tools as tools_module

    long_text = "x" * 5000
    original_row = {"doc_id": "d1", "score": 1.0, "text": long_text}

    async def fake_raw_search(client, query, limit=20):
        return [original_row]

    monkeypatch.setattr(tools_module, "raw_search", fake_raw_search)

    result = await dispatch_tool_call(
        "search_es", {"query": "gst exemption"}, gateway=None, es_client=object(), milvus_client=None,
    )

    returned_text = result["rows"][0]["text"]
    assert len(returned_text) < len(long_text)
    assert returned_text.startswith("x" * 500)
    # original row dict must not be mutated in place - it may be shared
    assert original_row["text"] == long_text
