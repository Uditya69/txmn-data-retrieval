import pytest

from agents.tools import TOOL_SCHEMAS, dispatch_tool_call


def test_tool_schemas_cover_all_four_tools_with_milvus_collection_enum():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {"search_es", "search_milvus_dense", "search_milvus_sparse", "lookup_doc"}
    dense_schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_milvus_dense")
    assert dense_schema["function"]["parameters"]["properties"]["collection"]["enum"] == [
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    ]


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
async def test_dispatch_search_milvus_dense_embeds_then_searches_one_collection(monkeypatch):
    import agents.tools as tools_module

    class FakeGateway:
        async def embed(self, role, text):
            assert role == "query_embed"
            return [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert collections == ["held"]
        assert dense_vector == [0.1, 0.2]
        return {"held": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9}]}

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_dense", {"collection": "held", "query": "gst"},
        gateway=FakeGateway(), es_client=None, milvus_client=object(),
    )

    assert result == {"rows": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9}]}


@pytest.mark.asyncio
async def test_dispatch_search_milvus_sparse_passes_none_dense_vector(monkeypatch):
    import agents.tools as tools_module

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert dense_vector is None
        assert sparse_query_text == "gst"
        assert collections == ["digest"]
        return {"digest": [{"chunk_id": "c2", "doc_id": "d2", "score": 3.1}]}

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_sparse", {"collection": "digest", "query": "gst"},
        gateway=object(), es_client=None, milvus_client=object(),
    )

    assert result == {"rows": [{"chunk_id": "c2", "doc_id": "d2", "score": 3.1}]}


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
