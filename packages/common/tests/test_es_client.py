import pytest
from common.es_client import raw_search, resolve_doc_id_allowlist, fetch_citations


class FakeAsyncES:
    def __init__(self, search_hits=None, mget_docs=None):
        self.search_hits = search_hits or []
        self.mget_docs = mget_docs or {}
        self.search_calls = []
        self.mget_calls = []

    async def search(self, index, query, size):
        self.search_calls.append(query)
        return {"hits": {"hits": self.search_hits}}

    async def mget(self, index, ids):
        self.mget_calls.append(ids)
        return {"docs": [{"_id": i, "found": True, "_source": self.mget_docs.get(i, {})} for i in ids]}


@pytest.mark.asyncio
async def test_raw_search_returns_doc_id_score_snippet():
    client = FakeAsyncES(search_hits=[
        {"_source": {"doc_id": "d1", "facts_text": "assessee claimed exemption"}, "_score": 4.2},
    ])

    results = await raw_search(client, "exemption claim", limit=20)

    assert results == [{"doc_id": "d1", "score": 4.2, "snippet": "assessee claimed exemption"}]


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_returns_none_when_no_filters():
    client = FakeAsyncES()
    assert await resolve_doc_id_allowlist(client, {}) is None


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_and_returns_doc_ids():
    client = FakeAsyncES(search_hits=[{"_source": {"doc_id": "d1"}}, {"_source": {"doc_id": "d2"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Supreme Court"})

    assert result == ["d1", "d2"]
    assert client.search_calls  # a query was actually issued


@pytest.mark.asyncio
async def test_fetch_citations_returns_doc_id_keyed_masterinfo_fields():
    client = FakeAsyncES(mget_docs={
        "d1": {"masterinfo": {"court": "Supreme Court", "citations": ["2020 SCC 1"]}},
    })

    result = await fetch_citations(client, ["d1"])

    assert result == {"d1": {"masterinfo": {"court": "Supreme Court", "citations": ["2020 SCC 1"]}}}
