import pytest
from common.es_client import raw_search, resolve_doc_id_allowlist, fetch_citations
from common.schemas import MASTERINFO_CITATION_FIELDS


def _filter_source(source: dict, fields: list[str]) -> dict:
    """Mimic Elasticsearch's `_source` include filtering for dotted field paths."""
    result: dict = {}
    for path in fields:
        parts = path.split(".")
        node = source
        found = True
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                found = False
                break
        if found:
            dest = result
            for part in parts[:-1]:
                dest = dest.setdefault(part, {})
            dest[parts[-1]] = node
    return result


class FakeAsyncES:
    def __init__(self, search_hits=None, mget_docs=None):
        self.search_hits = search_hits or []
        self.mget_docs = mget_docs or {}
        self.search_calls = []
        self.mget_calls = []

    async def search(self, index, query, size):
        self.search_calls.append(query)
        return {"hits": {"hits": self.search_hits}}

    async def mget(self, index, ids, _source=None):
        self.mget_calls.append({"ids": ids, "_source": _source})
        docs = []
        for i in ids:
            source = self.mget_docs.get(i, {})
            if _source is not None:
                source = _filter_source(source, _source)
            docs.append({"_id": i, "found": True, "_source": source})
        return {"docs": docs}


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
async def test_resolve_doc_id_allowlist_raises_on_unrecognized_filter_keys():
    client = FakeAsyncES()

    with pytest.raises(ValueError):
        await resolve_doc_id_allowlist(client, {"unknown_key": "whatever"})


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_section_term():
    client = FakeAsyncES(search_hits=[{"_source": {"doc_id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"section": "80C"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"masterinfo.section": "80C"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_partyname_match():
    client = FakeAsyncES(search_hits=[{"_source": {"doc_id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"party": "Reliance Industries"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"match": {"masterinfo.partyname": "Reliance Industries"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_party_only_filter_does_not_raise():
    client = FakeAsyncES(search_hits=[])

    # previously raised ValueError because "party" was not a recognized key
    result = await resolve_doc_id_allowlist(client, {"party": "Reliance Industries"})

    assert result == []


@pytest.mark.asyncio
async def test_fetch_citations_returns_doc_id_keyed_masterinfo_fields():
    client = FakeAsyncES(mget_docs={
        "d1": {
            "masterinfo": {"court": "Supreme Court", "citations": ["2020 SCC 1"]},
            "judgment_text": "irrelevant text that should not be returned",
        },
    })

    result = await fetch_citations(client, ["d1"])

    assert result == {"d1": {"masterinfo": {"court": "Supreme Court", "citations": ["2020 SCC 1"]}}}
    assert "judgment_text" not in result["d1"]
    # confirm the field restriction was actually passed through to ES (mget _source param)
    assert client.mget_calls[0]["_source"] == MASTERINFO_CITATION_FIELDS
