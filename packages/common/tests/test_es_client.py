import pytest
from common.config import Settings
from common.es_client import get_es_client, raw_search, resolve_doc_id_allowlist, fetch_citations, fetch_fullcontent
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
    def __init__(self, search_hits=None, mget_docs=None, index="test_index"):
        self.search_hits = search_hits or []
        self.mget_docs = mget_docs or {}
        self.search_calls = []
        self.mget_calls = []
        self.index = index

    async def search(self, index, query, size):
        self.search_calls.append(query)
        self.searched_index = index
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
async def test_raw_search_returns_doc_id_score_heading_subheading():
    # "id", not "doc_id", is the real ES document identifier field
    # (verified against the live researchindex_aic_test mapping).
    client = FakeAsyncES(search_hits=[
        {
            "_source": {
                "id": "d1",
                "heading": "[2022] 140 taxmann.com 136 (Punjab & Haryana)",
                "subheading": "Krishana Goel vs. Principal Chief Commissioner of Income-tax",
            },
            "_score": 4.2,
        },
    ], index="researchindex_aic_test")

    results = await raw_search(client, "exemption claim", limit=20)

    assert results == [{
        "doc_id": "d1",
        "score": 4.2,
        "heading": "[2022] 140 taxmann.com 136 (Punjab & Haryana)",
        "subheading": "Krishana Goel vs. Principal Chief Commissioner of Income-tax",
    }]
    # index comes from the client (sourced from Settings.es_index), never hardcoded
    assert client.searched_index == "researchindex_aic_test"


@pytest.mark.asyncio
async def test_raw_search_defaults_missing_heading_subheading_to_empty_string():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}, "_score": 1.0}])

    results = await raw_search(client, "query", limit=20)

    assert results == [{"doc_id": "d1", "score": 1.0, "heading": "", "subheading": ""}]


def test_get_es_client_reads_index_and_auth_from_settings():
    settings = Settings(
        milvus_uri="http://milvus:19530", milvus_token="root:Milvus",
        es_uri="https://es:9200", es_username="elastic", es_password="secret",
        es_index="researchindex_aic_test", es_verify_certs=False,
        gateway_url="http://model-gateway:8001",
    )

    client = get_es_client(settings)

    assert client.index == "researchindex_aic_test"


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_returns_none_when_no_filters():
    client = FakeAsyncES()
    assert await resolve_doc_id_allowlist(client, {}) is None


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_and_returns_doc_ids():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}, {"_source": {"id": "d2"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Supreme Court"})

    assert result == ["d1", "d2"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"masterinfo.info.court.name.keyword": "Supreme Court"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_raises_on_unrecognized_filter_keys():
    client = FakeAsyncES()

    with pytest.raises(ValueError):
        await resolve_doc_id_allowlist(client, {"unknown_key": "whatever"})


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_section_term():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"section": "80C"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"masterinfo.info.section.name.keyword": "80C"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_otherinfo_partyname_match():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"party": "Reliance Industries"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"match": {"otherinfo.partyname.name": {"query": "Reliance Industries", "operator": "and"}}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_falls_back_to_fuzzy_match_when_exact_term_misses():
    class ExactMissFuzzyHitES(FakeAsyncES):
        async def search(self, index, query, size):
            self.search_calls.append(query)
            if {"term": {"masterinfo.info.court.name.keyword": "Bombay High Court"}} in query["bool"]["must"]:
                return {"hits": {"hits": []}}
            return {"hits": {"hits": [{"_source": {"id": "d1"}}]}}

    client = ExactMissFuzzyHitES()

    result = await resolve_doc_id_allowlist(client, {"court": "Bombay High Court"})

    assert result == ["d1"]
    assert len(client.search_calls) == 2
    assert client.search_calls[1] == {
        "bool": {"must": [{"match": {"masterinfo.info.court.name": "Bombay High Court"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_drops_malformed_date_range_instead_of_querying_es():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Supreme Court", "date_range": "2020"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"masterinfo.info.court.name.keyword": "Supreme Court"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_raises_when_only_filter_is_malformed_date_range():
    client = FakeAsyncES()

    with pytest.raises(ValueError):
        await resolve_doc_id_allowlist(client, {"date_range": "2020"})


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_accepts_well_formed_date_range():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"date_range": {"gte": "2020-01-01", "lte": "2022-01-01"}})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"range": {"formatteddocumentdate": {"gte": "2020-01-01", "lte": "2022-01-01"}}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_party_only_filter_does_not_raise():
    client = FakeAsyncES(search_hits=[])

    # previously raised ValueError because "party" was not a recognized key
    result = await resolve_doc_id_allowlist(client, {"party": "Reliance Industries"})

    assert result == []


@pytest.mark.asyncio
async def test_fetch_fullcontent_returns_field_for_matching_doc_id():
    client = FakeAsyncES(search_hits=[{"_source": {"fullcontent": "<document>...</document>"}}])

    result = await fetch_fullcontent(client, "101010000000322113")

    assert result == "<document>...</document>"
    assert client.search_calls[0] == {"bool": {"must": [{"term": {"id": "101010000000322113"}}]}}


@pytest.mark.asyncio
async def test_fetch_fullcontent_returns_none_when_doc_not_found():
    client = FakeAsyncES(search_hits=[])

    result = await fetch_fullcontent(client, "missing")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_citations_returns_doc_id_keyed_masterinfo_fields():
    client = FakeAsyncES(mget_docs={
        "d1": {
            "masterinfo": {"info": {"court": "Supreme Court"}, "citations": ["2020 SCC 1"]},
            "otherinfo": {"judge": "J. Smith", "partyname": "State v. Doe"},
            "judgment_text": "irrelevant text that should not be returned",
        },
    })

    result = await fetch_citations(client, ["d1"])

    assert result == {
        "d1": {
            "masterinfo": {"info": {"court": "Supreme Court"}, "citations": ["2020 SCC 1"]},
            "otherinfo": {"judge": "J. Smith", "partyname": "State v. Doe"},
        }
    }
    assert "judgment_text" not in result["d1"]
    # confirm the field restriction was actually passed through to ES (mget _source param)
    assert client.mget_calls[0]["_source"] == MASTERINFO_CITATION_FIELDS
