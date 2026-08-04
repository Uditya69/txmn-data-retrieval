from elasticsearch import AsyncElasticsearch

from common.config import Settings
from common.schemas import MASTERINFO_CITATION_FIELDS

_RAW_SEARCH_FIELDS = [
    "facts_text", "held_text", "headnotes_text",
]


class IndexedESClient:
    """Wraps a real ES client with the index name it should query, sourced
    from Settings.es_index (env-driven) at construction time - no index name
    is ever hardcoded downstream."""

    def __init__(self, client: AsyncElasticsearch, index: str):
        self._client = client
        self.index = index

    def __getattr__(self, name):
        return getattr(self._client, name)


def get_es_client(settings: Settings) -> IndexedESClient:
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    client = AsyncElasticsearch(
        settings.es_uri, basic_auth=auth, verify_certs=settings.es_verify_certs,
    )
    return IndexedESClient(client, settings.es_index)


async def raw_search(client, query: str, limit: int = 20) -> list[dict]:
    body = {"multi_match": {"query": query, "fields": _RAW_SEARCH_FIELDS, "fuzziness": "AUTO"}}
    response = await client.search(index=client.index, query=body, size=limit)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        snippet = next((source[f] for f in _RAW_SEARCH_FIELDS if source.get(f)), "")
        results.append({"doc_id": source["id"], "score": hit["_score"], "snippet": snippet})
    return results


async def resolve_doc_id_allowlist(client, filters: dict) -> list[str] | None:
    if not filters:
        return None
    must = []
    if "court" in filters:
        must.append({"term": {"masterinfo.info.court.name.keyword": filters["court"]}})
    if "act" in filters:
        must.append({"term": {"masterinfo.info.act.name.keyword": filters["act"]}})
    if "section" in filters:
        must.append({"term": {"masterinfo.info.section.name.keyword": filters["section"]}})
    if "party" in filters:
        must.append({"match": {"otherinfo.partyname.name": filters["party"]}})
    if "date_range" in filters:
        must.append({"range": {"formatteddocumentdate": filters["date_range"]}})
    if not must:
        raise ValueError(f"No recognized filter keys in {filters!r}")
    response = await client.search(index=client.index, query={"bool": {"must": must}}, size=1000)
    return [hit["_source"]["id"] for hit in response["hits"]["hits"]]


async def fetch_citations(client, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    response = await client.mget(index=client.index, ids=doc_ids, _source=MASTERINFO_CITATION_FIELDS)
    return {
        doc["_id"]: doc["_source"]
        for doc in response["docs"]
        if doc.get("found")
    }
