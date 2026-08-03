from elasticsearch import AsyncElasticsearch

from common.config import Settings

_RAW_SEARCH_FIELDS = [
    "facts_text", "held_text", "headnotes_text", "judgment_text", "case_review_text",
]

_INDEX = "taxmann_caselaw"


def get_es_client(settings: Settings) -> AsyncElasticsearch:
    return AsyncElasticsearch(settings.es_url)


async def raw_search(client, query: str, limit: int = 20) -> list[dict]:
    body = {"multi_match": {"query": query, "fields": _RAW_SEARCH_FIELDS, "fuzziness": "AUTO"}}
    response = await client.search(index=_INDEX, query=body, size=limit)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        snippet = next((source[f] for f in _RAW_SEARCH_FIELDS if source.get(f)), "")
        results.append({"doc_id": source["doc_id"], "score": hit["_score"], "snippet": snippet})
    return results


async def resolve_doc_id_allowlist(client, filters: dict) -> list[str] | None:
    if not filters:
        return None
    must = []
    if "court" in filters:
        must.append({"term": {"masterinfo.court": filters["court"]}})
    if "act" in filters:
        must.append({"term": {"masterinfo.act": filters["act"]}})
    if "date_range" in filters:
        must.append({"range": {"masterinfo.date": filters["date_range"]}})
    response = await client.search(index=_INDEX, query={"bool": {"must": must}}, size=1000)
    return [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]


async def fetch_citations(client, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    response = await client.mget(index=_INDEX, ids=doc_ids)
    return {
        doc["_id"]: doc["_source"]
        for doc in response["docs"]
        if doc.get("found")
    }
