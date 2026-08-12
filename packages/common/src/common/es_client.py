from elasticsearch import AsyncElasticsearch

from common.config import Settings
from common.query_tokenizer import classify_query_shape, expand_query_synonyms, extract_boost_phrases
from common.schemas import MASTERINFO_CITATION_FIELDS

_BOOST_PROFILES = {
    "citation": {"heading": 5.0, "subheading": 3.0, "fullcontent": 1.0,
                 "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.5},
    "provision": {"heading": 2.0, "subheading": 3.0, "fullcontent": 1.0,
                  "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 2.5},
    "plain": {"heading": 2.0, "subheading": 2.0, "fullcontent": 1.5,
              "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.0},
}


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


_PHRASE_BOOST_FACTOR = 3.0


def _build_field_query(query: str, shape: str, boost_phrases: list[str] = ()) -> dict:
    """Query-shape-aware multi-field search (design doc section 1+3): every content field
    is searched (facts_text/held_text/headnotes_text are only 26-58% populated on the real
    index, so heading/subheading/fullcontent - 100% populated - must never be skipped),
    with boosts picked by the no-LLM query-shape classifier.

    boost_phrases (queryAnalyzer.js-ported merges like "Section 6"/"Delhi High Court", or a
    quoted span) get an extra phrase-match should clause layered on top - never replacing
    the loose per-field terms above, so a query that doesn't fully match the merge pipeline
    still falls back to plain recall instead of a legacy-style hard AND-of-all-tokens."""
    boosts = _BOOST_PROFILES[shape]
    fields = list(boosts.keys())
    should = [
        {"multi_match": {"query": query, "fields": [field], "boost": boost, "fuzziness": "AUTO"}}
        for field, boost in boosts.items()
    ]
    should += [
        {"multi_match": {"query": phrase, "fields": fields, "type": "phrase", "boost": _PHRASE_BOOST_FACTOR}}
        for phrase in boost_phrases
    ]
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _wrap_function_score(field_query: dict) -> dict:
    """Ranking fix (design doc section 2): court_boost/documenttypeboost/landmarkruling are
    real, populated, precomputed boost fields the live index already carries but nothing in
    this codebase used before. documenttypeboost/landmarkruling constants are centax's own
    already-tuned formula for these exact fields; court_boost's factor is new, sized to that
    field's own smaller value range (0-294)."""
    return {
        "function_score": {
            "query": {
                "bool": {
                    "must": [field_query],
                    "must_not": [{"term": {"landmarkruling": -10}}],
                }
            },
            "functions": [
                {"field_value_factor": {"field": "documenttypeboost", "factor": 0.2, "modifier": "sqrt", "missing": 0.0001}},
                {"field_value_factor": {"field": "court_boost", "factor": 0.01, "modifier": "none", "missing": 0.0001}},
                {"field_value_factor": {"field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0.0001}},
            ],
            "boost_mode": "multiply",
        }
    }


async def raw_search(client, query: str, limit: int = 20) -> list[dict]:
    shape = classify_query_shape(query)
    field_query = _build_field_query(
        expand_query_synonyms(query), shape, boost_phrases=extract_boost_phrases(query),
    )
    body = _wrap_function_score(field_query)
    response = await client.search(index=client.index, query=body, size=limit)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "doc_id": source["id"],
            "score": hit["_score"],
            "heading": source.get("heading", ""),
            "subheading": source.get("subheading", ""),
        })
    return results


# court/bench/section/act filters used to target masterinfo.info.{court,act,section,bench}
# .name - confirmed 0% populated across all 410,427 docs in the live index (every content
# group), so those filters silently matched nothing. Real signal lives elsewhere: see each
# helper below. judge/party/date_range are untouched - their fields are genuinely populated
# (otherinfo.judge.name 99.4%, otherinfo.partyname.name 100%, formatteddocumentdate 100%).
_FUZZY_FALLBACK_KEYS = {"court", "bench", "section", "judge"}

_COURT_HEADING_ALIASES = {
    "supreme court": "SC",
    "delhi high court": "Delhi",
    "bombay high court": "Bombay",
    "madras high court": "Madras",
    "calcutta high court": "Calcutta",
    "karnataka high court": "Karnataka",
    "gujarat high court": "Gujarat",
    "income tax appellate tribunal": "Trib.",
    "customs excise and service tax appellate tribunal": "CESTAT",
}


def _resolve_heading_term(value: str) -> str:
    """Courts/benches appear inside `heading` as an abbreviation (e.g. "(SC)" for Supreme
    Court - confirmed correlated with that court's own court_boost=294.8 value; "(Bombay)"
    for Bombay High Court), not in masterinfo.info.{court,bench}.name. This maps the full
    name AI Mode extracts to the literal abbreviation that appears in heading; an
    unrecognized value is passed through unchanged so a filter never silently drops a
    court/bench this map doesn't know about."""
    return _COURT_HEADING_ALIASES.get(value.strip().lower(), value)


def _section_heading_queries(value: str, phrase: bool) -> list[dict]:
    """ACT/RULE-group documents' `heading` field is the section/rule identifier itself,
    verbatim (e.g. "Section - 184", "Rule - 37CA") - the only real signal for a section
    filter, since masterinfo.info.section.name is confirmed 0% populated."""
    match_type = "match_phrase" if phrase else "match"
    num = value.strip()
    return [
        {match_type: {"heading": f"Section - {num}"}},
        {match_type: {"heading": f"Rule - {num}"}},
    ]


def _build_filter_must(filters: dict, fuzzy: bool) -> list[dict]:
    must = []
    match_type = "match" if fuzzy else "match_phrase"

    for key in ("court", "bench"):
        if key in filters:
            must.append({match_type: {"heading": _resolve_heading_term(filters[key])}})

    if "section" in filters:
        must.append({"bool": {"should": _section_heading_queries(filters["section"], phrase=not fuzzy)}})

    if "act" in filters:
        # No field in this index reliably links a document back to its specific parent Act
        # (masterinfo.info.act.name, incometaxactinfo, companyactinfo all confirmed 0%
        # populated; `categories` only gives subject area, not the Act itself) - this is a
        # best-effort full-text match, not an exact filter, until that data exists.
        must.append({"match": {"fullcontent": filters["act"]}})

    if "judge" in filters:
        must.append(
            {"match": {"otherinfo.judge.name": filters["judge"]}} if fuzzy
            else {"term": {"otherinfo.judge.name.keyword": filters["judge"]}}
        )

    if "party" in filters:
        # operator "and" requires every name token to match - a plain match
        # query ORs analyzed tokens, so "Meenaben Maheshchandra Patel" would
        # match any document naming a party with just "Patel" (a very common
        # surname), effectively returning almost the whole index unfiltered.
        must.append({"match": {"otherinfo.partyname.name": {"query": filters["party"], "operator": "and"}}})

    date_range = filters.get("date_range")
    if isinstance(date_range, dict) and ("gte" in date_range or "lte" in date_range):
        must.append({"range": {"formatteddocumentdate": date_range}})

    return must


async def resolve_doc_id_allowlist(client, filters: dict) -> list[str] | None:
    if not filters:
        return None
    must = _build_filter_must(filters, fuzzy=False)
    if not must:
        raise ValueError(f"No recognized filter keys in {filters!r}")
    response = await client.search(index=client.index, query={"bool": {"must": must}}, size=1000)
    hits = response["hits"]["hits"]
    if not hits and any(key in filters for key in _FUZZY_FALLBACK_KEYS):
        fuzzy_must = _build_filter_must(filters, fuzzy=True)
        response = await client.search(index=client.index, query={"bool": {"must": fuzzy_must}}, size=1000)
        hits = response["hits"]["hits"]
    return [hit["_source"]["id"] for hit in hits]


async def fetch_fullcontent(client, doc_id: str) -> str | None:
    response = await client.search(
        index=client.index, query={"bool": {"must": [{"term": {"id": doc_id}}]}}, size=1,
    )
    hits = response["hits"]["hits"]
    if not hits:
        return None
    return hits[0]["_source"]["fullcontent"]


async def fetch_document_metadata(client, doc_id: str) -> dict | None:
    """Header fields for the document reader (case title, citation, year).
    `masterinfo.info.court` is frequently empty across the corpus - the
    court/bench abbreviation lives inside `heading` instead (e.g. "...(SC)")
    - so this doesn't try to surface it as a separate structured field."""
    response = await client.search(
        index=client.index, query={"bool": {"must": [{"term": {"id": doc_id}}]}}, size=1,
    )
    hits = response["hits"]["hits"]
    if not hits:
        return None
    source = hits[0]["_source"]
    year = source.get("year")
    return {
        "heading": source.get("heading"),
        "subheading": source.get("subheading"),
        "year": year.get("name") if isinstance(year, dict) else year,
    }


async def fetch_citations(client, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    response = await client.mget(index=client.index, ids=doc_ids, _source=MASTERINFO_CITATION_FIELDS)
    return {
        doc["_id"]: doc["_source"]
        for doc in response["docs"]
        if doc.get("found")
    }
