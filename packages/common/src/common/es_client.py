from elasticsearch import AsyncElasticsearch

from common.config import Settings
from common.query_tokenizer import chunk_query, classify_query_shape, expand_query_synonyms
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


def _build_field_query(query: str, shape: str, chunks: list[dict] = ()) -> dict:
    """Query-shape-aware multi-field search (design doc section 1+3): every content field
    is searched (facts_text/held_text/headnotes_text are only 26-58% populated on the real
    index, so heading/subheading/fullcontent - 100% populated - must never be skipped),
    with boosts picked by the no-LLM query-shape classifier.

    chunks (see query_tokenizer.chunk_query - ported from centax-node's queryAnalyzer.js/
    searchTextElastic.js) add match_phrase-with-slop should clauses per field, one per chunk -
    never replacing the loose per-field multi_match terms above, so a query still falls back to
    plain OR-term recall (typos/fuzzy matches match_phrase can't tolerate) even where chunking
    finds nothing. Each phrase clause reuses the SAME per-field boost weight as that field's
    multi_match clause - chunking only changes matching precision (does "Dimension Data India"
    have to appear together, within `slop` positions, versus anywhere independently), not the
    boost scale. This replaces the older, narrower extract_boost_phrases mechanism (which only
    phrase-boosted the few explicitly-recognized merges - Section+number, court+city, citation
    triple, quotes - and left every other word, including an unrecognized party name, to compete
    as independent OR terms with no phrase treatment at all)."""
    boosts = _BOOST_PROFILES[shape]
    should = [
        {"multi_match": {"query": query, "fields": [field], "boost": boost, "fuzziness": "AUTO"}}
        for field, boost in boosts.items()
    ]
    for chunk in chunks:
        for field, boost in boosts.items():
            should.append({
                "match_phrase": {field: {"query": chunk["text"], "slop": chunk["proximity"], "boost": boost}},
            })
            if chunk.get("alt_text"):
                should.append({
                    "match_phrase": {field: {"query": chunk["alt_text"], "slop": chunk["proximity"], "boost": boost}},
                })
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _wrap_function_score(field_query: dict) -> dict:
    """Formula kept for reference/future re-tuning - NOT called by raw_search (see the comment
    there). Was ranking fix (design doc section 2): court_boost/documenttypeboost/landmarkruling
    are real, precomputed boost fields the live index carries; documenttypeboost/landmarkruling
    constants are centax's own already-tuned formula for these exact fields, court_boost's
    factor is new, sized to that field's own smaller value range (0-294).

    boost_mode "multiply" made every one of these functions load-bearing: a single function
    landing on (or defaulting to) 0 zeroed the *entire* relevance score, no matter how well the
    text matched. Hit twice on the real index:
      - landmarkruling is populated on only 2.1% of the corpus. A `missing` fallback of 0.0001
        compounded through log2p+factor+multiply into a ~30,000x penalty for the other 98%.
      - court_boost can be a real, present value of exactly 0 (seen on a live Supreme Court
        doc, and on 45.8% of the whole corpus - confirmed via `term: {court_boost: 0}` count).
        0.01 * 0 = 0 kills the product just the same, `missing` fallbacks don't even apply.
    Both were patched below (every function gated behind `{"range": {field: {"gt": 0}}}`,
    turning missing/zero into a neutral 1x instead of a score-killing near-zero) and verified
    fixed on the live index. But a full head-to-head Instant-mode eval run (53-query set,
    `evals/retrieval_cases.json`) with the patched formula still active (21/53 passed) versus
    the same run with this function_score wrapper skipped entirely (42/53 passed - pure BM25
    text relevance, no boost) showed boosting is net-negative even fully patched: the
    multiplicative documenttypeboost x court_boost x landmarkruling stack still routinely
    outweighs real query-text relevance by 10-50x for docs that have strong boost values but a
    weaker text match, burying better-matching docs that have modest/absent boost values. That
    result is why raw_search doesn't call this - not a missing-data bug this time, an
    architecture one (multiply-mode itself), left as a follow-up rather than further tuning
    factors/modifiers here."""
    def _boost_function(field: str, factor: float, modifier: str) -> dict:
        return {
            "filter": {"range": {field: {"gt": 0}}},
            "field_value_factor": {"field": field, "factor": factor, "modifier": modifier},
        }

    return {
        "function_score": {
            "query": {
                "bool": {
                    "must": [field_query],
                    "must_not": [{"term": {"landmarkruling": -10}}],
                }
            },
            "functions": [
                _boost_function("documenttypeboost", 0.2, "sqrt"),
                _boost_function("court_boost", 0.01, "none"),
                _boost_function("landmarkruling", 1.2, "log2p"),
            ],
            "boost_mode": "multiply",
        }
    }


def _exclude_blacklisted(field_query: dict) -> dict:
    """landmarkruling: -10 marks a hard-excluded/blacklisted doc (centax's own convention,
    see _wrap_function_score's must_not) - a content filter, not a ranking signal, so it
    stays even though the ranking boost that used to carry it (function_score) does not."""
    return {
        "bool": {
            "must": [field_query],
            "must_not": [{"term": {"landmarkruling": -10}}],
        }
    }


async def raw_search(client, query: str, limit: int = 20) -> list[dict]:
    shape = classify_query_shape(query)
    field_query = _build_field_query(
        expand_query_synonyms(query), shape, chunks=chunk_query(query),
    )
    # Deliberately NOT wrapped in _wrap_function_score(). Even after patching that formula's
    # missing/zero-value bugs (see its docstring), a 53-query eval run showed the patched
    # boosting still nearly halves pass rate versus plain text relevance (21/53 boosted vs
    # 42/53 unboosted - evals/retrieval_cases.json). Root cause is architectural
    # (boost_mode: "multiply" lets documenttypeboost/court_boost/landmarkruling outweigh real
    # text relevance by 10-50x), not another missing-data instance, so it's left disabled here
    # rather than patched further. Re-enabling requires redesigning the boost combination
    # (e.g. bounded/additive instead of multiplicative), not just flipping this back on. The
    # blacklist exclusion that used to ride along inside function_score's must_not is kept
    # independently via _exclude_blacklisted, since that's a content filter, not ranking.
    body = _exclude_blacklisted(field_query)
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
