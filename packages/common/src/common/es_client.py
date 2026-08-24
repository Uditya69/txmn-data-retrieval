import functools

from elasticsearch import AsyncElasticsearch

from common.config import Settings
from common.instant_classifier import effective_label
from common.instant_classifier.labels import boost_profile_key
from common.query_tokenizer import chunk_query, expand_query_synonyms
from common.schemas import ES_GROUP_FOR_COLLECTION, MASTERINFO_CITATION_FIELDS

import tiktoken

# Same tokenizer tm-dp/packages/data-pipeline/src/data_pipeline/chunking.py uses for its
# CHUNK_SIZE_TOKENS=1024 splitter cap - matching it here keeps ES-fallback snippets from
# being systematically under-scored by the reranker for carrying less context than the
# real Milvus chunks they compete against. See
# docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md.
_SNIPPET_TARGET_TOKENS = 1024


@functools.lru_cache(maxsize=1)
def _get_snippet_tokenizer():
    """Lazy, cached getter - tiktoken.get_encoding() fetches the BPE vocab file over the
    network on first use (cached to disk after). Must NOT run at module import time: this
    module is imported by retrieval_api at process startup, and a module-level constant here
    previously meant any network hiccup made the whole service fail to boot over a
    snippet-trimming helper on a fallback code path. Deferring to first real call turns that
    into (at worst) a failure local to sparse_fallback_search."""
    return tiktoken.get_encoding("cl100k_base")


def trim_to_token_budget(text: str, target_tokens: int = _SNIPPET_TARGET_TOKENS, center: bool = True) -> str:
    """Trims text to at most target_tokens tokens - never expands short text. Shared by
    every reranker call site in this repo (ES sparse-fallback snippets here, Instant
    mode's fulltext fetch in retrieval_api/instant/rerank.py) so a reranker never sees
    more text than the ~1024-token budget tm-dp's own chunker targets - full documents
    run tens of thousands of tokens, which otherwise made a single rerank call take
    9-12s and, on at least one real query, 422 against DeepInfra's rerank endpoint.

    center=True (default): trims evenly from both ends, for ES's own highlighted
    fragment - the best-scoring match can be anywhere in the (oversized) fragment ES
    returned, so centering keeps it regardless of where it landed.
    center=False: keeps only the head - for full document text with no highlighted
    match to center on, where per common/schemas.py's caselaws field order
    (case_summary, digest, headnotes, facts, held, ruling, metadata) the opening is the
    summary/headnote content most relevant to a reranker's judgment; centering would
    risk cutting it to keep an arbitrary middle slice instead."""
    tokenizer = _get_snippet_tokenizer()
    ids = tokenizer.encode(text)
    if len(ids) <= target_tokens:
        return text
    if not center:
        return tokenizer.decode(ids[:target_tokens])
    excess = len(ids) - target_tokens
    start = excess // 2
    return tokenizer.decode(ids[start : start + target_tokens])


def _cap_group_shares(hits: list[dict], limit: int, group_cap: int) -> list[dict]:
    """hits must already be in ES relevance order (ES's own default sort). Caps any single
    group's share of the top `limit` hits at `group_cap` - minority groups' hits are picked
    up naturally within this same single pass as encountered, in relevance order, without
    ever reaching back into an over-cap group's exclusions. With only one group present,
    this is a no-op past the limit slice - the cap only ever engages with 2+ groups in the
    same call. See "Per-group starvation cap" in
    docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md."""
    if len({hit["_group"] for hit in hits}) <= 1:
        return hits[:limit]

    taken_counts: dict[str, int] = {}
    kept: list[dict] = []
    for hit in hits:
        group = hit["_group"]
        if taken_counts.get(group, 0) < group_cap:
            kept.append(hit)
            taken_counts[group] = taken_counts.get(group, 0) + 1
        if len(kept) == limit:
            break
    return kept


_ES_FALLBACK_LIMIT = 20
_ES_FALLBACK_GROUP_CAP = 15
_ES_HIGHLIGHT_FRAGMENT_CHARS = 6000  # oversized on purpose - trim_to_token_budget cuts to ~1024 tokens after

_COLLECTION_FOR_ES_GROUP = {group: collection for collection, group in ES_GROUP_FOR_COLLECTION.items()}


def build_sparse_fallback_query_preview(
    query: str, groups: list[str], doc_id_allowlist: list[str] | None = None, boost: bool = False,
) -> dict:
    """The exact query body sparse_fallback_search sends to ES, without executing a search -
    same single-source-of-truth pattern as build_query_preview (see its own docstring for why):
    a caller building this independently could silently drift from what the real search sends.
    Powers the AI Mode trace panel's "Show ES query" block for the ai_milvus_sparse step, same
    as build_query_preview powers Instant mode's."""
    field_query = build_query_preview(query, boost=boost)["es_query"]
    must: list[dict] = [{"terms": {"groups.group.name.keyword": groups}}]
    if doc_id_allowlist:
        must.append({"terms": {"id": doc_id_allowlist}})
    must.append(field_query)
    return {"bool": {"must": must}}


async def sparse_fallback_search(
    client, query: str, groups: list[str], doc_id_allowlist: list[str] | None = None,
    limit: int = _ES_FALLBACK_LIMIT, group_cap: int = _ES_FALLBACK_GROUP_CAP,
    boost: bool = False,
) -> dict[str, list[dict]]:
    """ES fallback for lexical search on the Milvus collections whose sparse_vector was
    dropped. One ES call per query regardless of how many gap-collections are routed
    together - `groups` is the list of ES groups.group.name values to search (mapped from
    the routed gap-collections via ES_GROUP_FOR_COLLECTION), OR'd into one filter. Returns
    rows partitioned back into the same dict[collection, list[row]] shape
    common.milvus_client.hybrid_search returns, via the inverse of that same mapping. See
    docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md.

    `boost` mirrors Instant mode's raw_search toggle (_apply_boost) - safe to reuse here
    unchanged: this function's rows only ever get locally rank-sorted against each other
    (never raw-score-compared against Milvus - see retrieve.py::_flatten's interleave-by-rank),
    so a boosted ES score here can only change *which* ES-origin row ranks first among ES's
    own rows, never let ES outrank Milvus on score. See
    docs/superpowers/specs/2026-08-24-ai-mode-boosting-design.md."""
    # Requesting more than `limit` when multiple groups are routed gives _cap_group_shares
    # a real pool to draw from - without this, ES's own top-`limit` (sorted globally by
    # score) could already be dominated by one group before the cap ever sees the rest.
    fetch_size = limit if len(groups) <= 1 else limit * len(groups)

    response = await client.search(
        index=client.index,
        query=build_sparse_fallback_query_preview(query, groups, doc_id_allowlist, boost=boost),
        size=fetch_size,
        _source=["id", "groups.group.name"],
        highlight={"fields": {"fullcontent": {
            "fragment_size": _ES_HIGHLIGHT_FRAGMENT_CHARS, "number_of_fragments": 1,
        }}, "pre_tags": [""], "post_tags": [""]},
    )

    hits = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        group = source.get("groups", {}).get("group", {}).get("name")
        fragments = hit.get("highlight", {}).get("fullcontent")
        if group not in _COLLECTION_FOR_ES_GROUP or not fragments:
            continue
        hits.append({
            "_group": group, "_doc_id": source["id"], "_snippet": fragments[0], "_score": hit["_score"],
        })

    capped = _cap_group_shares(hits, limit, group_cap)

    by_collection: dict[str, list[dict]] = {}
    for hit in capped:
        collection = _COLLECTION_FOR_ES_GROUP[hit["_group"]]
        row = {
            "chunk_id": f"es:{hit['_doc_id']}:0",
            "doc_id": hit["_doc_id"],
            "text": trim_to_token_budget(hit["_snippet"]),
            "score": hit["_score"],
            "source": "es_fallback",
        }
        by_collection.setdefault(collection, []).append(row)
    return by_collection


_BOOST_PROFILES = {
    "KEYWORD": {"heading": 5.0, "subheading": 3.0, "fullcontent": 1.0,
                "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.5},
    "HYBRID": {"heading": 2.0, "subheading": 3.0, "fullcontent": 1.0,
               "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 2.5},
    "INTENT": {"heading": 2.0, "subheading": 2.0, "fullcontent": 1.5,
               "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.0},
}

# Boost magnitudes for the exact section-number phrase match only (chunk type "section"),
# ported from centax-node's legacy query (query_legacy.json) - confirmed live on the old
# platform to correctly rank a canonical "Section - 52" doc above every judgment that merely
# mentions "section 52" in its own body text. The gap between fields spans orders of
# magnitude, not the small 1-3x multipliers _BOOST_PROFILES uses for loose/fuzzy recall -
# that gap is the actual fix: it makes an exact heading match structurally undefeatable by
# BM25 term-frequency in a long document (a judgment repeating "section 52" 30+ times still
# can't outscore one exact heading hit). Deliberately excludes documenttypeboost/court_boost/
# landmarkruling - CLAUDE.md's hard-earned lesson is that boost_mode:"multiply" on those
# fields regresses eval pass rate even when patched; this only touches match_phrase boost
# weights on text fields already present in every doc, so there's no missing/zero-value
# fragility to inherit. Only applied to "section" chunks - the identity signal is meaningless
# for a court_city/citation/quoted chunk, where firing at this magnitude would misrank on any
# incidental heading match.
_SECTION_PHRASE_BOOSTS = {
    "heading": 100000.0,
    "subheading": 50000.0,
    "headnotes_text": 40000.0,
    "fullcontent": 1.0,
    "facts_text": 1.0,
    "held_text": 1.0,
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


# Current-edition preference for a specific act, ported from centax-node's legacy query
# (query_legacy.json embeds this as a `year:2026 AND subgroup:20042` compound should-clause
# with boost 80000.0, at the SAME scale as its own heading match_phrase boosts - not a small
# function_score multiplier). Belongs here, inside the should-list, not in _apply_boost's
# function_score: a first attempt put it there at weight 3.0 and it did nothing - verified live,
# the current-2025-edition doc stayed buried at rank ~108/200, because +3 is negligible next to
# the natural BM25 variance between 200+ near-identical "Section 52" heading matches. This
# should-clause competes at the SAME scale as _SECTION_PHRASE_BOOSTS instead (still additive,
# `bool` should-scoring is sum by default - no boost_mode:"multiply" risk, same safe mechanism
# _SECTION_PHRASE_BOOSTS already uses), sized well below an exact heading/subheading phrase
# match (100000/50000) so it only ever tiebreaks among docs that already matched the section
# number, never outranks a correct match to a *different* section. Only the current live edition
# (Income-tax Act, 2025, subgroup 111050000000020042) gets this - the 1961 edition keeps its
# small function_score bump (_STATIC_TAXONOMY_BOOSTS below), since nothing here should prefer
# an old edition over the current one.
_CURRENT_EDITION_SUBGROUP_ID = "111050000000020042"
_CURRENT_EDITION_SHOULD_BOOST = 20000.0


def _build_field_query(query: str, shape: str, chunks: list[dict] = (), boost_enabled: bool = False) -> dict:
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
    boosts = _BOOST_PROFILES[boost_profile_key(shape)]
    should = [
        {"multi_match": {"query": query, "fields": [field], "boost": boost, "fuzziness": "AUTO"}}
        for field, boost in boosts.items()
    ]
    for chunk in chunks:
        chunk_boosts = _SECTION_PHRASE_BOOSTS if chunk["type"] == "section" else boosts
        for field, boost in chunk_boosts.items():
            should.append({
                "match_phrase": {field: {"query": chunk["text"], "slop": chunk["proximity"], "boost": boost}},
            })
            if chunk.get("alt_text"):
                should.append({
                    "match_phrase": {field: {"query": chunk["alt_text"], "slop": chunk["proximity"], "boost": boost}},
                })
    if boost_enabled and any(chunk["type"] == "section" for chunk in chunks):
        should.append({
            "term": {
                "groups.group.subgroup.id": {
                    "value": _CURRENT_EDITION_SUBGROUP_ID, "boost": _CURRENT_EDITION_SHOULD_BOOST,
                },
            },
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
    factors/modifiers here.

    No separate landmarkruling:-10 exclusion here (an earlier version of this function had one,
    a top-level query must_not - since removed as a misreading of centax-node's actual source;
    see raw_search's comment for the full explanation). It isn't needed even for the boost:
    -10 already fails the `range: {gt: 0}` filter every function below is gated on, same as any
    missing/zero value, so a -10 doc already gets the neutral (1x, no boost) treatment - exactly
    centax-node's own "Don't add Function Score for blacklisted" comment describes, with no
    extra clause required."""
    def _boost_function(field: str, factor: float, modifier: str) -> dict:
        return {
            "filter": {"range": {field: {"gt": 0}}},
            "field_value_factor": {"field": field, "factor": factor, "modifier": modifier},
        }

    return {
        "function_score": {
            "query": field_query,
            "functions": [
                _boost_function("documenttypeboost", 0.2, "sqrt"),
                _boost_function("court_boost", 0.01, "none"),
                _boost_function("landmarkruling", 1.2, "log2p"),
            ],
            "boost_mode": "multiply",
        }
    }


# Recency ladder ported from centax-node's legacy query (query_legacy.json's function_score
# functions array) - formatteddocumentdate is confirmed 100% populated on the live index
# (docs/retrieval-flow-current-state.md), so this tier list is directly portable as-is.
# Weights kept at legacy's own scale (single/low-double-digit) even though legacy combined
# them under boost_mode "multiply" and _apply_boost below uses "sum" - under sum mode these
# numbers only ever *add* to a query's BM25/phrase-boost score, never multiply it, so unlike
# legacy there's no risk of a missing/zero date tier collapsing the whole score.
_RECENCY_TIERS = [
    ("now-1d", "now", 18.0),
    ("now-7d", "now-1d", 15.0),
    ("now-1M", "now-7d", 13.0),
    ("now-3M", "now-1M", 10.0),
    ("now-1y", "now-3M", 8.0),
    ("now-2y", "now-1y", 5.0),
    ("now-5y", "now-2y", 3.5),
    ("now-150y", "now-5y", 1.5),
]

# groups.group.name buckets to prefer when the query itself names a specific section/rule
# number (query_tokenizer.chunk_query's "section" chunk type - the same detector
# _SECTION_PHRASE_BOOSTS above already relies on): a query naming "Section 52" is looking
# for the statutory provision itself, not a judgment that happens to cite it. Scoped to
# that one existing detector rather than a general content-type classifier - Instant mode
# runs no LLM/intent classification of its own (extract_intent() is an AI-Mode-only,
# per-request network call), and chunk_query's section detector is already computed for
# every query regardless of this toggle, so reusing it costs nothing extra. See
# docs/superpowers/specs/... group/subgroup boosting discussion: legacy's own numeric
# taxonomy ids (groups.group.id/.subgroup.id/etc.) require resolving a query to the exact
# act/section's CMS node id, which this repo has no resolver for - deferred as a follow-up;
# this coarse groups.group.name boost is the buildable-today subset.
_STATUTORY_GROUPS = ["ACT", "RULE"]
_STATUTORY_GROUP_BOOST_WEIGHT = 8.0

# Static per-taxonomy-node boosts ported from centax-node's legacy query
# (query_legacy.json's function_score functions array: groups.group.id/groups.group.subgroup.id
# weight 2.0/3.0 entries) - unconditional, unlike _group_name_boost_functions above (which only
# fires for a section/rule-number query). Verified live against the real index before porting
# (see chat history/2026-08-24 audit): only 2 of the legacy sample's 5 id-boost entries still
# resolve to real docs - "groups.group.subgroup.id"=111050000000000064 (0 hits - that id is a
# *group* id, not a subgroup id) and "groups.group.id"=111050000000020048 (0 hits, dead/stale)
# are dropped. The other two - "groups.group.id"=111050000000000064 (ACT, 83,309 docs) and
# "groups.group.subgroup.id"=111050000000010687 (Income-tax Act 1961, 40,524 docs) - are kept
# here as small function_score tie-breakers. The third live id from the legacy sample,
# subgroup 111050000000020042 (Income-tax Act 2025 - the CURRENT edition), is deliberately NOT
# here: a first attempt put it in this list at weight 3.0 and verified live it did nothing - the
# current-edition doc stayed buried at rank ~108/200 for "Section 52", because +3 here is
# negligible next to natural BM25 variance between 200+ near-identical heading matches. It's
# instead a should-clause boost inside _build_field_query itself (_CURRENT_EDITION_SHOULD_BOOST,
# same scale as _SECTION_PHRASE_BOOSTS) - see that constant's comment for why it has to live
# there instead of here to actually move the ranking.
#
# Deliberately excludes centax-node's matching *penalty* functions (subcategory
# 111050000000017095 outside caselaws -> weight 0.03; subgroup 111050000000010567 Finance Acts
# minus year 2025 -> weight 0.02): those are only meaningful under boost_mode "multiply" (a
# near-zero weight suppresses a doc's score). Under this toggle's sum/additive design - the
# whole reason it doesn't reproduce _wrap_function_score's eval regression - every function can
# only ever add to a score, never suppress it, so there is no additive equivalent of a penalty.
_STATIC_TAXONOMY_BOOSTS = [
    ("groups.group.id", "111050000000000064", 2.0),  # ACT
    ("groups.group.subgroup.id", "111050000000010687", 2.0),  # Income-tax Act, 1961
]


def _recency_boost_functions() -> list[dict]:
    return [
        {"filter": {"range": {"formatteddocumentdate": {"gte": gte, "lte": lte}}}, "weight": weight}
        for gte, lte, weight in _RECENCY_TIERS
    ]


def _static_taxonomy_boost_functions() -> list[dict]:
    return [
        {"filter": {"term": {field: value}}, "weight": weight}
        for field, value, weight in _STATIC_TAXONOMY_BOOSTS
    ]


def _group_name_boost_functions(chunks: list[dict]) -> list[dict]:
    if not any(chunk["type"] == "section" for chunk in chunks):
        return []
    return [{
        "filter": {"terms": {"groups.group.name.keyword": _STATUTORY_GROUPS}},
        "weight": _STATUTORY_GROUP_BOOST_WEIGHT,
    }]


def _apply_boost(field_query: dict, chunks: list[dict]) -> dict:
    """Instant mode's opt-in `boost` toggle (raw_search's `boost` param). Additive
    (score_mode/boost_mode "sum"), deliberately never "multiply" - _wrap_function_score
    above is the same documenttypeboost/court_boost/landmarkruling formula design-doc
    section 2 built, but boost_mode "multiply" got it disabled: a single function
    landing on (or defaulting to) 0 zeroed the *entire* relevance score regardless of
    text match quality (see _wrap_function_score's docstring; 21/53 vs 42/53 eval pass
    rate with/without it). Sum mode can't reproduce that failure - every function here
    only ever adds a small, bounded amount on top of the query's own text-relevance
    score, so a doc with no boost signal at all just gets +0, never a score-killing
    multiplier. Still gated behind `gt: 0` filters for the sparse/zero-valued fields
    (landmarkruling 2.1% populated, court_boost a real 0 on 45.8% of the corpus) so
    "no signal" reads as +0, not a negative/degenerate field_value_factor output."""
    functions = [
        {
            "filter": {"range": {"documenttypeboost": {"gt": 0}}},
            "field_value_factor": {"field": "documenttypeboost", "factor": 0.2, "modifier": "sqrt"},
        },
        {
            "filter": {"range": {"court_boost": {"gt": 0}}},
            "field_value_factor": {"field": "court_boost", "factor": 0.01, "modifier": "none"},
        },
        {
            "filter": {"range": {"landmarkruling": {"gt": 0}}},
            "field_value_factor": {"field": "landmarkruling", "factor": 1.2, "modifier": "log2p"},
        },
        *_recency_boost_functions(),
        *_group_name_boost_functions(chunks),
        *_static_taxonomy_boost_functions(),
    ]
    return {
        "function_score": {
            "query": field_query,
            "functions": functions,
            "score_mode": "sum",
            "boost_mode": "sum",
        }
    }


def build_query_preview(query: str, boost: bool = False) -> dict:
    """The exact shape/chunk/ES-query breakdown raw_search uses for this query, without
    executing a search - single source of truth shared with raw_search (below) so the two
    can never drift apart (this is also why `boost` is a parameter here rather than raw_search
    wrapping the query itself after calling this - a caller that omitted `boost` here would
    silently show an unboosted preview for a boosted search). Powers the `/v1/query-analysis`
    endpoint (retrieval_api/query_analysis.py) and the Instant mode trace panel's "Show ES
    query" block - our equivalent of centax-node's own `/research-premium/api/v1/getLowLevelQuery`,
    for comparing query breakdowns side by side.

    boost=False (default): `es_query` is the unwrapped, plain BM25/phrase-boost query.
    boost=True: `es_query` is wrapped via _apply_boost() - see that function's docstring for
    why (additive/sum-mode, not the disabled multiply-mode _wrap_function_score)."""
    shape = effective_label(query)
    expanded_query = expand_query_synonyms(query)
    chunks = chunk_query(query)
    field_query = _build_field_query(expanded_query, shape, chunks=chunks, boost_enabled=boost)
    if boost:
        field_query = _apply_boost(field_query, chunks)
    return {
        "query": query,
        "shape": shape,
        "expanded_query": expanded_query if expanded_query != query else None,
        "chunks": chunks,
        "es_query": field_query,
    }


async def raw_search(client, query: str, limit: int = 20, boost: bool = False) -> list[dict]:
    field_query = build_query_preview(query, boost=boost)["es_query"]
    # No landmarkruling:-10 exclusion here either, deliberately - a previous version of this
    # function had one (`_exclude_blacklisted`, since removed), reasoning it preserved a
    # content filter that used to ride along inside centax-node's function_score must_not. That
    # was a misreading of the source: in centax-node's actual query (services/caselaws.js), that
    # must_not is scoped as a per-function `filter` *inside* one entry of function_score's
    # `functions` array - in ES semantics that only controls whether *that one function's* boost
    # applies to a doc (matching its own comment, "Don't add Function Score for blacklisted"). It
    # never excluded the doc from search results at all; centax-node shows these ~173 flagged
    # docs in results normally, just without the landmark-ruling boost bonus. Our prior version
    # hoisted that must_not to the top-level query instead, which - unlike centax-node - hid all
    # ~173 docs from every search entirely, a real regression with no source-of-truth backing it.
    # And now that boosting is off altogether, the original motivation (skip the boost for these
    # docs) is moot too: there's no boost being computed for anyone to skip.
    response = await client.search(index=client.index, query=field_query, size=limit)
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


async def fetch_fulltext_batch(client, doc_ids: list[str]) -> dict[str, str]:
    """Batched sibling of fetch_fullcontent for the Instant-mode reranker: one mget
    instead of N sequential searches, restricted to fullcontent (the field actually
    fed to the reranker). Returns full untrimmed text - trimming is the reranker call
    site's job (see trim_to_token_budget), not this generic fetch's."""
    if not doc_ids:
        return {}
    response = await client.mget(index=client.index, ids=doc_ids, _source=["fullcontent"])
    return {
        doc["_id"]: doc["_source"].get("fullcontent", "")
        for doc in response["docs"]
        if doc.get("found")
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
