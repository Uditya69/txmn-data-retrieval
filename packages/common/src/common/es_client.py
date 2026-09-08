import functools

from elasticsearch import AsyncElasticsearch

from common.config import Settings
from common.document_parser import strip_tags_to_text
from common.instant_classifier import effective_label
from common.instant_classifier.labels import boost_profile_key
from common.query_tokenizer import (
    chunk_query, default_instrument_kind, detect_group_signals, expand_query_normalizations,
    expand_query_synonyms, keyword_shape_group_filter,
)
from common.repotaxmannapi_boost_config import load_repotaxmannapi_boost_config
from common.repotaxmannapi_query_builder import build_should_clauses
from common.repotaxmannapi_scoring import ACT_GROUP_ID, build_function_score_functions
from common.repotaxmannapi_tokenizer import TokenType, tokenize
from common.schemas import (
    CATEGORY_DISPLAY_LABELS, ES_GROUP_FOR_COLLECTION, GROUP_DISPLAY_LABELS, MASTERINFO_CITATION_FIELDS,
)

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
# Document-reader highlighting (fetch_highlighted_fullcontent) needs the WHOLE
# document as one fragment, not a short reranker snippet - mirrors real prod's
# own FragmentSize(50000000) (FileContentElasticSearchResearch.cs:118).
_ES_DOCUMENT_HIGHLIGHT_FRAGMENT_CHARS = 5_000_000

# Operationally-variable years/current-editions/tuned-boost-weights behind every constant
# below this point that reads from `_BOOST_CONFIG` - see data/repotaxmannapi_boost_config.json
# and repotaxmannapi_boost_config.py's own docstrings for what belongs there vs stays a
# plain code constant here (structural taxonomy ids ported from repotaxmannapi's compile-time
# Constants.cs, which don't drift the way these do). Loaded once at import time, same as any
# other module-level constant - update the JSON file and restart the process to pick up a
# change, no code edit needed for these specific values.
_BOOST_CONFIG = load_repotaxmannapi_boost_config()

_COLLECTION_FOR_ES_GROUP = {group: collection for collection, group in ES_GROUP_FOR_COLLECTION.items()}


def build_sparse_fallback_query_preview(
    query: str, groups: list[str], doc_id_allowlist: list[str] | None = None, boost: bool = False,
) -> dict:
    """The exact query body sparse_fallback_search sends to ES, without executing a search -
    same single-source-of-truth pattern as build_query_preview (see its own docstring for why):
    a caller building this independently could silently drift from what the real search sends.
    Powers the AI Mode trace panel's "Show ES query" block for the ai_milvus_sparse step, same
    as build_query_preview powers Instant mode's."""
    # boost_source pinned to "sum" explicitly (2026-09-02) - immune to raw_search/
    # build_query_preview's own default changing underneath it. AI Mode's ES sparse-fallback
    # is a rank-based-only fusion mechanism (CLAUDE.md hard rule 3) with its own scoring
    # assumptions; it must never silently start using the repotaxmannapi multiply-mode
    # formula just because that became Instant mode's own new default elsewhere.
    field_query = build_query_preview(query, boost=boost, boost_source="sum")["es_query"]
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
            "text": trim_to_token_budget(strip_tags_to_text(hit["_snippet"])),
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

# Boost magnitudes for a chunk's own phrase match, applied to EVERY chunk type (text,
# citation, section - all of them), not just "section". This used to be gated to
# chunk["type"] == "section" only, reasoning that this magnitude "is meaningless for a
# court_city/citation/quoted chunk". That reasoning was never actually centax-node's own
# behavior - verified 2026-08-24 by reading centax-node's real source
# (services/searchTextElastic.js's default per-token branch, ~line 494-538): it applies this
# exact tier (heading/subheading/headnotestext at this same order of magnitude) to every
# QueryToken uniformly, with no type check at all - only `fullcontent` is kept small (5 there,
# 1 here), because that field is long/noisy and would drown in false-positive term-frequency
# hits otherwise. Confirmed live why the type=="section" gate was wrong: a case-citation query
# ("Commissioner of Customs Indian Oil 136 Taxman 491 demurrage section 14 and section 151A")
# buried the actual reported case (whose heading is the exact citation "136 Taxman 491") past
# rank 500, because its citation-chunk phrase match only got the small _BOOST_PROFILES weight
# (2-3) while two incidental section-number mentions in the same query got 100000 each and
# dominated with unrelated "Section 151A"/"Section 14" statutory-provision docs. Deliberately
# still excludes documenttypeboost/court_boost/landmarkruling (CLAUDE.md's boost_mode:
# "multiply" eval regression) - this only touches match_phrase boost weights on text fields
# already present in every doc, no missing/zero-value fragility to inherit.
_PHRASE_BOOSTS = {
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


# Edition-preference should-clause boosts - ported from repotaxmannapi's real production
# source (TaxmannAPI/Elastic/SearchTextElastic.cs, GlobalSearchResearch.cs), not guessed or
# carried over from centax-node's legacy query_legacy.json sample, which only covered a single
# edition (Income-tax Act 2025) and, when compared against production, its Act-vs-Rules
# instrument distinction turned out not to exist at all - centax-node/query_tokenizer.py's
# predecessor of this mechanism defaulted a bare "Rule N" to the Act too. Belongs here, inside
# the should-list, not in _apply_boost's function_score: a first attempt at a single-edition
# version put it in function_score at weight 3.0 and it did nothing - verified live, the
# current-2025-edition doc stayed buried at rank ~108/200 for "Section 52", because +3 is
# negligible next to the natural BM25 variance between 200+ near-identical heading matches.
# This should-clause competes at the SAME scale as _PHRASE_BOOSTS instead (still additive,
# `bool` should-scoring is sum by default - no boost_mode:"multiply" risk, same safe mechanism
# _PHRASE_BOOSTS already uses), sized well below an exact heading/subheading phrase match
# (100000/50000) so it only ever tiebreaks among docs that already matched the section/rule
# number, never outranks a correct match to a genuinely different section/rule.
#
# Both editions of each instrument get a should-clause (not just the current one) - ported
# from GlobalSearchResearch.cs:623-624, which weights the current edition higher (3) than the
# old one (2) rather than excluding the old edition entirely. All four subgroup ids verified
# live against this repo's own ES index (2026-09-01 investigation): "Income-tax Rules, 1962"
# returned real docs at subgroup 111050000000010121 (also confirmed against repotaxmannapi's
# Models/Research/AllAbout.cs:67); "Income-tax Rules, 2026" at subgroup 111050000000020129 (a
# real doc's own groups.group.subgroup - repotaxmannapi's BL/Constants.cs IncomeTaxRule2026
# constant, 103010000000002191, is a DIFFERENT id namespace - a `rule` associate id, not a
# groups.group.subgroup.id - and was not used here for that reason).
#
# Ratio (20000 current : 15000 old) mirrors GlobalSearchResearch.cs's 3:2 current:old weighting
# at _PHRASE_BOOSTS' should-clause scale rather than function_score's small-weight scale (see
# above for why the latter doesn't move ranking) - not itself independently verified live
# against this repo's index (unlike the subgroup ids, which are); a follow-up should confirm
# 15000 vs 20000 is the right gap once real query traffic is available.
_EDITION_BOOSTS_BY_INSTRUMENT_KIND = {
    kind: [(subgroup_id, boost) for subgroup_id, boost in pairs]
    for kind, pairs in _BOOST_CONFIG["edition_boosts_by_instrument_kind"].items()
}

# Group-signal boost - fixes a real query ("landmark Supreme Court ruling on GST") where
# generic "Words & Idioms" commentary docs (heading literally "Appellate power - Supreme
# Court", "Law - Declaration by Supreme Court", etc.) outscored actual GST case law by ~2x,
# because _PHRASE_BOOSTS' heading/subheading/headnotes_text tiers fire identically regardless
# of document type - a short commentary heading that happens to contain "Supreme Court" gets
# the same +100000 as a real citation match. centax-node's own queryAnalyzer.js/token
# dictionary (constants/token.js) recognizes RULING/JUDGEMENT/CASE/CITATION (-> CASELAWS),
# RULE, and ARTICLE (-> Experts Opinion; see query_tokenizer.detect_group_signals) as signals
# the query wants that content type specifically.
#
# Magnitude corrected 2026-09-01: originally ported centax-node's own constants (2,000,000 /
# 10,000,000) as-is, on the assumption they were already tuned. Checked against production's
# real source (repotaxmannapi/TaxmannAPI/Elastic/SearchTextElastic.cs:753) instead: the
# equivalent coarse group-id should-clause boost there is only 1000, sized as a genuine
# tiebreaker under its own 155000 heading phrase-boost tier (~0.6%) - not a value anywhere near
# large enough to outrank real text relevance. centax-node's 2,000,000/10,000,000 (20-100x
# *larger* than the 100000 heading tier they're supposed to sit under) was the actual root
# cause of a real bug: a bare "Rule 6" query got force-ranked by RULE-group membership alone,
# regardless of which unrelated Rule 6 (Motor Vehicles Rules, Customs Valuation Rules, etc.) -
# see git history/2026-09-01 investigation. Rescaled to preserve the correct proportion under
# this repo's own 100000 heading tier (was already scoped to skip section/citation chunks, so a
# citation-bearing query that merely mentions a rule number is unaffected either way) - CASELAWS
# keeps its original 5x-larger-than-RULE/ARTICLE ratio, just at the corrected base scale.
# ACT only ever arrives via keyword_shape_group_filter (detect_group_signals has no ACT
# entry - a bare "Section N" is a section-type chunk, which detect_group_signals always
# excludes, see its own docstring). Same 1000 tier as RULE: confirmed against the real
# .NET source (repotaxmannapi/TaxmannAPI/Elastic/SearchTextElastic.cs:751-766,
# GlobalSearchResearch.cs:613-621) that Section and Rule keyword-lookups share one code
# path with one boost, never a hard filter for either - see the correction below.
_GROUP_SIGNAL_SHOULD_BOOSTS = _BOOST_CONFIG["group_signal_should_boosts"]

# Real ES `groups.group.subgroup.id` -> the single "latest edition" year (`year.name.keyword`)
# global search narrows that subgroup to, ported from repotaxmannapi/TaxmannAPI/Elastic/
# SearchTextElastic.cs::GetGlobalSearchQuery (lines 638-786, called for every global search -
# SearchTextElastic.cs:281,301 - not shape/boost-gated the way the should-clause boosts above
# are). Real mechanism: `globalQuery &= (minusQuery || (CatIdsQuery && positiveFilters))` -
# `minusQuery` (line 734) lets every document NOT in one of these subgroups through
# unconditionally (case laws, commentary, most rules, etc.); a document that IS in one of
# these subgroups only survives if it also matches that subgroup's own `X && XYearFilter`
# pair (e.g. `IncomeTaxAct1961query = IncomeTaxAct1961YearFilter && IncomeTaxAct1961Filter`,
# lines 682-683,726) - i.e. only the current year's edition of that act/subgroup passes,
# every other yearly re-indexed edition is excluded from global search entirely. Without
# this, a bare "SECTION 52" query returns 100% Income-tax-Act documents - live-verified
# 2026-09-02: this repo's index carries ~40,500 separate Income-tax Act 1961 documents (one
# per year 1997-2026) all sharing the exact heading "Section - 52", each hitting
# _PHRASE_BOOSTS' 100000 heading-phrase tier identically - no should-clause boost/group
# signal can out-rank that many exact-heading duplicates, so the fix has to be a hard filter,
# matching production's own mechanism exactly, not a ranking change.
#
# Scoped to the 3 subgroups the real source actually hard-excludes via `minusQuery`
# (SearchTextElastic.cs:664-734), out of the ~15 real source references there - the others
# were checked and found to either not apply here (GST tariff's "latest edition" is a
# *runtime* dataset lookup, not a static subgroup+year pair - unported, needs its own
# separate investigation; Forms key off a formtype id, not a group id) or have zero live
# documents at all in this repo's index (Account Standard, AAA Model Report, Comparative
# Group, OECD Model Commentary - now hard-excluded anyway via
# _additional_exclusion_filters(), future-proofed for when that data is indexed) or carry no
# year field at all on any sampled doc (the 5 Rules groups, Companies Act 2013, CGST Act
# 2017 - consistent with the real source pairing only these 3 with an actual hard exclusion,
# everything else in its OR-list is should-boost only).
#
# CORRECTED 2026-09-03: Finance Act (general) was wrongly included here as a 4th
# hard-excluded subgroup. Direct re-read of `SearchTextElastic.cs:734` shows
# `minusQuery`'s AND-list never negates `FinanceActFilter`/`FinanceActYearFilter` at all -
# the line even has a dangling, commented-out `//!MinusFinanceActsFilter &&` where such a
# term would go, and no `MinusFinanceActsFilter` variable is ever defined anywhere in the
# file. `FinanceActFilter`/`FinanceActYearFilter` (line 690-691, boost 30000) are used
# exclusively inside the *positive* should-boost OR-list (`FinanceActBoostquery`, line 729,
# consumed at line 785) - production only soft-boosts the current year's general Finance Act
# edition, it never excludes older years' Finance Act documents from global search. This
# repo's `_edition_exclusion_filter` was hard-excluding every non-2025 Finance-Act-general
# document from every query - a real over-filtering bug, not a parity gap. See
# `_static_group_membership_should_clauses`'s `_FINANCE_ACT_GENERAL_*` constants below for
# the should-boost this subgroup actually gets. Only `MinusFinanceAct1994Filter` (a
# DIFFERENT subgroup - service tax, not general Finance Act) is genuinely hard-excluded
# unless current-year, matching the entry kept below.
#
# Year values are the real source's current ConfigurationManager.AppSettings values
# (Web.config) - 2026-09-03: moved into data/repotaxmannapi_boost_config.json (loaded as
# `_BOOST_CONFIG` above) rather than kept as inline Python literals, so these (and every
# other operationally-variable value in this file) can be updated by editing the JSON file
# directly, without a code change - the same "production itself only bumps this a few times
# a year via ops config, not code" reasoning as before, just externalized properly instead
# of scattered across module-level literals.
_LATEST_EDITION_ONLY_YEARS = _BOOST_CONFIG["latest_edition_years"]

# Additional hard-exclusion clauses folded into the real source's same `minusQuery`
# (SearchTextElastic.cs:664-734) that were never ported, beyond the year-gated subgroups
# above - found on a 2026-09-03 line-by-line re-read of the real file. Each is a genuine
# `bool.filter`-equivalent exclusion in production, distinct from the should-clause boosts
# above. Live-checked against this repo's own ES index (2026-09-03): `AAAModelReport`/
# `AccountStandard`/`ModelCommentaries` all have 0 matching docs today, and
# `parentheadings.hasfile` is a mapped-but-never-populated field (0 docs) - implemented
# anyway rather than skipped, since none of that is a reason to leave production's own logic
# unported; a doc it *should* have caught is just as easy to add to the corpus later as one
# that already exists.
#
# `MinusHasChildFilter` (line 706): `parentheadings.FirstOrDefault().hasfile == "no"` -
# skip stub/placeholder headings with no attached content file. Global, unconditional,
# unlike every other clause here - not edition/group-scoped at all.
#
# `MinusAAAModelFilterFilter` (line 672): `groups.group.id == AAAModelReport` - the entire
# AAA Model Report group excluded from global search, unconditionally (no year-gated
# re-admission counterpart anywhere in the real source's positive OR-list at line 785).
_AAA_MODEL_REPORT_GROUP_ID = "111050000000017485"

# `minusASQuery` (lines 666-668, 732): `groups.group.subgroup.id == AccountStandard AND
# year.id == "2015"` - unlike the subgroups above, this is an EXCLUDE-when-matches-year
# clause, not a reinstate-when-matches-year one: only the 2015 edition is excluded, every
# other year of Account Standard content passes through untouched (matches
# `year.id`, not `year.name.keyword`, per the real source's own field choice here).
_ACCOUNT_STANDARD_SUBGROUP_ID = _BOOST_CONFIG["account_standard"]["subgroup_id"]
_ACCOUNT_STANDARD_EXCLUDED_YEAR = _BOOST_CONFIG["account_standard"]["excluded_year"]

# `OecdCommentaryFilter`/`OecdModelLatestCommentaryFilter` (lines 703-704, 724): `groups.
# group.subgroup.subsubgroup.id == ModelCommentaries` is negated unconditionally in
# `minusQuery`, then re-admitted only for `year.id == "2017"` via the positive OR-list -
# i.e. only the 2017 edition of OECD Model Commentaries passes global search, every other
# year is excluded. Same double-gate shape as the 3 `_LATEST_EDITION_ONLY_YEARS` subgroups,
# just on `subsubgroup.id` + `year.id` instead of `subgroup.id` + `year.name.keyword`.
_OECD_MODEL_COMMENTARIES_SUBSUBGROUP_ID = _BOOST_CONFIG["oecd_model_commentaries"]["subsubgroup_id"]
_OECD_MODEL_COMMENTARIES_LATEST_YEAR = _BOOST_CONFIG["oecd_model_commentaries"]["latest_year"]

# `GstTarrifMinusFilter`/`EditionGoodsLatestFilter`/`EditionServicesLatestFilter`/
# `EditionCGSTplusSGSTLatestFilter` (SearchTextElastic.cs:670-671,676,718-720,734,785) -
# GST Tariff's "latest edition" exclusion. Real source excludes the WHOLE top-level Tariff
# group (`groups.group.id == "111050000000017179"`) via one `GstTarrifMinusFilter`, then
# re-admits per-subgroup via 3 separate edition filters (Goods/Services/CGST+SGST-combined -
# a live terms aggregation on `groups.group.id` confirms these 3 subgroups are the entire
# membership of that one group). This repo instead runs 3 independent subgroup-scoped
# exclusions with the same net effect (a doc outside all 3 subgroups always passes, matching
# every other exclusion in this file's shape) rather than one group-level exclude + 3
# reinstate clauses. Real source resolves the current subsubgroup id per variant via a
# *runtime* `GstTariffBL` dataset lookup (`type1Data`/`type2Data`/`type3Data.
# FirstOrDefault().MID`), not a static id - previously flagged as "needs its own separate
# investigation" and left unported. Live-checked 2026-09-03: real data exists (2998 Goods /
# 733 Services / 413 CGST+SGST docs across ~29-35 "Edition N" subsubgroups each), and every
# subsubgroup's own `name` is a clean, monotonic "Edition N [Goods|Services|(CGST + SGST)]"
# label - the current edition is simply the highest N, no runtime dataset needed to
# reconstruct it. Ids below are the current-highest-edition subsubgroups as of 2026-09-03
# (Goods: "Edition 42 Goods"; Services: "Edition 35 Services (IGST)"; CGST+SGST: "Edition 35
# Services (CGST + SGST)") - plain literals, same precedent as every other
# repotaxmannapi-ported magic id/year in this file (bumped by hand on the rare occasion a
# new tariff edition is published, not on every deploy).
#
# CGST+SGST was found missing entirely in a later audit pass (2026-09-03) - unlike the
# zero-doc future-proofing cases elsewhere in this file, this was a REAL gap: 413 live docs
# had no latest-edition filtering applied at all (neither the Goods nor Services filter's
# subgroup-id check matched them, so both let every CGST+SGST doc, of every edition,
# through unconditionally) until this was added.
_GST_TARIFF_GOODS_SUBGROUP_ID = _BOOST_CONFIG["gst_tariff"]["goods_subgroup_id"]
_GST_TARIFF_GOODS_LATEST_SUBSUBGROUP_ID = _BOOST_CONFIG["gst_tariff"]["goods_latest_subsubgroup_id"]
_GST_TARIFF_SERVICES_SUBGROUP_ID = _BOOST_CONFIG["gst_tariff"]["services_subgroup_id"]
_GST_TARIFF_SERVICES_LATEST_SUBSUBGROUP_ID = _BOOST_CONFIG["gst_tariff"]["services_latest_subsubgroup_id"]
_GST_TARIFF_CGST_SGST_SUBGROUP_ID = _BOOST_CONFIG["gst_tariff"]["cgst_sgst_subgroup_id"]
_GST_TARIFF_CGST_SGST_LATEST_SUBSUBGROUP_ID = _BOOST_CONFIG["gst_tariff"]["cgst_sgst_latest_subsubgroup_id"]

# `MinusFormTypequery`/`FormTypequery` (SearchTextElastic.cs:677, 680, 700-701, 733) -
# Forms' "latest edition" exclusion: every `formtype.id == "frmtyp002"` doc is excluded
# from global search unless it also matches `year.name == LattestFormYear` (a
# ConfigurationManager.AppSettings value, same "runtime app-setting, not a compile-time
# constant" shape as `_LATEST_EDITION_ONLY_YEARS`' Finance Act year). Previously flagged as
# "keys off a formtype id, not a group id - not ported, same reason [as GST Tariff]" -
# unlike GST Tariff, live-checked 2026-09-03 and confirmed still genuinely zero: 0/410,427
# docs have `masterinfo.info.formtype` populated at all, so there is no live Forms document
# to test this filter against. Implemented anyway per the same future-proofing call as
# every other zero-data exclusion above. `_FORMS_LATEST_YEAR`'s value IS confirmed real
# (2026-09-03) - read directly from `repotaxmannapi/TaxmannAPI/Web.config`'s own
# `LattestFormYear` key (value "2026", checked into this checkout, no need to ask the
# team) - not derived from any live document the way the GST Tariff edition ids above are,
# since none exist yet to check it against. Re-verify this against real Web.config (or
# actual indexed Forms data) once Forms content lands, in case the value has moved on by
# then - same as every other Web.config-sourced year in this file
# (`_LATEST_EDITION_ONLY_YEARS`'s entries were cross-checked the same way and all matched).
_FORMS_FORMTYPE_ID = _BOOST_CONFIG["forms"]["formtype_id"]
_FORMS_LATEST_YEAR = _BOOST_CONFIG["forms"]["latest_year"]


def _forms_latest_edition_filter() -> dict:
    """A doc that isn't this exact formtype passes unconditionally; one that is must also
    match the current form year - same should-reinstate shape as
    `_gst_tariff_latest_edition_filter`, keyed on `masterinfo.info.formtype.id` instead of
    `groups.group.subgroup.id` (Forms aren't identified by group/subgroup the way every
    other exclusion in this file is)."""
    return {
        "bool": {
            "should": [
                {"bool": {"must_not": [{"term": {"masterinfo.info.formtype.id": _FORMS_FORMTYPE_ID}}]}},
                {"bool": {"must": [
                    {"term": {"masterinfo.info.formtype.id": _FORMS_FORMTYPE_ID}},
                    {"term": {"year.name.keyword": _FORMS_LATEST_YEAR}},
                ]}},
            ],
            "minimum_should_match": 1,
        },
    }


def _additional_exclusion_filters() -> list[dict]:
    """The 4 extra hard-exclusion clauses above, each an independent `bool.filter` clause
    ANDed alongside `_edition_exclusion_filter()` - mirrors `minusQuery`'s own AND-of-NOTs
    structure (each individual `!Minus...Filter` term there is a separate top-level AND
    operand, not one combined should-list), so a document must pass every one of these
    independently, not just one of them."""
    return [
        {"bool": {"must_not": [{"term": {"parentheadings.hasfile.keyword": "no"}}]}},
        {"bool": {"must_not": [{"term": {"groups.group.id": _AAA_MODEL_REPORT_GROUP_ID}}]}},
        {
            "bool": {
                "must_not": [{
                    "bool": {
                        "must": [
                            {"term": {"groups.group.subgroup.id": _ACCOUNT_STANDARD_SUBGROUP_ID}},
                            {"term": {"year.id": _ACCOUNT_STANDARD_EXCLUDED_YEAR}},
                        ],
                    },
                }],
            },
        },
        {
            "bool": {
                "should": [
                    {"bool": {"must_not": [
                        {"term": {"groups.group.subgroup.subsubgroup.id": _OECD_MODEL_COMMENTARIES_SUBSUBGROUP_ID}},
                    ]}},
                    {"bool": {"must": [
                        {"term": {"groups.group.subgroup.subsubgroup.id": _OECD_MODEL_COMMENTARIES_SUBSUBGROUP_ID}},
                        {"term": {"year.id": _OECD_MODEL_COMMENTARIES_LATEST_YEAR}},
                    ]}},
                ],
                "minimum_should_match": 1,
            },
        },
        _gst_tariff_latest_edition_filter(
            _GST_TARIFF_GOODS_SUBGROUP_ID, _GST_TARIFF_GOODS_LATEST_SUBSUBGROUP_ID,
        ),
        _gst_tariff_latest_edition_filter(
            _GST_TARIFF_SERVICES_SUBGROUP_ID, _GST_TARIFF_SERVICES_LATEST_SUBSUBGROUP_ID,
        ),
        _gst_tariff_latest_edition_filter(
            _GST_TARIFF_CGST_SGST_SUBGROUP_ID, _GST_TARIFF_CGST_SGST_LATEST_SUBSUBGROUP_ID,
        ),
        _forms_latest_edition_filter(),
    ]


def _gst_tariff_latest_edition_filter(subgroup_id: str, latest_subsubgroup_id: str) -> dict:
    """A doc outside this subgroup passes unconditionally; one inside it must also match
    the current-edition subsubgroup id - same should-reinstate shape as
    `_edition_exclusion_filter`, just keyed on subsubgroup id instead of year."""
    return {
        "bool": {
            "should": [
                {"bool": {"must_not": [{"term": {"groups.group.subgroup.id": subgroup_id}}]}},
                {"bool": {"must": [
                    {"term": {"groups.group.subgroup.id": subgroup_id}},
                    {"term": {"groups.group.subgroup.subsubgroup.id": latest_subsubgroup_id}},
                ]}},
            ],
            "minimum_should_match": 1,
        },
    }


# Group-membership should-boosts ported from SearchTextElastic.cs::GetGlobalSearchQuery's
# positive OR-list (line 785, CatIdsQuery && (...)) - real, unscaled weights
# (GlobalSearchResearch.cs:690-698/722), verbatim per the user's explicit copy-paste
# instruction (2026-09-06 correction - a prior version scaled these ~0.25x, which is now
# reverted). Fire unconditionally (per-document subgroup membership only, no query-side
# gating).
_STATIC_GROUP_MEMBERSHIP_BOOSTS = [
    (subgroup_id, boost) for subgroup_id, boost in _BOOST_CONFIG["static_group_membership_boosts"]["boosts"]
]

# `FinanceActBoostquery` (`FinanceActFilter && FinanceActYearFilter`, both boost 30000,
# lines 690-691, 729) - unlike the 5 above, this one IS year-gated: only the current year's
# Finance-Act-general edition gets the boost, mirroring `_EDITION_BOOSTS_BY_INSTRUMENT_KIND`'s
# own current-vs-old should-clause pattern. NOT a hard exclusion (see
# `_LATEST_EDITION_ONLY_YEARS`'s 2026-09-03 correction above) - every year passes, this only
# tiebreaks among them.
_FINANCE_ACT_GENERAL_SUBGROUP_ID = _BOOST_CONFIG["finance_act_general"]["subgroup_id"]
_FINANCE_ACT_GENERAL_CURRENT_YEAR = _BOOST_CONFIG["finance_act_general"]["current_year"]
_FINANCE_ACT_GENERAL_BOOST = _BOOST_CONFIG["finance_act_general"]["boost"]


def _static_group_should_clauses() -> list[dict]:
    should = [
        {"term": {"groups.group.subgroup.id": {"value": subgroup_id, "boost": boost}}}
        for subgroup_id, boost in _STATIC_GROUP_MEMBERSHIP_BOOSTS
    ]
    should.append({
        "bool": {
            "must": [
                {"term": {"groups.group.subgroup.id": _FINANCE_ACT_GENERAL_SUBGROUP_ID}},
                {"term": {"year.name.keyword": _FINANCE_ACT_GENERAL_CURRENT_YEAR}},
            ],
            "boost": _FINANCE_ACT_GENERAL_BOOST,
        },
    })
    return should


def _edition_exclusion_filter() -> dict:
    """The hard `bool.filter` clause every global-search ES query gets, mirroring
    GetGlobalSearchQuery's `minusQuery || (CatIdsQuery && *YearFilter)` structure exactly:
    a document not in one of _LATEST_EDITION_ONLY_YEARS' subgroups passes unconditionally
    (first should-clause); one that is must additionally match that subgroup's own current
    year (remaining should-clauses) - see _LATEST_EDITION_ONLY_YEARS' comment for why only
    these 4 subgroups are ported."""
    should = [
        {"bool": {"must_not": [{"terms": {"groups.group.subgroup.id": list(_LATEST_EDITION_ONLY_YEARS)}}]}},
    ]
    for subgroup_id, year in _LATEST_EDITION_ONLY_YEARS.items():
        should.append({
            "bool": {
                "must": [
                    {"term": {"groups.group.subgroup.id": subgroup_id}},
                    {"term": {"year.name.keyword": year}},
                ],
            },
        })
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _build_field_query(query: str, shape: str, chunks: list[dict] = (), boost_enabled: bool = False) -> dict:
    """Query-shape-aware multi-field search (design doc section 1+3): every content field
    is searched (facts_text/held_text/headnotes_text are only 26-58% populated on the real
    index, so heading/subheading/fullcontent - 100% populated - must never be skipped),
    with boosts picked by the no-LLM query-shape classifier.

    chunks (see query_tokenizer.chunk_query - ported from centax-node's queryAnalyzer.js/
    searchTextElastic.js) add match_phrase-with-slop should clauses per field, one per chunk -
    never replacing the loose per-field multi_match terms above, so a query still falls back to
    plain OR-term recall (typos/fuzzy matches match_phrase can't tolerate) even where chunking
    finds nothing. Every chunk's phrase clause uses `_PHRASE_BOOSTS` (heading/subheading/
    headnotes_text at 100000/50000/40000, fullcontent/facts_text/held_text at 1.0) regardless of
    chunk type (text, citation, section - all of them) - ported from centax-node's real behavior
    (searchTextElastic.js's default per-token branch applies this exact tier to every QueryToken
    uniformly, no type check). An earlier version of this function gated the big tier to
    chunk["type"] == "section" only, reusing the small per-shape `boosts` weight for every other
    chunk type - verified live to be wrong: a case-citation query's citation-chunk phrase match
    (the exact reported citation, its strongest identifying signal) only got weight 2-3 under
    that scheme, while an incidental section-number mention elsewhere in the same query got
    100000 and buried the real case past rank 500 behind unrelated statutory-provision docs. This
    replaces the older, narrower extract_boost_phrases mechanism (which only phrase-boosted the
    few explicitly-recognized merges - Section+number, court+city, citation triple, quotes - and
    left every other word, including an unrecognized party name, to compete as independent OR
    terms with no phrase treatment at all)."""
    boosts = _BOOST_PROFILES[boost_profile_key(shape)]
    should = [
        {"multi_match": {"query": query, "fields": [field], "boost": boost, "fuzziness": "AUTO"}}
        for field, boost in boosts.items()
    ]
    for chunk in chunks:
        for field, boost in _PHRASE_BOOSTS.items():
            should.append({
                "match_phrase": {field: {"query": chunk["text"], "slop": chunk["proximity"], "boost": boost}},
            })
            if chunk.get("alt_text"):
                should.append({
                    "match_phrase": {field: {"query": chunk["alt_text"], "slop": chunk["proximity"], "boost": boost}},
                })
    if boost_enabled:
        instrument_kind = default_instrument_kind(chunks, query)
        for subgroup_id, edition_boost in _EDITION_BOOSTS_BY_INSTRUMENT_KIND.get(instrument_kind, []):
            should.append({
                "term": {
                    "groups.group.subgroup.id": {"value": subgroup_id, "boost": edition_boost},
                },
            })
        should.extend(_static_group_should_clauses())
    for group_name in detect_group_signals(chunks):
        should.append({
            "term": {
                "groups.group.name.keyword": {
                    "value": group_name, "boost": _GROUP_SIGNAL_SHOULD_BOOSTS[group_name],
                },
            },
        })
    # Bare-anchor-lookup queries ("Rule 6", "Section 54F") get the same group-signal
    # should-clause boost as detect_group_signals above, not a hard filter - see
    # keyword_shape_group_filter's docstring for why this is scoped narrowly to
    # shape=="KEYWORD" and doesn't reproduce detect_group_signals' section-chunk exclusion
    # regression.
    #
    # CORRECTED 2026-09-02 (was a hard `bool.filter` term clause, wrongly excluding every
    # non-matching-group document from the result entirely - "SECTION 52" returned Acts
    # only, never the case laws/commentary that cite it, contradicting the real product's
    # own UI, which shows a mix). The prior comment here claimed this was "confirmed live
    # (2026-09-01) as the actual mechanism behind centax-node's own 'correct' Rule-lookup
    # results" - re-investigated directly against the real .NET source
    # (repotaxmannapi/TaxmannAPI/Elastic/SearchTextElastic.cs:751-766,
    # GlobalSearchResearch.cs:613-621) and that claim does not hold: `iGroupID`'s only two
    # real destinations are a `match_phrase(boost=1000)` should-clause and a `function_score`
    # weight function - never a `bool.filter`/`must` term. Section and Rule share the
    # identical code path there (same variable, same reduced single-digit multiplier, no
    # filter-vs-boost distinction between them). No hard `groups.group.name`/`.id` filter
    # exists anywhere in the real source for this signal, for any group - confirmed by a
    # full grep of every `groups.*` reference in GlobalSearchResearch.cs. (centax-node, a
    # separate JS system not present in this checkout, may behave differently - if that
    # claim was ever re-checked, it wasn't against this repo's actual reference source.)
    hard_group = keyword_shape_group_filter(shape, chunks)
    if hard_group is not None:
        should.append({
            "term": {
                "groups.group.name.keyword": {
                    "value": hard_group, "boost": _GROUP_SIGNAL_SHOULD_BOOSTS[hard_group],
                },
            },
        })
    # Latest-edition-only hard filter (see _edition_exclusion_filter's own comment) - applies
    # to every query, not gated by shape/boost, matching GetGlobalSearchQuery's own
    # unconditional application to all global search. _additional_exclusion_filters() (the
    # 4 extra minusQuery clauses) shares the same unconditional scope.
    return {
        "bool": {
            "should": should, "minimum_should_match": 1,
            "filter": [_edition_exclusion_filter(), *_additional_exclusion_filters()],
        },
    }


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
    `evals/datasets/retrieval_cases.json`) with the patched formula still active (21/53 passed) versus
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


# Recency ladder - corrected 2026-08-25. Originally ported from query_legacy.json (a saved
# sample query, not live code) - that turned out to be the wrong source. centax-node's actual
# live generic-search path (searchText.js's main query builder, function_score wrapping at
# searchText.js:1078) calls searchText.js::functionAging(), a completely different 11-tier
# ladder (verified by reading the real function body, not a sample query file). formatteddocumentdate
# is confirmed 100% populated on the live index (docs/retrieval-flow-current-state.md), so this
# tier list is directly portable as-is. Weights kept at legacy's own scale (single/low-double-digit)
# even though legacy combines them under boost_mode "multiply" and _apply_boost below uses "sum" -
# under sum mode these numbers only ever *add* to a query's BM25/phrase-boost score, never
# multiply it, so unlike legacy there's no risk of a missing/zero date tier collapsing the whole
# score. Deliberately excludes functionAging's groupId-conditional extras (year.name-range boosts,
# categories.id/categories.subcategory.id EXCISE/STATE_GST penalty weights) - those only fire
# when the UI itself scopes a search to one explicit group/category (a filter parameter this
# repo's raw_search has no equivalent of yet), and the category weights are Centax-product-specific
# taxonomy (several literally commented out in centax's own source) with no bearing on Taxmann's
# index.
_RECENCY_TIERS = [
    ("now-6M", "now", 11.0),
    ("now-1y", "now-6M", 10.0),
    ("now-2y", "now-1y", 9.0),
    ("now-3y", "now-2y", 8.0),
    ("now-6y", "now-3y", 7.0),
    ("now-11y", "now-6y", 6.0),
    ("now-26y", "now-11y", 5.0),
    ("now-35y", "now-26y", 4.0),
    ("now-51y", "now-35y", 3.0),
    ("now-65y", "now-51y", 2.0),
    ("now-85y", "now-65y", 1.0),
]

# groups.group.name buckets to prefer when the query itself names a specific section/rule
# number (query_tokenizer.chunk_query's "section" chunk type - the same detector
# _PHRASE_BOOSTS above already relies on): a query naming "Section 52" is looking
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
# are dropped. The other, "groups.group.id"=111050000000000064 (ACT, 83,309 docs), is kept here
# as a small function_score tie-breaker. Its sibling entry (subgroup 111050000000010687,
# Income-tax Act 1961) was removed 2026-09-01: now covered, at the correct much-larger scale, by
# _EDITION_BOOSTS_BY_INSTRUMENT_KIND's should-clause boost instead (see that constant's comment
# for why a small function_score weight doesn't move ranking at all) - keeping both here would
# double-count the same subgroup.
#
# Deliberately excludes centax-node's matching *penalty* functions (subcategory
# 111050000000017095 outside caselaws -> weight 0.03; subgroup 111050000000010567 Finance Acts
# minus year 2025 -> weight 0.02): those are only meaningful under boost_mode "multiply" (a
# near-zero weight suppresses a doc's score). Under this toggle's sum/additive design - the
# whole reason it doesn't reproduce _wrap_function_score's eval regression - every function can
# only ever add to a score, never suppress it, so there is no additive equivalent of a penalty.
_STATIC_TAXONOMY_BOOSTS = [
    (field, value, weight) for field, value, weight in _BOOST_CONFIG["static_taxonomy_boosts"]["boosts"]
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
    "no signal" reads as +0, not a negative/degenerate field_value_factor output.

    viewcount added 2026-09-01, ported from repotaxmannapi's real production source
    (TaxmannAPI/Elastic/GlobalSearchResearch.cs) - factor/modifier (0.0000018, log2p)
    copied as-is from there. Verified the field exists and is populated in this repo's
    own index (e.g. a real doc: viewcount 335) before porting. Same gt:0 gate as
    court_boost/landmarkruling: a doc with viewcount exactly 0 (common, not just
    "missing") gets +0 here rather than relying on log2p(0)==0 implicitly - under
    repotaxmannapi's boost_mode:multiply that same log2p(0)==0 zeroes the ENTIRE score
    for a low-traffic doc regardless of text relevance (verified by formula, not by a
    live query - no execution access to that system); this repo's sum-mode design can't
    reproduce that failure by construction, which is the whole reason it exists."""
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
        {
            "filter": {"range": {"viewcount": {"gt": 0}}},
            "field_value_factor": {"field": "viewcount", "factor": 0.0000018, "modifier": "log2p"},
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


def _and_of_or_groups(or_groups: list[list[dict]]) -> dict:
    """AND together a field's per-token OR-groups (each itself OR'd internally) -
    SearchTextElastic.cs's queryHeadingAnd &= (alt1 || alt2) pattern, repeated per token,
    now reconstructed from build_should_clauses's per-field-tier output. A single OR-group
    degenerates to that group directly (no wrapping needed) when there's only one token's
    worth of contribution to this field - matches today's single-token-query behavior
    exactly."""
    wrapped = [
        or_group[0] if len(or_group) == 1 else {"bool": {"should": or_group, "minimum_should_match": 1}}
        for or_group in or_groups
    ]
    return wrapped[0] if len(wrapped) == 1 else {"bool": {"must": wrapped}}


def _build_repotaxmannapi_field_query(query: str) -> dict:
    """The ported legacy .NET multiply-mode function_score - shared by build_query_preview
    (the "sum" vs "repotaxmannapi" branch) and raw_search_grouped (below), so the scored
    `query` half of both a flat search and a bucketed one can never drift apart. See
    raw_search's docstring for the group_id resolution this mirrors (first-classified-token-
    wins, TaxmannQueryAnalizer.cs's SetPrimaryTag gate).

    ALWAYS FunctionScore (2026-09-06 correction) - verified directly against
    GlobalSearchResearch.cs (repotaxmannapi/TaxmannAPI/Elastic/GlobalSearchResearch.cs:38-40,
    599, 1016-1017): `searchContainer = stext.query` (the exact bool query
    SearchTextElastic.cs builds - same should/must shape this function produces) is wrapped in
    `.FunctionScore(...)` unconditionally, every single request, no branch anywhere that skips
    it for a whole-quoted-phrase query. A prior version of this function had a
    skip-FunctionScore-entirely branch for that shape, based on a single live trace of
    `"571/Ahd/2016 vide order dated 02-04-2026"` that appeared to show a plain unwrapped
    `bool` query - that trace was misread (or captured a stale/broken state): the real source
    cannot structurally produce that shape. Removed; every return path here now goes through
    `function_score`.

    _edition_exclusion_filter (2026-09-02): the real source's edition-exclusion mechanism
    (SearchTextElastic.cs::GetGlobalSearchQuery) applies to ALL global search regardless of
    which per-token GetQuery branch built the should-clauses - it's not specific to the
    "sum"-mode builder. Without this here, this builder floods on the exact same bug
    _build_field_query had before the fix: a bare "SECTION 52" surfaced ~40,500 separate
    Income-tax Act 1961 yearly editions (live-verified 2026-09-02, re-checked against this
    builder specifically after the boost_source default flip made it reachable through the
    UI for the first time)."""
    tokens = tokenize(query, is_global=True)
    phrase_tokens = [t for t in tokens if t.type == TokenType.PHRASE_WORD]
    other_tokens = [t for t in tokens if t.type != TokenType.PHRASE_WORD]
    # Resolved once, up front, so both build_should_clauses calls below (should-list) and
    # build_function_score_functions further down (functions stack) use the identical
    # group_id - first-classified-token-wins, TaxmannQueryAnalizer.cs's SetPrimaryTag gate
    # (see this function's own docstring, "group_id resolution").
    group_id = next((t.group_id for t in tokens if t.group_id != "0"), "0")
    # ACT_GROUP_ID fallback (2026-09-07 fix) - mirrors SearchTextElastic.cs:283-284 exactly,
    # NOT a generic "nothing classified -> assume Acts" rule (that would be too broad - a
    # multi-word query like "foo bar baz" leaves searchFields.Count > 1 in the real source
    # and must NOT get this fallback, only the separate, already-ported "not tokens" ->
    # groups.group.url=="act" boost further below when EVERY per-token query also came back
    # null). The real gate is narrower and specific:
    #   if (SearchProcess.searchFields.Count == 1 && SearchProcess.searchFields[0].QType == "N")
    #       SearchProcess.iGroupID = SearchProcess.GetGroupID().ToString();  // "0" -> "acts"
    # i.e. the ENTIRE query must tokenize to exactly one token, and that one token must be
    # Numeric-typed (TokenType.NUMBER) - exactly a bare section/rule number with nothing else
    # in the query ("148", "270A", ...). A bare number never matches the token dictionary
    # (iTagNo stays "0"), so group_id would otherwise stay "0" here too, silently skipping
    # every group_id-gated boost below (the w/w0/w1/wc groupBoost functions in
    # build_function_score_functions, and the GroupFilterquery should-clause in the
    # whole-quoted-phrase branch further down) for exactly this query shape. Verified live
    # 2026-09-07 (hot_query_smoke_test.py): 8 of Taxmann's top-10 hot queries are bare
    # section numbers - this was the actual reason our index buried the current Income-tax
    # Act edition's own section text under unrelated recent case law for every one of them,
    # not the multiply-mode boost formula itself (prod runs the identical BoostMode.Multiply
    # on every query - re-verified against every *ElasticSearchResearch.cs controller, not
    # just this one - and still ranks the Act section first, because it gets this same
    # fallback and we didn't).
    if group_id == "0" and len(tokens) == 1 and tokens[0].type == TokenType.NUMBER:
        group_id = ACT_GROUP_ID
    per_field = build_should_clauses(other_tokens, is_global=True, is_excus=False, group_id=group_id)
    if phrase_tokens:
        # Double-quoted text in the search bar (e.g. `"section 52"`) is extracted by
        # `tokenize()` into PHRASE_WORD tokens (TaxmannQueryAnalizer.cs's quote-extraction,
        # SearchTextElastic.cs:1068-1082's PH/isExcus branch) - these need the
        # `.phrase_search` field variant + no analyzer, rendered via a second
        # `is_excus=True` call and merged in, since `build_should_clauses` renders an
        # entire call's tokens under one mode (see its docstring).
        phrase_per_field = build_should_clauses(phrase_tokens, is_global=True, is_excus=True, group_id=group_id)
        for field, or_groups in phrase_per_field.items():
            per_field.setdefault(field, []).extend(or_groups)

    # Per-field AND-of-OR assembly (2026-09-06 restructure): each field-tier's own
    # per-token OR-groups get ANDed together via `_and_of_or_groups`, then all field-tier
    # results get OR'd together as the top-level `should` - matching SearchTextElastic.cs's
    # queryFieldAnd/queryFieldOr accumulation exactly (see build_should_clauses's own
    # docstring and the 2026-09-06 parity plan's Task 2). The special
    # "_fullcontent_minus" key is popped out and applied as a must_not against the
    # fullcontent field-tier (SearchTextElastic.cs:1169: queryFullcontentAnd &&
    # queryFullcontentOr && !queryFullcontentMinusOr - the SUB-clause is a NEGATION of
    # this field-tier's own contribution, not a positive boost - see Task 3).
    minus_groups = per_field.pop("_fullcontent_minus", [])
    should: list[dict] = []
    for field, or_groups in per_field.items():
        field_query = _and_of_or_groups(or_groups)
        if field == "fullcontent" and minus_groups:
            minus_query = _and_of_or_groups(minus_groups)
            field_query = {"bool": {"must": [field_query], "must_not": [minus_query]}}
        should.append(field_query)

    # headnotestext supplemental clause (2026-09-07 fix, additive, NOT a replacement for
    # the _HEADNOTES_TEXT_FIELD ("headnotes_text", underscore) tiers above - those stay
    # untouched, still the byte-exact ported formula. Separate, real bug found this
    # session: `headnotes_text` is a field THIS repo's own transform_full.py derives by
    # joining raw["headnotes"][]["text"] (transform_full.py:91-93), which stops at
    # "[In favour of X]" - the source XML's own <headnote> tag genuinely ends there too,
    # confirmed against a real prod fullcontent pull. Real prod's own `headnotestext`
    # field (no underscore) is DIFFERENT - not the raw <headnote> tag content at all, an
    # enriched field their indexer builds separately (headnote text + a repeated copy +
    # an appended `~~`-delimited keyword/metadata tail from the doc's other fields) - and
    # that enriched value already exists in our raw ingestion source under the sibling
    # key `headnotestext` too (confirmed against a real pre-index record), copied
    # verbatim into our own ES index by transform_full.py's own `_UNWANTED`-but-indexed
    # list (line ~52) - already live, right now, just never queried by anything.
    # `headnotes_text.phrase_search` exists (mapped, snowball-analyzed); `headnotestext`
    # has neither a `.phrase_search` sub-field nor a non-default analyzer (confirmed via
    # `_mapping/field`) - adding a proper multi-field to it needs an ES mapping change +
    # reindex, an infra change out of scope for a query-code fix. This clause is the safe,
    # additive alternative: a plain `match` (no phrase, no analyzer override - works fine
    # against `headnotestext`'s own default mapping) at the same 65000 boost the primary
    # headnotes_text tier uses, so real prod's fuller content still contributes real score
    # even though the ported per-token branch machinery above never touches this field.
    if query.strip():
        should.append({"match": {"headnotestext": {"query": query, "boost": 65000}}})

    if not other_tokens and phrase_tokens:
        # Whole query is a double-quoted phrase (no unquoted tokens at all) - live-captured
        # against the real production endpoint (2026-09-04,
        # `"571/Ahd/2016 vide order dated 02-04-2026"`), the 5 phrase-field clauses sit in
        # their OWN nested `bool.must` entry, not flattened into the same top-level `should`
        # list as the static group-membership boosts - real capture shows
        # `bool.must: [{bool: {should: [5 phrase clauses]}}, {bool: {should:
        # [exclusion-passthrough, group-boost]}}]`. This is load-bearing, not cosmetic: ES
        # defaults a bare `bool.should` (no sibling `must`/`filter`) to
        # `minimum_should_match: 1`, so flattening the group-boost `term` clauses into the
        # same should-list as the phrase clauses (as the general branch below does) lets a
        # document match the query via a group-boost clause ALONE, with zero text relevance -
        # reproduced live: `"571/Ahd/2016 vide order dated 02-04-2026"` returned every GST Act
        # section (each a member of the boosted CGST Act 2017 subgroup, score ~67000 from the
        # boost alone, no phrase match at all) ranked above the one real matching case (score
        # ~6, genuinely phrase-matched). Putting the phrase should-list inside its own `must`
        # entry makes ES apply that same default-1 rule to it alone (a real text match becomes
        # mandatory), while the static group boosts move to a plain top-level `should` - which
        # ES makes score-only, no minimum required, once a sibling `must` exists. Still wrapped
        # in `function_score` below like every other shape (see this function's own docstring,
        # 2026-09-06 correction) - only the inner bool shape differs from the general branch.
        top_level_should = list(_static_group_should_clauses())
        # GroupFilterquery, GlobalSearchResearch.cs:751-754: whenever a non-"0" group_id was
        # resolved (this is repotaxmannapi's own global-search path, search.IsGlobal is always
        # true here), boost documents whose groups.group.id matches it.
        if group_id != "0":
            top_level_should.append({"match_phrase": {"groups.group.id": {"query": group_id, "boost": 1000}}})
        # Fallback act-url boost, GlobalSearchResearch.cs:755-758: when the tokenizer produced
        # no usable tokens at all (every per-token query was null - `nullcount ==
        # queries.Count`), boost groups.group.url == "act" instead of leaving the query
        # entirely boost-less.
        if not tokens:
            top_level_should.append({"match": {"groups.group.url": {"query": "act", "boost": 1000}}})
        bool_query = {
            "bool": {
                "must": [{"bool": {"should": should, "minimum_should_match": 1}}],
                "should": top_level_should,
                "filter": [_edition_exclusion_filter(), *_additional_exclusion_filters()],
            },
        }
    else:
        should.extend(_static_group_should_clauses())
        # GroupFilterquery, GlobalSearchResearch.cs:751-754: whenever a non-"0" group_id was
        # resolved (this is repotaxmannapi's own global-search path, search.IsGlobal is always
        # true here), boost documents whose groups.group.id matches it.
        if group_id != "0":
            should.append({"match_phrase": {"groups.group.id": {"query": group_id, "boost": 1000}}})
        # Fallback act-url boost, GlobalSearchResearch.cs:755-758: when the tokenizer produced
        # no usable tokens at all (every per-token query was null - `nullcount ==
        # queries.Count`), boost groups.group.url == "act" instead of leaving the query
        # entirely boost-less.
        if not tokens:
            should.append({"match": {"groups.group.url": {"query": "act", "boost": 1000}}})
        bool_query = {
            "bool": {
                "should": should, "minimum_should_match": 1,
                "filter": [_edition_exclusion_filter(), *_additional_exclusion_filters()],
            },
        }
    return {
        "function_score": {
            "query": bool_query,
            "functions": build_function_score_functions(
                group_id,
                latest_finance_act_year=_BOOST_CONFIG["latest_finance_act_year_for_multiply_formula"]["value"],
            ),
            "score_mode": "multiply",
            "boost_mode": "multiply",
        },
    }


def build_query_preview(query: str, boost: bool = True, boost_source: str = "repotaxmannapi") -> dict:
    """The exact shape/chunk/ES-query breakdown raw_search uses for this query, without
    executing a search - single source of truth shared with raw_search (below) so the two
    can never drift apart (this is also why `boost`/`boost_source` are parameters here rather
    than raw_search wrapping the query itself after calling this - a caller that omitted them
    here would silently show a preview that doesn't match the search actually run). Powers the
    `/v1/query-analysis` endpoint (retrieval_api/query_analysis.py) and the Instant mode trace
    panel's "Show ES query" block - our equivalent of centax-node's own
    `/research-premium/api/v1/getLowLevelQuery`, for comparing query breakdowns side by side.

    `chunks`/`shape`/`expanded_query` always reflect this repo's own chunk_query() breakdown,
    regardless of boost_source - repotaxmannapi's own tokenizer runs a structurally different
    per-token classification (RepotaxmannapiToken, not QueryChunk) that has no equivalent
    display shape here; only `es_query` (the field actually rendered as "Show ES query") changes
    with boost_source, since that's the one raw_search itself sends to ES.

    boost=True, boost_source="repotaxmannapi" (default, 2026-09-02 - explicit, deliberate
    user override of this repo's own earlier eval-driven recommendation; CLAUDE.md and
    SESSION_CONTEXT.md both reserved this exact "promote repotaxmannapi to the real
    default" decision for the user to make, and this is that decision made): `es_query` is
    the ported legacy .NET multiply-mode formula, byte-exact - see raw_search's docstring
    for the group_id resolution this mirrors. The user's own words: "keep everything same
    [as repotaxmannapi]... just copy paste boostings, logics to query everything" - no
    blended/tuned formula, the real production query construction as the one true default.
    boost=True, boost_source="sum": `es_query` is wrapped via _apply_boost() instead - this
    repo's own additive formula, still available, just no longer the default.
    boost=False: `es_query` is the unwrapped, plain BM25/phrase-boost query - still
    available for any caller that explicitly wants it (e.g. a future A/B comparison)."""
    shape = effective_label(query)
    expanded_query = expand_query_normalizations(expand_query_synonyms(query))
    chunks = chunk_query(query)
    if boost and boost_source == "repotaxmannapi":
        field_query = _build_repotaxmannapi_field_query(query)
    else:
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


def build_keyword_search_query_preview(
    query: str, doc_id_allowlist: list[str] | None = None, boost: bool = False,
) -> dict:
    """The exact query body keyword_mode_search sends to ES, without executing a search -
    same single-source-of-truth pattern as build_sparse_fallback_query_preview. Unlike that
    function, no `groups.group.name` filter: AI Mode's keyword-tagged path (see
    query_tokenizer.classify_intent_mode) is a precise anchor lookup (section/citation/
    court/Act name) that can legitimately live in any content group, not just the 5
    Milvus-sparse-gap collections sparse_fallback_search targets.

    boost_source pinned to "sum" explicitly (2026-09-02), same reason as
    build_sparse_fallback_query_preview - immune to build_query_preview's own default
    changing underneath it; AI Mode's keyword-anchor lookup must stay on this repo's
    additive formula regardless of what Instant mode's raw_search defaults to."""
    field_query = build_query_preview(query, boost=boost, boost_source="sum")["es_query"]
    must: list[dict] = []
    if doc_id_allowlist:
        must.append({"terms": {"id": doc_id_allowlist}})
    must.append(field_query)
    return {"bool": {"must": must}}


async def keyword_mode_search(
    client, query: str, doc_id_allowlist: list[str] | None = None, limit: int = 20,
    boost: bool = False,
) -> list[dict]:
    """ES-only retrieval for AI Mode's "keyword"-tagged queries (classify_intent_mode) -
    a precise anchor lookup skips Milvus dense/sparse + RRF entirely, same highlight-based
    text-fragment shape as sparse_fallback_search so the caller can feed rows straight into
    synthesize() without a chunk_id (no chunk-level candidate set to dedupe/rerank here)."""
    response = await client.search(
        index=client.index,
        query=build_keyword_search_query_preview(query, doc_id_allowlist, boost=boost),
        size=limit,
        _source=["id"],
        highlight={"fields": {"fullcontent": {
            "fragment_size": _ES_HIGHLIGHT_FRAGMENT_CHARS, "number_of_fragments": 1,
        }}, "pre_tags": [""], "post_tags": [""]},
    )
    results = []
    for hit in response["hits"]["hits"]:
        fragments = hit.get("highlight", {}).get("fullcontent")
        if not fragments:
            continue
        results.append({
            "doc_id": hit["_source"]["id"], "score": hit["_score"],
            "text": trim_to_token_budget(strip_tags_to_text(fragments[0])),
        })
    return results


async def raw_search(
    client, query: str, limit: int = 20, boost: bool = True, boost_source: str = "repotaxmannapi",
    page: int = 1, page_size: int | None = None,
) -> list[dict]:
    """boost_source selects which boost formula `boost=True` applies:
    - "repotaxmannapi" (default, 2026-09-02 - explicit user override, see
      build_query_preview's docstring for the exact instruction): repotaxmannapi's own
      multiply-mode function_score (build_function_score_functions), wrapping should-clauses
      built from the repotaxmannapi tokenizer/query-builder pair instead of this repo's own
      chunk_query/_build_field_query - the byte-exact ported .NET formula, not this repo's
      own tuned approximation. group_id is resolved from the tokenized query - the
      first token (in tokenize() order) whose group_id != "0", mirroring the real source's
      "first classification wins" behavior: TaxmannQueryAnalizer.cs's SetPrimaryTag
      (288-317) is gated by `if (iTagNo == "0")` (line 294) and is the only setter called
      per-token in the token-parsing loop (~1599-1909), so only the first token that
      classifies to a non-"0" tag/group ever sets it - every later call in the same query is
      a no-op. (ReSetPrimaryTag, unconditional, is only ever called once for the special
      "EXPERTSOPINION" case at line 1980 - not part of general per-token iteration.)
      GlobalSearchResearch.cs:595-596 (duplicated at 988-989, and in
      GlobalSearchResearchMobileApp.cs:64-65) then takes that single query-level iGroupID
      verbatim: `if (stext.iGroupID != "0") groupid = stext.iGroupID;`. "0" (no group
      signal) if no token has one.
    - "sum": this repo's own additive function_score (_apply_boost, via build_query_preview)
      - still available for any caller that explicitly passes it, just no longer the
      default. AI Mode's ES sparse-fallback (sparse_fallback_search/
      build_sparse_fallback_query_preview) pins this explicitly, unaffected by this default
      change - see its own comment.
    Neither has any effect when boost=False (there is nothing to select a formula for).

    page/page_size (added 2026-09-02, hidden-by-default pagination for the Instant-mode
    UI): page_size=None (every existing caller) reproduces prior behavior exactly -
    `size=limit`, no `from_` sent at all. Passing page_size switches to real ES paging -
    `from_=(page-1)*page_size`, `size=page_size` - and `limit` is ignored in that case."""
    # Real C# semantics (SetPrimaryTag's `iTagNo == "0"` gate) actually lock in the FIRST
    # *classified* token's group_id even if that token's own id were "0" - never falling
    # through to a later token's non-"0" id. build_query_preview's `next(...)` instead skips
    # "0" tokens looking for the first non-"0" one, which diverges from that edge case - but
    # it's unreachable with real data: verified zero entries in
    # repotaxmannapi_token_dictionary.json have tag_no != "0" (i.e. "classified") with
    # group_id == "0", so a classified token's group_id is never "0" in practice.
    field_query = build_query_preview(query, boost=boost, boost_source=boost_source)["es_query"]
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
    # _source filtering (2026-09-07): this loop below only ever reads id/heading/subheading
    # (plus, below, the highlight-derived text) off each hit, but without an explicit
    # `source` filter ES returns the FULL document - including `fullcontent` (a judgment's
    # entire text, can run tens of KB) - for every one of `limit`/`page_size` hits.
    # Root-caused a real, reproducible intermittent ConnectionTimeout on a live long-phrase
    # citation query: ES's own `took` was 16ms (the query itself is cheap), but the
    # unfiltered response body was large enough that transfer over the network to this
    # remote node blew the client's ~10s default timeout on 3 of 4 tries - a short/
    # section-shaped query usually matches shorter documents and stayed under it, making the
    # failure look query-specific rather than infra-wide. Adding this filter (already applied
    # to raw_search_grouped's top_hits sub-agg below, just never mirrored here) cut the same
    # query's response to ~5KB and 0.05-0.15s, consistently. Real source's own GetSearchResult
    # (GlobalSearchResearch.cs:658-678) always source-filters too - this was a straight gap in
    # the port, not a deliberate omission.
    #
    # Compatible with the highlight request below despite excluding fullcontent from
    # `_source` here - ES computes highlights from the document's actual stored data
    # server-side, independent of the `_source` response filter (that filter only controls
    # what's serialized back into the client-facing `_source` object). keyword_mode_search
    # above already relies on exactly this: `_source=["id"]` with `highlight` on fullcontent
    # in the same call, shipped and working.
    source_fields = ["id", "heading", "subheading"]
    # Same highlight shape keyword_mode_search/sparse_fallback_search already use (oversized
    # fragment_size, trim_to_token_budget does the real cutting) - added here so raw_search's
    # rows carry real match-context text too, not just doc_id/score/heading/subheading. Every
    # reranker call site in this repo should be able to use a candidate's own row["text"]
    # instead of a separate whole-document refetch (see instant/rerank.py); previously
    # raw_search was the one gap forcing that refetch.
    highlight = {"fields": {"fullcontent": {
        "fragment_size": _ES_HIGHLIGHT_FRAGMENT_CHARS, "number_of_fragments": 1,
    }}, "pre_tags": [""], "post_tags": [""]}
    if page_size is not None:
        response = await client.search(
            index=client.index, query=field_query, size=page_size, from_=(page - 1) * page_size,
            _source=source_fields, highlight=highlight,
        )
    else:
        response = await client.search(
            index=client.index, query=field_query, size=limit, _source=source_fields, highlight=highlight,
        )
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        fragments = hit.get("highlight", {}).get("fullcontent")
        results.append({
            "doc_id": source["id"],
            "score": hit["_score"],
            "heading": source.get("heading", ""),
            "subheading": source.get("subheading", ""),
            "text": trim_to_token_budget(strip_tags_to_text(fragments[0])) if fragments else "",
        })
    return results


# Live-verified via a direct terms aggregation against the real index (2026-09-02): the
# only groups.group.name values that exist are ACT/RULE/CASELAWS/COMMENTARY/"Experts
# Opinion"/Tariff - no "Comparative" or "Article" group is indexed in this repo at all,
# unlike the reference product's UI. Fixed priority order for raw_search_grouped's
# sections: statutory text first, then case law, then secondary sources. A group name not
# in this list (a future new content type) still appears, appended after these six rather
# than silently dropped - see raw_search_grouped.
_GROUPED_SECTION_PRIORITY = ["ACT", "RULE", "CASELAWS", "COMMENTARY", "Experts Opinion", "Tariff"]


async def raw_search_grouped(client, query: str, limit_per_group: int = 5) -> dict[str, list[dict]]:
    """Instant mode's repotaxmannapi-mode sectioned result view - buckets hits by
    groups.group.name.keyword (the real content-type field) via an ES terms+top_hits
    aggregation, one independent full-corpus scan per bucket - NOT a regroup of the flat
    top-N hits raw_search returns. That regroup-only approach was tried and reverted: for
    a bare "SECTION 52" query the flat top-20 window is entirely Act documents (score
    dominance under multiply-mode, verified byte-identical to a real captured production
    response), so regrouping only that window left every other section (Rules, Caselaws,
    Commentary, Articles) empty even though those content types have real matches deeper
    in the corpus - "only Income Tax Acts showing" was the visible symptom. This
    aggregation scans independently per content type instead, so a section's docs aren't
    bounded by whatever the flat list's top-20 window happened to contain.

    Always uses the repotaxmannapi (multiply-mode) query - grouping has no sum-mode
    equivalent, this is exclusively part of the repotaxmannapi replica path, unlike
    raw_search's boost_source switch.

    `size: 0` on the main query - no flat hits list is fetched here at all, only the
    aggregation's buckets, each independently scanning the full filtered match set via its
    own `top_hits` sub-aggregation.

    Returns an ordered dict (see _GROUPED_SECTION_PRIORITY) with only the groups that
    actually matched - a group with zero hits for this query is omitted entirely, not
    returned as an empty list."""
    field_query = _build_repotaxmannapi_field_query(query)
    response = await client.search(
        index=client.index,
        query=field_query,
        size=0,
        aggs={
            "by_group": {
                "terms": {"field": "groups.group.name.keyword", "size": 20},
                "aggs": {
                    "top": {
                        "top_hits": {
                            "size": limit_per_group,
                            "sort": [{"_score": {"order": "desc"}}],
                            "_source": ["id", "heading", "subheading"],
                        },
                    },
                },
            },
        },
    )
    buckets: dict[str, list[dict]] = {}
    for bucket in response["aggregations"]["by_group"]["buckets"]:
        buckets[bucket["key"]] = [
            {
                "doc_id": hit["_source"]["id"],
                "score": hit["_score"],
                "heading": hit["_source"].get("heading", ""),
                "subheading": hit["_source"].get("subheading", ""),
            }
            for hit in bucket["top"]["hits"]["hits"]
        ]
    ordered: dict[str, list[dict]] = {}
    for group_name in _GROUPED_SECTION_PRIORITY:
        if group_name in buckets:
            ordered[group_name] = buckets.pop(group_name)
    ordered.update(buckets)
    return ordered


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


async def fetch_highlighted_fullcontent(client, doc_id: str, query: str) -> str | None:
    """Document-reader parity with real prod's server-side highlighting
    (repotaxmannapi/TaxmannAPI/Elastic/FileContentElasticSearchResearch.cs:118 -
    `.Highlight(h => h.PreTags(...).PostTags(...).MaxAnalyzedOffset(1000000)
    .Fields(f => f.Field(fullcontent).Type(Plain).NumberOfFragments(1)
    .Fragmenter(Simple).FragmentSize(50000000)))`, then FileContent
    ElasticSearchResearch.cs:1627-1628 swaps the placeholder pre/post tags for
    `<span class="researchdochighlight">...</span>` before the response ever
    leaves the server) - the whole document gets highlighted once, server-side,
    using the real query's own analyzer/stemming, not the frontend re-guessing
    with a naive word-split regex against whatever text it already has.

    `<mark>`/`</mark>` used directly as the highlight tags rather than prod's
    placeholder-then-swap dance - `<mark>` is already well-formed XML on its
    own, no separate swap step needed before document_parser.parse_fullcontent
    re-parses this as XML (see _HIGHLIGHT_TAGS there). `number_of_fragments: 1`
    + a huge `fragment_size` mirrors prod's own "one fragment covering nearly
    the whole document" shape - this is the reader view, not a search-result
    snippet (contrast raw_search's own highlight call, a genuine short
    fragment for the reranker).

    The document is selected by `id` ALONE (guarantees exactly one hit
    regardless of whether `query` happens to match `fullcontent` at all - a
    card can rank on `heading`/`subheading` with zero fullcontent overlap);
    `highlight_query` computes highlighting independently of what selected the
    document.

    A PLAIN `match` on `fullcontent`, not build_query_preview's full ranking
    query - verified against the real source (FileContentElasticSearchResearch.
    cs:71): `fcontentSearch.Match(m => m.Field(f => f.fullcontent).Query
    (elasticFilterSearch.searchtext))` - no fuzziness, no should-clause fan-out
    across a dozen fields/boosts. An earlier version of this function reused
    build_query_preview(query, boost=False)'s "sum" ranking-query builder
    instead, live-tested via Chrome: for query "section 148" it highlighted
    "mention"/"mentioning" too - genuinely explainable (that builder's
    multi_match carries `fuzziness: AUTO`, and "mention" is exactly
    Levenshtein-distance 2 from "section", AUTO's allowed distance for a
    7-letter term), but exactly the kind of confusing, seemingly-unrelated
    highlight a reader has no way to make sense of. Prod's own highlight query
    was never that broad to begin with - matching it exactly fixes this too.

    Quoted query -> `match_phrase`, not `match` (2026-09-07 fix, live-verified
    against real prod, taxmann.com/research): a plain search ("section 148")
    highlights "section"/"148" as independent words everywhere in the
    document; a quoted exact-phrase search ('"section 148"') highlights ONLY
    the contiguous phrase - other bare "section"/"148" occurrences elsewhere
    are left alone. `match` (OR-of-words) already reproduces the plain-search
    case; a query wrapped in double quotes needs `match_phrase` instead,
    quotes stripped before querying, or an exact-phrase search here would
    still highlight individual words, same bug as the fuzzy-match one above."""
    stripped = query.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        field_query = {"match_phrase": {"fullcontent": {"query": stripped[1:-1]}}}
    else:
        field_query = {"match": {"fullcontent": {"query": query}}}
    response = await client.search(
        index=client.index, query={"term": {"id": doc_id}}, size=1,
        _source=["fullcontent"],
        highlight={
            "fields": {"fullcontent": {
                "type": "plain",
                "number_of_fragments": 1,
                "fragment_size": _ES_DOCUMENT_HIGHLIGHT_FRAGMENT_CHARS,
                "highlight_query": field_query,
            }},
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
    )
    hits = response["hits"]["hits"]
    if not hits:
        return None
    fragments = hits[0].get("highlight", {}).get("fullcontent")
    return fragments[0] if fragments else hits[0]["_source"]["fullcontent"]


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


_BARE_ACT_CATEGORY_URL = "bare-act"


def _format_repotaxmannapi_date(raw: str | None) -> str | None:
    """Port of `GlobalSearchIndexController.cs::GetDateString` - parses a `yyyyMMdd`
    string and reformats it `dd-MM-yyyy`; returns None (matching the real method's silent
    fallthrough on a failed `DateTime.TryParseExact`) for anything else, including None/
    empty input. Used only by the CirNot reshaping below - see that block's own comment for
    why `displaydocumentdatestring` (the field this reads) is fetched at all despite being
    confirmed dead corpus-wide."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[6:8]}-{raw[4:6]}-{raw[0:4]}"


async def fetch_doc_categories(client, doc_ids: list[str]) -> dict[str, dict]:
    """Instant mode's per-card metadata fetch: "category | group" badge, plus (this
    session, 2026-09-02) every other real, populated ES field confirmed usable on the
    result card - judge/party/citation/associates are CASELAWS-oriented and simply
    absent for other content types (same *ngIf-style fallback the reference product's
    own card uses); date/viewcount/boost-debug-numbers apply to every content type.
    Batched sibling of fetch_citations - one mget for every doc_id across ES, Milvus
    dense/sparse, and reranked cards, so metadata is consistent regardless of which
    engine surfaced a given doc (Milvus itself carries none of these fields).
    `categories` is a populated-on-every-doc but multi-valued list (verified live: a doc
    can carry 4+ subject-area tags at once) with no reliable primary flag -
    `isprimarycat` is only set on ~20% of the corpus (81k/410k docs). Picks the
    isprimarycat=1 entry when present, else the first entry.

    Bare-act override, ported from the real .NET source
    (repotaxmannapi/TaxmannAPI/Controllers/ResearchElastic/GlobalSearchIndexController.cs:
    309-334): if the picked entry's `url` is the generic "bare-act" bucket and the doc
    carries more than one category, the real system discards that pick and uses
    `categories[1]` (the next, more specific subject-area tag) instead - live-verified
    2026-09-02 this is a common pattern (196/200 sampled bare-act-tagged docs have
    isprimarycat unset and a more specific category, e.g. "Company Law"/"Account & Audit",
    sitting right after "Bare Act" in the array) that the old first-entry-only fallback
    was silently mislabeling as "Indian Acts & Rules" instead of the real, more specific
    subject. A doc whose ONLY category is Bare Act still correctly resolves to
    "Indian Acts & Rules" via CATEGORY_DISPLAY_LABELS - the override only fires when a
    better answer actually exists at index 1.

    Raw category/group values are run through CATEGORY_DISPLAY_LABELS/GROUP_DISPLAY_LABELS;
    a value with no entry there is passed through unchanged rather than guessed at.

    `act_name` (this session): the specific Act/Rule instrument a document belongs to (e.g.
    "Companies Act, 2013", "Central Goods and Services Tax Act, 2017") - distinct from the
    category/group badge and from `associates.act`/`referenced_act` (those are
    cross-references to OTHER acts this section relates to, never the doc's own act). The
    real .NET source's field for this (`masterinfo.info.act[].name`, `GlobalSearchIndex
    Controller.cs:247-251`) is confirmed dead (0% populated) on this index - a live audit
    2026-09-02 found `groups.group.subgroup.name` 100% populated (83,309/83,309) on every
    ACT-group document instead, and it already carries genuine per-instrument names (e.g.
    "Finance Acts, 2025", "Companies Act, 2013") - used here as the substitute source field,
    not what the real source itself binds to, but the only live-populated equivalent.
    ACT/RULE-only (`groups.group.subgroup.name` is this port's own established "which
    edition/instrument" field elsewhere, e.g. _EDITION_BOOSTS_BY_INSTRUMENT_KIND) - absent
    for CASELAWS/COMMENTARY/other groups, same absent-safe fallback as every other field
    here.

    `is_unreported` (2026-09-03): the real DTO's `isuro` boolean
    (`GlobalSearchIndexController.cs:362`, `c.isuro ?? false`) - flags a case law as an
    Unreported ruling, the field the real product's "Include Unreported Case Laws" toggle
    filters on (`CaselawsElasticSearchResearch.cs:58`, default-included). Confirmed 0%
    populated on this index today (live-checked 2026-09-03, same dead-field pattern as
    masterinfo.info.{court,bench,act,section}.name) - fetched and exposed anyway, per the
    same future-proofing call made for AAAModelReport/AccountStandard/OECD Model
    Commentaries above: if this field is ever populated, the card starts showing it with no
    code change needed. Only the boolean itself is ported here, not the include/exclude
    filter toggle - that's a UI-driven query parameter this repo's raw_search has no
    equivalent of, out of scope by the user's own instruction (query logic only, no new
    filters/UI).

    Confirmed dead fields (0% populated, live-audited 2026-09-02/03) are deliberately never
    fetched here: masterinfo.info.{court,bench,act,section}.name, masterinfo.citations.*,
    searchcitation/searchiltcitation formattedcitation, url,
    searchboosttext, boostpopularity, incometaxactinfo/companyactinfo/incometaxruleinfo.
    `displaydocumentdatestring` was also confirmed dead corpus-wide as of 2026-09-02, but
    IS fetched again below for the CirNot reshaping added 2026-09-03 - see that block's own
    comment for why (CirNot itself has 0 live docs, so the corpus-wide dead-field finding
    can't be independently re-checked against CirNot specifically). `InfavourOf`/`CourtName`/
    `AuthorName` (the real DTO's remaining 3 caselaw fields, `GlobalSearchIndexController.
    cs:359-361`) are NOT ported - live-checked 2026-09-03: `masterinfo`/`masterinfo.info` are
    completely empty objects on every sampled document (not just these 3 specific
    subfields - the whole structure), and unlike `act_name`/`referenced_act`/etc. above, no
    substitute field carrying equivalent data exists anywhere else in this index (checked
    `otherinfo.*` directly - only `judge`/`partyname`/`fullcitation`/`counselname`/
    `appealno`/`asstyr` exist there, no court/author/in-favour-of equivalent). Genuinely
    dead with no workaround, not a future-proofing candidate the way is_unreported etc. are.
    (Tariff-group cards no longer lack type-specific fields - see `tariff_name` below,
    2026-09-03.)

    `tariff_name`/`commentary_topic` (2026-09-03): per-content-type "additionHeading"
    reshaping ported from GlobalSearchIndexController.cs's `switch (c.groups.group.url)` -
    see the inline comment at their call site for exactly which of that switch's 9+ cases
    have live data in this repo's index at all (only ACT/RULE/Commentary/Tariff) and which
    field each one actually reads (the real source's own field is dead for ACT/RULE,
    already covered by `act_name` instead).

    `form_name`/`dta_name`/`heading_override`/`subheading_override`/`shortcontent_override`
    (2026-09-03): the remaining 6 branches of that same switch (Form, DTA, CBDT, News,
    Bill/Ordinances/Report/ListingInformalReport/PracticeProcedure, CirNot,
    StandardGuidanceNotes/FinancialsAndDisclosures - 7 group urls sharing 6 branches, the
    last 2 share one) - see the inline comment at their call site. ALL 7 have 0 live
    documents in this index today (live-checked 2026-09-03) - implemented anyway per this
    session's own future-proofing precedent (AAAModelReport/AccountStandard/OECD Model
    Commentaries/GST Tariff/Forms exclusions), but with a caveat those don't share: the
    *field paths* themselves (`masterinfo.iltinfoes.*`, `parentheadings.name`/`.pname`,
    `masterinfo.info.company.name`, `masterinfo.info.form`) have never been checked against
    a single real document of these types, unlike every other field in this file (checked
    against real, if sparse, live data even when the VALUE distribution later turned out
    zero/dead). Treat these 6 branches as unverified against real data entirely, not just
    "implemented ahead of volume" - re-read `GlobalSearchIndexController.cs:244-297`
    directly and cross-check every field path against a real document once any of these 7
    group urls get indexed, don't assume this port is already correct. See
    docs/pending-data-followups.md."""
    if not doc_ids:
        return {}
    response = await client.mget(
        index=client.index, ids=doc_ids,
        _source=[
            "categories.name", "categories.isprimarycat", "categories.url",
            "groups.group.name", "groups.group.url", "groups.group.subgroup.name",
            "groups.group.subgroup.subsubgroup.name",
            "groups.group.subgroup.subsubgroup.subsubsubgroup.name",
            "otherinfo.judge.name", "otherinfo.partyname.name", "otherinfo.fullcitation.name",
            "formatteddocumentdate", "viewcount", "documenttypeboost", "court_boost", "isuro",
            "associates.act.name", "associates.section.name", "associates.casereferred.name",
            # Per-content-type "additionHeading"/heading-reshape fields for the 7 zero-doc
            # group urls (Form/DTA/CBDT/News/Bill-family/CirNot/StandardGuidanceNotes-
            # FinancialsAndDisclosures) - see this function's own docstring for the
            # "unverified against real data" caveat on all of these.
            "heading", "subheading", "shortcontent",
            "masterinfo.info.form.name",
            "masterinfo.iltinfoes.country1.name", "masterinfo.iltinfoes.country2.name",
            "parentheadings.name", "parentheadings.pname",
            "displaydocumentdatestring",
            "masterinfo.info.company.name", "year.name",
        ],
    )
    results: dict[str, dict] = {}
    for doc in response["docs"]:
        if not doc.get("found"):
            continue
        source = doc["_source"]
        categories = source.get("categories") or []
        primary = next((c for c in categories if c.get("isprimarycat") == 1), None)
        picked = primary or (categories[0] if categories else None)
        if picked and picked.get("url") == _BARE_ACT_CATEGORY_URL and len(categories) > 1:
            picked = categories[1]
        category_name = picked["name"] if picked else None
        category = CATEGORY_DISPLAY_LABELS.get(category_name, category_name)
        group_node = source.get("groups", {}).get("group", {})
        group_name = group_node.get("name")
        group = GROUP_DISPLAY_LABELS.get(group_name, group_name)
        entry: dict = {"category": category, "group": group}

        # CategoryUrl/GroupUrl (GlobalSearchIndexController.cs:348-364's DTO) - live-checked
        # 2026-09-03: both 100% populated (410,427/410,427), unlike most other raw url/id
        # fields in this file that turned out dead - the real slugs the reference product's
        # own category/group navigation uses (e.g. "direct-tax-laws", "goods-services-tax",
        # "bare-act"; "caselaws", "act", "rule", "commentary", "experts-opinion", "tariff").
        # `category_url` reflects the SAME picked entry as `category` above (post bare-act
        # override), not raw categories[0] - the two must always name the same category.
        subgroup_node = group_node.get("subgroup", {})
        subsubgroup_node = subgroup_node.get("subsubgroup", {})
        group_url = group_node.get("url")
        if picked and picked.get("url"):
            entry["category_url"] = picked["url"]
        if group_url:
            entry["group_url"] = group_url

        act_name = subgroup_node.get("name")
        if act_name:
            entry["act_name"] = act_name

        # Per-content-type "additionHeading" reshaping, ported from the real .NET source
        # (GlobalSearchIndexController.cs:244-297's `switch (c.groups.group.url)`) - live
        # data-checked per group (2026-09-03) before porting, unlike a blind port of the
        # whole switch: only 4 of its 9+ cases have ANY documents in this repo's index at
        # all (ACT/RULE/Commentary/Tariff - confirmed via a live groups.group.url terms
        # aggregation; DTA/CBDT/Form/News/Bill-Ordinances-Report/CirNot/StandardGuidance
        # Notes-FinancialsAndDisclosures all return 0 docs), and even within those 4 the
        # real source's own field isn't always the one actually populated here:
        #   - ACT/RULE: real additionHeading source is `masterinfo.info.act[0].name` /
        #     `masterinfo.info.rule[0].name` - both confirmed 0% populated (same dead-field
        #     pattern as masterinfo.info.{court,bench,act,section}.name elsewhere in this
        #     file). No separate additionHeading is added for these two groups - `act_name`
        #     above (groups.group.subgroup.name, already live-verified 100% populated on
        #     every ACT-group doc) already serves the identical purpose for both, so this
        #     isn't a gap to fill, just a different-but-equivalent field the doc already
        #     exposes. The real source's Finance-Act year-suffix special case is skipped
        #     for the same reason - the substitute subgroup.name values already observed
        #     live (e.g. "Finance Acts, 2025") already bake the edition year in, so
        #     appending it again would just duplicate it, not add real information.
        #   - Tariff: `groups.group.subgroup.name` + `.subsubgroup.name`, both live-verified
        #     100% populated (4144/4144) - fully portable as `tariff_name`.
        #   - Commentary: `groups.group.subgroup.subsubgroup.name` is only 76% populated
        #     (20,768/27,291 live-verified) - when present, used as `commentary_topic`
        #     (+ subsubsubgroup.name suffix, itself only 23% populated, appended only when
        #     present); when absent, falls back to `groups.group.subgroup.name` (100%
        #     populated for every Commentary doc, e.g. "Commentaries") rather than omitting
        #     the field outright - coarser than the real subsubgroup-level topic, but still
        #     real, populated data instead of nothing.
        if group_url == "tariff":
            tariff_name = subgroup_node.get("name") or ""
            subsubgroup_name = subsubgroup_node.get("name")
            if subsubgroup_name:
                tariff_name = f"{tariff_name} - {subsubgroup_name}" if tariff_name else subsubgroup_name
            if tariff_name:
                entry["tariff_name"] = tariff_name
        elif group_url == "commentary":
            subsubgroup_name = subsubgroup_node.get("name")
            commentary_topic = subsubgroup_name or subgroup_node.get("name")
            if subsubgroup_name:
                subsubsubgroup_name = subsubgroup_node.get("subsubsubgroup", {}).get("name")
                if subsubsubgroup_name:
                    commentary_topic = f"{commentary_topic} - {subsubsubgroup_name}"
            if commentary_topic:
                entry["commentary_topic"] = commentary_topic
        elif group_url == "form":
            # `masterinfo.info.form[0].name` - unverified, see this function's docstring.
            form_list = (source.get("masterinfo") or {}).get("info", {}).get("form") or []
            form_name = form_list[0].get("name") if form_list else None
            if form_name:
                entry["form_name"] = form_name
        elif group_url == "dta":
            # DTA (tax treaty): additionHeading = country1 (+ " - " + country2); heading
            # becomes "<old subheading> : <old heading>" (only when subheading is
            # non-empty); subheading becomes the subsubsubgroup name. All unverified.
            ilt_list = (source.get("masterinfo") or {}).get("iltinfoes") or []
            first_ilt = ilt_list[0] if ilt_list else {}
            country1 = (first_ilt.get("country1") or {}).get("name") or ""
            country2 = (first_ilt.get("country2") or {}).get("name")
            dta_name = f"{country1} - {country2}" if country2 else country1
            if dta_name:
                entry["dta_name"] = dta_name
            raw_heading = source.get("heading") or ""
            raw_subheading = source.get("subheading") or ""
            entry["heading_override"] = f"{raw_subheading} : {raw_heading}" if raw_subheading else raw_heading
            subsubsubgroup_name = subsubgroup_node.get("subsubsubgroup", {}).get("name")
            entry["subheading_override"] = subsubsubgroup_name or ""
        elif group_url == "cbdt":
            # additionHeading = parentheading name (+ " of " + its pname); heading falls
            # back to subheading if blank; subheading becomes the old shortcontent;
            # shortcontent is cleared. All unverified.
            parentheadings = source.get("parentheadings") or []
            first_parent = parentheadings[0] if parentheadings else {}
            parent_name = first_parent.get("name") or ""
            parent_pname = first_parent.get("pname")
            cbdt_name = f"{parent_name} of {parent_pname}" if parent_pname else parent_name
            if cbdt_name:
                entry["cbdt_name"] = cbdt_name
            raw_heading = source.get("heading") or ""
            raw_subheading = source.get("subheading") or ""
            entry["heading_override"] = raw_heading or raw_subheading
            entry["subheading_override"] = source.get("shortcontent") or ""
            entry["shortcontent_override"] = ""
        elif group_url == "news":
            entry["subheading_override"] = source.get("shortcontent") or ""
        elif group_url in ("bill", "ordinances", "report", "listinginformal-report", "practice-procedure"):
            # heading gets a " - <parent heading name>" suffix; subheading becomes
            # shortcontent. All unverified.
            parentheadings = source.get("parentheadings") or []
            parent_name = parentheadings[0].get("name") if parentheadings else None
            raw_heading = source.get("heading") or ""
            entry["heading_override"] = f"{raw_heading} - {parent_name}" if parent_name else raw_heading
            entry["subheading_override"] = source.get("shortcontent") or ""
        elif group_url == "cirnot":
            # heading gets a " - Dated <dd-MM-yyyy>" suffix, from displaydocumentdatestring
            # (confirmed dead corpus-wide as of 2026-09-02 - see this function's own
            # docstring for why it's fetched here anyway). All unverified.
            date_str = _format_repotaxmannapi_date(source.get("displaydocumentdatestring"))
            raw_heading = source.get("heading") or ""
            entry["heading_override"] = f"{raw_heading} - Dated {date_str}" if date_str else raw_heading
        elif group_url in ("standard-guidance-notes", "financials-and-disclosures"):
            # heading is fully rebuilt: "<company name> <subheading> : <year>". All
            # unverified.
            company_list = (source.get("masterinfo") or {}).get("info", {}).get("company") or []
            company_name = company_list[0].get("name") if company_list else None
            raw_subheading = source.get("subheading") or ""
            year_name = source.get("year", {}).get("name")
            year_suffix = f" : {year_name}" if year_name else ""
            entry["heading_override"] = f"{company_name or ''} {raw_subheading}{year_suffix}".strip()

        otherinfo = source.get("otherinfo") or {}
        judges = [j["name"] for j in otherinfo.get("judge") or [] if j.get("name")]
        if judges:
            entry["judge"] = judges
        parties = [p["name"] for p in otherinfo.get("partyname") or [] if p.get("name")]
        if parties:
            entry["party"] = parties
        citations = [c["name"] for c in otherinfo.get("fullcitation") or [] if c.get("name")]
        if citations:
            entry["fullcitation"] = citations[0]

        if source.get("formatteddocumentdate") is not None:
            entry["date"] = source["formatteddocumentdate"]
        if source.get("viewcount") is not None:
            entry["viewcount"] = source["viewcount"]
        if source.get("documenttypeboost") is not None:
            entry["documenttypeboost"] = source["documenttypeboost"]
        if source.get("court_boost") is not None:
            entry["court_boost"] = source["court_boost"]
        if source.get("isuro") is not None:
            entry["is_unreported"] = source["isuro"]

        associates = source.get("associates") or {}
        acts = [a["name"] for a in associates.get("act") or [] if a.get("name")]
        if acts:
            entry["referenced_act"] = acts
        sections = [s["name"] for s in associates.get("section") or [] if s.get("name")]
        if sections:
            entry["referenced_section"] = sections
        cases = [c["name"] for c in associates.get("casereferred") or [] if c.get("name")]
        if cases:
            entry["cases_referred"] = cases

        results[doc["_id"]] = entry
    return results
