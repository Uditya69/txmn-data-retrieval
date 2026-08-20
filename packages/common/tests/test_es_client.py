import pytest
from common.config import Settings
from common.es_client import (
    get_es_client,
    raw_search,
    resolve_doc_id_allowlist,
    fetch_citations,
    fetch_fullcontent,
    fetch_document_metadata,
    build_query_preview,
)
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
        self.highlight_calls = []
        self.source_calls = []
        self.mget_calls = []
        self.index = index

    async def search(self, index, query, size, highlight=None, _source=None):
        self.search_calls.append(query)
        self.highlight_calls.append(highlight)
        self.source_calls.append(_source)
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


@pytest.mark.asyncio
async def test_raw_search_expands_known_abbreviation_into_multi_match_query_text():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "ACIT order on depreciation", limit=20)

    sent_query = client.search_calls[0]
    multi_match_query = sent_query["bool"]["should"][0]["multi_match"]["query"]
    assert "ACIT order on depreciation" in multi_match_query
    assert "ASSISTANT COMMISSIONER INCOME TAX" in multi_match_query


def _heading_phrase_clauses(should: list[dict]) -> list[dict]:
    return [c for c in should if "match_phrase" in c and "heading" in c["match_phrase"]]


@pytest.mark.asyncio
async def test_raw_search_adds_exact_match_phrase_clause_for_merged_keyword_number():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "Section 6 of Income Tax Act", limit=20)

    sent_query = client.search_calls[0]
    phrase_clauses = _heading_phrase_clauses(sent_query["bool"]["should"])
    section_clauses = [c for c in phrase_clauses if c["match_phrase"]["heading"]["query"] == "Section 6"]
    assert section_clauses
    assert section_clauses[0]["match_phrase"]["heading"]["slop"] == 0


@pytest.mark.asyncio
async def test_raw_search_chunks_unrecognized_word_run_into_a_text_phrase_clause():
    """The gap chunk_query closes: words with no Section/Court/citation keyword
    anchor still get grouped into one match_phrase (slop=5), not left as loose
    independent OR terms - see chunk_query's own docstring for why."""
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "can a company claim depreciation on goodwill", limit=20)

    sent_query = client.search_calls[0]
    phrase_clauses = _heading_phrase_clauses(sent_query["bool"]["should"])
    text_clauses = [c for c in phrase_clauses if c["match_phrase"]["heading"]["slop"] == 5]
    assert text_clauses
    # stopword-strip removes "on" before chunking, same as extract_boost_phrases did
    assert "on" not in text_clauses[0]["match_phrase"]["heading"]["query"].split()


@pytest.mark.asyncio
async def test_raw_search_adds_zero_padded_alternative_clause_for_short_section_numbers():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "section 92C ITES comparables", limit=20)

    sent_query = client.search_calls[0]
    phrase_clauses = _heading_phrase_clauses(sent_query["bool"]["should"])
    queries = {c["match_phrase"]["heading"]["query"] for c in phrase_clauses}
    assert "section 92C" in queries
    assert "section 092C" in queries


@pytest.mark.asyncio
async def test_raw_search_queries_heading_subheading_fullcontent_not_just_sparse_fields():
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20)

    query = client.search_calls[0]
    should_fields = {
        clause["multi_match"]["fields"][0] if "multi_match" in clause else None
        for clause in query["bool"]["should"]
    }
    for field in ("heading", "subheading", "fullcontent"):
        assert field in should_fields, f"{field} missing from should clauses: {should_fields}"


def test_build_query_preview_matches_what_raw_search_actually_sends():
    """Single source of truth: raw_search calls build_query_preview() internally (see its
    own source) rather than recomputing shape/chunks/query separately, specifically so the
    /v1/query-analysis endpoint (backed by this function) can never drift from what a real
    search actually does."""
    preview = build_query_preview("Section 6 of Income Tax Act")

    assert preview["query"] == "Section 6 of Income Tax Act"
    assert preview["shape"] == "provision"
    assert any(c["type"] == "section" and c["text"] == "Section 6" for c in preview["chunks"])
    assert "bool" in preview["es_query"]


def test_build_query_preview_omits_expanded_query_when_unchanged():
    preview = build_query_preview("can a company claim depreciation")
    assert preview["expanded_query"] is None


def test_build_query_preview_includes_expanded_query_when_synonym_applies():
    preview = build_query_preview("ACIT order on depreciation")
    assert preview["expanded_query"] is not None
    assert "ASSISTANT COMMISSIONER INCOME TAX" in preview["expanded_query"]


@pytest.mark.asyncio
async def test_raw_search_does_not_wrap_query_in_function_score():
    """raw_search deliberately skips _wrap_function_score - see its docstring and the
    comment in raw_search: a 53-query eval run showed the patched boost formula still
    nearly halves pass rate versus plain BM25 text relevance (21/53 boosted vs 42/53
    unboosted), an architectural (boost_mode: multiply) issue, not a missing-data one."""
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20)

    query = client.search_calls[0]
    assert "function_score" not in query
    assert "bool" in query


@pytest.mark.asyncio
async def test_raw_search_does_not_exclude_landmarkruling_blacklisted_docs():
    """A prior version of raw_search hoisted centax-node's landmarkruling:-10 handling into
    a top-level query must_not, believing it preserved a content-exclusion filter. That was a
    misreading of the source: in centax-node's actual query (services/caselaws.js), that
    must_not is scoped as a per-function `filter` *inside* one function_score function entry -
    it only controls whether that one function's boost applies, matching its own comment
    ("Don't add Function Score for blacklisted"). It never excluded the doc from results at
    all. Our prior top-level version was a real regression (hid ~173 legitimate docs from
    every search) with no source-of-truth backing it, and is now removed - raw_search must
    send the field_query as-is, no must_not anywhere, no landmarkruling clause at all."""
    client = FakeAsyncES(search_hits=[])

    await raw_search(client, "exemption claim", limit=20)

    query = client.search_calls[0]
    assert query == {"bool": {"should": query["bool"]["should"], "minimum_should_match": 1}}
    assert "must_not" not in query["bool"]
    assert "landmarkruling" not in str(query)


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
async def test_resolve_doc_id_allowlist_queries_heading_for_court_and_returns_doc_ids():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}, {"_source": {"id": "d2"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Supreme Court"})

    assert result == ["d1", "d2"]
    # "Supreme Court" resolves to "SC" - the abbreviation that actually appears in
    # heading (masterinfo.info.court.name is confirmed 0% populated on the live index).
    assert client.search_calls[0] == {
        "bool": {"must": [{"match_phrase": {"heading": "SC"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_passes_through_unmapped_court_names():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Patna High Court"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"match_phrase": {"heading": "Patna High Court"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_raises_on_unrecognized_filter_keys():
    client = FakeAsyncES()

    with pytest.raises(ValueError):
        await resolve_doc_id_allowlist(client, {"unknown_key": "whatever"})


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_heading_for_section_or_rule():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"section": "80C"})

    assert result == ["d1"]
    # ACT/RULE-group docs' heading is the section/rule identifier itself, verbatim
    # (masterinfo.info.section.name is confirmed 0% populated on the live index).
    assert client.search_calls[0] == {
        "bool": {"must": [{"bool": {"should": [
            {"match_phrase": {"heading": "Section - 80C"}},
            {"match_phrase": {"heading": "Rule - 80C"}},
        ]}}]}
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
async def test_resolve_doc_id_allowlist_falls_back_to_fuzzy_match_when_phrase_misses():
    class PhraseMissFuzzyHitES(FakeAsyncES):
        async def search(self, index, query, size):
            self.search_calls.append(query)
            if {"match_phrase": {"heading": "Bombay"}} in query["bool"]["must"]:
                return {"hits": {"hits": []}}
            return {"hits": {"hits": [{"_source": {"id": "d1"}}]}}

    client = PhraseMissFuzzyHitES()

    result = await resolve_doc_id_allowlist(client, {"court": "Bombay High Court"})

    assert result == ["d1"]
    assert len(client.search_calls) == 2
    assert client.search_calls[1] == {
        "bool": {"must": [{"match": {"heading": "Bombay"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_drops_malformed_date_range_instead_of_querying_es():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Supreme Court", "date_range": "2020"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"match_phrase": {"heading": "SC"}}]}
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
async def test_fetch_document_metadata_returns_heading_subheading_and_year_name():
    client = FakeAsyncES(search_hits=[{"_source": {
        "heading": "[2021] 130 taxmann.com 199 (AAR - KARNATAKA)[09-07-2021]",
        "subheading": "B.G. Shirke Constructions Technology (P.) Ltd., In re vs.",
        "year": {"name": "2021", "id": "2021"},
    }}])

    result = await fetch_document_metadata(client, "101010000000316021")

    assert result == {
        "heading": "[2021] 130 taxmann.com 199 (AAR - KARNATAKA)[09-07-2021]",
        "subheading": "B.G. Shirke Constructions Technology (P.) Ltd., In re vs.",
        "year": "2021",
    }


@pytest.mark.asyncio
async def test_fetch_document_metadata_returns_none_when_doc_not_found():
    client = FakeAsyncES(search_hits=[])

    result = await fetch_document_metadata(client, "missing")

    assert result is None


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_heading_for_bench():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"bench": "Principal Bench"})

    assert result == ["d1"]
    # "Principal Bench" has no alias entry, passed through unchanged
    # (masterinfo.info.bench.name is confirmed 0% populated on the live index).
    assert client.search_calls[0] == {
        "bool": {"must": [{"match_phrase": {"heading": "Principal Bench"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_otherinfo_judge_term():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"judge": "D.Y. Chandrachud"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"otherinfo.judge.name.keyword": "D.Y. Chandrachud"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_fullcontent_for_act_as_best_effort():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"act": "CGST Act"})

    assert result == ["d1"]
    # No field in the index reliably links a doc to its specific parent Act (confirmed
    # 0% populated: masterinfo.info.act.name, incometaxactinfo, companyactinfo) - this is
    # a best-effort full-text match, not an exact filter, until that data exists.
    assert client.search_calls[0] == {
        "bool": {"must": [{"match": {"fullcontent": "CGST Act"}}]}
    }


@pytest.mark.asyncio
async def test_fetch_citations_returns_doc_id_keyed_masterinfo_fields():
    client = FakeAsyncES(mget_docs={
        "d1": {
            "heading": "[2020] 1 SCC 1 (SC)",
            "subheading": "State v. Doe",
            "masterinfo": {"info": {"court": "Supreme Court"}, "citations": ["2020 SCC 1"]},
            "otherinfo": {"judge": "J. Smith", "partyname": "State v. Doe"},
            "judgment_text": "irrelevant text that should not be returned",
        },
    })

    result = await fetch_citations(client, ["d1"])

    assert result == {
        "d1": {
            "heading": "[2020] 1 SCC 1 (SC)",
            "subheading": "State v. Doe",
            "masterinfo": {"info": {"court": "Supreme Court"}, "citations": ["2020 SCC 1"]},
            "otherinfo": {"judge": "J. Smith", "partyname": "State v. Doe"},
        }
    }
    assert "judgment_text" not in result["d1"]
    # confirm the field restriction was actually passed through to ES (mget _source param)
    assert client.mget_calls[0]["_source"] == MASTERINFO_CITATION_FIELDS


def test_trim_to_token_budget_returns_short_text_unchanged():
    from common.es_client import trim_to_token_budget

    text = "short text well under budget"
    assert trim_to_token_budget(text, target_tokens=1024) == text


def test_trim_to_token_budget_centers_and_trims_oversized_text():
    from common.es_client import trim_to_token_budget
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    # "filler" repeated gives before/after segments that are token-exact symmetric by
    # construction (unlike numbered "wordN" placeholders, whose token length varies with
    # N's digit count and drifts MARKER's true offset away from the token-count midpoint).
    before = " ".join(["filler"] * 2000)
    after = " ".join(["filler"] * 2000)
    text = f"{before} MARKER {after}"

    trimmed = trim_to_token_budget(text, target_tokens=100)

    trimmed_tokens = tokenizer.encode(trimmed)
    assert len(trimmed_tokens) == 100
    assert "MARKER" in trimmed


def test_trim_to_token_budget_keeps_head_when_center_is_false():
    from common.es_client import trim_to_token_budget
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = "HEAD " + " ".join(["filler"] * 2000) + " TAIL"

    trimmed = trim_to_token_budget(text, target_tokens=100, center=False)

    assert len(tokenizer.encode(trimmed)) == 100
    assert "HEAD" in trimmed
    assert "TAIL" not in trimmed


def test_cap_group_shares_single_group_is_unaffected():
    from common.es_client import _cap_group_shares

    hits = [{"_group": "CASELAWS", "id": i} for i in range(20)]
    result = _cap_group_shares(hits, limit=20, group_cap=15)

    assert result == hits


def test_cap_group_shares_trims_dominant_group_and_backfills_from_minority():
    from common.es_client import _cap_group_shares

    # 18 CASELAWS hits (relevance rank 1-18) + 2 Experts Opinion hits (rank 19-20) -
    # naive top-20 would return 18 CASELAWS + 2 Experts Opinion. The cap should trim
    # CASELAWS to 15 and backfill the 3 freed slots from Experts Opinion's next-best
    # hits (which don't exist here beyond the 2 already present, so this asserts what
    # DOES exist survives and CASELAWS is capped, not that phantom hits appear).
    caselaws_hits = [{"_group": "CASELAWS", "id": f"cl{i}"} for i in range(18)]
    eo_hits = [{"_group": "Experts Opinion", "id": f"eo{i}"} for i in range(2)]
    hits = caselaws_hits + eo_hits  # already in relevance order

    result = _cap_group_shares(hits, limit=20, group_cap=15)

    result_caselaws = [h for h in result if h["_group"] == "CASELAWS"]
    result_eo = [h for h in result if h["_group"] == "Experts Opinion"]
    assert len(result_caselaws) == 15
    assert result_caselaws == caselaws_hits[:15]
    assert result_eo == eo_hits


@pytest.mark.asyncio
async def test_sparse_fallback_search_filters_by_group_and_partitions_by_collection():
    client = FakeAsyncES(search_hits=[
        {
            "_source": {"id": "d1", "groups": {"group": {"name": "CASELAWS"}}},
            "_score": 9.0,
            "highlight": {"fullcontent": ["snippet about the ruling"]},
        },
        {
            "_source": {"id": "d2", "groups": {"group": {"name": "Experts Opinion"}}},
            "_score": 7.0,
            "highlight": {"fullcontent": ["snippet about the article"]},
        },
    ], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    result = await sparse_fallback_search(client, "query text", groups=["CASELAWS", "Experts Opinion"])

    assert result == {
        "ruling": [{
            "chunk_id": "es:d1:0", "doc_id": "d1", "text": "snippet about the ruling",
            "score": 9.0, "source": "es_fallback",
        }],
        "article_section": [{
            "chunk_id": "es:d2:0", "doc_id": "d2", "text": "snippet about the article",
            "score": 7.0, "source": "es_fallback",
        }],
    }


@pytest.mark.asyncio
async def test_sparse_fallback_search_applies_doc_id_allowlist_and_highlight_config():
    client = FakeAsyncES(search_hits=[], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    await sparse_fallback_search(client, "query text", groups=["ACT"], doc_id_allowlist=["d1", "d2"])

    query = client.search_calls[0]
    must_clauses = query["bool"]["must"]
    assert {"terms": {"groups.group.name.keyword": ["ACT"]}} in must_clauses
    assert {"terms": {"id": ["d1", "d2"]}} in must_clauses


@pytest.mark.asyncio
async def test_sparse_fallback_search_strips_highlight_markup_tags():
    """Default ES highlighting wraps matches in <em>...</em> - that raw markup must never
    reach the reranker/LLM as if it were clean document text, and unclosed/split tags could
    eat into trim_to_token_budget's centered cut. pre_tags/post_tags=[""] disables the
    wrapping while keeping fragment selection/centering."""
    client = FakeAsyncES(search_hits=[], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    await sparse_fallback_search(client, "query text", groups=["CASELAWS"])

    highlight = client.highlight_calls[0]
    assert highlight["pre_tags"] == [""]
    assert highlight["post_tags"] == [""]


@pytest.mark.asyncio
async def test_sparse_fallback_search_restricts_source_to_id_and_group():
    """sparse_fallback_search only ever reads source['id'] and
    source['groups']['group']['name'] - fullcontent (the full legal document text, 100%
    populated) must not be fetched needlessly for every one of up to 100 hits per query."""
    client = FakeAsyncES(search_hits=[], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    await sparse_fallback_search(client, "query text", groups=["CASELAWS"])

    assert client.source_calls[0] == ["id", "groups.group.name"]


@pytest.mark.asyncio
async def test_sparse_fallback_search_skips_hits_missing_highlight_or_unknown_group():
    client = FakeAsyncES(search_hits=[
        {"_source": {"id": "d1", "groups": {"group": {"name": "CASELAWS"}}}, "_score": 9.0, "highlight": {}},
        {"_source": {"id": "d2", "groups": {"group": {"name": "Nonsense Group"}}}, "_score": 8.0,
         "highlight": {"fullcontent": ["x"]}},
    ], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    result = await sparse_fallback_search(client, "query text", groups=["CASELAWS"])

    assert result == {}


def test_cap_group_shares_backfills_to_full_limit_when_minority_group_has_more():
    from common.es_client import _cap_group_shares

    # 18 CASELAWS + 10 Experts Opinion, all in relevance order (interleaved isn't
    # required - the function must not assume a particular pre-existing interleave).
    caselaws_hits = [{"_group": "CASELAWS", "id": f"cl{i}"} for i in range(18)]
    eo_hits = [{"_group": "Experts Opinion", "id": f"eo{i}"} for i in range(10)]
    hits = caselaws_hits + eo_hits

    result = _cap_group_shares(hits, limit=20, group_cap=15)

    assert len(result) == 20
    result_caselaws = [h for h in result if h["_group"] == "CASELAWS"]
    result_eo = [h for h in result if h["_group"] == "Experts Opinion"]
    assert len(result_caselaws) == 15
    assert len(result_eo) == 5
    assert result_caselaws == caselaws_hits[:15]
    assert result_eo == eo_hits[:5]
