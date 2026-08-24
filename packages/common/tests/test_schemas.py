from common.schemas import (
    MILVUS_COLLECTIONS,
    CHUNKED_COLLECTIONS,
    BM25_SOURCE_FIELD,
    MASTERINFO_CITATION_FIELDS,
    collections_for_intent,
)


def test_eleven_collections():
    assert set(MILVUS_COLLECTIONS) == {
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
        "act_section", "rule_section", "article_section", "commentary_section",
    }


def test_chunked_collections_match_verified_code_behavior():
    assert CHUNKED_COLLECTIONS == {
        "digest", "facts", "held", "ruling",
        "act_section", "rule_section", "article_section", "commentary_section",
    }
    assert "case_summary" not in CHUNKED_COLLECTIONS
    assert "headnotes" not in CHUNKED_COLLECTIONS
    assert "metadata" not in CHUNKED_COLLECTIONS


def test_bm25_source_field_metadata_uses_heading_subheading():
    assert BM25_SOURCE_FIELD["metadata"] == "heading_subheading_text"
    assert BM25_SOURCE_FIELD["ruling"] == "text"


def test_masterinfo_citation_fields():
    # Paths verified against the real production ES index mapping
    # (researchindex_aic_test) - court/bench live under masterinfo.info,
    # judge/partyname live under otherinfo, not masterinfo. heading/subheading
    # are top-level - added so the AI Mode UI can show a case title on cited-doc
    # cards without a second ES round trip.
    assert MASTERINFO_CITATION_FIELDS == [
        "heading", "subheading",
        "masterinfo.citations", "masterinfo.info.court", "masterinfo.info.bench",
        "otherinfo.judge", "otherinfo.partyname",
    ]


def test_collections_for_intent_empty_list_returns_all_collections():
    assert collections_for_intent([]) == MILVUS_COLLECTIONS


def test_collections_for_intent_single_category_routes_to_its_group():
    assert collections_for_intent(["acts"]) == ["act_section"]
    assert collections_for_intent(["rules"]) == ["rule_section"]
    assert collections_for_intent(["articles"]) == ["article_section"]
    assert collections_for_intent(["commentary"]) == ["commentary_section"]


def test_collections_for_intent_caselaws_routes_to_original_seven():
    assert collections_for_intent(["caselaws"]) == [
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    ]


def test_collections_for_intent_multi_category_unions_groups():
    result = collections_for_intent(["acts", "caselaws"])

    assert set(result) == {
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
        "act_section",
    }


def test_collections_for_intent_result_order_follows_milvus_collections():
    result = collections_for_intent(["acts", "caselaws"])

    assert result == [c for c in MILVUS_COLLECTIONS if c in set(result)]


def test_collections_for_intent_tariff_only_falls_back_to_all_collections():
    """tariff_section isn't in MILVUS_COLLECTIONS yet - a tariff-only tag has
    nothing to route to, so it must fall back to searching everything rather
    than an empty collection list (which would search nothing)."""
    assert collections_for_intent(["tariff"]) == MILVUS_COLLECTIONS


def test_collections_for_intent_unrecognized_tag_only_falls_back_to_all_collections():
    assert collections_for_intent(["not_a_real_category"]) == MILVUS_COLLECTIONS


def test_es_group_for_collection_covers_every_sparse_missing_collection():
    from common.schemas import ES_GROUP_FOR_COLLECTION, MILVUS_COLLECTIONS, SPARSE_VECTOR_COLLECTIONS

    gap_collections = set(MILVUS_COLLECTIONS) - SPARSE_VECTOR_COLLECTIONS
    assert set(ES_GROUP_FOR_COLLECTION.keys()) == gap_collections


def test_es_group_for_collection_values():
    from common.schemas import ES_GROUP_FOR_COLLECTION

    assert ES_GROUP_FOR_COLLECTION == {
        "ruling": "CASELAWS",
        "act_section": "ACT",
        "rule_section": "RULE",
        "article_section": "Experts Opinion",
        "commentary_section": "COMMENTARY",
    }
