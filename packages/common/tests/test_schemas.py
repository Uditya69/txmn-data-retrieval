from common.schemas import (
    MILVUS_COLLECTIONS,
    CHUNKED_COLLECTIONS,
    BM25_SOURCE_FIELD,
    MASTERINFO_CITATION_FIELDS,
)


def test_seven_collections():
    assert set(MILVUS_COLLECTIONS) == {
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    }


def test_chunked_collections_match_verified_code_behavior():
    assert CHUNKED_COLLECTIONS == {"digest", "facts", "held", "ruling"}
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
