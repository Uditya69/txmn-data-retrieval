MILVUS_COLLECTIONS = [
    "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    "act_section", "rule_section", "article_section", "commentary_section",
]

CHUNKED_COLLECTIONS = {
    "digest", "facts", "held", "ruling",
    "act_section", "rule_section", "article_section", "commentary_section",
}

BM25_SOURCE_FIELD = {name: "text" for name in MILVUS_COLLECTIONS}
BM25_SOURCE_FIELD["metadata"] = "heading_subheading_text"

# "ruling" and the act/rule/article/commentary section collections have no
# sparse_vector field in Milvus - the ingestion pipeline dropped their BM25
# Function. Sparse search must skip them rather than query a field that
# doesn't exist.
SPARSE_VECTOR_COLLECTIONS = set(MILVUS_COLLECTIONS) - {
    "ruling", "act_section", "rule_section", "article_section", "commentary_section",
}

MASTERINFO_CITATION_FIELDS = [
    "heading",
    "subheading",
    "masterinfo.citations",
    "masterinfo.info.court",
    "masterinfo.info.bench",
    "otherinfo.judge",
    "otherinfo.partyname",
]

# Maps each intent category tag to the Milvus collection(s) it routes to. "tariff"
# has no entry - tariff_section isn't in MILVUS_COLLECTIONS yet (parked in the
# ingestion pipeline's _disabled_collections, not live) - a tariff-only intent tag
# falls through collections_for_intent's fallback instead. "caselaws" maps to the
# original 7 collections including metadata - its fields (landmark_ruling, doc-level
# heading/subheading) are case-doc-specific, not a generic cross-category collection.
CATEGORY_COLLECTIONS: dict[str, list[str]] = {
    "caselaws": ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"],
    "acts": ["act_section"],
    "rules": ["rule_section"],
    "articles": ["article_section"],
    "commentary": ["commentary_section"],
}


def collections_for_intent(intent: list[str]) -> list[str]:
    """Which Milvus collections to search for a given intent category list.
    Empty/unrecognized-only intent (nothing confidently tagged, a tariff-only
    tag, or a value CATEGORY_COLLECTIONS has no entry for) falls back to
    searching every collection - never worse than the old always-search-
    everything behavior. Multi-category intent unions its groups. Return
    order follows MILVUS_COLLECTIONS's own order, not intent's tag order, so
    trace/log output stays collection-order-stable regardless of tag order.
    """
    if not intent:
        return list(MILVUS_COLLECTIONS)
    routed = {collection for tag in intent for collection in CATEGORY_COLLECTIONS.get(tag, [])}
    return [c for c in MILVUS_COLLECTIONS if c in routed] or list(MILVUS_COLLECTIONS)
