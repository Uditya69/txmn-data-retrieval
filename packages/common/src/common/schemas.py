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
