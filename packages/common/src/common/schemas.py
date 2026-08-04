MILVUS_COLLECTIONS = [
    "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
]

CHUNKED_COLLECTIONS = {"digest", "facts", "held", "ruling"}

BM25_SOURCE_FIELD = {name: "text" for name in MILVUS_COLLECTIONS}
BM25_SOURCE_FIELD["metadata"] = "heading_subheading_text"

MASTERINFO_CITATION_FIELDS = [
    "masterinfo.citations",
    "masterinfo.info.court",
    "masterinfo.info.bench",
    "otherinfo.judge",
    "otherinfo.partyname",
]
