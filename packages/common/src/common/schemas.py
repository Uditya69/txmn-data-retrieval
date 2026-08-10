MILVUS_COLLECTIONS = [
    "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
]

CHUNKED_COLLECTIONS = {"digest", "facts", "held", "ruling"}

BM25_SOURCE_FIELD = {name: "text" for name in MILVUS_COLLECTIONS}
BM25_SOURCE_FIELD["metadata"] = "heading_subheading_text"

# "ruling" has no sparse_vector field in Milvus - the ingestion pipeline
# dropped its BM25 Function for that collection. Sparse search must skip it
# rather than query a field that no longer exists.
SPARSE_VECTOR_COLLECTIONS = set(MILVUS_COLLECTIONS) - {"ruling"}

MASTERINFO_CITATION_FIELDS = [
    "masterinfo.citations",
    "masterinfo.info.court",
    "masterinfo.info.bench",
    "otherinfo.judge",
    "otherinfo.partyname",
]
