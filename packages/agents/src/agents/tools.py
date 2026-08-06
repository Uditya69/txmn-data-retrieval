from common.es_client import fetch_citations, raw_search
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_es",
            "description": "Full-text search over the Elasticsearch case-law index (facts, held, headnotes).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_milvus_dense",
            "description": "Dense embedding similarity search within one Milvus collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "enum": MILVUS_COLLECTIONS},
                    "query": {"type": "string"},
                },
                "required": ["collection", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_milvus_sparse",
            "description": "BM25 sparse search within one Milvus collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "enum": MILVUS_COLLECTIONS},
                    "query": {"type": "string"},
                },
                "required": ["collection", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_doc",
            "description": "Fetch citation metadata for a doc_id already seen from another tool's results.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
]


_TEXT_TRUNCATE_LEN = 500


def _truncate_rows(rows: list[dict]) -> list[dict]:
    """Truncate each row's `text` field before it's json.dumps'd into the
    agent's message history - the full multi-KB chunk text would otherwise
    get replayed into every subsequent turn's tool-result message. Returns
    new dicts; never mutates the input rows in place since they may be
    shared with other callers (e.g. trace_utils.collection_trace)."""
    truncated = []
    for row in rows:
        text = row.get("text")
        if isinstance(text, str) and len(text) > _TEXT_TRUNCATE_LEN:
            row = {**row, "text": text[:_TEXT_TRUNCATE_LEN] + "..."}
        truncated.append(row)
    return truncated


async def dispatch_tool_call(name: str, arguments: dict, *, gateway, es_client, milvus_client) -> dict:
    if name == "search_es":
        rows = await raw_search(es_client, arguments["query"])
        return {"rows": _truncate_rows(rows)}
    if name == "search_milvus_dense":
        collection = arguments["collection"]
        vector = await gateway.embed(role="query_embed", text=arguments["query"])
        result = await hybrid_search(milvus_client, [collection], vector, arguments["query"])
        return {"rows": _truncate_rows(result.get(collection, []))}
    if name == "search_milvus_sparse":
        collection = arguments["collection"]
        result = await hybrid_search(milvus_client, [collection], None, arguments["query"])
        return {"rows": _truncate_rows(result.get(collection, []))}
    if name == "lookup_doc":
        citations = await fetch_citations(es_client, [arguments["doc_id"]])
        return {"citation": citations.get(arguments["doc_id"])}
    raise ValueError(f"unknown tool: {name}")
