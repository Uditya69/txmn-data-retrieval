from datetime import datetime, timezone

_VECTOR_INDEX_NAME = "semantic_cache_vector_index"
_NUM_CANDIDATES = 100


async def lookup(
    collection, mode: str, query_embedding: list[float], threshold: float,
) -> dict | None:
    pipeline = [
        {
            "$vectorSearch": {
                "index": _VECTOR_INDEX_NAME,
                "path": "query_embedding",
                "queryVector": query_embedding,
                "numCandidates": _NUM_CANDIDATES,
                "limit": 1,
                "filter": {"mode": mode},
            },
        },
        {
            "$project": {
                "result": 1,
                "score": {"$meta": "vectorSearchScore"},
            },
        },
    ]
    docs = [doc async for doc in collection.aggregate(pipeline)]
    if not docs:
        return None
    top = docs[0]
    if top["score"] < threshold:
        return None
    return top["result"]


async def write(
    collection, mode: str, query_text: str, query_embedding: list[float], result: dict,
) -> None:
    await collection.insert_one({
        "mode": mode,
        "query_text": query_text,
        "query_embedding": query_embedding,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
