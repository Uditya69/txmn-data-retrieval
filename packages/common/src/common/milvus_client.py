import asyncio
import json

from pymilvus import MilvusClient

from common.config import Settings
from common.schemas import BM25_SOURCE_FIELD

# "metadata" is a doc-level collection (one row per document, keyed by doc_id
# itself) - every other collection is chunked with its own chunk_id primary key.
_DOC_LEVEL_COLLECTION = "metadata"


def get_milvus_client(settings: Settings) -> MilvusClient:
    return MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token, db_name=settings.milvus_db)


def _doc_id_filter(doc_id_allowlist: list[str] | None) -> str | None:
    if not doc_id_allowlist:
        return None
    quoted = ", ".join(f'"{d}"' for d in doc_id_allowlist)
    return f"doc_id in [{quoted}]"


def _search_one(
    client, collection: str, dense_vector, sparse_query_text: str, limit: int, filter_expr: str | None
) -> list[dict]:
    if dense_vector is not None:
        anns_field = "dense_vector"
        data = [dense_vector]
    else:
        anns_field = "sparse_vector"
        data = [sparse_query_text]
    text_field = BM25_SOURCE_FIELD[collection]
    hits = client.search(
        collection_name=collection,
        data=data,
        anns_field=anns_field,
        limit=limit,
        filter=filter_expr,
        output_fields=["doc_id", text_field],
    )[0]
    is_doc_level = collection == _DOC_LEVEL_COLLECTION
    return [
        {
            "chunk_id": h["doc_id"] if is_doc_level else h["chunk_id"],
            "doc_id": h["entity"]["doc_id"],
            "text": h["entity"][text_field],
            "score": h["distance"],
        }
        for h in hits
    ]


async def hybrid_search(
    client,
    collections: list[str],
    dense_vector: list[float] | None,
    sparse_query_text: str,
    doc_id_allowlist: list[str] | None = None,
    limit: int = 50,
) -> dict[str, list[dict]]:
    filter_expr = _doc_id_filter(doc_id_allowlist)
    results = await asyncio.gather(*[
        asyncio.to_thread(_search_one, client, collection, dense_vector, sparse_query_text, limit, filter_expr)
        for collection in collections
    ])
    return dict(zip(collections, results))
