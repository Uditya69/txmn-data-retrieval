# packages/retrieval-api/src/retrieval_api/instant/search.py
import asyncio

from common.es_client import raw_search
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS


async def _run_es(es_client, query: str) -> tuple[list[dict] | None, str | None]:
    try:
        return await raw_search(es_client, query), None
    except Exception as exc:  # noqa: BLE001 - branch isolation is the point
        return None, str(exc)


async def _run_milvus(gateway, milvus_client, query: str) -> tuple[dict | None, str | None]:
    try:
        dense_vector = await gateway.embed(role="query_embed", text=query)
        result = await hybrid_search(
            milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector, sparse_query_text=query,
        )
        return result, None
    except Exception as exc:  # noqa: BLE001 - branch isolation is the point
        return None, str(exc)


async def run_instant(gateway, es_client, milvus_client, query: str) -> dict:
    (es_result, es_error), (milvus_result, milvus_error) = await asyncio.gather(
        _run_es(es_client, query),
        _run_milvus(gateway, milvus_client, query),
    )
    return {
        "es": es_result,
        "es_error": es_error,
        "milvus": milvus_result,
        "milvus_error": milvus_error,
    }
