# packages/retrieval-api/src/retrieval_api/instant/search.py
import asyncio

from langfuse import get_client

from common.es_client import raw_search
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.score_cutoff import elbow_cutoff
from retrieval_api.trace_utils import collection_trace


def _apply_elbow_cutoff(rows: list[dict]) -> list[dict]:
    """Trims the long decimal-score tail ES/Milvus hand back untouched -
    Instant has no reranker, so this is the only score-based pruning in
    that path. No max_keep: unlike AI Mode's reranked chunks (which feed
    an LLM prompt and need a hard ceiling), this is a UI preview list."""
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    cutoff = elbow_cutoff([row["score"] for row in ranked])
    return ranked[:cutoff]


def _apply_elbow_cutoff_per_collection(by_collection: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {collection: _apply_elbow_cutoff(rows) for collection, rows in by_collection.items()}


async def _run_es(es_client, query: str, on_step: OnStep | None) -> tuple[list[dict] | None, str | None]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="retriever", name="search-es", input={"query": query}) as span:
        try:
            results = _apply_elbow_cutoff(await raw_search(es_client, query))
            span.update(output={"num_hits": len(results)})
            if on_step is not None:
                await on_step("es_search", {"hits": results})
            return results, None
        except Exception as exc:  # noqa: BLE001 - branch isolation is the point
            span.update(level="ERROR", status_message=str(exc))
            return None, str(exc)


async def _run_milvus(
    gateway, milvus_client, query: str, on_step: OnStep | None,
) -> tuple[dict | None, dict | None, str | None]:
    """Runs dense (Voyage embedding) and sparse (Milvus-native BM25) search
    against every collection - the same two passes AI Mode's retrieve()
    does - so Instant's trace surfaces exactly what each retriever fetched,
    not just the dense results Instant's merged card list is built from."""
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-milvus", input={"query": query},
    ) as span:
        try:
            dense_vector = await gateway.embed(role="query_embed", text=query)
            dense_result, sparse_result = await asyncio.gather(
                hybrid_search(
                    milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector, sparse_query_text=query,
                ),
                hybrid_search(
                    milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None, sparse_query_text=query,
                ),
            )
            dense_result = _apply_elbow_cutoff_per_collection(dense_result)
            sparse_result = _apply_elbow_cutoff_per_collection(sparse_result)
            span.update(output={
                collection: {"dense": len(dense_result[collection]), "sparse": len(sparse_result.get(collection, []))}
                for collection in dense_result
            })
            if on_step is not None:
                await on_step("milvus_dense", collection_trace(dense_result))
                await on_step("milvus_sparse", collection_trace(sparse_result))
            return dense_result, sparse_result, None
        except Exception as exc:  # noqa: BLE001 - branch isolation is the point
            span.update(level="ERROR", status_message=str(exc))
            return None, None, str(exc)


async def run_instant(gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="instant-search", input={"query": query}):
        (es_result, es_error), (milvus_dense, milvus_sparse, milvus_error) = await asyncio.gather(
            _run_es(es_client, query, on_step),
            _run_milvus(gateway, milvus_client, query, on_step),
        )
    return {
        "es": es_result,
        "es_error": es_error,
        "milvus": milvus_dense,
        "milvus_sparse": milvus_sparse,
        "milvus_error": milvus_error,
    }
