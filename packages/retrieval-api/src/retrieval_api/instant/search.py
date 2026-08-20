# packages/retrieval-api/src/retrieval_api/instant/search.py
import asyncio

from langfuse import get_client

from common.es_client import build_query_preview, raw_search
from common.milvus_client import hybrid_search
from common.query_tokenizer import classify_query_shape
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.instant.rerank import rerank_instant_results
from retrieval_api.score_cutoff import elbow_cutoff
from retrieval_api.trace_utils import collection_trace, ranked_trace, collection_ranked_trace

_ES_LIMIT = 20  # kept in a name so the trace input and the raw_search() call can't drift apart


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
    shape = classify_query_shape(query)
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-es",
        input={"query": query, "query_shape": shape, "limit": _ES_LIMIT},
    ) as span:
        try:
            raw_results = await raw_search(es_client, query, limit=_ES_LIMIT)
            results = _apply_elbow_cutoff(raw_results)
            span.update(output={
                "hits_before_cutoff": len(raw_results),
                "hits_after_cutoff": len(results),
                # full pre-cutoff ranking, not just what survives the elbow -
                # this is what lets a gold doc_id's rank (or its absence within
                # the fetched window) be read straight off the trace.
                "top_hits": ranked_trace(raw_results, top_n=_ES_LIMIT),
            })
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
            # snapshot pre-cutoff ranks before the elbow trims them, same reason
            # as _run_es: this is the only place that can show a gold doc_id's
            # rank when the elbow cutoff is what dropped it, versus the search
            # itself never surfacing it at all.
            dense_pre_cutoff, sparse_pre_cutoff = dense_result, sparse_result
            dense_result = _apply_elbow_cutoff_per_collection(dense_result)
            sparse_result = _apply_elbow_cutoff_per_collection(sparse_result)
            span.update(output={
                "dense": collection_ranked_trace(dense_pre_cutoff),
                "sparse": collection_ranked_trace(sparse_pre_cutoff),
                "after_cutoff": {
                    collection: {"dense": len(dense_result[collection]), "sparse": len(sparse_result.get(collection, []))}
                    for collection in dense_result
                },
            })
            if on_step is not None:
                await on_step("milvus_dense", collection_trace(dense_result))
                await on_step("milvus_sparse", collection_trace(sparse_result))
            return dense_result, sparse_result, None
        except Exception as exc:  # noqa: BLE001 - branch isolation is the point
            span.update(level="ERROR", status_message=str(exc))
            return None, None, str(exc)


async def run_instant(
    gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None,
    rrf: bool = False, rerank: bool = False,
) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="instant-search", input={"query": query}):
        # build_query_preview is the same function raw_search() calls internally (and that
        # backs the standalone /v1/query-analysis endpoint) - using it here rather than
        # independently recomputing shape/chunks means this trace step can never drift from
        # what the real ES query actually was, the way the older analyze_query()-based version
        # of this step did (it used its own separate, pre-chunk_query pipeline, so it never
        # showed an unrecognized word run like "Dimension Data India" grouped into one phrase
        # the way the real query - and /v1/query-analysis - already did).
        if on_step is not None:
            await on_step("query_analysis", build_query_preview(query))

        (es_result, es_error), (milvus_dense, milvus_sparse, milvus_error) = await asyncio.gather(
            _run_es(es_client, query, on_step),
            _run_milvus(gateway, milvus_client, query, on_step),
        )
        result = {
            "es": es_result,
            "es_error": es_error,
            "milvus": milvus_dense,
            "milvus_sparse": milvus_sparse,
            "milvus_error": milvus_error,
        }
        if not rrf and not rerank:
            return result

        # Milvus isn't consulted at all when rrf is off (see rerank_instant_results),
        # so a Milvus-side failure shouldn't block that path.
        reranked_error = es_error or (milvus_error if rrf else None)
        reranked = []
        if reranked_error is None:
            with langfuse.start_as_current_observation(
                as_type="chain", name="rerank", input={"query": query, "rrf": rrf, "rerank": rerank},
            ) as rerank_span:
                try:
                    reranked = await rerank_instant_results(
                        gateway, es_client, query, classify_query_shape(query),
                        es_result or [], milvus_dense or {}, milvus_sparse or {},
                        rrf=rrf, rerank=rerank,
                    )
                    rerank_span.update(output={"num_reranked": len(reranked)})
                    if on_step is not None:
                        await on_step("instant_reranked", {"hits": reranked})
                except Exception as exc:  # noqa: BLE001 - branch isolation is the point
                    reranked_error = str(exc)
                    rerank_span.update(level="ERROR", status_message=reranked_error)
        result["reranked"] = reranked
        result["reranked_error"] = reranked_error
    return result
