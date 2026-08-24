# packages/retrieval-api/src/retrieval_api/instant/search.py
import asyncio

from langfuse import get_client

from common.es_client import build_query_preview, fetch_doc_categories, raw_search
from common.instant_classifier import effective_label_with_confidence
from common.instant_classifier.labels import routing_plan
from common.legal_lexicon import fuzzy_correct_query
from common.milvus_client import hybrid_search
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


def _all_doc_ids(
    es_result: list[dict] | None, milvus_dense: dict[str, list[dict]] | None, milvus_sparse: dict[str, list[dict]] | None,
) -> list[str]:
    """Union of doc_ids across every source Instant mode can show a card for - the
    reranked list is a fusion of exactly these three, so it needs no separate pass."""
    ids: set[str] = {row["doc_id"] for row in es_result or []}
    for by_collection in (milvus_dense, milvus_sparse):
        for rows in (by_collection or {}).values():
            ids.update(row["doc_id"] for row in rows)
    return list(ids)


async def _run_es(
    es_client, query: str, on_step: OnStep | None, boost: bool = False,
) -> tuple[list[dict] | None, str | None]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-es",
        input={"query": query, "limit": _ES_LIMIT, "boost": boost},
    ) as span:
        try:
            raw_results = await raw_search(es_client, query, limit=_ES_LIMIT, boost=boost)
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
    rrf: bool = False, auto_route: bool = False, boost: bool = False,
) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span", name="instant-search", input={"query": query},
    ) as instant_span:
        # Corrects misspelled court/journal abbreviations before anything else touches the
        # query, so the classifier, ES, and Milvus all search/route on the same corrected
        # text rather than each needing to apply this independently.
        corrected_query, corrections = fuzzy_correct_query(query)
        query_correction_trace = {"original": query, "corrected": corrected_query, "corrections": corrections}
        instant_span.update(metadata={"query_correction": query_correction_trace})
        if on_step is not None:
            await on_step("query_correction", query_correction_trace)
        query = corrected_query

        # build_query_preview is the same function raw_search() calls internally (and that
        # backs the standalone /v1/query-analysis endpoint) - using it here rather than
        # independently recomputing shape/chunks means this trace step can never drift from
        # what the real ES query actually was, the way the older analyze_query()-based version
        # of this step did (it used its own separate, pre-chunk_query pipeline, so it never
        # showed an unrecognized word run like "Dimension Data India" grouped into one phrase
        # the way the real query - and /v1/query-analysis - already did).
        if on_step is not None:
            await on_step("query_analysis", build_query_preview(query, boost=boost))

        label, confidence = effective_label_with_confidence(query)
        plan = routing_plan(label) if auto_route else {"es": True, "milvus": True, "fuse": False}
        # Surfaced in both trace systems - without this, a skipped ES/Milvus call (auto_route)
        # is indistinguishable from one that ran and legitimately found nothing, and the raw
        # model confidence (as opposed to the post-threshold label) is otherwise unobservable
        # anywhere, since effective_label() alone discards it.
        classifier_trace = {
            "label": label, "confidence": confidence, "auto_route": auto_route, "plan": plan,
        }
        instant_span.update(metadata={"classifier": classifier_trace})
        if on_step is not None:
            await on_step("classifier", classifier_trace)

        es_task = _run_es(es_client, query, on_step, boost=boost) if plan["es"] else None
        milvus_task = _run_milvus(gateway, milvus_client, query, on_step) if plan["milvus"] else None

        if es_task is not None and milvus_task is not None:
            (es_result, es_error), (milvus_dense, milvus_sparse, milvus_error) = await asyncio.gather(
                es_task, milvus_task,
            )
        elif es_task is not None:
            es_result, es_error = await es_task
            milvus_dense, milvus_sparse, milvus_error = None, None, None
        else:
            es_result, es_error = None, None
            milvus_dense, milvus_sparse, milvus_error = await milvus_task

        result = {
            "query_correction": query_correction_trace,
            "es": es_result,
            "es_error": es_error,
            "milvus": milvus_dense,
            "milvus_sparse": milvus_sparse,
            "milvus_error": milvus_error,
        }

        # Runs alongside reranking below, not after - a separate mget by doc_id, so it has
        # no dependency on the fuse step's own output.
        doc_meta_task = asyncio.create_task(
            fetch_doc_categories(es_client, _all_doc_ids(es_result, milvus_dense, milvus_sparse)),
        )

        effective_rrf = plan["fuse"] if auto_route else rrf

        # Whichever side was actually skipped (by plan, above) has its error left at None,
        # so this naturally reduces to "the error from whichever source(s) ran" in every case.
        reranked_error = es_error or milvus_error
        reranked = []
        if reranked_error is None:
            with langfuse.start_as_current_observation(
                as_type="chain", name="instant-fuse", input={"query": query, "rrf": effective_rrf},
            ) as rerank_span:
                try:
                    reranked = await rerank_instant_results(
                        label, es_result or [], milvus_dense or {}, milvus_sparse or {},
                        rrf=effective_rrf, plan=plan, on_step=on_step,
                    )
                    rerank_span.update(output={"num_reranked": len(reranked)})
                    if on_step is not None:
                        await on_step("instant_reranked", {"hits": reranked})
                except Exception as exc:  # noqa: BLE001 - branch isolation is the point
                    reranked_error = str(exc)
                    rerank_span.update(level="ERROR", status_message=reranked_error)
        result["reranked"] = reranked
        result["reranked_error"] = reranked_error

        # A badge is a nice-to-have, not a reason to fail the whole search - degrade to no
        # badges rather than propagate an mget error out of run_instant.
        try:
            result["doc_meta"] = await doc_meta_task
        except Exception:  # noqa: BLE001 - branch isolation is the point
            result["doc_meta"] = {}
    return result
