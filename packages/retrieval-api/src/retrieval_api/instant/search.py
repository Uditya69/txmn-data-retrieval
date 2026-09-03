# packages/retrieval-api/src/retrieval_api/instant/search.py
import asyncio

from langfuse import get_client

from common.es_client import build_query_preview, fetch_doc_categories, raw_search, raw_search_grouped
from common.instant_classifier import effective_label_with_confidence
from common.instant_classifier.labels import routing_plan
from common.legal_lexicon import fuzzy_correct_query
from common.milvus_client import hybrid_search
from common.query_tokenizer import build_dense_sparse_query, chunk_query
from common.repotaxmannapi_tokenizer import is_whole_query_exact_phrase
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.instant.rerank import rerank_instant_results
from retrieval_api.score_cutoff import elbow_cutoff
from retrieval_api.trace_utils import collection_trace, ranked_trace, collection_ranked_trace

_ES_LIMIT = 20  # kept in a name so the trace input and the raw_search() call can't drift apart


def _apply_elbow_cutoff(rows: list[dict]) -> list[dict]:
    """Trims the long decimal-score tail ES/Milvus hand back untouched - applied to each
    retriever's raw hits regardless of whether the (opt-in) cross-encoder reranker later
    re-sorts them. No max_keep: unlike AI Mode's reranked chunks (which feed an LLM prompt
    and need a hard ceiling), this is a UI preview list."""
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    cutoff = elbow_cutoff([row["score"] for row in ranked])
    return ranked[:cutoff]


def _apply_elbow_cutoff_per_collection(by_collection: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {collection: _apply_elbow_cutoff(rows) for collection, rows in by_collection.items()}


def _all_doc_ids(
    es_result: list[dict] | None, milvus_dense: dict[str, list[dict]] | None, milvus_sparse: dict[str, list[dict]] | None,
    grouped_es: dict[str, list[dict]] | None = None,
) -> list[str]:
    """Union of doc_ids across every source Instant mode can show a card for - the
    reranked list is a fusion of exactly these three, so it needs no separate pass.
    grouped_es (raw_search_grouped's sectioned results) is a fourth, independent source -
    its own ES call, not part of the es_result/reranked fusion - so its doc_ids need adding
    here too or fetch_doc_categories would never resolve badges for cards a section shows
    but the flat es_result/reranked list doesn't."""
    ids: set[str] = {row["doc_id"] for row in es_result or []}
    for by_collection in (milvus_dense, milvus_sparse):
        for rows in (by_collection or {}).values():
            ids.update(row["doc_id"] for row in rows)
    for rows in (grouped_es or {}).values():
        ids.update(row["doc_id"] for row in rows)
    return list(ids)


async def _run_es(
    es_client, query: str, on_step: OnStep | None, boost: bool = True, skip_cutoff: bool = False,
    boost_source: str = "repotaxmannapi", page: int = 1, page_size: int | None = None,
) -> tuple[list[dict] | None, str | None]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-es",
        input={"query": query, "limit": _ES_LIMIT, "boost": boost, "boost_source": boost_source},
    ) as span:
        try:
            raw_results = await raw_search(
                es_client, query, limit=_ES_LIMIT, boost=boost, boost_source=boost_source,
                page=page, page_size=page_size,
            )
            # KEYWORD-shape queries are precise anchor lookups whose results span steep
            # boost-tier gaps by design (heading:100000 vs fullcontent:1 in _PHRASE_BOOSTS,
            # common/es_client.py) - the elbow's ratio test misreads a legit lower-tier
            # match as a score cliff and prunes it, silently dropping correct answers ES
            # itself already ranked and returned. HYBRID/INTENT keep the elbow: they lean
            # on dense fusion rather than showing this raw ES ranking as-is.
            #
            # page_size is not None (2026-09-02, hidden-by-default pagination): the elbow's
            # ratio test assumes a flat top-N window starting at rank 1 - on page 2+ it would
            # evaluate over an arbitrary mid-corpus score window with no relationship to real
            # relevance, silently pruning results in a way that isn't even deterministic
            # w.r.t. page size. Skip it entirely for any paged request, same as skip_cutoff.
            results = raw_results if skip_cutoff or page_size is not None else _apply_elbow_cutoff(raw_results)
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


_ES_GROUPED_LIMIT_PER_GROUP = 5


async def _run_es_grouped(
    es_client, query: str, on_step: OnStep | None,
) -> tuple[dict[str, list[dict]] | None, str | None]:
    """repotaxmannapi-mode's sectioned result view (raw_search_grouped, common/es_client.py) -
    always the multiply-mode formula, no sum-mode equivalent, so this is only ever called
    when boost_source == "repotaxmannapi" (see run_instant). A separate ES call from
    _run_es's flat search, not a filter/regroup of its results - an independent full-corpus
    aggregation per content type, so a section's docs aren't bounded by whatever the flat
    top-20 window happened to contain. A regroup-of-flat-hits version was tried and
    reverted: for a bare "SECTION 52" query the flat top-20 is entirely Act documents
    (multiply-mode score dominance, verified byte-identical to a real captured production
    response), so regrouping only that window left every other section empty even though
    those content types have real matches elsewhere in the corpus."""
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-es-grouped",
        input={"query": query, "limit_per_group": _ES_GROUPED_LIMIT_PER_GROUP},
    ) as span:
        try:
            grouped = await raw_search_grouped(es_client, query, limit_per_group=_ES_GROUPED_LIMIT_PER_GROUP)
            span.update(output={
                "groups": {name: len(rows) for name, rows in grouped.items()},
            })
            if on_step is not None:
                await on_step("es_grouped", {"groups": grouped})
            return grouped, None
        except Exception as exc:  # noqa: BLE001 - branch isolation is the point
            span.update(level="ERROR", status_message=str(exc))
            return None, str(exc)


async def _run_milvus(
    gateway, milvus_client, query: str, on_step: OnStep | None, milvus_sparse_enabled: bool = False,
) -> tuple[dict | None, dict | None, str | None]:
    """Runs dense (Voyage embedding) and, when enabled, sparse (Milvus-native BM25) search
    against every collection - the same two passes AI Mode's retrieve() does - so Instant's
    trace surfaces exactly what each retriever fetched, not just the dense results Instant's
    merged card list is built from.

    milvus_sparse_enabled (common.config.Settings.milvus_sparse_enabled) is off by default -
    same env-only kill switch AI Mode's retrieve() uses, no separate UI toggle. Instant has no
    ES sparse-fallback mechanism (that's an AI-Mode-only gap-collection concern), so disabling
    this skips the native sparse pass entirely with nothing else to fall back to.

    `query` here is already the cleaned dense/sparse search text (see run_instant's
    build_dense_sparse_query call) - not necessarily the user's raw sentence."""
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-milvus", input={"query": query},
    ) as span:
        try:
            dense_vector = await gateway.embed(role="query_embed", text=query)
            if milvus_sparse_enabled:
                dense_result, sparse_result = await asyncio.gather(
                    hybrid_search(
                        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector, sparse_query_text=query,
                    ),
                    hybrid_search(
                        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None, sparse_query_text=query,
                    ),
                )
            else:
                dense_result = await hybrid_search(
                    milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector, sparse_query_text=query,
                )
                sparse_result = {}
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
                # Nothing ran the sparse pass at all when disabled - an empty "Milvus
                # sparse search" card with zero collections would just be noise (same
                # reasoning as AI Mode's ai_milvus_sparse step, retrieve.py).
                if milvus_sparse_enabled:
                    await on_step("milvus_sparse", collection_trace(sparse_result))
            return dense_result, sparse_result, None
        except Exception as exc:  # noqa: BLE001 - branch isolation is the point
            span.update(level="ERROR", status_message=str(exc))
            return None, None, str(exc)


async def run_instant(
    gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None,
    rrf: bool = False, rerank: bool = False, auto_route: bool = False, boost: bool = True,
    milvus_sparse_enabled: bool = False, boost_source: str = "repotaxmannapi",
    page: int = 1, page_size: int | None = None,
) -> dict:
    """boost_source (common/es_client.py::raw_search) affects the `query_analysis` trace
    step's `es_query` preview and the actual ES search below; nothing else in this
    function's control flow (label/plan/milvus/rrf) reads it. `boost_source="repotaxmannapi"`
    additionally runs `_run_es_grouped` (see below) - an independent per-content-type
    aggregation, not a regroup of `es_result`."""
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
            await on_step("query_analysis", build_query_preview(query, boost=boost, boost_source=boost_source))

        # Instant mode has no LLM query-rewrite step (unlike AI Mode's extract_intent) to strip
        # conversational scaffolding before searching - without this, "section 55" and "what is
        # section 55" send identical text to ES's phrase-boost clauses (chunk_query already
        # cleans those - see _PHRASE_BOOSTS) but different, noise-diluted text to Milvus
        # dense/sparse, which searched the raw sentence verbatim. ES keeps searching the raw
        # `query` (its own pipeline already handles this); only Milvus gets the cleaned text.
        milvus_query = build_dense_sparse_query(chunk_query(query), fallback=query)

        # Exact-phrase fast path (2026-09-04): a search bar query that is ENTIRELY one
        # double-quoted phrase (e.g. `"section 52"`) is an unambiguous exact-lookup request -
        # running it through the ML shape classifier and Milvus dense search anyway (as
        # every query previously did) has actively hurt results in practice: for a real
        # long-phrase headnote-text query, the classifier mislabeled it HYBRID and routed to
        # `{"es": True, "milvus": True, "fuse": True}`, but ES's own `es_query.bool.must`
        # (correctly) requires a genuine phrase match to even return a hit at all, so
        # `es_search` came back empty while Milvus's semantic search - which has no such
        # exact-match requirement - filled the entire result list with topically-similar but
        # NOT phrase-matching cases. The user asked to quote an exact phrase specifically to
        # rule those out; running semantic search anyway silently defeated the intent. Skips
        # `effective_label_with_confidence`'s model call entirely (not just its routing
        # output) - this is a structural property of the query text, not something the
        # shape classifier is any better positioned to judge than a plain tokenize() call.
        if is_whole_query_exact_phrase(query):
            label, confidence = "KEYWORD", 1.0
            plan = {"es": True, "milvus": False, "fuse": False}
        else:
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

        es_task = (
            _run_es(
                es_client, query, on_step, boost=boost, skip_cutoff=label == "KEYWORD",
                boost_source=boost_source, page=page, page_size=page_size,
            )
            if plan["es"] else None
        )
        milvus_task = (
            _run_milvus(gateway, milvus_client, milvus_query, on_step, milvus_sparse_enabled=milvus_sparse_enabled)
            if plan["milvus"] else None
        )
        # Sectioned result view - repotaxmannapi mode only, no sum-mode equivalent.
        # Runs alongside es_task/milvus_task, not after: independent ES call, no
        # dependency on either's output.
        grouped_task = (
            _run_es_grouped(es_client, query, on_step)
            if plan["es"] and boost and boost_source == "repotaxmannapi" else None
        )

        # Order here must match `tasks`' construction order above - gathered results are
        # consumed in the same left-to-right order they were gathered in.
        tasks = [t for t in (es_task, milvus_task, grouped_task) if t is not None]
        gathered = iter(await asyncio.gather(*tasks))
        es_result, es_error = next(gathered) if es_task is not None else (None, None)
        milvus_dense, milvus_sparse, milvus_error = (
            next(gathered) if milvus_task is not None else (None, None, None)
        )
        grouped_es, grouped_es_error = (
            next(gathered) if grouped_task is not None else (None, None)
        )

        result = {
            "query_correction": query_correction_trace,
            "es": es_result,
            "es_error": es_error,
            "milvus": milvus_dense,
            "milvus_sparse": milvus_sparse,
            "milvus_error": milvus_error,
            "grouped_es": grouped_es,
            "grouped_es_error": grouped_es_error,
        }

        # Runs alongside reranking below, not after - a separate mget by doc_id, so it has
        # no dependency on the fuse step's own output.
        doc_meta_task = asyncio.create_task(
            fetch_doc_categories(es_client, _all_doc_ids(es_result, milvus_dense, milvus_sparse, grouped_es)),
        )

        effective_rrf = plan["fuse"] if auto_route else rrf

        # Whichever side was actually skipped (by plan, above) has its error left at None,
        # so this naturally reduces to "the error from whichever source(s) ran" in every case.
        reranked_error = es_error or milvus_error
        reranked = []
        if reranked_error is None:
            with langfuse.start_as_current_observation(
                as_type="chain", name="instant-fuse", input={"query": query, "rrf": effective_rrf, "rerank": rerank},
            ) as rerank_span:
                try:
                    reranked = await rerank_instant_results(
                        gateway, es_client, query, label,
                        es_result or [], milvus_dense or {}, milvus_sparse or {},
                        rrf=effective_rrf, rerank=rerank, plan=plan, on_step=on_step,
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
