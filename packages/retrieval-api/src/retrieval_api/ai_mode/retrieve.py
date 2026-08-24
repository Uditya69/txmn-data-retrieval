import asyncio
from itertools import zip_longest

from common.es_client import build_sparse_fallback_query_preview, sparse_fallback_search
from common.milvus_client import hybrid_search
from common.schemas import ES_GROUP_FOR_COLLECTION, SPARSE_VECTOR_COLLECTIONS, collections_for_intent
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.trace_utils import collection_trace


def rrf_merge(
    dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60,
    dense_weight: float = 1.0, sparse_weight: float = 1.0,
) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked_list, weight in ((dense_ranked, dense_weight), (sparse_ranked, sparse_weight)):
        for rank, row in enumerate(ranked_list, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            rows.setdefault(chunk_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[chunk_id], "rrf_score": score} for chunk_id, score in ordered]


def _flatten(by_collection: dict[str, list[dict]], default_origin: str = "milvus") -> list[dict]:
    all_rows = []
    for collection, rows in by_collection.items():
        for row in rows:
            row.setdefault("collection", collection)
            row["origin"] = "es" if row.get("source") == "es_fallback" else default_origin
            all_rows.append(row)
    es_rows = [row for row in all_rows if row.get("source") == "es_fallback"]
    if not es_rows:
        return sorted(all_rows, key=lambda row: row["score"], reverse=True)

    native_rows = [row for row in all_rows if row.get("source") != "es_fallback"]
    ranked_native = sorted(native_rows, key=lambda row: row["score"], reverse=True)
    ranked_es = sorted(es_rows, key=lambda row: row["score"], reverse=True)

    interleaved: list[dict] = []
    for native_row, es_row in zip_longest(ranked_native, ranked_es):
        if native_row is not None:
            interleaved.append(native_row)
        if es_row is not None:
            interleaved.append(es_row)
    return interleaved


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    es_client,
    search_query: str,
    doc_id_allowlist: list[str] | None,
    intent: list[str] | None = None,
    on_step: OnStep | None = None,
    boost: bool = False,
    raw_query: str | None = None,
) -> list[dict]:
    collections = collections_for_intent(intent or [])
    gap_collections = [
        c for c in collections if c not in SPARSE_VECTOR_COLLECTIONS and c in ES_GROUP_FOR_COLLECTION
    ]

    dense_vector = await gateway.embed(role="query_embed", text=search_query)

    # ES fallback searches the user's own words, not the LLM-rewritten search_query - unlike
    # Milvus (which benefits from the rewrite's added context/synonyms for embedding + its own
    # BM25 pass), ES already has its own query-shape classification, phrase chunking, and
    # synonym expansion (_build_field_query) tuned against real user phrasing - a rewrite step
    # on top of that risks drifting the exact section/citation/term match the boost toggle's
    # phrase boosts depend on. Falls back to search_query if no raw_query was given (keeps
    # existing callers/tests working unchanged).
    es_query_text = raw_query if raw_query is not None else search_query
    es_groups = [ES_GROUP_FOR_COLLECTION[c] for c in gap_collections]
    # Tracks whichever doc_id_allowlist the ES fallback call actually last ran with (None
    # once/if the zero-hit-allowlist retry below fires) - needed to rebuild an accurate query
    # preview for the trace step after the fact, without re-running the search.
    effective_allowlist = doc_id_allowlist

    async def _run_es_fallback(allowlist):
        if not gap_collections:
            return {}
        groups = es_groups
        try:
            return await sparse_fallback_search(
                es_client, es_query_text, groups, doc_id_allowlist=allowlist, boost=boost,
            )
        except Exception as exc:
            # ES was never in this path before this branch - a fallback path degrades
            # gracefully (same "no ES rows" shape the rest of the pipeline already
            # handles, e.g. zero gap collections routed), it doesn't escalate an ES
            # hiccup (timeout, 5xx, index.highlight.max_analyzed_offset on a very long
            # judgment) into a total query failure for dense/native-sparse results that
            # would otherwise have been fine.
            if on_step is not None:
                await on_step("es_fallback_degraded", {
                    "reason": "sparse_fallback_search raised; continuing without ES-fallback rows",
                    "error": str(exc),
                })
            return {}

    dense_by_collection, sparse_by_collection, es_sparse_by_collection = await asyncio.gather(
        hybrid_search(
            milvus_client, collections=collections, dense_vector=dense_vector,
            sparse_query_text=search_query, doc_id_allowlist=doc_id_allowlist, limit=50,
        ),
        hybrid_search(
            milvus_client, collections=collections, dense_vector=None,
            sparse_query_text=search_query, doc_id_allowlist=doc_id_allowlist, limit=50,
        ),
        _run_es_fallback(doc_id_allowlist),
    )
    sparse_by_collection.update(es_sparse_by_collection)

    # Circuit breaker: a resolved doc_id_allowlist that's non-empty but the wrong kind of
    # document for these collections silently zeroes every collection even though an
    # unfiltered search would find real matches. If the allowlist was non-empty but
    # produced zero hits everywhere, retry once unfiltered rather than returning nothing -
    # the embedding is already computed, so this only costs the extra round-trips.
    # Retries against the SAME routed collection set - a routed-but-genuinely-wrong-
    # category query should surface as zero results, not silently widen to every
    # collection (that would defeat the point of routing).
    if doc_id_allowlist and not any(dense_by_collection.values()) and not any(sparse_by_collection.values()):
        effective_allowlist = None
        if on_step is not None:
            await on_step("filter_fallback", {
                "reason": "doc_id_allowlist matched zero Milvus results across every routed collection; retrying unfiltered",
                "doc_id_allowlist_count": len(doc_id_allowlist),
            })
        dense_by_collection, sparse_by_collection, es_sparse_by_collection = await asyncio.gather(
            hybrid_search(
                milvus_client, collections=collections, dense_vector=dense_vector,
                sparse_query_text=search_query, doc_id_allowlist=None, limit=50,
            ),
            hybrid_search(
                milvus_client, collections=collections, dense_vector=None,
                sparse_query_text=search_query, doc_id_allowlist=None, limit=50,
            ),
            _run_es_fallback(None),
        )
        sparse_by_collection.update(es_sparse_by_collection)

    # Named distinctly from Instant mode's identically-shaped "milvus_dense"/"milvus_sparse"/
    # "rrf_merge" steps (instant/search.py) - both pipelines share one traceSteps array over
    # the websocket (ws.py), split into Instant vs. AI Mode panes by step name on the
    # frontend (ChatMessageView.tsx's INSTANT_STEP_NAMES) - identical names would misroute
    # AI Mode's own retrieval trace into the Instant pane.
    if on_step is not None:
        await on_step("ai_milvus_dense", {**collection_trace(dense_by_collection), "query": search_query})
        sparse_trace = {**collection_trace(sparse_by_collection), "milvus_query": search_query}
        if gap_collections:
            # es_query_text/boost/effective_allowlist are exactly what the real ES-fallback
            # call(s) feeding this step's rows used - build_sparse_fallback_query_preview
            # reconstructs the query body without re-running the search, same
            # single-source-of-truth pattern Instant mode's query_analysis step uses.
            sparse_trace["es_query"] = es_query_text
            sparse_trace["es_query_body"] = build_sparse_fallback_query_preview(
                es_query_text, es_groups, doc_id_allowlist=effective_allowlist, boost=boost,
            )
        await on_step("ai_milvus_sparse", sparse_trace)

    # RRF fusion weight is always neutral - category does not drive dense/sparse
    # weighting (considered during brainstorming, explicitly rejected; see
    # docs/superpowers/specs/2026-08-14-category-collection-routing-design.md).
    merged = rrf_merge(
        _flatten(dense_by_collection, "milvus_dense"), _flatten(sparse_by_collection, "milvus_sparse"),
    )

    if on_step is not None:
        top_candidates = [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "rrf_score": row["rrf_score"],
                "text_preview": row["text"][:200],
                "origin": row.get("origin"),
                "collection": row.get("collection"),
            }
            for row in merged[:15]
        ]
        await on_step("ai_rrf_merge", {
            "candidate_count": len(merged), "top_candidates": top_candidates,
            "dense_weight": 1.0, "sparse_weight": 1.0,
        })

    return merged
