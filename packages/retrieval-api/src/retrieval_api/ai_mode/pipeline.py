from langfuse import get_client

from common.config import get_settings
from common.es_client import fetch_citations, keyword_mode_search
from common.query_tokenizer import build_dense_sparse_query, chunk_query, classify_intent_mode
from retrieval_api.ai_mode.intent import extract_intent, OnStep
from retrieval_api.ai_mode.keyword_expansion import expand_keyword_terms
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist
from retrieval_api.ai_mode.retrieve import retrieve
from retrieval_api.ai_mode.citations import rerank_and_prefetch
from retrieval_api.ai_mode.synthesize import synthesize

# ES-only top-N handed to synthesize() for a "keyword"-tagged query - same as
# rerank_and_prefetch's _NO_RERANK_TOP_N, so the synthesis prompt's chunk count stays
# comparable between the two paths.
_KEYWORD_MODE_TOP_N = 5
_KEYWORD_MODE_ES_LIMIT = 20


async def run_ai_mode(
    gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None,
    persona_context: str = "", boost: bool = False,
) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="ai-mode", input={"query": query}) as root_span:
        try:
            # A precise anchor lookup (bare section/rule/article ref, citation, court name,
            # Act name - see classify_intent_mode) needs no semantic recall: ES alone already
            # resolves it. classify_intent_mode is pure lexical/regex logic with no SLM
            # dependency, so this runs BEFORE extract_intent, on the raw query - letting the
            # keyword branch skip the SLM call entirely rather than pay for a search_query
            # rewrite/intent/filters extraction it never uses. (extract_intent never rewrites
            # `original_query` away from the raw query it's given, so classifying on `query`
            # here is identical to classifying on intent_result["original_query"] later.)
            mode = classify_intent_mode(query)

            if mode == "keyword":
                # Every surviving chunk (post chunk_query's own filler/stopword strip) is a
                # precise anchor - but the raw query still carries whatever conversational
                # scaffolding ("what is", "tell me about") the user typed around it. Searching
                # ES with that raw text dilutes the query with noise words the anchor check
                # itself already discarded; build_dense_sparse_query reconstructs the same
                # cleaned anchor-only text Instant mode's own dense/sparse pass uses (see its
                # docstring), so ES gets "section 55", not "what is section 55".
                chunks = chunk_query(query)
                keyword_query = build_dense_sparse_query(chunks, fallback=query)

                # Experimental, off by default (common.config.Settings.
                # keyword_mode_expansion_enabled) - lets an SLM add up to 2 genuinely-confident
                # legal keywords to broaden ES recall, without paying for (or risking) a full
                # extract_intent()-style rewrite. Appended as extra OR terms, never replacing
                # the cleaned anchor text above.
                if get_settings().keyword_mode_expansion_enabled:
                    with langfuse.start_as_current_observation(
                        as_type="chain", name="keyword-expansion", input={"query": keyword_query},
                    ) as span:
                        added_keywords = await expand_keyword_terms(gateway, keyword_query, on_step=on_step)
                        span.update(output={"added_keywords": added_keywords})
                    if added_keywords:
                        keyword_query = f"{keyword_query} {' '.join(added_keywords)}"

                with langfuse.start_as_current_observation(
                    as_type="chain", name="keyword-search", input={"query": keyword_query},
                ) as span:
                    # No SLM call means no extracted filters - a precise anchor lookup relies
                    # on ES's own phrase-boost match against the anchor text (chunk_query's
                    # section/court_city/citation/quoted chunks) for precision instead.
                    rows = await keyword_mode_search(
                        es_client, keyword_query, doc_id_allowlist=None,
                        limit=_KEYWORD_MODE_ES_LIMIT, boost=boost,
                    )
                    top_rows = sorted(rows, key=lambda row: row["score"], reverse=True)[:_KEYWORD_MODE_TOP_N]
                    top_chunks = [{"doc_id": row["doc_id"], "text": row["text"]} for row in top_rows]
                    citations = await fetch_citations(es_client, [row["doc_id"] for row in top_rows])
                    span.update(output={"num_top_chunks": len(top_chunks), "num_citations": len(citations)})
                if on_step is not None:
                    await on_step("keyword_search", {
                        "query": keyword_query, "mode": mode,
                        "candidate_count": len(rows), "top_doc_ids": [row["doc_id"] for row in top_rows],
                    })
                intent_categories: list[str] = []
            else:
                with langfuse.start_as_current_observation(
                    as_type="chain", name="extract-intent", input={"query": query},
                ) as span:
                    intent_result = await extract_intent(gateway, query, on_step=on_step, persona_context=persona_context)
                    span.update(output=intent_result)

                with langfuse.start_as_current_observation(
                    as_type="retriever", name="resolve-allowlist", input={"filters": intent_result["filters"]},
                ) as span:
                    doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"], on_step=on_step)
                    span.update(output={"num_allowed": None if doc_id_allowlist is None else len(doc_id_allowlist)})

                intent_categories = intent_result["intent"]

                with langfuse.start_as_current_observation(
                    as_type="chain", name="retrieve", input={"search_query": intent_result["search_query"]},
                ) as span:
                    candidates = await retrieve(
                        gateway, milvus_client, es_client, intent_result["search_query"], doc_id_allowlist,
                        intent_result["intent"], on_step=on_step, boost=boost,
                        raw_query=intent_result["original_query"],
                        milvus_sparse_enabled=get_settings().milvus_sparse_enabled,
                    )
                    span.update(output={"num_candidates": len(candidates)})

                with langfuse.start_as_current_observation(
                    as_type="chain", name="rerank-and-prefetch", input={"query": query, "num_candidates": len(candidates)},
                ) as span:
                    top_chunks, citations = await rerank_and_prefetch(
                        gateway, es_client, query, candidates, on_step=on_step,
                        rerank_enabled=get_settings().ai_mode_rerank_enabled,
                    )
                    span.update(output={"num_top_chunks": len(top_chunks), "num_citations": len(citations)})

            with langfuse.start_as_current_observation(as_type="chain", name="synthesize", input={"query": query}) as span:
                synthesis = await synthesize(
                    gateway, es_client, query, top_chunks, citations, on_step=on_step,
                    persona_context=persona_context,
                )
                span.update(output=synthesis["answer"])
                if synthesis.get("reasoning"):
                    span.update(metadata={"reasoning": synthesis["reasoning"]})

            result = {
                "ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"],
                "intent": intent_categories,
            }
            if synthesis.get("reasoning"):
                result["reasoning"] = synthesis["reasoning"]
            root_span.update(output=result)
            return result
        except Exception as exc:  # noqa: BLE001 - AI Mode failure must never crash Instant's result
            root_span.update(level="ERROR", status_message=str(exc), output={"ok": False, "error": str(exc)})
            return {"ok": False, "error": str(exc)}
