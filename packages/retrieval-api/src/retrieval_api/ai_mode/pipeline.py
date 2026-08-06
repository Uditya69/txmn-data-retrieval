from langfuse import get_client

from retrieval_api.ai_mode.intent import extract_intent, OnStep
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist
from retrieval_api.ai_mode.retrieve import retrieve
from retrieval_api.ai_mode.citations import rerank_and_prefetch
from retrieval_api.ai_mode.synthesize import synthesize


async def run_ai_mode(gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="ai-mode", input={"query": query}) as root_span:
        try:
            with langfuse.start_as_current_observation(
                as_type="chain", name="extract-intent", input={"query": query},
            ) as span:
                intent_result = await extract_intent(gateway, query, on_step=on_step)
                span.update(output=intent_result)

            with langfuse.start_as_current_observation(
                as_type="retriever", name="resolve-allowlist", input={"filters": intent_result["filters"]},
            ) as span:
                doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"], on_step=on_step)
                span.update(output={"num_allowed": None if doc_id_allowlist is None else len(doc_id_allowlist)})

            with langfuse.start_as_current_observation(
                as_type="chain", name="retrieve", input={"rewritten_query": intent_result["rewritten_query"]},
            ) as span:
                candidates = await retrieve(
                    gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist, on_step=on_step
                )
                span.update(output={"num_candidates": len(candidates)})

            with langfuse.start_as_current_observation(
                as_type="chain", name="rerank-and-prefetch", input={"query": query, "num_candidates": len(candidates)},
            ) as span:
                top_chunks, citations = await rerank_and_prefetch(gateway, es_client, query, candidates, on_step=on_step)
                span.update(output={"num_top_chunks": len(top_chunks), "num_citations": len(citations)})

            with langfuse.start_as_current_observation(as_type="chain", name="synthesize", input={"query": query}) as span:
                synthesis = await synthesize(gateway, es_client, query, top_chunks, citations, on_step=on_step)
                span.update(output=synthesis["answer"])
                if synthesis.get("reasoning"):
                    span.update(metadata={"reasoning": synthesis["reasoning"]})

            result = {"ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"]}
            if synthesis.get("reasoning"):
                result["reasoning"] = synthesis["reasoning"]
            root_span.update(output=result)
            return result
        except Exception as exc:  # noqa: BLE001 - AI Mode failure must never crash Instant's result
            root_span.update(level="ERROR", status_message=str(exc), output={"ok": False, "error": str(exc)})
            return {"ok": False, "error": str(exc)}
