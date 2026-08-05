from retrieval_api.ai_mode.intent import extract_intent, OnStep
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist
from retrieval_api.ai_mode.retrieve import retrieve
from retrieval_api.ai_mode.citations import rerank_and_prefetch
from retrieval_api.ai_mode.synthesize import synthesize


async def run_ai_mode(gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None) -> dict:
    try:
        intent_result = await extract_intent(gateway, query, on_step=on_step)
        doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"], on_step=on_step)
        candidates = await retrieve(
            gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist, on_step=on_step
        )
        top_chunks, citations = await rerank_and_prefetch(gateway, es_client, query, candidates, on_step=on_step)
        synthesis = await synthesize(gateway, es_client, query, top_chunks, citations, on_step=on_step)
        return {"ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"]}
    except Exception as exc:  # noqa: BLE001 - AI Mode failure must never crash Instant's result
        return {"ok": False, "error": str(exc)}
