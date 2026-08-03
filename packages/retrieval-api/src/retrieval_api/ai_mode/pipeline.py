from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist
from retrieval_api.ai_mode.retrieve import retrieve
from retrieval_api.ai_mode.citations import rerank_and_prefetch
from retrieval_api.ai_mode.synthesize import synthesize


async def run_ai_mode(gateway, es_client, milvus_client, query: str) -> dict:
    try:
        intent_result = await extract_intent(gateway, query)
        doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"])
        candidates = await retrieve(gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist)
        top_chunks, citations = await rerank_and_prefetch(gateway, es_client, query, candidates)
        synthesis = await synthesize(gateway, es_client, query, top_chunks, citations)
        return {"ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"]}
    except Exception as exc:  # noqa: BLE001 - AI Mode failure must never crash Instant's result
        return {"ok": False, "error": str(exc)}
