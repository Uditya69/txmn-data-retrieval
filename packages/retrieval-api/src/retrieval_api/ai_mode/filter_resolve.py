from common.es_client import resolve_doc_id_allowlist
from retrieval_api.ai_mode.intent import OnStep


async def resolve_allowlist(es_client, filters: dict, on_step: OnStep | None = None) -> list[str] | None:
    result = await resolve_doc_id_allowlist(es_client, filters)

    if on_step is not None:
        sample = (result or [])[:10]
        await on_step("filters_resolved", {"filters": filters, "doc_id_count": len(result or []), "doc_id_sample": sample})

    return result
