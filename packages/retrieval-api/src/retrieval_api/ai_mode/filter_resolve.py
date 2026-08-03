from common.es_client import resolve_doc_id_allowlist


async def resolve_allowlist(es_client, filters: dict) -> list[str] | None:
    return await resolve_doc_id_allowlist(es_client, filters)
