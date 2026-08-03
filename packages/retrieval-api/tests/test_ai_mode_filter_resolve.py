import pytest
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist


@pytest.mark.asyncio
async def test_resolve_allowlist_returns_none_for_empty_filters():
    assert await resolve_allowlist(es_client=object(), filters={}) is None


@pytest.mark.asyncio
async def test_resolve_allowlist_delegates_to_common_es_client(monkeypatch):
    import retrieval_api.ai_mode.filter_resolve as module

    async def fake_resolve(client, filters):
        assert filters == {"court": "Supreme Court"}
        return ["d1", "d2"]

    monkeypatch.setattr(module, "resolve_doc_id_allowlist", fake_resolve)

    result = await resolve_allowlist(es_client=object(), filters={"court": "Supreme Court"})

    assert result == ["d1", "d2"]
