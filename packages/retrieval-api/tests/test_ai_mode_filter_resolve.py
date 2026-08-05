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


@pytest.mark.asyncio
async def test_resolve_allowlist_emits_filters_resolved_step_with_no_filters():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await resolve_allowlist(es_client=object(), filters={}, on_step=on_step)

    assert result is None
    assert steps == [("filters_resolved", {"filters": {}, "doc_id_count": 0, "doc_id_sample": []})]


@pytest.mark.asyncio
async def test_resolve_allowlist_emits_filters_resolved_step_with_matches(monkeypatch):
    import retrieval_api.ai_mode.filter_resolve as module

    async def fake_resolve(client, filters):
        return ["d1", "d2", "d3"]

    monkeypatch.setattr(module, "resolve_doc_id_allowlist", fake_resolve)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await resolve_allowlist(es_client=object(), filters={"court": "Supreme Court"}, on_step=on_step)

    assert result == ["d1", "d2", "d3"]
    assert steps == [("filters_resolved", {
        "filters": {"court": "Supreme Court"}, "doc_id_count": 3, "doc_id_sample": ["d1", "d2", "d3"],
    })]
