import pytest

import retrieval_api.admin_eval.adapters.collection_routing as adapter


def _cases():
    return [
        {"id": "R01", "expect": "confident", "query": "case law for Ramesh Gupta vs. Income-tax Officer", "expected_categories": ["caselaws"]},
        {"id": "R02", "expect": "vague", "query": "capital gains", "expected_categories": []},
    ]


@pytest.mark.asyncio
async def test_run_marks_wrong_confident_tag_as_fail(monkeypatch):
    monkeypatch.setattr(adapter, "load_routing_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        if query == "capital gains":
            return {"search_query": query, "intent": ["commentary"], "filters": {}}  # wrong non-empty tag
        return {"search_query": query, "intent": ["caselaws"], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "pass"
    assert case_events[0]["detail"]["outcome"] == "exact"
    assert case_events[1]["status"] == "fail"
    assert case_events[1]["detail"]["outcome"] == "wrong"


@pytest.mark.asyncio
async def test_run_treats_safe_empty_as_pass(monkeypatch):
    monkeypatch.setattr(adapter, "load_routing_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert all(c["status"] == "pass" for c in case_events)
