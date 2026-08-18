import pytest

import retrieval_api.admin_eval.adapters.intent as adapter


def _cases():
    return [
        {
            "id": "F01", "query": "What did the Bombay High Court decide about ITC under Rule 6(3)(c)?",
            "expected_filters": {"court": "Bombay High Court"}, "expected_categories": ["rules", "caselaws"],
        },
    ]


@pytest.mark.asyncio
async def test_run_yields_pass_case(monkeypatch):
    monkeypatch.setattr(adapter, "load_intent_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": ["rules", "caselaws"], "filters": {"court": "Bombay High Court"}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "pass"
    assert events[-1] == {"type": "done", "summary": {"total": 1, "passed": 1}}


@pytest.mark.asyncio
async def test_run_yields_fail_case_on_filter_mismatch(monkeypatch):
    monkeypatch.setattr(adapter, "load_intent_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": ["rules", "caselaws"], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "fail"
    assert case_events[0]["detail"]["filters"]["ok"] is False
