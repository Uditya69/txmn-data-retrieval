import pytest

import retrieval_api.admin_eval.adapters.slm_intent as adapter


def _cases():
    return [
        {
            "id": "S01", "query": "case law for Ramesh Gupta vs. Income-tax Officer",
            "expect": "confident", "expected_categories": ["caselaws"],
            "expected_filters": {"party": "Ramesh Gupta"}, "rewrite_must_contain": ["Ramesh Gupta"],
        },
        {
            "id": "S02", "query": "what is this", "expect": "vague",
            "expected_categories": [], "expected_filters": {}, "rewrite_must_contain": [],
        },
    ]


@pytest.mark.asyncio
async def test_run_yields_case_progress_and_done_events(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        if "Ramesh Gupta" in query:
            return {"search_query": "Ramesh Gupta vs. Income-tax Officer", "intent": ["caselaws"], "filters": {"party": "Ramesh Gupta"}}
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]

    case_events = [e for e in events if e["type"] == "case"]
    progress_events = [e for e in events if e["type"] == "progress"]
    done_events = [e for e in events if e["type"] == "done"]

    assert [c["id"] for c in case_events] == ["S01", "S02"]
    assert case_events[0]["status"] == "pass"
    assert case_events[1]["status"] == "pass"  # safe-empty categories
    assert progress_events[-1] == {"type": "progress", "done": 2, "total": 2, "percent": 100}
    assert done_events == [{"type": "done", "summary": {"total": 2, "passed": 2}}]


@pytest.mark.asyncio
async def test_run_respects_limit(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", 1)]
    case_events = [e for e in events if e["type"] == "case"]
    assert [c["id"] for c in case_events] == ["S01"]


@pytest.mark.asyncio
async def test_run_yields_error_status_and_continues(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())

    call_count = {"n": 0}

    async def fake_extract_intent(gateway, query, model=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("gateway unreachable")
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "error"
    assert "gateway unreachable" in case_events[0]["detail"]["error"]
    assert case_events[1]["status"] == "pass"
