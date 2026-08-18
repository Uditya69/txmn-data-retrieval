import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.admin_eval.router as router_module


@pytest.fixture(autouse=True)
def _reset_router_state():
    router_module._running.clear()
    router_module._cache.clear()
    yield
    router_module._running.clear()
    router_module._cache.clear()


async def _fake_suite_run(gateway_url, limit):
    yield {"type": "case", "id": "T1", "query": "q1", "status": "pass", "detail": {}}
    yield {"type": "progress", "done": 1, "total": 1, "percent": 100}
    yield {"type": "done", "summary": {"total": 1, "passed": 1}}


async def _failing_suite_run(gateway_url, limit):
    if False:
        yield  # pragma: no cover - makes this an async generator
    raise FileNotFoundError("evals/routing_cases.json not found")


async def _partial_then_failing_suite_run(gateway_url, limit):
    yield {"type": "case", "id": "T1", "query": "q1", "status": "pass", "detail": {}}
    raise RuntimeError("boom")


CASES = [
    {"id": f"T{i}", "query": f"q{i}"} for i in range(1, 6)
]


async def _limit_recording_suite_run(gateway_url, limit):
    cases = CASES[:limit] if limit is not None else CASES
    for case in cases:
        yield {"type": "case", "id": case["id"], "query": case["query"], "status": "pass", "detail": {}}
    yield {"type": "progress", "done": len(cases), "total": len(cases), "percent": 100}
    yield {"type": "done", "summary": {"total": len(cases), "passed": len(cases)}}


def test_ws_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: False)
    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "slm_intent", "token": "wrong"})
        message = ws.receive_json()
        assert message == {"type": "error", "reason": "unauthorized"}


def test_ws_rejects_unknown_suite(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "not_a_real_suite", "token": "t"})
        message = ws.receive_json()
        assert message == {"type": "error", "reason": "unknown_suite"}


def test_ws_rejects_already_running_suite(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _fake_suite_run})
    router_module._running.add("fake_suite")

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t"})
        message = ws.receive_json()
        assert message == {"type": "error", "reason": "already_running"}


def test_ws_streams_events_and_populates_cache(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _fake_suite_run})
    monkeypatch.setattr(router_module, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t"})
        messages = [ws.receive_json() for _ in range(3)]

    assert messages[0]["type"] == "case"
    assert messages[1]["type"] == "progress"
    assert messages[2] == {"type": "done", "summary": {"total": 1, "passed": 1}}
    assert "fake_suite" not in router_module._running
    assert router_module._cache["fake_suite"]["summary"] == {"total": 1, "passed": 1}
    assert router_module._cache["fake_suite"]["cases"] == [messages[0]]


def test_cache_read_returns_null_before_any_run(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "t"})
    assert response.status_code == 200
    assert response.json() is None


def test_cache_read_returns_populated_run(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    router_module._cache["slm_intent"] = {"summary": {"total": 1, "passed": 1}, "cases": []}
    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "t"})
    assert response.status_code == 200
    assert response.json() == {"summary": {"total": 1, "passed": 1}, "cases": []}


def test_cache_read_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: False)
    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "wrong"})
    assert response.status_code == 403


def test_ws_surfaces_run_level_exception_before_any_yield(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _failing_suite_run})
    monkeypatch.setattr(router_module, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t"})
        message = ws.receive_json()

    assert message["type"] == "error"
    assert "run failed" in message["reason"]
    assert "evals/routing_cases.json not found" in message["reason"]
    assert "fake_suite" not in router_module._running


def test_ws_surfaces_run_level_exception_after_partial_yield(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(
        router_module.SUITES, "fake_suite", {"name": "Fake", "run": _partial_then_failing_suite_run}
    )
    monkeypatch.setattr(router_module, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t"})
        first = ws.receive_json()
        second = ws.receive_json()

    assert first["type"] == "case"
    assert second["type"] == "error"
    assert "boom" in second["reason"]
    assert "fake_suite" not in router_module._running


def test_ws_rejects_when_admin_secret_unset_real_predicate(monkeypatch):
    import common.config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": None})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "slm_intent", "token": "anything"})
        message = ws.receive_json()

    assert message == {"type": "error", "reason": "unauthorized"}


def test_cache_read_rejects_when_admin_secret_unset_real_predicate(monkeypatch):
    import common.config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": None})())

    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "anything"})
    assert response.status_code == 403


@pytest.mark.parametrize("bad_limit", ["abc", -1, 0])
def test_ws_normalizes_invalid_limit_to_no_limit(monkeypatch, bad_limit):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _limit_recording_suite_run})
    monkeypatch.setattr(router_module, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t", "limit": bad_limit})
        messages = [ws.receive_json() for _ in range(len(CASES) + 2)]

    case_messages = [m for m in messages if m["type"] == "case"]
    assert len(case_messages) == len(CASES)
    assert messages[-1]["summary"]["total"] == len(CASES)


def test_ws_applies_valid_limit(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _limit_recording_suite_run})
    monkeypatch.setattr(router_module, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t", "limit": 2})
        messages = [ws.receive_json() for _ in range(2 + 2)]

    case_messages = [m for m in messages if m["type"] == "case"]
    assert len(case_messages) == 2
