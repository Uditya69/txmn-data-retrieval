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
