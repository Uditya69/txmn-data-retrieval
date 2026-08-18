from fastapi import APIRouter, Header, HTTPException, WebSocket

from common.config import get_settings
from retrieval_api.admin_eval.auth import is_valid_admin_token
from retrieval_api.admin_eval.registry import SUITES

router = APIRouter()

# Process-local run state - single admin user, single machine (see spec's
# "no queueing" non-goal). _running tracks suite ids currently streaming;
# _cache holds the last completed run per suite for the read-only endpoint.
_running: set[str] = set()
_cache: dict[str, dict] = {}


@router.websocket("/ws/admin-eval")
async def admin_eval(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    token = message.get("token")
    suite = message.get("suite")
    limit = message.get("limit")

    if not is_valid_admin_token(token):
        await websocket.send_json({"type": "error", "reason": "unauthorized"})
        await websocket.close(code=4403)
        return
    if suite not in SUITES:
        await websocket.send_json({"type": "error", "reason": "unknown_suite"})
        await websocket.close(code=4404)
        return
    if suite in _running:
        await websocket.send_json({"type": "error", "reason": "already_running"})
        await websocket.close(code=4409)
        return

    _running.add(suite)
    gateway_url = get_settings().gateway_url
    cases: list[dict] = []
    try:
        async for event in SUITES[suite]["run"](gateway_url, limit):
            if event["type"] == "case":
                cases.append(event)
            await websocket.send_json(event)
            if event["type"] == "done":
                _cache[suite] = {"summary": event["summary"], "cases": cases}
    finally:
        _running.discard(suite)


@router.get("/admin/api/eval-runs/{suite}")
def get_eval_run(suite: str, x_admin_token: str | None = Header(default=None)):
    if not is_valid_admin_token(x_admin_token):
        raise HTTPException(status_code=403)
    if suite not in SUITES:
        raise HTTPException(status_code=404)
    return _cache.get(suite)
