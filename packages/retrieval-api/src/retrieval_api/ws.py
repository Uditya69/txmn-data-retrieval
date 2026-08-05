import asyncio
import logging

from fastapi import APIRouter, WebSocket

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.search import run_instant
from retrieval_api.ai_mode.pipeline import run_ai_mode

router = APIRouter()

logger = logging.getLogger(__name__)


def get_gateway_client(settings) -> GatewayClient:
    return GatewayClient(base_url=settings.gateway_url)


async def _emit_trace_step(send, step: str, data: dict) -> None:
    """Swallows any exception from `send` (e.g. the client disconnected
    mid-stream) - a dead trace channel must never fail the AI Mode
    pipeline or its final answer."""
    try:
        await send({"type": "ai_mode_trace", "step": step, "data": data})
    except Exception as exc:
        logger.debug("trace step %r dropped: %s", step, exc)


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"
    trace = message.get("trace", False)

    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        milvus_client = None

    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await _emit_trace_step(send, step, data)

    try:
        instant_task = (
            asyncio.create_task(run_instant(gateway, es_client, milvus_client, query))
            if mode in ("instant", "both") else None
        )
        ai_mode_task = (
            asyncio.create_task(
                run_ai_mode(
                    gateway, es_client, milvus_client, query,
                    on_step=emit_trace_step if trace else None,
                )
            )
            if mode in ("ai_mode", "both") else None
        )

        if instant_task is not None:
            instant_result = await instant_task
            await send({"type": "instant_result", **instant_result})

        if ai_mode_task is not None:
            ai_mode_result = await ai_mode_task
            if ai_mode_result["ok"]:
                await send({
                    "type": "ai_mode_done", "answer": ai_mode_result["answer"], "citations": ai_mode_result["citations"],
                })
            else:
                await send({"type": "ai_mode_error", "error": ai_mode_result["error"]})

        await websocket.close()
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
