import asyncio

from fastapi import APIRouter, WebSocket

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.search import run_instant
from retrieval_api.ai_mode.pipeline import run_ai_mode

router = APIRouter()


def get_gateway_client(settings) -> GatewayClient:
    return GatewayClient(base_url=settings.gateway_url)


async def _emit_trace_step(send, step: str, data: dict) -> None:
    """Swallows any exception from `send` (e.g. the client disconnected
    mid-stream) - a dead trace channel must never fail the AI Mode
    pipeline or its final answer."""
    try:
        await send({"type": "ai_mode_trace", "step": step, "data": data})
    except Exception:
        pass


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"

    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        milvus_client = None

    send_lock = asyncio.Lock()
    # Guards message ORDER on the wire (not thread-safety - send_lock already
    # covers that). Both tasks are created concurrently below so their real
    # work overlaps, but if instant is running at all, ai_mode's trace steps
    # must not reach the client before instant_result does. Without this,
    # asyncio's call_soon scheduling can let ai_mode's first on_step -> send
    # reach the websocket queue before instant's completion callback even
    # resumes this handler, delivering ai_mode_trace ahead of instant_result.
    instant_sent = asyncio.Event()
    if mode not in ("instant", "both"):
        instant_sent.set()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await instant_sent.wait()
        await _emit_trace_step(send, step, data)

    try:
        instant_task = (
            asyncio.create_task(run_instant(gateway, es_client, milvus_client, query))
            if mode in ("instant", "both") else None
        )
        ai_mode_task = (
            asyncio.create_task(run_ai_mode(gateway, es_client, milvus_client, query, on_step=emit_trace_step))
            if mode in ("ai_mode", "both") else None
        )

        if instant_task is not None:
            instant_result = await instant_task
            await send({"type": "instant_result", **instant_result})
        instant_sent.set()

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
