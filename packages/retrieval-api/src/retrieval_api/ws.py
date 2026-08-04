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


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]

    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        milvus_client = None

    try:
        instant_task = asyncio.create_task(run_instant(gateway, es_client, milvus_client, query))
        ai_mode_task = asyncio.create_task(run_ai_mode(gateway, es_client, milvus_client, query))

        instant_result = await instant_task
        await websocket.send_json({"type": "instant_result", **instant_result})

        ai_mode_result = await ai_mode_task
        if ai_mode_result["ok"]:
            await websocket.send_json({
                "type": "ai_mode_done", "answer": ai_mode_result["answer"], "citations": ai_mode_result["citations"],
            })
        else:
            await websocket.send_json({"type": "ai_mode_error", "error": ai_mode_result["error"]})

        await websocket.close()
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
