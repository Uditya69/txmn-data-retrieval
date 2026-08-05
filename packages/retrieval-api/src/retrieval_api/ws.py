import asyncio

from fastapi import APIRouter, WebSocket
from langfuse import get_client

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
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"

    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        milvus_client = None

    langfuse = get_client()
    try:
        with langfuse.start_as_current_observation(
            as_type="span", name="ws-search", input={"query": query, "mode": mode},
        ) as root_span:
            instant_task = (
                asyncio.create_task(run_instant(gateway, es_client, milvus_client, query))
                if mode in ("instant", "both") else None
            )
            ai_mode_task = (
                asyncio.create_task(run_ai_mode(gateway, es_client, milvus_client, query))
                if mode in ("ai_mode", "both") else None
            )

            # Root observation output: kept to what a reviewer needs at a
            # glance (the answer), per-branch errors go to metadata instead
            # of duplicating the full nested result payloads.
            output: dict = {}

            if instant_task is not None:
                instant_result = await instant_task
                output["instant_ok"] = instant_result["es_error"] is None and instant_result["milvus_error"] is None
                root_span.update(metadata={
                    "instant_es_error": instant_result["es_error"] or "",
                    "instant_milvus_error": instant_result["milvus_error"] or "",
                })
                await websocket.send_json({"type": "instant_result", **instant_result})

            if ai_mode_task is not None:
                ai_mode_result = await ai_mode_task
                if ai_mode_result["ok"]:
                    output["answer"] = ai_mode_result["answer"]
                    await websocket.send_json({
                        "type": "ai_mode_done", "answer": ai_mode_result["answer"], "citations": ai_mode_result["citations"],
                    })
                else:
                    output["ai_mode_error"] = ai_mode_result["error"]
                    await websocket.send_json({"type": "ai_mode_error", "error": ai_mode_result["error"]})

            root_span.update(output=output)
            # Distributed trace: model-gateway's generations auto-export on
            # their own periodic schedule and can be ingested before this
            # span's end-of-request flush, so the trace-level io "mirrors
            # the root observation" default is a race - pin it explicitly.
            root_span.set_trace_io(input={"query": query, "mode": mode}, output=output)

        await websocket.close()
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
        langfuse.flush()
