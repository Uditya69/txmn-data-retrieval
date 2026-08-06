import asyncio
import logging

from fastapi import APIRouter, WebSocket
from langfuse import get_client

from agents.pipeline import run_agentic_search
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

    langfuse = get_client()
    try:
        with langfuse.start_as_current_observation(
            as_type="span", name="ws-search", input={"query": query, "mode": mode},
        ) as root_span:
            instant_task = (
                asyncio.create_task(
                    run_instant(
                        gateway, es_client, milvus_client, query,
                        on_step=emit_trace_step if trace else None,
                    )
                )
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
                await send({"type": "instant_result", **instant_result})

            if ai_mode_task is not None:
                ai_mode_result = await ai_mode_task
                if ai_mode_result["ok"]:
                    output["answer"] = ai_mode_result["answer"]
                    ai_mode_message = {
                        "type": "ai_mode_done", "answer": ai_mode_result["answer"], "citations": ai_mode_result["citations"],
                    }
                    if ai_mode_result.get("reasoning"):
                        ai_mode_message["reasoning"] = ai_mode_result["reasoning"]
                    await send(ai_mode_message)
                else:
                    output["ai_mode_error"] = ai_mode_result["error"]
                    await send({"type": "ai_mode_error", "error": ai_mode_result["error"]})

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


@router.websocket("/ws/agent")
async def agent_search(websocket: WebSocket):
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

    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await _emit_trace_step(send, step, data)

    try:
        result = await run_agentic_search(gateway, es_client, milvus_client, query, on_step=emit_trace_step)
        if result["ok"]:
            await send({"type": "agent_done", "answer": result["answer"], "doc_ids": result["doc_ids"]})
        else:
            await send({"type": "agent_unverifiable", "invalid_doc_ids": result["invalid_doc_ids"]})
        await websocket.close()
    except Exception as exc:
        await send({"type": "agent_error", "error": f"{type(exc).__name__}: {exc}"})
        await websocket.close()
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
