import asyncio
import logging

from fastapi import APIRouter, WebSocket
from langfuse import get_client

from agents.pipeline import run_agentic_search
from auth.config import get_auth_settings
from auth.security import decode_access_token
from chat.config import get_chat_settings
from chat.db import get_conversations_collection, get_mongo_client as get_chat_mongo_client
from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_personas_collection
from persona.prompt import render_persona_context
from persona.repository import get_persona
from semantic_cache.config import get_semantic_cache_settings
from semantic_cache.db import get_semantic_cache_collection, get_mongo_client as get_cache_mongo_client
from semantic_cache.repository import lookup as cache_lookup, write as cache_write
from retrieval_api.ai_mode.chat_signal import record_conversation_turn
from retrieval_api.ai_mode.persona_signal import record_persona_signal
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.search import run_instant
from retrieval_api.ai_mode.pipeline import run_ai_mode

router = APIRouter()

logger = logging.getLogger(__name__)

# Holds strong references to fire-and-forget background tasks (e.g. the
# persona-signal write below) - asyncio's event loop only keeps a weak
# reference to a task, so an unreferenced task can be garbage-collected
# mid-execution. Each task removes itself from this set via its done callback.
_background_tasks: set[asyncio.Task] = set()

# Mirrors the frontend's `titleFromQuestion` (packages/web/src/App.tsx) so a
# persisted conversation's title doesn't visibly change (get longer) the
# moment `remoteConversations.refresh()` swaps the locally-generated title
# out for the server's stored one.
_TITLE_MAX_LEN = 48


def _title_from_query(query: str) -> str:
    return f"{query[:_TITLE_MAX_LEN]}…" if len(query) > _TITLE_MAX_LEN else query


def get_gateway_client(settings) -> GatewayClient:
    return GatewayClient(base_url=settings.gateway_url)


def _resolve_user_id(access_token: str | None) -> str | None:
    if not access_token:
        return None
    user_id = decode_access_token(access_token, get_auth_settings())
    if user_id is None:
        # A token WAS sent but didn't decode - most commonly an expired token
        # (JWT_EXPIRY_MINUTES, currently 60) reused past its lifetime, since the
        # web client never checks expiry client-side and just keeps resending
        # whatever's in localStorage. Falls through to guest behavior silently
        # (personas.py's caller treats user_id=None exactly like no token was
        # sent at all) - this log line is the only visible sign that happened,
        # so a "persona stopped updating" report can be told apart from "user
        # was never logged in" instead of looking identical.
        logger.warning("access_token present but failed to decode (likely expired) - proceeding as guest")
    return user_id


async def _emit_trace_step(send, step: str, data: dict) -> None:
    """Swallows any exception from `send` (e.g. the client disconnected
    mid-stream) - a dead trace channel must never fail the AI Mode
    pipeline or its final answer."""
    try:
        await send({"type": "ai_mode_trace", "step": step, "data": data})
    except Exception as exc:
        logger.warning("trace step %r dropped: %s", step, exc)


async def _safe_cache_write(collection, mode: str, query: str, query_embedding: list[float], result: dict) -> None:
    """Swallows any exception from `cache_write` - this runs as a
    fire-and-forget background task (see the persona-signal write's
    identical pattern in record_persona_signal), so an unguarded failure
    (e.g. an unreachable Atlas cluster) would otherwise only surface via
    asyncio's noisy default unretrieved-task-exception handler instead of
    this module's own logging."""
    try:
        await cache_write(collection, mode, query, query_embedding, result)
    except Exception:
        logger.warning("semantic cache write failed for mode %r", mode, exc_info=True)


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"
    trace = message.get("trace", False)
    access_token = message.get("access_token")
    user_id = _resolve_user_id(access_token)
    conversation_id = message.get("conversation_id")

    settings = get_settings()
    # Client can independently ask for Instant's RRF fusion and/or cross-encoder rerank.
    # instant_mode_rerank_enabled (env, default true) is a server-side kill switch on the
    # cross-encoder call specifically (the costly/fragile external DeepInfra call) - false
    # forces it off regardless of what the client requested. RRF is cheap local rank math
    # with no external dependency, so it isn't gated by that switch. Same kill-switch
    # pattern as AI Mode's rerank flag.
    rrf = message.get("rrf", False)
    rerank = message.get("rerank", False) and settings.instant_mode_rerank_enabled
    auto_route = message.get("auto_route", False) and settings.instant_mode_auto_route_enabled
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        logger.exception("Milvus connection failed; proceeding without Milvus for this request")
        milvus_client = None

    personas_collection = None
    persona_context = ""
    if user_id is not None:
        try:
            persona_settings = get_persona_settings()
            mongo_client = get_mongo_client(persona_settings)
            personas_collection = get_personas_collection(mongo_client, persona_settings)
            persona = await get_persona(personas_collection, user_id)
            persona_context = render_persona_context(persona)
        except Exception:
            # A down/unreachable persona store must never crash the request -
            # mirrors the milvus_client resilience pattern above. Degrade to
            # no persona context (as if the user were a guest for this request)
            # rather than failing the whole /ws/search round-trip.
            logger.exception("Persona lookup failed for user %r; proceeding without persona context", user_id)
            personas_collection = None
            persona_context = ""

    # Semantic cache: an unreachable/misconfigured Atlas cluster must never
    # crash the request - resolving settings/client/collection and the
    # subsequent embed + lookup calls are all best-effort and fail open to a
    # normal (uncached) pipeline run, matching the milvus_client/persona
    # resilience pattern above.
    cache_collection = None
    query_embedding = None
    instant_cache_hit = None
    ai_mode_cache_hit = None
    # Each (auto_route, rrf, rerank) combination produces different result content - a
    # separate cache key per combination, same as the single "instant_rerank" key before
    # rrf/rerank were split, now extended with a third toggle.
    instant_cache_key = (
        f"instant_auto_route_{auto_route}_rrf_{rrf}_rerank_{rerank}"
    )
    try:
        cache_settings = get_semantic_cache_settings()
        if cache_settings.semantic_cache_enabled:
            cache_mongo_client = get_cache_mongo_client(cache_settings)
            cache_collection = get_semantic_cache_collection(cache_mongo_client, cache_settings)
    except Exception:
        logger.exception("Semantic cache setup failed; proceeding without cache")
        cache_collection = None

    if cache_collection is not None:
        try:
            query_embedding = await gateway.embed(role="query_embed", text=query)
        except Exception:
            logger.exception("Query embedding for semantic cache failed; proceeding without cache")
            query_embedding = None

    if query_embedding is not None:
        if mode in ("instant", "both"):
            try:
                instant_cache_hit = await cache_lookup(
                    cache_collection, instant_cache_key, query_embedding,
                    cache_settings.semantic_cache_threshold,
                )
            except Exception:
                logger.exception("Semantic cache lookup failed for instant mode; proceeding without cache")
        if mode in ("ai_mode", "both"):
            try:
                ai_mode_cache_hit = await cache_lookup(
                    cache_collection, "ai_mode", query_embedding,
                    cache_settings.semantic_cache_threshold,
                )
            except Exception:
                logger.exception("Semantic cache lookup failed for ai_mode; proceeding without cache")

    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await _emit_trace_step(send, step, data)

    if access_token and user_id is None:
        # A token was sent but didn't decode (see _resolve_user_id's log line) -
        # surface it to the client instead of silently proceeding as guest, so a
        # stale localStorage token (most commonly expired - JWT_EXPIRY_MINUTES)
        # doesn't go unnoticed indefinitely.
        await send({"type": "session_expired"})

    langfuse = get_client()
    try:
        with langfuse.start_as_current_observation(
            as_type="span", name="ws-search", input={"query": query, "mode": mode},
        ) as root_span:
            instant_task = (
                asyncio.create_task(
                    run_instant(
                        gateway, es_client, milvus_client, query,
                        on_step=emit_trace_step if trace else None, rrf=rrf, rerank=rerank,
                        auto_route=auto_route,
                    )
                )
                if mode in ("instant", "both") and instant_cache_hit is None else None
            )
            ai_mode_task = (
                asyncio.create_task(
                    run_ai_mode(
                        gateway, es_client, milvus_client, query,
                        on_step=emit_trace_step if trace else None,
                        persona_context=persona_context,
                    )
                )
                if mode in ("ai_mode", "both") and ai_mode_cache_hit is None else None
            )

            # Root observation output: kept to what a reviewer needs at a
            # glance (the answer), per-branch errors go to metadata instead
            # of duplicating the full nested result payloads.
            output: dict = {}

            if instant_cache_hit is not None or instant_task is not None:
                if instant_cache_hit is not None:
                    instant_result = instant_cache_hit
                else:
                    instant_result = await instant_task
                    if (
                        cache_collection is not None
                        and instant_result["es_error"] is None
                        and instant_result["milvus_error"] is None
                    ):
                        write_task = asyncio.create_task(
                            _safe_cache_write(
                                cache_collection, instant_cache_key, query, query_embedding, instant_result,
                            )
                        )
                        _background_tasks.add(write_task)
                        write_task.add_done_callback(_background_tasks.discard)
                output["instant_ok"] = instant_result["es_error"] is None and instant_result["milvus_error"] is None
                root_span.update(metadata={
                    "instant_es_error": instant_result["es_error"] or "",
                    "instant_milvus_error": instant_result["milvus_error"] or "",
                })
                await send({"type": "instant_result", **instant_result})

            if ai_mode_cache_hit is not None or ai_mode_task is not None:
                if ai_mode_cache_hit is not None:
                    ai_mode_result = ai_mode_cache_hit
                else:
                    ai_mode_result = await ai_mode_task
                    if cache_collection is not None and ai_mode_result["ok"]:
                        write_task = asyncio.create_task(
                            _safe_cache_write(cache_collection, "ai_mode", query, query_embedding, ai_mode_result)
                        )
                        _background_tasks.add(write_task)
                        write_task.add_done_callback(_background_tasks.discard)

                if ai_mode_result["ok"]:
                    output["answer"] = ai_mode_result["answer"]
                    ai_mode_message = {
                        "type": "ai_mode_done", "answer": ai_mode_result["answer"], "citations": ai_mode_result["citations"],
                    }
                    if ai_mode_result.get("reasoning"):
                        ai_mode_message["reasoning"] = ai_mode_result["reasoning"]
                    await send(ai_mode_message)

                    if user_id is not None and personas_collection is not None:
                        task = asyncio.create_task(
                            record_persona_signal(
                                personas_collection, gateway, user_id, query,
                                categories=ai_mode_result.get("intent", []),
                            )
                        )
                        _background_tasks.add(task)
                        task.add_done_callback(_background_tasks.discard)

                    if user_id is not None and conversation_id is not None:
                        try:
                            chat_settings = get_chat_settings()
                            chat_mongo_client = get_chat_mongo_client(chat_settings)
                            conversations_collection = get_conversations_collection(chat_mongo_client, chat_settings)
                            chat_task = asyncio.create_task(
                                record_conversation_turn(
                                    conversations_collection, conversation_id, user_id, _title_from_query(query),
                                    [
                                        {"role": "user", "text": query},
                                        {"role": "assistant", "text": ai_mode_result["answer"]},
                                    ],
                                )
                            )
                            _background_tasks.add(chat_task)
                            chat_task.add_done_callback(_background_tasks.discard)
                        except Exception:
                            # A down/unreachable chat store must never crash the request -
                            # mirrors the persona lookup's resilience pattern above.
                            logger.exception("Failed to schedule conversation write for user %r", user_id)
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
    except Exception as exc:
        # A dead/closed connection (client gone before the final send) used to blow
        # out of this function raw - uvicorn logged an unhandled traceback and the
        # pipeline's own result was never recorded anywhere as "computed but
        # undelivered". Log it explicitly so that failure mode is diagnosable
        # instead of looking like nothing happened.
        logger.warning("ws /ws/search failed for query %r: %s", query, exc, exc_info=True)
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
        logger.exception("Milvus connection failed; proceeding without Milvus for this request")
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
