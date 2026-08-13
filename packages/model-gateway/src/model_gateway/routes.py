from fastapi import APIRouter, HTTPException, Request
from langfuse import get_client
from pydantic import BaseModel

from model_gateway.adapters.deepinfra import DeepInfraAdapter
from model_gateway.adapters.voyage import VoyageAdapter
from model_gateway.config import build_role_model_map, build_role_provider_map, get_gateway_settings

router = APIRouter()

ROLE_MODEL_MAP: dict[str, str] = build_role_model_map(get_gateway_settings())
ROLE_PROVIDER_MAP: dict[str, str] = build_role_provider_map()

# Matches the headers retrieval_api.gateway_client sets so this generation
# nests under the caller's trace instead of starting a new one.
_TRACE_ID_HEADER = "x-langfuse-trace-id"
_PARENT_SPAN_ID_HEADER = "x-langfuse-parent-observation-id"


def _trace_context_from_headers(request: Request) -> dict[str, str] | None:
    trace_id = request.headers.get(_TRACE_ID_HEADER)
    parent_span_id = request.headers.get(_PARENT_SPAN_ID_HEADER)
    if not trace_id or not parent_span_id:
        return None
    return {"trace_id": trace_id, "parent_span_id": parent_span_id}


def get_adapter(provider: str):
    settings = get_gateway_settings()
    if provider == "voyage":
        return VoyageAdapter(api_key=settings.voyage_api_key)
    return DeepInfraAdapter(api_key=settings.deepinfra_api_key)


def _resolve(role: str) -> tuple[str, str]:
    if role not in ROLE_MODEL_MAP or role not in ROLE_PROVIDER_MAP:
        raise HTTPException(status_code=400, detail=f"unknown role: {role}")
    return ROLE_MODEL_MAP[role], ROLE_PROVIDER_MAP[role]


@router.get("/v1/models/{role}")
async def get_model(role: str):
    if role not in ROLE_MODEL_MAP:
        raise HTTPException(status_code=404, detail=f"unknown role: {role}")
    return {"role": role, "model": ROLE_MODEL_MAP[role]}


class ChatRequest(BaseModel):
    role: str
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | None = None
    model: str | None = None
    response_format: dict | None = None


class EmbedRequest(BaseModel):
    role: str
    text: str
    model: str | None = None


class RerankRequest(BaseModel):
    role: str
    query: str
    documents: list[str]
    model: str | None = None


@router.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    default_model, provider = _resolve(req.role)
    model = req.model or default_model
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=f"chat:{req.role}",
        model=model,
        input=req.messages,
        metadata={"provider": provider, "has_tools": bool(req.tools)},
        trace_context=_trace_context_from_headers(request),
    ) as generation:
        content, usage_details, reasoning, tool_calls = await get_adapter(provider).chat(
            model, req.messages, req.tools, req.tool_choice, req.response_format,
        )
        generation.update(output=content if content is not None else {"tool_calls": tool_calls}, usage_details=usage_details)
        if reasoning:
            generation.update(metadata={"reasoning": reasoning})
    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls}


@router.post("/v1/embed")
async def embed(req: EmbedRequest, request: Request):
    model, provider = _resolve(req.role)
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="embedding",
        name=f"embed:{req.role}",
        model=model,
        input=req.text,
        metadata={"provider": provider},
        trace_context=_trace_context_from_headers(request),
    ) as generation:
        embedding, usage_details = await get_adapter(provider).embed(model, req.text)
        generation.update(output={"dimensions": len(embedding)}, usage_details=usage_details)
    return {"embedding": embedding}


@router.post("/v1/rerank")
async def rerank(req: RerankRequest, request: Request):
    # DeepInfra's rerank endpoint 422s on an empty documents list ("the number of queries and
    # documents must be the same") - nothing to rerank anyway, so short-circuit before it ever
    # reaches the adapter, rather than let that surface as an unhandled 500 to the caller.
    if not req.documents:
        return {"scores": []}
    default_model, provider = _resolve(req.role)
    model = req.model or default_model
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=f"rerank:{req.role}",
        model=model,
        input={"query": req.query, "documents": req.documents},
        metadata={"provider": provider, "num_documents": len(req.documents)},
        trace_context=_trace_context_from_headers(request),
    ) as generation:
        scores = await get_adapter(provider).rerank(model, req.query, req.documents)
        generation.update(output=scores)
    return {"scores": scores}
