from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model_gateway.adapters.deepinfra import DeepInfraAdapter
from model_gateway.adapters.voyage import VoyageAdapter
from model_gateway.config import build_role_model_map, build_role_provider_map, get_gateway_settings

router = APIRouter()

ROLE_MODEL_MAP: dict[str, str] = build_role_model_map(get_gateway_settings())
ROLE_PROVIDER_MAP: dict[str, str] = build_role_provider_map()


def get_adapter(provider: str):
    settings = get_gateway_settings()
    if provider == "voyage":
        return VoyageAdapter(api_key=settings.voyage_api_key)
    return DeepInfraAdapter(api_key=settings.deepinfra_api_key)


def _resolve(role: str) -> tuple[str, str]:
    if role not in ROLE_MODEL_MAP or role not in ROLE_PROVIDER_MAP:
        raise HTTPException(status_code=400, detail=f"unknown role: {role}")
    return ROLE_MODEL_MAP[role], ROLE_PROVIDER_MAP[role]


class ChatRequest(BaseModel):
    role: str
    messages: list[dict]


class EmbedRequest(BaseModel):
    role: str
    text: str


class RerankRequest(BaseModel):
    role: str
    query: str
    documents: list[str]


@router.post("/v1/chat")
async def chat(req: ChatRequest):
    model, provider = _resolve(req.role)
    content = await get_adapter(provider).chat(model, req.messages)
    return {"content": content}


@router.post("/v1/embed")
async def embed(req: EmbedRequest):
    model, provider = _resolve(req.role)
    embedding = await get_adapter(provider).embed(model, req.text)
    return {"embedding": embedding}


@router.post("/v1/rerank")
async def rerank(req: RerankRequest):
    model, provider = _resolve(req.role)
    scores = await get_adapter(provider).rerank(model, req.query, req.documents)
    return {"scores": scores}
