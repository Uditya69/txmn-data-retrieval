import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.config import get_auth_settings
from auth.db import ensure_refresh_token_indexes, get_mongo_client, get_refresh_tokens_collection
from auth.router import router as auth_router
from chat.router import router as chat_router
from retrieval_api.admin_eval.router import router as admin_eval_router
from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router
from retrieval_api.query_analysis import router as query_analysis_router
from retrieval_api.intent_analysis import router as intent_analysis_router
from retrieval_api.ai_mode_analysis import router as ai_mode_analysis_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_index is idempotent - cheap no-op on every restart once the indexes
    # already exist. See auth/db.py::ensure_refresh_token_indexes for what/why.
    # A down/unreachable Mongo at startup must never crash the app (mirrors the
    # same degrade-don't-crash pattern ws.py already uses for persona/milvus) -
    # worst case the indexes just don't get (re-)created until a future restart
    # when Mongo is reachable again.
    try:
        settings = get_auth_settings()
        client = get_mongo_client(settings)
        await ensure_refresh_token_indexes(get_refresh_tokens_collection(client, settings))
    except Exception:
        logger.exception("Failed to ensure refresh-token indexes at startup - continuing without them")
    yield


app = FastAPI(title="retrieval-api", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)
app.include_router(router)
app.include_router(documents_router)
app.include_router(query_analysis_router)
app.include_router(intent_analysis_router)
app.include_router(ai_mode_analysis_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_eval_router)
