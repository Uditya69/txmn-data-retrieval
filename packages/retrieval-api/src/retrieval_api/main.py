from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.router import router as auth_router
from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router
from retrieval_api.query_analysis import router as query_analysis_router
from retrieval_api.intent_analysis import router as intent_analysis_router
from retrieval_api.ai_mode_analysis import router as ai_mode_analysis_router

app = FastAPI(title="retrieval-api")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)
app.include_router(router)
app.include_router(documents_router)
app.include_router(query_analysis_router)
app.include_router(intent_analysis_router)
app.include_router(ai_mode_analysis_router)
app.include_router(auth_router)
