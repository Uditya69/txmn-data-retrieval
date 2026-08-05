from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router

app = FastAPI(title="retrieval-api")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)
app.include_router(router)
app.include_router(documents_router)
