from fastapi import FastAPI

from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router

app = FastAPI(title="retrieval-api")
app.include_router(router)
app.include_router(documents_router)
