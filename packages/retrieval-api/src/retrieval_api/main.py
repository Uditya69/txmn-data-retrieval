from fastapi import FastAPI

from retrieval_api.ws import router

app = FastAPI(title="retrieval-api")
app.include_router(router)
