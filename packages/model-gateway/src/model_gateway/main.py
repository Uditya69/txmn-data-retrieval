from fastapi import FastAPI

from model_gateway.routes import router

app = FastAPI(title="model-gateway")
app.include_router(router)
