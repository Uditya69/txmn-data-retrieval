FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --package model-gateway
CMD ["uv", "run", "--package", "model-gateway", "uvicorn", "model_gateway.main:app", "--host", "0.0.0.0", "--port", "8001"]
