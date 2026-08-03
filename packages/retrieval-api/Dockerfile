FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --package retrieval-api
CMD ["uv", "run", "--package", "retrieval-api", "uvicorn", "retrieval_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
