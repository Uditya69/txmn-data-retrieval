FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY evals ./evals
RUN uv sync --package retrieval-api
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN uv run --package retrieval-api python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
CMD ["uv", "run", "--package", "retrieval-api", "uvicorn", "retrieval_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
