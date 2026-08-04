FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --package web
CMD ["uv", "run", "--package", "web", "streamlit", "run", "packages/web/src/web/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
