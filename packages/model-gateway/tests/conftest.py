import os

os.environ.setdefault("DEEPINFRA_API_KEY", "test-deepinfra-key")
os.environ.setdefault("DEEPINFRA_CHAT_MODEL_SLM", "test-deepinfra-slm-model")
os.environ.setdefault("DEEPINFRA_CHAT_MODEL_SYNTHESIS", "test-deepinfra-synthesis-model")
os.environ.setdefault("DEEPINFRA_RERANK_MODEL", "test-rerank-model")
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage-key")
os.environ.setdefault("VOYAGE_EMBED_MODEL", "test-voyage-embed-model")
os.environ.setdefault("LOCAL_API_KEY", "test-local-key")
os.environ.setdefault("LOCAL_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("LOCAL_CHAT_MODEL_SLM", "test-local-slm-model")
os.environ.setdefault("LOCAL_CHAT_MODEL_SYNTHESIS", "test-local-synthesis-model")
