from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    milvus_uri: str
    milvus_token: str
    milvus_db: str = "aic"
    es_uri: str
    es_username: str | None = None
    es_password: str | None = None
    es_index: str = "taxmann_caselaw"
    es_verify_certs: bool = True
    gateway_url: str
    # A/B eval switch: disables AI Mode's rerank-and-prefetch's DeepInfra reranker call
    # when False, falling back to top-N-by-rrf_score straight out of RRF fusion. Lets us
    # compare AI Mode quality with/without the reranker step without a code change.
    ai_mode_rerank_enabled: bool = True
    # Kill switch for Instant mode's classifier-driven automatic backend routing
    # (the `auto_route` field on the /ws/search message) - same pattern as
    # ai_mode_rerank_enabled above. False here forces every request onto
    # today's always-both-backends behavior regardless of what the client asks for.
    instant_mode_auto_route_enabled: bool = True
    # Kill switch for native Milvus sparse (BM25 Function) search - shared by both AI Mode's
    # retrieve() and Instant mode's _run_milvus, off by default. AI Mode's ES sparse-fallback
    # (sparse_fallback_search, for collections with no sparse_vector at all) is a separate
    # mechanism and is NOT gated by this - it keeps running regardless. Env-only: no UI toggle
    # exists for this one.
    milvus_sparse_enabled: bool = False
    # Gates the local-only admin eval-runner UI (retrieval_api/admin_eval/) - unset
    # (the default) disables that feature entirely, so no deployment needs to think
    # about it unless it opts in.
    admin_secret: str | None = None
    # Dev-only visibility switch: when True, AI Mode's synthesis-step model reasoning
    # (already captured via gateway_client.chat_with_reasoning / synthesize.py, and
    # always logged to Langfuse regardless of this flag) is also included in the
    # ai_mode_done websocket message sent to the client. False in production so raw
    # chain-of-thought never reaches an end user; flip true in .env for local/dev only.
    expose_reasoning: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
