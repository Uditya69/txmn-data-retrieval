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
    # Kill switch for Instant mode's opt-in reranker. Instant's rerank is normally a
    # per-request choice (the `rerank` field on the /ws/search message) - this flag sits
    # above that and forces it off server-wide regardless of what the client asks for,
    # same pattern as ai_mode_rerank_enabled above. False here means Instant never
    # reranks, even if a client sends rerank: true.
    instant_mode_rerank_enabled: bool = True
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
