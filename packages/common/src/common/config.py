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
    # Kill switch for Instant mode's classifier-driven automatic backend routing
    # (the `auto_route` field on the /ws/search message) - same pattern as
    # instant_mode_rerank_enabled above. False here forces every request onto
    # today's always-both-backends behavior regardless of what the client asks for.
    instant_mode_auto_route_enabled: bool = True
    # Gates the local-only admin eval-runner UI (retrieval_api/admin_eval/) - unset
    # (the default) disables that feature entirely, so no deployment needs to think
    # about it unless it opts in.
    admin_secret: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
