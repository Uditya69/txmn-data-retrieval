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
    # Gates the local-only admin eval-runner UI (retrieval_api/admin_eval/) - unset
    # (the default) disables that feature entirely, so no deployment needs to think
    # about it unless it opts in.
    admin_secret: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
