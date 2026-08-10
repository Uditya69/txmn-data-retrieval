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
    intent_rrf_weighting_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
