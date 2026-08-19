from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SemanticCacheSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongo_uri: str
    mongo_db: str
    semantic_cache_threshold: float = 0.95


@lru_cache
def get_semantic_cache_settings() -> SemanticCacheSettings:
    return SemanticCacheSettings()
