from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class PersonaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongo_uri: str
    mongo_db: str


@lru_cache
def get_persona_settings() -> PersonaSettings:
    return PersonaSettings()
