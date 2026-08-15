from functools import lru_cache

from pydantic_settings import BaseSettings


class PersonaSettings(BaseSettings):
    mongo_uri: str
    mongo_db: str


@lru_cache
def get_persona_settings() -> PersonaSettings:
    return PersonaSettings()
