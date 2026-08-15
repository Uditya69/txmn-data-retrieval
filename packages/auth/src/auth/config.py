from functools import lru_cache

from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    mongo_uri: str
    mongo_db: str
    jwt_secret: str
    jwt_expiry_minutes: int


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
