from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    # frozen=True: makes AuthSettings hashable (pydantic's default __eq__ sets
    # __hash__ = None unless frozen), which auth.db.get_mongo_client relies on
    # for its @lru_cache keyed by settings.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongo_uri: str
    mongo_db: str
    jwt_secret: str = Field(min_length=32)
    jwt_expiry_minutes: int
    refresh_token_expiry_days: int = 30


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
