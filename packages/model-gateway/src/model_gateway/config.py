from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepinfra_api_key: str
    chat_model_slm: str
    chat_model_synthesis: str
    rerank_model: str
    voyage_api_key: str
    voyage_embed_model: str


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


def build_role_model_map(settings: GatewaySettings) -> dict[str, str]:
    return {
        "slm": settings.chat_model_slm,
        "synthesis": settings.chat_model_synthesis,
        "query_embed": settings.voyage_embed_model,
        "reranker": settings.rerank_model,
    }


def build_role_provider_map() -> dict[str, str]:
    return {
        "slm": "deepinfra",
        "synthesis": "deepinfra",
        "reranker": "deepinfra",
        "query_embed": "voyage",
    }
