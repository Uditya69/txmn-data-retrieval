from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepinfra_api_key: str
    deepinfra_chat_model_agent: str
    deepinfra_rerank_model: str
    voyage_api_key: str
    voyage_embed_model: str
    local_api_key: str
    local_base_url: str
    local_chat_model_slm: str
    local_chat_model_synthesis: str


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


def build_role_model_map(settings: GatewaySettings) -> dict[str, str]:
    return {
        "slm": settings.local_chat_model_slm,
        "synthesis": settings.local_chat_model_synthesis,
        "agent_chat": settings.deepinfra_chat_model_agent,
        "query_embed": settings.voyage_embed_model,
        "reranker": settings.deepinfra_rerank_model,
    }


def build_role_provider_map() -> dict[str, str]:
    return {
        "slm": "local",
        "synthesis": "local",
        "agent_chat": "deepinfra",
        "reranker": "deepinfra",
        "query_embed": "voyage",
    }
