from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepinfra_api_key: str
    deepinfra_chat_model_slm: str
    deepinfra_chat_model_synthesis: str
    deepinfra_rerank_model: str
    voyage_api_key: str
    voyage_embed_model: str
    local_api_key: str
    local_base_url: str
    local_chat_model_slm: str
    local_chat_model_synthesis: str
    # Switches slm+synthesis between the self-hosted adapter and DeepInfra.
    # Both roles moved to "local" together in one commit, so one switch
    # flips both back in lockstep - see the design note in .env.example.
    chat_provider: str = "deepinfra"


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


def build_role_model_map(settings: GatewaySettings) -> dict[str, str]:
    slm_model = settings.deepinfra_chat_model_slm if settings.chat_provider == "deepinfra" else settings.local_chat_model_slm
    synthesis_model = (
        settings.deepinfra_chat_model_synthesis if settings.chat_provider == "deepinfra" else settings.local_chat_model_synthesis
    )
    return {
        "slm": slm_model,
        "synthesis": synthesis_model,
        "query_embed": settings.voyage_embed_model,
        "reranker": settings.deepinfra_rerank_model,
    }


def build_role_provider_map(settings: GatewaySettings) -> dict[str, str]:
    return {
        "slm": settings.chat_provider,
        "synthesis": settings.chat_provider,
        "reranker": "deepinfra",
        "query_embed": "voyage",
    }
