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
    # Per-role kill switch for a Thinking model's <think> chain-of-thought (sent as
    # chat_template_kwargs.enable_thinking - vLLM's standard toggle for Qwen3's hybrid
    # thinking mode; DeepInfra passes the same field through for the Qwen3 models it
    # hosts). Independent per role because slm and synthesis have different reasons to
    # want reasoning off: slm's extract_intent only ever consumes the final JSON, so
    # reasoning is pure latency/token cost there once you don't need to read the trace;
    # synthesis's answer quality may depend on it more. Defaults to True (today's
    # behavior, unchanged) - flip off to compare latency/token cost/output quality
    # with vs without reasoning for a given role, or if a role's reasoning trace turns
    # out to add cost without changing the final answer often enough to justify it.
    slm_reasoning_enabled: bool = True
    synthesis_reasoning_enabled: bool = True


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


def build_role_reasoning_map(settings: GatewaySettings) -> dict[str, bool]:
    return {
        "slm": settings.slm_reasoning_enabled,
        "synthesis": settings.synthesis_reasoning_enabled,
    }
