from langfuse import get_client

from model_gateway.adapters.deepinfra import DeepInfraAdapter
from model_gateway.adapters.local import LocalAdapter
from model_gateway.adapters.local_rerank import LocalRerankAdapter
from model_gateway.adapters.voyage import VoyageAdapter
from model_gateway.config import (
    build_role_model_map,
    build_role_provider_map,
    build_role_reasoning_map,
    get_gateway_settings,
)


def _get_adapter(provider: str, settings):
    if provider == "voyage":
        return VoyageAdapter(api_key=settings.voyage_api_key)
    if provider == "local":
        return LocalAdapter(base_url=settings.local_base_url, api_key=settings.local_api_key)
    if provider == "local_rerank":
        return LocalRerankAdapter(base_url=settings.local_rerank_base_url, api_key=settings.local_api_key)
    return DeepInfraAdapter(api_key=settings.deepinfra_api_key)


class UnknownRoleError(ValueError):
    pass


class GatewayClient:
    """In-process replacement for the old model-gateway HTTP service. Same role-based
    interface (chat/embed/rerank against a "role" like "slm"/"reranker"/"query_embed"),
    but resolves the provider/model/adapter directly rather than over an HTTP hop -
    retrieval-api and model-gateway now share one process, so there's no separate
    service boundary (and no trace-header forwarding) left to bridge.
    """

    def __init__(self, trace_enabled: bool = True, reasoning_overrides: dict[str, bool] | None = None):
        self._trace_enabled = trace_enabled
        settings = get_gateway_settings()
        self._settings = settings
        self._role_model_map = build_role_model_map(settings)
        self._role_provider_map = build_role_provider_map(settings)
        self._role_reasoning_map = build_role_reasoning_map(settings)
        # Callers that resolve slm/synthesis reasoning through a Mongo-backed
        # feature-flag layer (see retrieval_api.admin.feature_flags) pass their
        # merged env/Mongo/default values here rather than model-gateway reading
        # Mongo itself - model-gateway has no Mongo dependency of its own.
        if reasoning_overrides:
            self._role_reasoning_map = {**self._role_reasoning_map, **reasoning_overrides}

    def _resolve(self, role: str) -> tuple[str, str]:
        if role not in self._role_model_map or role not in self._role_provider_map:
            raise UnknownRoleError(f"unknown role: {role}")
        return self._role_model_map[role], self._role_provider_map[role]

    async def get_model(self, role: str) -> str:
        if role not in self._role_model_map:
            raise UnknownRoleError(f"unknown role: {role}")
        return self._role_model_map[role]

    async def chat(
        self, role: str, messages: list[dict], model: str | None = None,
        response_format: dict | None = None, temperature: float | None = None,
    ) -> str:
        content, _reasoning = await self.chat_with_reasoning(
            role, messages, model=model, response_format=response_format, temperature=temperature,
        )
        return content

    async def chat_with_reasoning(
        self, role: str, messages: list[dict], model: str | None = None,
        response_format: dict | None = None, temperature: float | None = None,
    ) -> tuple[str, str | None]:
        default_model, provider = self._resolve(role)
        resolved_model = model or default_model
        reasoning_enabled = self._role_reasoning_map.get(role, True)
        adapter = _get_adapter(provider, self._settings)

        if not self._trace_enabled:
            content, _usage, reasoning = await adapter.chat(
                resolved_model, messages, response_format, temperature,
                role=role, reasoning_enabled=reasoning_enabled,
            )
            return content, reasoning

        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=f"chat:{role}",
            model=resolved_model,
            input=messages,
            metadata={"provider": provider, "reasoning_enabled": reasoning_enabled},
        ) as generation:
            content, usage_details, reasoning = await adapter.chat(
                resolved_model, messages, response_format, temperature,
                role=role, reasoning_enabled=reasoning_enabled,
            )
            generation.update(output=content, usage_details=usage_details)
            if reasoning:
                generation.update(metadata={"reasoning": reasoning})
        return content, reasoning

    async def embed(self, role: str, text: str) -> list[float]:
        default_model, provider = self._resolve(role)
        adapter = _get_adapter(provider, self._settings)

        if not self._trace_enabled:
            embedding, _usage = await adapter.embed(default_model, text)
            return embedding

        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="embedding",
            name=f"embed:{role}",
            model=default_model,
            input=text,
            metadata={"provider": provider},
        ) as generation:
            embedding, usage_details = await adapter.embed(default_model, text)
            generation.update(output={"dimensions": len(embedding)}, usage_details=usage_details)
        return embedding

    async def rerank(
        self, role: str, query: str, documents: list[str], model: str | None = None,
        instruction: str | None = None,
    ) -> list[float]:
        # DeepInfra's rerank endpoint 422s on an empty documents list - nothing to rerank
        # anyway, so short-circuit before it ever reaches the adapter.
        if not documents:
            return []
        default_model, provider = self._resolve(role)
        resolved_model = model or default_model
        adapter = _get_adapter(provider, self._settings)

        if not self._trace_enabled:
            return await adapter.rerank(resolved_model, query, documents, instruction=instruction)

        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=f"rerank:{role}",
            model=resolved_model,
            input={"query": query, "documents": documents},
            metadata={"provider": provider, "num_documents": len(documents), "instruction": instruction},
        ) as generation:
            scores = await adapter.rerank(resolved_model, query, documents, instruction=instruction)
            generation.update(output=scores)
        return scores
