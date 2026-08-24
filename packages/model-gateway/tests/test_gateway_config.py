from model_gateway.config import GatewaySettings, build_role_model_map, build_role_provider_map


def _settings(**overrides):
    defaults = dict(
        deepinfra_api_key="k",
        deepinfra_chat_model_slm="deepinfra-slm-model",
        deepinfra_chat_model_synthesis="deepinfra-synthesis-model",
        deepinfra_rerank_model="rerank-model",
        voyage_api_key="k",
        voyage_embed_model="embed-model",
        local_api_key="k",
        local_base_url="http://localhost:8000/v1",
        local_chat_model_slm="local-slm-model",
        local_chat_model_synthesis="local-synthesis-model",
    )
    defaults.update(overrides)
    return GatewaySettings(**defaults)


def test_chat_provider_defaults_to_deepinfra_for_slm_and_synthesis():
    settings = _settings()

    model_map = build_role_model_map(settings)
    provider_map = build_role_provider_map(settings)

    assert provider_map["slm"] == "deepinfra"
    assert provider_map["synthesis"] == "deepinfra"
    assert model_map["slm"] == "deepinfra-slm-model"
    assert model_map["synthesis"] == "deepinfra-synthesis-model"


def test_chat_provider_local_switches_slm_and_synthesis_to_local_models():
    settings = _settings(chat_provider="local")

    model_map = build_role_model_map(settings)
    provider_map = build_role_provider_map(settings)

    assert provider_map["slm"] == "local"
    assert provider_map["synthesis"] == "local"
    assert model_map["slm"] == "local-slm-model"
    assert model_map["synthesis"] == "local-synthesis-model"


def test_chat_provider_never_affects_query_embed_or_reranker():
    settings = _settings(chat_provider="local")

    provider_map = build_role_provider_map(settings)

    assert provider_map["query_embed"] == "voyage"
    assert provider_map["reranker"] == "deepinfra"
