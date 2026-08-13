from model_gateway.config import GatewaySettings, build_role_model_map, build_role_provider_map


def test_agent_chat_role_maps_to_its_own_model_and_deepinfra():
    settings = GatewaySettings(
        deepinfra_api_key="k",
        deepinfra_chat_model_agent="agent-model",
        deepinfra_rerank_model="rerank-model",
        voyage_api_key="k",
        voyage_embed_model="embed-model",
        local_api_key="k",
        local_base_url="http://localhost:8000/v1",
        local_chat_model_slm="slm-model",
        local_chat_model_synthesis="synthesis-model",
    )

    model_map = build_role_model_map(settings)
    provider_map = build_role_provider_map()

    assert model_map["agent_chat"] == "agent-model"
    assert provider_map["agent_chat"] == "deepinfra"
