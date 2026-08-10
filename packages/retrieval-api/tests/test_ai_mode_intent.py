import json
from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.intent import extract_intent


@pytest.mark.asyncio
async def test_extract_intent_parses_json_object_response():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "BNS section 103 murder punishment",
        "intent": "provision_lookup",
        "filters": {"act": "BNS"},
    })

    result = await extract_intent(gateway, "IPC 302 punishment")

    assert result == {
        "rewritten_query": "IPC 302 punishment",
        "intent": "provision_lookup",
        "filters": {},
    }
    gateway.chat.assert_awaited_once()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_intent_requests_json_object_response_format():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_response_is_wrapped_in_markdown_fence():
    """json_object response_format should prevent this from a compliant model,
    but if a model still wraps its output, fall back rather than regex-guessing
    the JSON out of prose - that guesswork is exactly what schema mode replaces."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = "```json\n" + json.dumps({
        "rewritten_query": "capital gains set off against carried forward business losses",
        "intent": "conceptual",
        "filters": {},
    }) + "\n```"

    result = await extract_intent(gateway, "set off capital gains against brought forward business losses")

    assert result == {
        "rewritten_query": "set off capital gains against brought forward business losses",
        "intent": "unknown",
        "filters": {},
    }


@pytest.mark.asyncio
async def test_extract_intent_falls_back_to_plain_search_on_unparseable_response():
    """Covers SLM refusals too (e.g. Llama declining a named-party query) -
    AI Mode should degrade to plain semantic search, not fail outright."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = "I cannot provide case law for that person."

    result = await extract_intent(gateway, "some query")

    assert result == {"rewritten_query": "some query", "intent": "unknown", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_system_prompt_includes_schema_context_and_new_fields():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "q", "intent": "conceptual", "filters": {},
    })

    await extract_intent(gateway, "some query")

    system_message = gateway.chat.await_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "facts" in system_message["content"]
    assert "Supreme Court" in system_message["content"]
    assert '"section"' in system_message["content"]
    assert '"gte"' in system_message["content"]
    assert '"lte"' in system_message["content"]
    assert '"bench"' in system_message["content"]
    assert '"judge"' in system_message["content"]
    assert "citation_lookup" in system_message["content"]
    assert "provision_lookup" in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_emits_intent_step_when_on_step_given():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "original query normalized",
        "intent": "conceptual",
        "filters": {"act": "CGST Act"},
    })
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await extract_intent(gateway, "original query", on_step=on_step)

    assert result == {"rewritten_query": "original query normalized", "intent": "conceptual", "filters": {}}
    assert steps == [("intent", {
        "query": "original query",
        "rewritten_query": "original query normalized",
        "intent": "conceptual",
        "filters": {},
    })]


@pytest.mark.asyncio
async def test_extract_intent_skips_on_step_when_none():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    result = await extract_intent(gateway, "q")  # no on_step passed

    assert result == {"rewritten_query": "q", "intent": "conceptual", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_act_and_preserves_legal_identifier():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "case law for Bharatiya Nyaya Sanhita about scrap sale",
        "intent": "conceptual",
        "filters": {},
    })

    result = await extract_intent(gateway, "80HH scrap sale yes useless drum sale no")

    assert result["rewritten_query"] == "80HH scrap sale yes useless drum sale no"


@pytest.mark.asyncio
async def test_extract_intent_rejects_expansion_of_ambiguous_acronym():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "software royalty Profit and Excess India USA DTAA",
        "intent": "conceptual",
        "filters": {},
    })

    result = await extract_intent(gateway, "software royalty PE India USA DTAA")

    assert result["rewritten_query"] == "software royalty PE India USA DTAA"


@pytest.mark.asyncio
async def test_extract_intent_drops_unknown_null_and_empty_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "trade training takeover Kolkata",
        "intent": "conceptual",
        "filters": {"city": "Kolkata", "act": None, "court": "", "section": "37(1)"},
    })

    result = await extract_intent(gateway, "trade training takeover Kolkata section 37(1)")

    assert result["filters"] == {"section": "37(1)"}


@pytest.mark.asyncio
async def test_extract_intent_extracts_bench_and_judge_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
        "intent": "conceptual",
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    })

    result = await extract_intent(
        gateway,
        "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
    )

    assert result["filters"] == {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"}


@pytest.mark.asyncio
async def test_extract_intent_drops_bench_and_judge_when_not_literally_in_query():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "Modvat credit ruling",
        "intent": "conceptual",
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    })

    result = await extract_intent(gateway, "Modvat credit ruling")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_coerces_unrecognized_intent_label_to_unknown():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "q", "intent": "section_lookup", "filters": {},
    })

    result = await extract_intent(gateway, "q")

    assert result["intent"] == "unknown"


@pytest.mark.asyncio
async def test_extract_intent_accepts_each_allowed_intent_label():
    for label in ["citation_lookup", "provision_lookup", "conceptual", "unknown"]:
        gateway = AsyncMock()
        gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": label, "filters": {}})

        result = await extract_intent(gateway, "q")

        assert result["intent"] == label


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_shape_is_invalid():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": None,
        "intent": ["tax"],
        "filters": "none",
    })

    result = await extract_intent(gateway, "original")

    assert result == {"rewritten_query": "original", "intent": "unknown", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_year_and_court():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "income tax deduction in 2024 decided by Delhi High Court",
        "intent": "conceptual",
        "filters": {},
    })

    result = await extract_intent(gateway, "income tax deduction")

    assert result["rewritten_query"] == "income tax deduction"


@pytest.mark.asyncio
async def test_extract_intent_preserves_user_supplied_year_and_section_numbers():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "income tax section 80HH deduction in 1985",
        "intent": "provision_lookup",
        "filters": {"section": "80HH"},
    })

    result = await extract_intent(gateway, "1985 income tax section 80HH deduction")

    assert result["rewritten_query"] == "income tax section 80HH deduction in 1985"
    assert result["filters"] == {"section": "80HH"}


@pytest.mark.asyncio
async def test_extract_intent_rejects_lossy_rewrite():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "sale of scrap under 80HH",
        "intent": "conceptual",
        "filters": {},
    })

    query = "80HH scrap sale yes useless drum sale no metallic wire factory"
    result = await extract_intent(gateway, query)

    assert result["rewritten_query"] == query


@pytest.mark.asyncio
async def test_extract_intent_requests_model_for_slm_role():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    gateway.get_model.assert_awaited_once_with(role="slm")


@pytest.mark.asyncio
async def test_extract_intent_uses_llama_tuned_prompt_for_llama_model():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    system_message = gateway.chat.await_args.kwargs["messages"][0]
    assert "Forbidden rewrites" in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_warns_when_model_has_no_tuned_prompt(monkeypatch):
    import retrieval_api.ai_mode.intent as intent_module

    captured = {}
    monkeypatch.setattr(intent_module, "get_client", lambda: type(
        "FakeLangfuseClient", (), {"update_current_span": staticmethod(lambda **kw: captured.update(kw))},
    )())
    gateway = AsyncMock()
    gateway.get_model.return_value = "some-brand-new-model"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    assert captured.get("level") == "WARNING"


@pytest.mark.asyncio
async def test_extract_intent_drops_non_iso_or_invented_date_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "income tax cases",
        "intent": "conceptual",
        "filters": {"date_range": {"gte": "not specified", "lte": "2024-12-31"}},
    })

    result = await extract_intent(gateway, "income tax cases")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_forwards_model_override_and_skips_get_model():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "candidate model test",
        "intent": "conceptual",
        "filters": {},
    })

    await extract_intent(gateway, "candidate model test", model="google/gemma-4-E4B-it")

    gateway.get_model.assert_not_awaited()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["model"] == "google/gemma-4-E4B-it"
