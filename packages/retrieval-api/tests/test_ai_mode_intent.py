import json
from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.intent import _build_chunk_context, _has_legal_anchor, build_lexicon_check, extract_intent
from persona.prompt import RELEVANCE_INSTRUCTION


@pytest.mark.asyncio
async def test_extract_intent_parses_json_object_response():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "BNS section 103 murder punishment",
        "intent": ["acts"],
        "filters": {"act": "BNS"},
    }), None)

    result = await extract_intent(gateway, "section 302 punishment")

    assert result == {
        "original_query": "section 302 punishment",
        "search_query": "section 302 punishment",  # rewrite rejected: <60% token overlap with input
        "intent": ["acts"],
        "filters": {},
        "reasoning": None,
    }
    gateway.chat_with_reasoning.assert_awaited_once()
    call_kwargs = gateway.chat_with_reasoning.await_args.kwargs
    assert call_kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_intent_requests_json_object_response_format():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q")

    call_kwargs = gateway.chat_with_reasoning.await_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_response_is_wrapped_in_markdown_fence():
    """json_object response_format should prevent this from a compliant model,
    but if a model still wraps its output, fall back rather than regex-guessing
    the JSON out of prose - that guesswork is exactly what schema mode replaces."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = ("```json\n" + json.dumps({
        "search_query": "capital gains set off against carried forward business losses",
        "intent": ["caselaws"],
        "filters": {},
    }) + "\n```", None)

    result = await extract_intent(gateway, "set off capital gains against brought forward business losses")

    assert result == {
        "original_query": "set off capital gains against brought forward business losses",
        "search_query": "set off capital gains against brought forward business losses",
        "intent": [],
        "filters": {},
        "reasoning": None,
    }


@pytest.mark.asyncio
async def test_extract_intent_falls_back_to_plain_search_on_unparseable_response():
    """Covers SLM refusals too (e.g. Llama declining a named-party query) -
    AI Mode should degrade to plain semantic search, not fail outright."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = ("I cannot provide case law for that person.", None)

    result = await extract_intent(gateway, "some query")

    assert result == {
        "original_query": "some query", "search_query": "some query", "intent": [], "filters": {},
        "reasoning": None,
    }


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_response_is_none():
    """A provider can return a null/empty completion for `content`; json.loads(None)
    raises TypeError (not JSONDecodeError) and must still degrade to the fallback
    instead of propagating an unhandled exception."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (None, None)

    result = await extract_intent(gateway, "some query")

    assert result == {
        "original_query": "some query", "search_query": "some query", "intent": [], "filters": {},
        "reasoning": None,
    }


@pytest.mark.asyncio
async def test_extract_intent_system_prompt_includes_schema_context_and_new_fields():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "some query")

    system_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "facts" in system_message["content"]
    assert "Supreme Court" in system_message["content"]
    assert '"section"' in system_message["content"]
    assert '"gte"' in system_message["content"]
    assert '"lte"' in system_message["content"]
    assert '"bench"' in system_message["content"]
    assert '"judge"' in system_message["content"]
    for category in ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]:
        assert f'"{category}"' in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_emits_intent_step_when_on_step_given():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "section 80HH normalized",
        "intent": ["caselaws"],
        "filters": {"act": "CGST Act"},
    }), None)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await extract_intent(gateway, "section 80HH normalized", on_step=on_step)

    assert result == {
        "original_query": "section 80HH normalized",
        "search_query": "section 80HH normalized",
        "intent": ["caselaws"],
        "filters": {},
        "reasoning": None,
    }
    assert steps == [("intent", {
        "query": "section 80HH normalized",
        "original_query": "section 80HH normalized",
        "search_query": "section 80HH normalized",
        "intent": ["caselaws"],
        "filters": {},
        "reasoning": None,
    })]


@pytest.mark.asyncio
async def test_extract_intent_skips_on_step_when_none():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    result = await extract_intent(gateway, "q")  # no on_step passed

    assert result == {"original_query": "q", "search_query": "q", "intent": [], "filters": {}, "reasoning": None}


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_act_and_preserves_legal_identifier():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "case law for Bharatiya Nyaya Sanhita about scrap sale",
        "intent": ["caselaws"],
        "filters": {},
        "reasoning": None,
    }), None)

    result = await extract_intent(gateway, "80HH scrap sale yes useless drum sale no")

    assert result["search_query"] == "80HH scrap sale yes useless drum sale no"


@pytest.mark.asyncio
async def test_extract_intent_rejects_expansion_of_ambiguous_acronym():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "software royalty Profit and Excess India USA DTAA",
        "intent": [],
        "filters": {},
    }), None)

    result = await extract_intent(gateway, "software royalty PE India USA DTAA")

    assert result["search_query"] == "software royalty PE India USA DTAA"


@pytest.mark.asyncio
async def test_extract_intent_drops_unknown_null_and_empty_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "trade training takeover Kolkata section 37(1)",
        "intent": ["acts"],
        "filters": {"city": "Kolkata", "act": None, "court": "", "section": "37(1)"},
    }), None)

    result = await extract_intent(gateway, "trade training takeover Kolkata section 37(1)")

    # "section" is dropped unconditionally now (see Task 2 Step 4), independent
    # of what "unknown null and empty filters" covers - only "city" (unrecognized
    # key), "act": None, and "court": "" are this test's actual subject.
    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_extracts_bench_and_judge_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
        "intent": ["caselaws"],
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    }), None)

    result = await extract_intent(
        gateway,
        "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
    )

    assert result["filters"] == {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"}


@pytest.mark.asyncio
async def test_extract_intent_drops_bench_and_judge_when_not_literally_in_query():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "Modvat credit ruling",
        "intent": ["caselaws"],
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    }), None)

    result = await extract_intent(gateway, "Modvat credit ruling")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_drops_unrecognized_category_values():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "q", "intent": ["acts", "not_a_real_category"], "filters": {},
    }), None)

    result = await extract_intent(gateway, "section 80HH deduction")

    assert result["intent"] == ["acts"]


@pytest.mark.asyncio
async def test_extract_intent_dedupes_category_values():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "q", "intent": ["acts", "acts", "caselaws"], "filters": {},
    }), None)

    result = await extract_intent(gateway, "section 80HH deduction")

    assert sorted(result["intent"]) == ["acts", "caselaws"]


@pytest.mark.asyncio
async def test_extract_intent_accepts_multi_label_category():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "case law on section 54F exemption eligibility",
        "intent": ["acts", "caselaws"],
        "filters": {},
    }), None)

    result = await extract_intent(gateway, "case law on section 54F exemption eligibility")

    assert sorted(result["intent"]) == ["acts", "caselaws"]


@pytest.mark.asyncio
async def test_extract_intent_accepts_each_allowed_category_label():
    for label in ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]:
        gateway = AsyncMock()
        gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [label], "filters": {}}), None)

        result = await extract_intent(gateway, "section 80HH deduction")

        assert result["intent"] == [label]


@pytest.mark.asyncio
async def test_extract_intent_accepts_empty_category_list():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    result = await extract_intent(gateway, "q")

    assert result["intent"] == []


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_shape_is_invalid():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": None,
        "intent": "acts",  # not a list - malformed shape
        "filters": "none",
    }), None)

    result = await extract_intent(gateway, "original")

    assert result == {
        "original_query": "original", "search_query": "original", "intent": [], "filters": {},
        "reasoning": None,
    }


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_year_and_court():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "income tax deduction in 2024 decided by Delhi High Court",
        "intent": ["caselaws"],
        "filters": {},
    }), None)

    result = await extract_intent(gateway, "income tax deduction")

    assert result["search_query"] == "income tax deduction"


@pytest.mark.asyncio
async def test_extract_intent_preserves_user_supplied_year_and_section_numbers():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "income tax section 80HH deduction in 1985",
        "intent": ["acts"],
        "filters": {"section": "80HH"},
    }), None)

    result = await extract_intent(gateway, "1985 income tax section 80HH deduction")

    assert result["search_query"] == "income tax section 80HH deduction in 1985"
    # "section" is dropped unconditionally now, regardless of category (Task 2 Step 4).
    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_always_drops_section_filter_regardless_of_category():
    """The old intent!=provision_lookup gate compared against a value that no
    longer exists post category-rewrite - made explicit instead: "section" is
    now unconditionally stripped from filters, for every category including
    "acts" (where the old gate would have kept it)."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "section 92C text",
        "intent": ["acts"],
        "filters": {"section": "92C"},
    }), None)

    result = await extract_intent(gateway, "section 92C text")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_rejects_lossy_rewrite():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "sale of scrap under 80HH",
        "intent": ["acts"],
        "filters": {},
    }), None)

    query = "80HH scrap sale yes useless drum sale no metallic wire factory"
    result = await extract_intent(gateway, query)

    assert result["search_query"] == query


@pytest.mark.asyncio
async def test_extract_intent_requests_model_for_slm_role():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q")

    gateway.get_model.assert_awaited_once_with(role="slm")


@pytest.mark.asyncio
async def test_extract_intent_uses_llama_tuned_prompt_for_llama_model():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q")

    system_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][0]
    assert "CONSERVATIVE search normalization" in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_uses_qwen3_tuned_prompt_for_qwen3_model():
    gateway = AsyncMock()
    gateway.get_model.return_value = "qwen3"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q")

    system_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][0]
    assert "CONSERVATIVE search normalization" in system_message["content"]
    assert "Section 52" in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_warns_when_model_has_no_tuned_prompt(monkeypatch):
    import retrieval_api.ai_mode.intent as intent_module

    captured = {}
    monkeypatch.setattr(intent_module, "get_client", lambda: type(
        "FakeLangfuseClient", (), {"update_current_span": staticmethod(lambda **kw: captured.update(kw))},
    )())
    gateway = AsyncMock()
    gateway.get_model.return_value = "some-brand-new-model"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q")

    assert captured.get("level") == "WARNING"


@pytest.mark.asyncio
async def test_extract_intent_drops_non_iso_or_invented_date_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "income tax cases",
        "intent": ["caselaws"],
        "filters": {"date_range": {"gte": "not specified", "lte": "2024-12-31"}},
    }), None)

    result = await extract_intent(gateway, "income tax cases")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_forwards_model_override_and_skips_get_model():
    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "candidate model test",
        "intent": [],
        "filters": {},
    }), None)

    await extract_intent(gateway, "candidate model test", model="google/gemma-4-E4B-it")

    gateway.get_model.assert_not_awaited()
    call_kwargs = gateway.chat_with_reasoning.await_args.kwargs
    assert call_kwargs["model"] == "google/gemma-4-E4B-it"


def test_build_chunk_context_returns_none_for_text_only_query():
    assert _build_chunk_context("capital gains set off business losses") is None


def test_build_chunk_context_projects_and_filters_chunks():
    result = _build_chunk_context('1995 taxmann.com 569 Delhi High Court "capital gains"')

    assert result is not None
    spans = json.loads(result)
    assert isinstance(spans, list)
    for span in spans:
        assert set(span.keys()) == {"text", "type"}
        assert span["type"] != "text"
    types = {span["type"] for span in spans}
    assert "citation" in types
    assert "court_city" in types


def test_build_chunk_context_drops_stopword_led_court_city_spans():
    """merge_court_city merges any token immediately before court/high/tribunal,
    including stopwords ("the court", "the tribunal") - injecting these as
    authoritative structural spans nudges the SLM toward a bogus court filter
    that _sanitize_filters can't catch (its substring check trivially passes
    a stopword phrase that IS literally in the query)."""
    result = _build_chunk_context("assessee approached the court after the tribunal order")

    spans = json.loads(result) if result is not None else []
    texts = [span["text"] for span in spans]
    assert "the court" not in texts
    assert "the tribunal" not in texts


def test_build_chunk_context_does_not_escape_non_ascii_text():
    """json.dumps defaults to ensure_ascii=True, which would \\uXXXX-escape a
    non-ASCII span - undercutting the "already present in the query above"
    framing since the injected block would then visually diverge from the
    query text shown right above it."""
    result = _build_chunk_context('"café royalty" ruling')

    assert result is not None
    assert "café royalty" in result
    assert "\\u" not in result


def test_build_chunk_context_keeps_real_court_name_spans():
    result = _build_chunk_context("Delhi High Court ruling on capital gains")

    assert result is not None
    spans = json.loads(result)
    texts = [span["text"] for span in spans]
    assert "Delhi High Court" in texts


def test_build_chunk_context_drops_alt_text_and_proximity():
    result = _build_chunk_context("Rule 57A applicability")

    assert result is not None
    spans = json.loads(result)
    for span in spans:
        assert "alt_text" not in span
        assert "proximity" not in span


@pytest.mark.asyncio
async def test_extract_intent_appends_lexicon_check_when_no_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "capital gains", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "capital gains")

    user_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][1]["content"]
    assert "Lexicon check:" in user_message
    assert "no known legal term" in user_message


@pytest.mark.asyncio
async def test_extract_intent_omits_lexicon_check_when_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "Delhi High Court ruling", "intent": ["caselaws"], "filters": {},
    }), None)

    await extract_intent(gateway, "Delhi High Court ruling")

    user_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][1]["content"]
    assert "Lexicon check:" not in user_message


@pytest.mark.asyncio
async def test_extract_intent_appends_chunk_context_when_spans_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "1995 taxmann.com 569 Delhi High Court",
        "intent": ["caselaws"],
        "filters": {},
    }), None)

    await extract_intent(gateway, "1995 taxmann.com 569 Delhi High Court")

    call_kwargs = gateway.chat_with_reasoning.await_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert user_message.startswith("1995 taxmann.com 569 Delhi High Court\n\n")
    assert "Structural spans already present in the query above" in user_message
    assert '"type": "citation"' in user_message
    assert '"type": "court_city"' in user_message


@pytest.mark.asyncio
async def test_extract_intent_user_message_unchanged_when_no_spans_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "capital gains treatment",
        "intent": ["caselaws"],
        "filters": {},
    }), None)

    await extract_intent(gateway, "capital gains treatment")

    call_kwargs = gateway.chat_with_reasoning.await_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert user_message.startswith("capital gains treatment")
    assert "Structural spans already present" not in user_message
    assert "Lexicon check:" in user_message


@pytest.mark.asyncio
async def test_extract_intent_includes_persona_context_in_user_message():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q", persona_context="This user frequently asks about caselaws.")

    user_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][1]["content"]
    assert "This user frequently asks about caselaws." in user_message
    assert RELEVANCE_INSTRUCTION in user_message


@pytest.mark.asyncio
async def test_extract_intent_omits_persona_block_when_context_empty():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({"search_query": "q", "intent": [], "filters": {}}), None)

    await extract_intent(gateway, "q")

    user_message = gateway.chat_with_reasoning.await_args.kwargs["messages"][1]["content"]
    assert RELEVANCE_INSTRUCTION not in user_message


def test_has_legal_anchor_true_when_chunk_context_present():
    assert _has_legal_anchor("Delhi High Court ruling", '[{"text": "Delhi High Court", "type": "court_city"}]') is True


def test_has_legal_anchor_true_when_lexicon_synonym_matches():
    # "ACIT" expands via the legal lexicon (see common/legal_lexicon.py's synonyms table) -
    # expand_query_synonyms appends the expansion, so the returned string differs from the input.
    assert _has_legal_anchor("ACIT order challenged", None) is True


def test_has_legal_anchor_true_when_shape_is_not_plain():
    # "section 80HH" matches SECTION_PATTERN inside classify_query_shape -> shape="provision",
    # even with chunk_context=None passed explicitly (simulating no structural spans found).
    assert _has_legal_anchor("explain section 80HH", None) is True


def test_has_legal_anchor_false_for_bare_topic_words():
    assert _has_legal_anchor("capital gains", None) is False


def test_too_vague_to_tag_true_when_no_anchor():
    from retrieval_api.ai_mode.intent import _too_vague_to_tag
    assert _too_vague_to_tag("capital gains", None) is True


def test_too_vague_to_tag_false_when_anchor_present():
    from retrieval_api.ai_mode.intent import _too_vague_to_tag
    assert _too_vague_to_tag("explain section 80HH", None) is False


def test_too_vague_to_tag_true_for_anchor_free_fact_pattern_question():
    """Documents the accepted tradeoff from the design spec's "Explicit ruling" section:
    a fact-pattern/scenario question with zero literal anchor is force-emptied by this
    floor even though extract_intent's caselaws category signal would otherwise tag it -
    deliberate, not a bug. A future change to _too_vague_to_tag has to consciously break
    this test to reintroduce fact-pattern tagging."""
    from retrieval_api.ai_mode.intent import _too_vague_to_tag
    assert _too_vague_to_tag("gift from father taxable?", None) is True


@pytest.mark.asyncio
async def test_extract_intent_forces_empty_intent_when_no_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "capital gains", "intent": ["commentary"], "filters": {},
    }), None)

    result = await extract_intent(gateway, "capital gains")

    assert result["intent"] == []


@pytest.mark.asyncio
async def test_extract_intent_keeps_intent_when_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat_with_reasoning.return_value = (json.dumps({
        "search_query": "section 80HH deduction", "intent": ["acts"], "filters": {},
    }), None)

    result = await extract_intent(gateway, "section 80HH deduction")

    assert result["intent"] == ["acts"]


def test_build_lexicon_check_reports_anchor_and_chunks_for_citation_query():
    result = build_lexicon_check("32 ITR 190 Provident Investment managing agency section 12B capital gains")

    assert result["has_anchor"] is True
    # CITATION_PATTERN requires a 4-digit year-like number; "32" is only 2 digits, so
    # this query's shape falls through to "provision" via SECTION_PATTERN matching
    # "section 12B" - the chunk-level "citation" type still gets recognized separately
    # by chunk_query's own (looser) citation detection, which is what the assertions
    # below check.
    assert result["shape"] == "provision"
    types = {chunk["type"] for chunk in result["chunks"]}
    assert "citation" in types
    assert "section" in types


def test_build_lexicon_check_reports_no_anchor_for_bare_topic_words():
    result = build_lexicon_check("capital gains")

    assert result["has_anchor"] is False
    assert result["shape"] == "plain"
    assert result["chunks"] == []
