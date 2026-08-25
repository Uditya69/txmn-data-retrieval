import json
from unittest.mock import AsyncMock

import pytest

from retrieval_api.ai_mode.keyword_expansion import _validate_keywords, expand_keyword_terms


def test_validate_keywords_caps_at_two():
    result = _validate_keywords("section 55", ["alpha", "beta", "gamma"])
    assert result == ["alpha", "beta"]


def test_validate_keywords_drops_non_string_values():
    result = _validate_keywords("section 55", ["alpha", 5, None, {"a": 1}])
    assert result == ["alpha"]


def test_validate_keywords_drops_terms_already_in_the_query_case_insensitively():
    result = _validate_keywords("Cost of Acquisition", ["cost of acquisition", "indexation"])
    assert result == ["indexation"]


def test_validate_keywords_dedupes_case_insensitively():
    result = _validate_keywords("section 55", ["Indexation", "indexation"])
    assert result == ["Indexation"]


def test_validate_keywords_drops_blank_strings():
    result = _validate_keywords("section 55", ["  ", "indexation"])
    assert result == ["indexation"]


def test_validate_keywords_returns_empty_for_non_list_input():
    assert _validate_keywords("section 55", None) == []
    assert _validate_keywords("section 55", "indexation") == []


@pytest.mark.asyncio
async def test_expand_keyword_terms_returns_validated_keywords_from_gateway():
    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = (
        json.dumps({"keywords": ["cost of improvement", "capital asset"]}), None,
    )

    result = await expand_keyword_terms(gateway, "section 55")

    assert result == ["cost of improvement", "capital asset"]
    gateway.chat_with_reasoning.assert_awaited_once()
    assert gateway.chat_with_reasoning.call_args.kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_expand_keyword_terms_degrades_to_empty_list_on_malformed_json():
    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("not json", None)

    result = await expand_keyword_terms(gateway, "section 55")

    assert result == []


@pytest.mark.asyncio
async def test_expand_keyword_terms_degrades_to_empty_list_when_gateway_raises():
    gateway = AsyncMock()
    gateway.chat_with_reasoning.side_effect = RuntimeError("gateway down")

    result = await expand_keyword_terms(gateway, "section 55")

    assert result == []


@pytest.mark.asyncio
async def test_expand_keyword_terms_emits_trace_step():
    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = (json.dumps({"keywords": ["indexation"]}), None)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await expand_keyword_terms(gateway, "section 55", on_step=on_step)

    assert steps == [
        ("keyword_expansion", {"query": "section 55", "added_keywords": ["indexation"], "reasoning": None}),
    ]


@pytest.mark.asyncio
async def test_expand_keyword_terms_trace_includes_reasoning_even_when_nothing_added():
    """The common case (model decides not to add anything) is exactly where seeing why
    matters most - reasoning must show up in the trace regardless of added_keywords."""
    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = (
        json.dumps({"keywords": []}), "Already a precise anchor; no synonym adds real recall here.",
    )

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await expand_keyword_terms(gateway, "Section 55", on_step=on_step)

    assert result == []
    assert steps == [(
        "keyword_expansion",
        {
            "query": "Section 55", "added_keywords": [],
            "reasoning": "Already a precise anchor; no synonym adds real recall here.",
        },
    )]


@pytest.mark.asyncio
async def test_expand_keyword_terms_trace_reasoning_is_none_on_malformed_json():
    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("not json", "some reasoning that came with garbage output")

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await expand_keyword_terms(gateway, "section 55", on_step=on_step)

    # reasoning is still captured even though the JSON body failed to parse - the model's
    # own explanation of what it was trying to do is still useful for debugging that failure.
    assert steps[0][1]["reasoning"] == "some reasoning that came with garbage output"
    assert steps[0][1]["added_keywords"] == []
