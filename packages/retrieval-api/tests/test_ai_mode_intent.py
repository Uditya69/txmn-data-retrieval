import json
from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.intent import extract_intent


@pytest.mark.asyncio
async def test_extract_intent_parses_json_response():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "BNS section 103 murder punishment",
        "intent": "section_lookup",
        "filters": {"act": "BNS"},
    })

    result = await extract_intent(gateway, "IPC 302 punishment")

    assert result == {
        "rewritten_query": "BNS section 103 murder punishment",
        "intent": "section_lookup",
        "filters": {"act": "BNS"},
    }
    gateway.chat.assert_awaited_once()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_intent_strips_markdown_code_fence():
    gateway = AsyncMock()
    gateway.chat.return_value = "```json\n" + json.dumps({
        "rewritten_query": "capital gains set off against carried forward business losses",
        "intent": "taxation",
        "filters": {"act": "Income-tax Act, 1961"},
    }) + "\n```"

    result = await extract_intent(gateway, "set off capital gains against brought forward business losses")

    assert result == {
        "rewritten_query": "capital gains set off against carried forward business losses",
        "intent": "taxation",
        "filters": {"act": "Income-tax Act, 1961"},
    }


@pytest.mark.asyncio
async def test_extract_intent_strips_leading_prose_and_code_fence():
    gateway = AsyncMock()
    gateway.chat.return_value = "Here is a JSON object with the requested information:\n\n```\n" + json.dumps({
        "rewritten_query": "CGST meaning",
        "intent": "taxation",
        "filters": {"act": "CGST Act"},
    }) + "\n```"

    result = await extract_intent(gateway, "what is cgst")

    assert result == {
        "rewritten_query": "CGST meaning",
        "intent": "taxation",
        "filters": {"act": "CGST Act"},
    }


@pytest.mark.asyncio
async def test_extract_intent_falls_back_to_plain_search_on_unparseable_response():
    """Covers SLM refusals too (e.g. Llama declining a named-party query) -
    AI Mode should degrade to plain semantic search, not fail outright."""
    gateway = AsyncMock()
    gateway.chat.return_value = "I cannot provide case law for that person."

    result = await extract_intent(gateway, "some query")

    assert result == {"rewritten_query": "some query", "intent": "unknown", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_system_prompt_includes_schema_context():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "q", "intent": "x", "filters": {},
    })

    await extract_intent(gateway, "some query")

    system_message = gateway.chat.await_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "facts" in system_message["content"]
    assert "Supreme Court" in system_message["content"]
    assert '"section"' in system_message["content"]
    assert '"gte"' in system_message["content"]
    assert '"lte"' in system_message["content"]
