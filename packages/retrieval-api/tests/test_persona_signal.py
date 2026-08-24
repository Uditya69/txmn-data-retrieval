import json
from unittest.mock import AsyncMock

import pytest

from persona.repository import get_persona
from retrieval_api.ai_mode.persona_signal import extract_expertise_signal, record_persona_signal


@pytest.mark.asyncio
async def test_extract_expertise_signal_parses_valid_json():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({"expertise_level": "practitioner", "query_style": "precise-citation"})

    result = await extract_expertise_signal(gateway, "Section 54F capital gains exemption query")

    assert result == {"expertise_level": "practitioner", "query_style": "precise-citation"}
    assert gateway.chat.call_args.kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_expertise_signal_returns_empty_dict_on_malformed_json():
    gateway = AsyncMock()
    gateway.chat.return_value = "not valid json"

    result = await extract_expertise_signal(gateway, "some query")

    assert result == {}


@pytest.mark.asyncio
async def test_record_persona_signal_writes_merged_persona(fake_personas_collection):
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({"expertise_level": "student"})
    personas = fake_personas_collection

    await record_persona_signal(personas, gateway, "user-1", "query text", categories=["rules"])

    stored = await get_persona(personas, "user-1")
    assert stored["expertise_level"] == "student"
    assert stored["category_affinity"]["rules"] == 1.0


@pytest.mark.asyncio
async def test_record_persona_signal_swallows_gateway_errors(fake_personas_collection):
    gateway = AsyncMock()
    gateway.chat.side_effect = RuntimeError("gateway unreachable")
    personas = fake_personas_collection

    await record_persona_signal(personas, gateway, "user-1", "query text", categories=["rules"])

    assert await get_persona(personas, "user-1") is None
