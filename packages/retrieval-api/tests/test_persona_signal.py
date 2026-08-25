import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from persona.repository import get_current_snapshot
from retrieval_api.ai_mode.persona_signal import extract_query_understanding, record_persona_signal

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _understanding_json(**overrides):
    base = {
        "concepts": ["director liability"],
        "legal_entities": ["IBC"],
        "research_objective": ["determine liability"],
        "specificity": 0.8,
        "confidence": 0.9,
    }
    base.update(overrides)
    return json.dumps(base)


@pytest.mark.asyncio
async def test_extract_query_understanding_parses_valid_json():
    gateway = AsyncMock()
    gateway.chat.return_value = _understanding_json()

    result = await extract_query_understanding(gateway, "Can a director be liable under IBC?")

    assert result["concepts"] == ["director liability"]
    assert result["legal_entities"] == ["IBC"]
    assert gateway.chat.call_args.kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_query_understanding_returns_empty_dict_on_malformed_json():
    gateway = AsyncMock()
    gateway.chat.return_value = "not valid json"

    result = await extract_query_understanding(gateway, "some query")

    assert result == {}


@pytest.mark.asyncio
async def test_record_persona_signal_creates_topic_from_valid_understanding(fake_events_collection, fake_topics_collection, persona_settings):
    gateway = AsyncMock()
    gateway.chat.return_value = _understanding_json()
    gateway.embed.return_value = [1.0, 0.0, 0.0]

    await record_persona_signal(
        fake_events_collection, fake_topics_collection, gateway, "user-1", "query text",
        ["rules"], T0, persona_settings,
    )

    snapshot = await get_current_snapshot(fake_topics_collection, "user-1")
    assert len(snapshot) == 1
    assert snapshot[0]["categories"] == ["rules"]
    assert gateway.embed.call_args.kwargs["role"] == "query_embed"


@pytest.mark.asyncio
async def test_record_persona_signal_swallows_gateway_errors(fake_events_collection, fake_topics_collection, persona_settings):
    gateway = AsyncMock()
    gateway.chat.side_effect = RuntimeError("gateway unreachable")

    await record_persona_signal(
        fake_events_collection, fake_topics_collection, gateway, "user-1", "query text",
        ["rules"], T0, persona_settings,
    )

    assert await get_current_snapshot(fake_topics_collection, "user-1") == []


@pytest.mark.asyncio
async def test_record_persona_signal_degrades_on_unreachable_event_store(fake_topics_collection, persona_settings):
    class _BrokenEventsCollection:
        async def insert_one(self, doc):
            raise RuntimeError("event store unreachable")

        def find(self, filter):
            raise RuntimeError("event store unreachable")

    gateway = AsyncMock()
    gateway.chat.return_value = _understanding_json()
    gateway.embed.return_value = [1.0, 0.0, 0.0]

    # Must not raise - the write is fire-and-forget and a down event store
    # must degrade silently, matching the pre-timeline implementation's
    # persona-store resilience pattern.
    await record_persona_signal(
        _BrokenEventsCollection(), fake_topics_collection, gateway, "user-1", "query text",
        ["rules"], T0, persona_settings,
    )

    assert await get_current_snapshot(fake_topics_collection, "user-1") == []


@pytest.mark.asyncio
async def test_record_persona_signal_writes_nothing_on_malformed_understanding(fake_events_collection, fake_topics_collection, persona_settings):
    gateway = AsyncMock()
    gateway.chat.return_value = "not valid json"
    gateway.embed.return_value = [1.0, 0.0, 0.0]

    await record_persona_signal(
        fake_events_collection, fake_topics_collection, gateway, "user-1", "query text",
        ["rules"], T0, persona_settings,
    )

    assert await get_current_snapshot(fake_topics_collection, "user-1") == []
    assert fake_events_collection.documents == []
