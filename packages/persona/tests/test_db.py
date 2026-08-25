import pytest

from persona.config import get_persona_settings
from persona.db import (
    ensure_persona_indexes, get_mongo_client, get_persona_events_collection,
    get_persona_topics_collection, get_personas_collection,
)


def test_get_personas_collection_selects_configured_db_and_collection_name():
    # get_mongo_client is now @lru_cache'd (process-wide singleton per settings) -
    # do not close it here, it's shared with other tests/call sites in this process.
    settings = get_persona_settings()
    client = get_mongo_client(settings)
    collection = get_personas_collection(client, settings)
    assert collection.name == "personas"
    assert collection.database.name == settings.mongo_db


def test_get_mongo_client_caches_client():
    """Verify that the same settings object returns the same cached client."""
    settings = get_persona_settings()
    client1 = get_mongo_client(settings)
    client2 = get_mongo_client(settings)
    assert client1 is client2, "Expected cached client to return the same object"


def test_get_persona_events_and_topics_collections_select_configured_names():
    settings = get_persona_settings()
    client = get_mongo_client(settings)
    events = get_persona_events_collection(client, settings)
    topics = get_persona_topics_collection(client, settings)
    assert events.name == "persona_events"
    assert topics.name == "persona_topics"
    assert events.database.name == settings.mongo_db


class _FakeCollection:
    def __init__(self):
        self.index_calls = []

    async def create_index(self, *args, **kwargs):
        self.index_calls.append((args, kwargs))


class _FakeDb(dict):
    def __missing__(self, key):
        self[key] = _FakeCollection()
        return self[key]


class _FakeClient(dict):
    def __missing__(self, key):
        self[key] = _FakeDb()
        return self[key]


@pytest.mark.asyncio
async def test_ensure_persona_indexes_creates_expected_indexes():
    settings = get_persona_settings()
    client = _FakeClient()

    await ensure_persona_indexes(client, settings)

    events = client[settings.mongo_db]["persona_events"]
    topics = client[settings.mongo_db]["persona_topics"]
    assert any(call[0] == ([("user_id", 1), ("timestamp", 1)],) for call in events.index_calls)
    assert any(call[0] == ([("user_id", 1), ("topic_id", 1)],) for call in events.index_calls)
    assert any(
        call[0] == ([("user_id", 1), ("topic_id", 1)],) and call[1].get("unique") is True
        for call in topics.index_calls
    )
