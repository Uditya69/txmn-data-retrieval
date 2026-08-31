import pytest

from chat.config import get_chat_settings
from chat.db import ensure_retrieval_traces_indexes, get_mongo_client, get_retrieval_traces_collection


def test_get_retrieval_traces_collection_selects_configured_db_and_collection_name():
    settings = get_chat_settings()
    client = get_mongo_client(settings)
    collection = get_retrieval_traces_collection(client, settings)
    assert collection.name == "retrieval_traces"
    assert collection.database.name == settings.mongo_db


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
async def test_ensure_retrieval_traces_indexes_creates_expected_indexes():
    settings = get_chat_settings()
    client = _FakeClient()

    await ensure_retrieval_traces_indexes(client, settings)

    traces = client[settings.mongo_db]["retrieval_traces"]
    assert any(call[0] == ([("conversation_id", 1), ("created_at", 1)],) for call in traces.index_calls)
    assert any(call[0] == ([("user_id", 1), ("created_at", 1)],) for call in traces.index_calls)
