from unittest.mock import AsyncMock

import pytest

from auth.config import get_auth_settings
from auth.db import ensure_refresh_token_indexes, get_mongo_client, get_users_collection


@pytest.mark.asyncio
async def test_ensure_refresh_token_indexes_creates_unique_and_ttl_indexes():
    refresh_tokens = AsyncMock()

    await ensure_refresh_token_indexes(refresh_tokens)

    refresh_tokens.create_index.assert_any_call("token_hash", unique=True)
    refresh_tokens.create_index.assert_any_call("expires_at", expireAfterSeconds=0)


def test_get_users_collection_selects_configured_db_and_collection_name():
    # get_mongo_client is now @lru_cache'd (process-wide singleton per settings) -
    # do not close it here, it's shared with other tests/call sites in this process.
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    collection = get_users_collection(client, settings)
    assert collection.name == "users"
    assert collection.database.name == settings.mongo_db


def test_get_mongo_client_is_cached_per_settings():
    settings = get_auth_settings()
    assert get_mongo_client(settings) is get_mongo_client(settings)
