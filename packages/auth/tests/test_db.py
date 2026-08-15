from auth.config import get_auth_settings
from auth.db import get_mongo_client, get_users_collection


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
