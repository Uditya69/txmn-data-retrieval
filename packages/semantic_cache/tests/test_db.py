from semantic_cache.config import get_semantic_cache_settings
from semantic_cache.db import get_mongo_client, get_semantic_cache_collection


def test_get_semantic_cache_collection_selects_configured_db_and_collection_name():
    settings = get_semantic_cache_settings()
    client = get_mongo_client(settings)
    collection = get_semantic_cache_collection(client, settings)
    assert collection.name == "semantic_cache"
    assert collection.database.name == settings.mongo_db


def test_get_mongo_client_caches_client():
    settings = get_semantic_cache_settings()
    client1 = get_mongo_client(settings)
    client2 = get_mongo_client(settings)
    assert client1 is client2, "Expected cached client to return the same object"
