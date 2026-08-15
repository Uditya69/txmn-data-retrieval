from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_personas_collection


def test_get_personas_collection_selects_configured_db_and_collection_name():
    settings = get_persona_settings()
    client = get_mongo_client(settings)
    try:
        collection = get_personas_collection(client, settings)
        assert collection.name == "personas"
        assert collection.database.name == settings.mongo_db
    finally:
        client.close()


def test_get_mongo_client_caches_client():
    """Verify that the same settings object returns the same cached client."""
    settings = get_persona_settings()
    client1 = get_mongo_client(settings)
    client2 = get_mongo_client(settings)
    try:
        assert client1 is client2, "Expected cached client to return the same object"
    finally:
        client1.close()
