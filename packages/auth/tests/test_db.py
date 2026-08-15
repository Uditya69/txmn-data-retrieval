from auth.config import get_auth_settings
from auth.db import get_mongo_client, get_users_collection


def test_get_users_collection_selects_configured_db_and_collection_name():
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    try:
        collection = get_users_collection(client, settings)
        assert collection.name == "users"
        assert collection.database.name == settings.mongo_db
    finally:
        client.close()
