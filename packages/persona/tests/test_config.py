from persona.config import get_persona_settings


def test_settings_load_from_env():
    settings = get_persona_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
