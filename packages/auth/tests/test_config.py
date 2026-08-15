from auth.config import get_auth_settings


def test_settings_load_from_env():
    settings = get_auth_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
    assert settings.jwt_secret == "test-jwt-secret"
    assert settings.jwt_expiry_minutes == 60
