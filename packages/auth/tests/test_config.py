import pytest
from pydantic import ValidationError

from auth.config import AuthSettings, get_auth_settings


def test_settings_load_from_env():
    settings = get_auth_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
    assert settings.jwt_secret == "test-jwt-secret-that-is-at-least-32-characters-long"
    assert settings.jwt_expiry_minutes == 60


def test_short_jwt_secret_raises_validation_error(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "too-short")
    with pytest.raises(ValidationError):
        AuthSettings()


def test_empty_jwt_secret_raises_validation_error(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "")
    with pytest.raises(ValidationError):
        AuthSettings()
