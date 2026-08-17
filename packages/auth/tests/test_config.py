import pytest
from pydantic import ValidationError

from auth.config import AuthSettings, get_auth_settings


def test_settings_load_from_env(monkeypatch):
    # get_auth_settings is @lru_cache'd - whichever test/import happens to construct
    # it first in the whole session wins for every later call, regardless of what
    # conftest.py's os.environ.setdefault() does afterward (that only guards the
    # FIRST construction, and offers no protection once a stale instance is cached).
    # Setting the env vars here and clearing the cache makes this test deterministic
    # regardless of session-wide call/collection order or what other packages'
    # tests happen to run before it.
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-auth-db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-that-is-at-least-32-characters-long")
    monkeypatch.setenv("JWT_EXPIRY_MINUTES", "60")
    get_auth_settings.cache_clear()
    settings = get_auth_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
    assert settings.jwt_secret == "test-jwt-secret-that-is-at-least-32-characters-long"
    assert settings.jwt_expiry_minutes == 60
    get_auth_settings.cache_clear()


def test_short_jwt_secret_raises_validation_error(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "too-short")
    with pytest.raises(ValidationError):
        AuthSettings()


def test_empty_jwt_secret_raises_validation_error(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "")
    with pytest.raises(ValidationError):
        AuthSettings()
