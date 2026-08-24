from persona.config import get_persona_settings


def test_settings_load_from_env(monkeypatch):
    # get_persona_settings is @lru_cache'd - whichever test/import happens to
    # construct it first in the whole session wins for every later call, regardless
    # of what conftest.py's os.environ.setdefault() does afterward (that only guards
    # the FIRST construction, no protection once a stale instance is cached). Setting
    # the env vars here and clearing the cache makes this test deterministic
    # regardless of session-wide call/collection order.
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-auth-db")
    get_persona_settings.cache_clear()
    settings = get_persona_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
    get_persona_settings.cache_clear()
