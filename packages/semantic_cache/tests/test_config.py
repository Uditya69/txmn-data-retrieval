from semantic_cache.config import get_semantic_cache_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-semantic-cache-db")
    get_semantic_cache_settings.cache_clear()
    settings = get_semantic_cache_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-semantic-cache-db"
    get_semantic_cache_settings.cache_clear()


def test_settings_default_threshold(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-semantic-cache-db")
    monkeypatch.delenv("SEMANTIC_CACHE_THRESHOLD", raising=False)
    get_semantic_cache_settings.cache_clear()
    settings = get_semantic_cache_settings()
    assert settings.semantic_cache_threshold == 0.95
    get_semantic_cache_settings.cache_clear()


def test_settings_threshold_overridable(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-semantic-cache-db")
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "0.9")
    get_semantic_cache_settings.cache_clear()
    settings = get_semantic_cache_settings()
    assert settings.semantic_cache_threshold == 0.9
    get_semantic_cache_settings.cache_clear()
