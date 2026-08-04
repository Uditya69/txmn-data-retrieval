from common.config import Settings, get_settings


def test_settings_reads_from_env(monkeypatch):
    # `uv run` auto-loads this repo's .env into the process environment
    # before pytest even starts, independent of pydantic-settings' own
    # dotenv handling - explicitly clear the optional ES_* vars this test
    # asserts defaults for, so a populated .env can never leak into this
    # "not set" case (also disable pydantic-settings' own .env file read,
    # for the same reason, belt and braces).
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in ("ES_USERNAME", "ES_PASSWORD", "ES_INDEX", "ES_VERIFY_CERTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "root:Milvus")
    monkeypatch.setenv("ES_URI", "http://es:9200")
    monkeypatch.setenv("GATEWAY_URL", "http://model-gateway:8001")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.milvus_uri == "http://milvus:19530"
    assert settings.milvus_db == "aic"  # default
    assert settings.es_uri == "http://es:9200"
    assert settings.es_username is None  # default
    assert settings.es_index == "taxmann_caselaw"  # default
    assert settings.es_verify_certs is True  # default
    assert settings.gateway_url == "http://model-gateway:8001"


def test_settings_reads_es_auth_and_index_overrides(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "root:Milvus")
    monkeypatch.setenv("ES_URI", "https://es:9200")
    monkeypatch.setenv("ES_USERNAME", "elastic")
    monkeypatch.setenv("ES_PASSWORD", "secret")
    monkeypatch.setenv("ES_INDEX", "researchindex_aic_test")
    monkeypatch.setenv("ES_VERIFY_CERTS", "false")
    monkeypatch.setenv("GATEWAY_URL", "http://model-gateway:8001")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.es_username == "elastic"
    assert settings.es_password == "secret"
    assert settings.es_index == "researchindex_aic_test"
    assert settings.es_verify_certs is False
