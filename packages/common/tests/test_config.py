import os
from common.config import Settings, get_settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "root:Milvus")
    monkeypatch.setenv("ES_URL", "http://es:9200")
    monkeypatch.setenv("GATEWAY_URL", "http://model-gateway:8001")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.milvus_uri == "http://milvus:19530"
    assert settings.milvus_db == "aic"  # default
    assert settings.es_url == "http://es:9200"
    assert settings.gateway_url == "http://model-gateway:8001"
