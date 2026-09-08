from unittest.mock import AsyncMock

import pytest

import retrieval_api.admin.feature_flags as feature_flags


@pytest.fixture(autouse=True)
def _reset_overrides():
    feature_flags._mongo_overrides = {}
    yield
    feature_flags._mongo_overrides = {}


def test_effective_uses_default_when_no_env_or_mongo_override(monkeypatch):
    monkeypatch.delenv("AI_MODE_RERANK_ENABLED", raising=False)
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY",
        {**feature_flags.FLAG_REGISTRY, "ai_mode_rerank_enabled": lambda: True},
    )
    assert feature_flags.effective("ai_mode_rerank_enabled") is True


def test_effective_uses_mongo_override_when_no_env_var(monkeypatch):
    monkeypatch.delenv("AI_MODE_RERANK_ENABLED", raising=False)
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY",
        {**feature_flags.FLAG_REGISTRY, "ai_mode_rerank_enabled": lambda: True},
    )
    feature_flags._mongo_overrides = {"ai_mode_rerank_enabled": False}
    assert feature_flags.effective("ai_mode_rerank_enabled") is False


def test_env_var_wins_over_mongo_override(monkeypatch):
    # Settings already resolved AI_MODE_RERANK_ENABLED=true from the env at
    # construction time - the default getter below stands in for that.
    monkeypatch.setenv("AI_MODE_RERANK_ENABLED", "true")
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY",
        {**feature_flags.FLAG_REGISTRY, "ai_mode_rerank_enabled": lambda: True},
    )
    feature_flags._mongo_overrides = {"ai_mode_rerank_enabled": False}
    assert feature_flags.effective("ai_mode_rerank_enabled") is True


def test_effective_raises_for_unknown_flag():
    with pytest.raises(KeyError):
        feature_flags.effective("not_a_real_flag")


def test_describe_flags_reports_default_env_mongo_and_effective(monkeypatch):
    monkeypatch.delenv("AI_MODE_RERANK_ENABLED", raising=False)
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"ai_mode_rerank_enabled": lambda: True},
    )
    feature_flags._mongo_overrides = {"ai_mode_rerank_enabled": False}
    rows = feature_flags.describe_flags()
    assert rows == [{
        "name": "ai_mode_rerank_enabled",
        "default": True,
        "env_override": False,
        "mongo_override": False,
        "effective": False,
        "options": None,
    }]


def test_describe_flags_reports_options_for_string_valued_flags(monkeypatch):
    monkeypatch.delenv("CHAT_PROVIDER", raising=False)
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"chat_provider": lambda: "deepinfra"},
    )
    monkeypatch.setattr(feature_flags, "VALID_VALUES", {"chat_provider": {"deepinfra", "local"}})
    rows = feature_flags.describe_flags()
    assert rows == [{
        "name": "chat_provider",
        "default": "deepinfra",
        "env_override": False,
        "mongo_override": None,
        "effective": "deepinfra",
        "options": ["deepinfra", "local"],
    }]


@pytest.mark.asyncio
async def test_set_flag_override_rejects_invalid_value_for_enum_flag(monkeypatch):
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"chat_provider": lambda: "deepinfra"},
    )
    monkeypatch.setattr(feature_flags, "VALID_VALUES", {"chat_provider": {"deepinfra", "local"}})
    with pytest.raises(ValueError):
        await feature_flags.set_flag_override("chat_provider", "not-a-real-provider")


@pytest.mark.asyncio
async def test_set_flag_override_accepts_valid_string_value(monkeypatch):
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"chat_provider": lambda: "deepinfra"},
    )
    monkeypatch.setattr(feature_flags, "VALID_VALUES", {"chat_provider": {"deepinfra", "local"}})
    collection = AsyncMock()
    collection.find_one.return_value = {"_id": "flags", "chat_provider": "local"}
    monkeypatch.setattr(feature_flags, "get_flags_collection", lambda: collection)

    await feature_flags.set_flag_override("chat_provider", "local")

    collection.update_one.assert_awaited_once_with(
        {"_id": "flags"}, {"$set": {"chat_provider": "local"}}, upsert=True,
    )
    assert feature_flags._mongo_overrides == {"chat_provider": "local"}


@pytest.mark.asyncio
async def test_refresh_flag_overrides_loads_doc_minus_id_and_unknown_keys(monkeypatch):
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"ai_mode_rerank_enabled": lambda: True},
    )
    collection = AsyncMock()
    collection.find_one.return_value = {
        "_id": "flags", "ai_mode_rerank_enabled": False, "not_a_real_flag": True,
    }
    monkeypatch.setattr(feature_flags, "get_flags_collection", lambda: collection)

    await feature_flags.refresh_flag_overrides()

    assert feature_flags._mongo_overrides == {"ai_mode_rerank_enabled": False}


@pytest.mark.asyncio
async def test_set_flag_override_upserts_and_refreshes(monkeypatch):
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"ai_mode_rerank_enabled": lambda: True},
    )
    collection = AsyncMock()
    collection.find_one.return_value = {"_id": "flags", "ai_mode_rerank_enabled": False}
    monkeypatch.setattr(feature_flags, "get_flags_collection", lambda: collection)

    await feature_flags.set_flag_override("ai_mode_rerank_enabled", False)

    collection.update_one.assert_awaited_once_with(
        {"_id": "flags"}, {"$set": {"ai_mode_rerank_enabled": False}}, upsert=True,
    )
    assert feature_flags._mongo_overrides == {"ai_mode_rerank_enabled": False}


@pytest.mark.asyncio
async def test_set_flag_override_none_clears_via_unset(monkeypatch):
    monkeypatch.setattr(
        feature_flags, "FLAG_REGISTRY", {"ai_mode_rerank_enabled": lambda: True},
    )
    collection = AsyncMock()
    collection.find_one.return_value = {"_id": "flags"}
    monkeypatch.setattr(feature_flags, "get_flags_collection", lambda: collection)

    await feature_flags.set_flag_override("ai_mode_rerank_enabled", None)

    collection.update_one.assert_awaited_once_with(
        {"_id": "flags"}, {"$unset": {"ai_mode_rerank_enabled": ""}}, upsert=True,
    )


@pytest.mark.asyncio
async def test_set_flag_override_rejects_unknown_flag():
    with pytest.raises(KeyError):
        await feature_flags.set_flag_override("not_a_real_flag", True)
