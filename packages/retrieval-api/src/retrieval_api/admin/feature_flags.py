"""Runtime feature-flag resolution: env var > Mongo override > code default.

Registry entries point back at the package that actually owns each flag's
default (common.config.Settings / model_gateway.config.GatewaySettings /
semantic_cache.config.SemanticCacheSettings) - this module never redefines a
default, it only decides whether an env var or a Mongo override should win
over it.

Mongo overrides live in one document (`_id: "flags"`) in the same Mongo
deployment chat/persona/auth/semantic_cache already use - reusing chat's
client/db rather than opening a fifth connection pool for one small
collection.
"""

import os
from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

import chat.config
import chat.db
import common.config
import model_gateway.config
import semantic_cache.config

_FLAGS_DOC_ID = "flags"

# name -> zero-arg getter for the code-level default (settings already parsed
# any env var into this value, so "the settings value" and "the env-var value
# when one is set" are the same read). Calls go through the module object
# (chat.config.get_chat_settings(), not a directly-imported name) so tests can
# monkeypatch the underlying get_*_settings function - see CLAUDE.md's
# "monkeypatch + direct imports don't mix" note.
FLAG_REGISTRY = {
    "ai_mode_rerank_enabled": lambda: common.config.get_settings().ai_mode_rerank_enabled,
    "instant_mode_auto_route_enabled": lambda: common.config.get_settings().instant_mode_auto_route_enabled,
    "instant_mode_rerank_enabled": lambda: common.config.get_settings().instant_mode_rerank_enabled,
    "instant_mode_rrf_enabled": lambda: common.config.get_settings().instant_mode_rrf_enabled,
    "milvus_sparse_enabled": lambda: common.config.get_settings().milvus_sparse_enabled,
    "keyword_mode_expansion_enabled": lambda: common.config.get_settings().keyword_mode_expansion_enabled,
    "expose_reasoning": lambda: common.config.get_settings().expose_reasoning,
    "slm_reasoning_enabled": lambda: model_gateway.config.get_gateway_settings().slm_reasoning_enabled,
    "synthesis_reasoning_enabled": lambda: model_gateway.config.get_gateway_settings().synthesis_reasoning_enabled,
    "semantic_cache_enabled": lambda: semantic_cache.config.get_semantic_cache_settings().semantic_cache_enabled,
}

# Module-level cache of Mongo overrides - refreshed at startup and after every
# admin write, so a read-path `effective()` call is a plain dict lookup, never
# an awaited Mongo round trip.
_mongo_overrides: dict[str, bool] = {}


def _has_env_override(name: str) -> bool:
    return name.upper() in os.environ


@lru_cache
def get_flags_collection() -> AsyncIOMotorCollection:
    settings = chat.config.get_chat_settings()
    client: AsyncIOMotorClient = chat.db.get_mongo_client(settings)
    return client[settings.mongo_db]["feature_flags"]


async def refresh_flag_overrides() -> None:
    global _mongo_overrides
    doc = await get_flags_collection().find_one({"_id": _FLAGS_DOC_ID}) or {}
    doc.pop("_id", None)
    _mongo_overrides = {k: v for k, v in doc.items() if k in FLAG_REGISTRY}


async def set_flag_override(name: str, value: bool | None) -> None:
    """`value=None` clears the Mongo override, falling back to env/default."""
    if name not in FLAG_REGISTRY:
        raise KeyError(f"unknown flag {name!r}")
    collection = get_flags_collection()
    if value is None:
        await collection.update_one({"_id": _FLAGS_DOC_ID}, {"$unset": {name: ""}}, upsert=True)
    else:
        await collection.update_one({"_id": _FLAGS_DOC_ID}, {"$set": {name: value}}, upsert=True)
    await refresh_flag_overrides()


def effective(name: str) -> bool:
    if name not in FLAG_REGISTRY:
        raise KeyError(f"unknown flag {name!r}")
    default_value = FLAG_REGISTRY[name]()
    if _has_env_override(name):
        return default_value
    if name in _mongo_overrides:
        return _mongo_overrides[name]
    return default_value


def describe_flags() -> list[dict]:
    """One row per flag for the admin UI: default, whether an env var pins it,
    the raw Mongo override (if any), and the resulting effective value."""
    rows = []
    for name in FLAG_REGISTRY:
        rows.append({
            "name": name,
            "default": FLAG_REGISTRY[name](),
            "env_override": _has_env_override(name),
            "mongo_override": _mongo_overrides.get(name),
            "effective": effective(name),
        })
    return rows
