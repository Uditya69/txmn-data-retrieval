from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from semantic_cache.config import SemanticCacheSettings


@lru_cache
def get_mongo_client(settings: SemanticCacheSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_semantic_cache_collection(
    client: AsyncIOMotorClient, settings: SemanticCacheSettings,
) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["semantic_cache"]
