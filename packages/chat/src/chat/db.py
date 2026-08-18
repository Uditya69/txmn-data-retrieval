from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from chat.config import ChatSettings


@lru_cache
def get_mongo_client(settings: ChatSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_conversations_collection(client: AsyncIOMotorClient, settings: ChatSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["conversations"]
