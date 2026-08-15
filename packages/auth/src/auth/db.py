from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from auth.config import AuthSettings


def get_mongo_client(settings: AuthSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_users_collection(client: AsyncIOMotorClient, settings: AuthSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["users"]
