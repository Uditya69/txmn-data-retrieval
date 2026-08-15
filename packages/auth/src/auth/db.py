from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from auth.config import AuthSettings


@lru_cache
def get_mongo_client(settings: AuthSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_users_collection(client: AsyncIOMotorClient, settings: AuthSettings) -> AsyncIOMotorCollection:
    # NOTE: a real deployment needs `await collection.create_index("email", unique=True)`
    # at startup - service.py's signup is check-then-insert, which has a known race
    # window (two concurrent signups for the same email can both succeed) until that
    # unique index exists. Not added here - see finding write-up for scope rationale.
    return client[settings.mongo_db]["users"]
