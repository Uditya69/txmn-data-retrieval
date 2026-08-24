from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from persona.config import PersonaSettings


@lru_cache
def get_mongo_client(settings: PersonaSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_personas_collection(client: AsyncIOMotorClient, settings: PersonaSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["personas"]
