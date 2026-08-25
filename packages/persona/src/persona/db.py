from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from persona.config import PersonaSettings


@lru_cache
def get_mongo_client(settings: PersonaSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_personas_collection(client: AsyncIOMotorClient, settings: PersonaSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["personas"]


def get_persona_events_collection(client: AsyncIOMotorClient, settings: PersonaSettings) -> AsyncIOMotorCollection:
    """Append-only per-query event history. Never updated in place after insert."""
    return client[settings.mongo_db]["persona_events"]


def get_persona_topics_collection(client: AsyncIOMotorClient, settings: PersonaSettings) -> AsyncIOMotorCollection:
    """Derived/cache layer (state, score cache, episodes) - always rebuildable from
    persona_events alone; never the sole source of truth. See design.md decision #1."""
    return client[settings.mongo_db]["persona_topics"]


async def ensure_persona_indexes(client: AsyncIOMotorClient, settings: PersonaSettings) -> None:
    events = get_persona_events_collection(client, settings)
    topics = get_persona_topics_collection(client, settings)
    await events.create_index([("user_id", 1), ("timestamp", 1)])
    await events.create_index([("user_id", 1), ("topic_id", 1)])
    await topics.create_index([("user_id", 1), ("topic_id", 1)], unique=True)
