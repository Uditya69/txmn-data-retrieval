from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from chat.config import ChatSettings


@lru_cache
def get_mongo_client(settings: ChatSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_conversations_collection(client: AsyncIOMotorClient, settings: ChatSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["conversations"]


def get_retrieval_traces_collection(client: AsyncIOMotorClient, settings: ChatSettings) -> AsyncIOMotorCollection:
    """Append-only per-turn retrieval trace (Instant results, AI Mode RRF/rerank/citations,
    Langfuse trace ids) - never updated in place, one document per instant/ai_mode run."""
    return client[settings.mongo_db]["retrieval_traces"]


async def ensure_retrieval_traces_indexes(client: AsyncIOMotorClient, settings: ChatSettings) -> None:
    retrieval_traces = get_retrieval_traces_collection(client, settings)
    await retrieval_traces.create_index([("conversation_id", 1), ("created_at", 1)])
    await retrieval_traces.create_index([("user_id", 1), ("created_at", 1)])
