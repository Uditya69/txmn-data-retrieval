from datetime import datetime, timezone


async def create_conversation(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "_id": conversation_id,
        "user_id": user_id,
        "title": title,
        "messages": messages,
        "created_at": now,
        "updated_at": now,
    }
    await conversations.replace_one({"_id": conversation_id, "user_id": user_id}, doc, upsert=True)
    return doc


async def append_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> dict:
    existing = await conversations.find_one({"_id": conversation_id, "user_id": user_id})
    if existing is None:
        return await create_conversation(conversations, conversation_id, user_id, title, messages)

    doc = {
        **existing,
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await conversations.replace_one({"_id": conversation_id}, doc, upsert=True)
    return doc


async def list_conversations(conversations, user_id: str) -> list[dict]:
    cursor = conversations.find({"user_id": user_id}).sort("updated_at", -1)
    return [doc async for doc in cursor]


async def get_conversation(conversations, conversation_id: str, user_id: str) -> dict | None:
    return await conversations.find_one({"_id": conversation_id, "user_id": user_id})


async def delete_conversation(conversations, conversation_id: str, user_id: str) -> bool:
    result = await conversations.delete_one({"_id": conversation_id, "user_id": user_id})
    return result.deleted_count > 0
