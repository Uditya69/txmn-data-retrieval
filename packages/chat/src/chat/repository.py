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
    """Appends `messages` (only the new turn's messages, not the full
    history) onto the conversation's existing message list, creating the
    conversation (with just this turn's messages) if it doesn't exist yet.
    """
    existing = await conversations.find_one({"_id": conversation_id, "user_id": user_id})
    if existing is None:
        return await create_conversation(conversations, conversation_id, user_id, title, messages)

    doc = {
        **existing,
        "messages": existing["messages"] + messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await conversations.replace_one({"_id": conversation_id, "user_id": user_id}, doc, upsert=True)
    return doc


async def list_conversations(conversations, user_id: str) -> list[dict]:
    cursor = conversations.find({"user_id": user_id}).sort("updated_at", -1)
    return [doc async for doc in cursor]


async def get_conversation(conversations, conversation_id: str, user_id: str) -> dict | None:
    return await conversations.find_one({"_id": conversation_id, "user_id": user_id})


async def delete_conversation(conversations, conversation_id: str, user_id: str) -> bool:
    result = await conversations.delete_one({"_id": conversation_id, "user_id": user_id})
    return result.deleted_count > 0


async def save_retrieval_trace(
    retrieval_traces, conversation_id: str, user_id: str, mode: str, query: str,
    langfuse_trace_id: str | None, langfuse_observation_id: str | None,
    instant: dict | None = None, ai_mode: dict | None = None,
) -> dict:
    """Persists one Instant- or AI Mode- retrieval trace. One document per
    turn per mode (a "both" request writes two documents) - a conversation
    can have many turns, so this is never keyed by conversation_id alone;
    created_at + mode distinguish documents within the same conversation.
    """
    doc = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "mode": mode,
        "query": query,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "langfuse_trace_id": langfuse_trace_id,
        "langfuse_observation_id": langfuse_observation_id,
        "instant": instant,
        "ai_mode": ai_mode,
    }
    await retrieval_traces.insert_one(doc)
    return doc
