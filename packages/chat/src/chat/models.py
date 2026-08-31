from pydantic import BaseModel


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[dict]
    created_at: str
    updated_at: str


def to_summary(doc: dict) -> ConversationSummary:
    return ConversationSummary(id=doc["_id"], title=doc["title"], updated_at=doc["updated_at"])


def to_detail(doc: dict) -> ConversationDetail:
    return ConversationDetail(
        id=doc["_id"], title=doc["title"], messages=doc["messages"],
        created_at=doc["created_at"], updated_at=doc["updated_at"],
    )


def to_trace_summary(doc: dict) -> dict:
    """No `id` field - insert_one's generated `_id` never gets read back onto
    `doc` by this repo's fakes, and the frontend only ever needs these
    positionally (oldest-first, zipped against conversations.messages)."""
    return {
        "mode": doc["mode"],
        "query": doc["query"],
        "created_at": doc["created_at"],
        "instant": doc.get("instant"),
        "ai_mode": doc.get("ai_mode"),
    }
