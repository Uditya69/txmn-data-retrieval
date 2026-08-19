from fastapi import APIRouter, Depends, HTTPException

from auth.dependency import get_current_user_id
from chat.config import get_chat_settings
from chat.db import get_conversations_collection, get_mongo_client
from chat.models import ConversationDetail, ConversationSummary, to_detail, to_summary
from chat.repository import delete_conversation, get_conversation, list_conversations

router = APIRouter(prefix="/conversations", tags=["chat"])


def get_conversations_dependency():
    settings = get_chat_settings()
    client = get_mongo_client(settings)
    return get_conversations_collection(client, settings)


def _require_user_id(user_id: str | None) -> str:
    if user_id is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user_id


@router.get("", response_model=list[ConversationSummary])
async def list_conversations_route(
    user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    docs = await list_conversations(conversations, user_id)
    return [to_summary(doc) for doc in docs]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_route(
    conversation_id: str, user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    doc = await get_conversation(conversations, conversation_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return to_detail(doc)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation_route(
    conversation_id: str, user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    deleted = await delete_conversation(conversations, conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
