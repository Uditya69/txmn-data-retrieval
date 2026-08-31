from fastapi import APIRouter, Depends, HTTPException

from auth.dependency import get_current_user_id
from chat.config import get_chat_settings
from chat.db import get_conversations_collection, get_retrieval_traces_collection, get_mongo_client
from chat.models import ConversationDetail, ConversationSummary, to_detail, to_summary, to_trace_summary
from chat.repository import delete_conversation, get_conversation, list_conversations, list_retrieval_traces

router = APIRouter(prefix="/conversations", tags=["chat"])


def get_conversations_dependency():
    settings = get_chat_settings()
    client = get_mongo_client(settings)
    return get_conversations_collection(client, settings)


def get_retrieval_traces_dependency():
    settings = get_chat_settings()
    client = get_mongo_client(settings)
    return get_retrieval_traces_collection(client, settings)


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


@router.get("/{conversation_id}/traces")
async def list_retrieval_traces_route(
    conversation_id: str, user_id: str | None = Depends(get_current_user_id),
    retrieval_traces=Depends(get_retrieval_traces_dependency),
):
    """Oldest-first per mode - the frontend zips these positionally against
    the conversation's own messages (see chat/repository.py's
    list_retrieval_traces docstring). Doesn't 404 on an empty/missing
    conversation - an absent or trace-less conversation just returns []."""
    user_id = _require_user_id(user_id)
    docs = await list_retrieval_traces(retrieval_traces, conversation_id, user_id)
    return [to_trace_summary(doc) for doc in docs]


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation_route(
    conversation_id: str, user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    deleted = await delete_conversation(conversations, conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
