import pytest
from pymongo.errors import DuplicateKeyError

from chat.repository import (
    append_turn,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
)


@pytest.mark.asyncio
async def test_create_conversation_stores_document(fake_conversations_collection):
    conversations = fake_conversations_collection
    doc = await create_conversation(conversations, "conv-1", "user-1", "first question", [{"role": "user", "text": "hi"}])
    assert doc["_id"] == "conv-1"
    assert doc["user_id"] == "user-1"
    assert doc["title"] == "first question"
    assert doc["messages"] == [{"role": "user", "text": "hi"}]
    assert doc["created_at"] == doc["updated_at"]


@pytest.mark.asyncio
async def test_create_conversation_does_not_clobber_another_users_doc_with_same_id(fake_conversations_collection):
    # Regression test: the frontend used to generate conversation ids from a
    # module-scoped counter that reset on every page load, so two different
    # users' first conversations could collide on the same id. Scoping the
    # upsert filter by user_id (not just _id) means a same-id collision from
    # a different user can no longer silently overwrite the first user's
    # title/messages/user_id - it now fails loudly instead of corrupting data.
    conversations = fake_conversations_collection
    first = await create_conversation(conversations, "conv-1", "user-1", "user 1's question", [{"role": "user", "text": "hi"}])

    with pytest.raises(DuplicateKeyError):
        await create_conversation(conversations, "conv-1", "user-2", "user 2's question", [{"role": "user", "text": "hey"}])

    # user-1's original document must be completely intact.
    unchanged = await get_conversation(conversations, "conv-1", "user-1")
    assert unchanged == first
    assert unchanged["title"] == "user 1's question"
    assert unchanged["user_id"] == "user-1"
    # user-2 never got a document written under this id.
    assert await get_conversation(conversations, "conv-1", "user-2") is None


@pytest.mark.asyncio
async def test_append_turn_extends_existing_conversation(fake_conversations_collection):
    # append_turn's contract: the caller passes only THIS turn's new messages,
    # not the full history - append_turn is responsible for concatenating
    # them onto whatever's already stored.
    conversations = fake_conversations_collection
    await create_conversation(
        conversations, "conv-1", "user-1", "first question",
        [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}],
    )
    updated = await append_turn(
        conversations, "conv-1", "user-1", "first question",
        [{"role": "user", "text": "how are you"}, {"role": "assistant", "text": "good"}],
    )
    assert updated["messages"] == [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "hello"},
        {"role": "user", "text": "how are you"},
        {"role": "assistant", "text": "good"},
    ]
    assert updated["updated_at"] >= updated["created_at"]


@pytest.mark.asyncio
async def test_append_turn_creates_conversation_when_absent(fake_conversations_collection):
    conversations = fake_conversations_collection
    doc = await append_turn(conversations, "conv-new", "user-1", "q", [{"role": "user", "text": "q"}])
    assert doc["_id"] == "conv-new"
    assert doc["messages"] == [{"role": "user", "text": "q"}]


@pytest.mark.asyncio
async def test_append_turn_third_call_appends_onto_two_prior_turns(fake_conversations_collection):
    # Regression test for the bug where append_turn REPLACED the messages
    # field with whatever the caller passed instead of appending - a third
    # write must retain both prior turns, not just the most recent one.
    conversations = fake_conversations_collection
    await create_conversation(
        conversations, "conv-1", "user-1", "q",
        [{"role": "user", "text": "t1u"}, {"role": "assistant", "text": "t1a"}],
    )
    await append_turn(
        conversations, "conv-1", "user-1", "q",
        [{"role": "user", "text": "t2u"}, {"role": "assistant", "text": "t2a"}],
    )
    updated = await append_turn(
        conversations, "conv-1", "user-1", "q",
        [{"role": "user", "text": "t3u"}, {"role": "assistant", "text": "t3a"}],
    )
    assert [m["text"] for m in updated["messages"]] == ["t1u", "t1a", "t2u", "t2a", "t3u", "t3a"]


@pytest.mark.asyncio
async def test_list_conversations_returns_only_callers_own_newest_first(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "q1", [])
    await create_conversation(conversations, "conv-2", "user-1", "q2", [])
    await create_conversation(conversations, "conv-3", "user-2", "other user", [])

    result = await list_conversations(conversations, "user-1")

    assert [c["_id"] for c in result] == ["conv-2", "conv-1"]


@pytest.mark.asyncio
async def test_get_conversation_returns_none_for_wrong_user(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "q", [])
    assert await get_conversation(conversations, "conv-1", "user-2") is None
    assert await get_conversation(conversations, "conv-1", "user-1") is not None


@pytest.mark.asyncio
async def test_delete_conversation_removes_only_owners_document(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "q", [])
    assert await delete_conversation(conversations, "conv-1", "user-2") is False
    assert await delete_conversation(conversations, "conv-1", "user-1") is True
    assert await get_conversation(conversations, "conv-1", "user-1") is None
