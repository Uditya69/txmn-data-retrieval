import pytest

from chat.repository import get_conversation
from retrieval_api.ai_mode.chat_signal import record_conversation_turn


@pytest.mark.asyncio
async def test_record_conversation_turn_writes_conversation(fake_conversations_collection):
    conversations = fake_conversations_collection

    await record_conversation_turn(conversations, "conv-1", "user-1", "gst rate", [{"role": "user", "text": "gst rate"}])

    stored = await get_conversation(conversations, "conv-1", "user-1")
    assert stored is not None
    assert stored["messages"] == [{"role": "user", "text": "gst rate"}]


@pytest.mark.asyncio
async def test_record_conversation_turn_swallows_errors():
    class BrokenCollection:
        async def find_one(self, filter):
            raise RuntimeError("mongo unreachable")

    # Must not raise - background task failures must never propagate.
    await record_conversation_turn(BrokenCollection(), "conv-1", "user-1", "q", [])
