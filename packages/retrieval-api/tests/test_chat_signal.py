import pytest

from chat.repository import get_conversation
from retrieval_api.ai_mode.chat_signal import record_conversation_turn, record_retrieval_trace


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


@pytest.mark.asyncio
async def test_record_retrieval_trace_writes_instant_document(fake_retrieval_traces_collection):
    traces = fake_retrieval_traces_collection

    await record_retrieval_trace(
        traces, "conv-1", "user-1", "instant", "gst rate", "trace-1", "obs-1",
        instant={"reranked": []},
    )

    assert len(traces.documents) == 1
    stored = traces.documents[0]
    assert stored["mode"] == "instant"
    assert stored["langfuse_trace_id"] == "trace-1"
    assert stored["instant"] == {"reranked": []}


@pytest.mark.asyncio
async def test_record_retrieval_trace_swallows_errors():
    class BrokenCollection:
        async def insert_one(self, document):
            raise RuntimeError("mongo unreachable")

    # Must not raise - background task failures must never propagate.
    await record_retrieval_trace(BrokenCollection(), "conv-1", "user-1", "ai_mode", "q", None, None)
