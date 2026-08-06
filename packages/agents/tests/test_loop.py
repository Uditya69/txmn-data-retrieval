import pytest

from agents.loop import build_initial_messages, run_agent_loop


def test_build_initial_messages_has_system_and_user_turns():
    messages = build_initial_messages("what is the rate for GST on X")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "what is the rate for GST on X"}


@pytest.mark.asyncio
async def test_loop_returns_answer_immediately_when_no_tool_calls():
    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            return {"content": "final answer, no tools needed", "tool_calls": None, "reasoning": None}

    result = await run_agent_loop(
        FakeGateway(), es_client=None, milvus_client=None,
        messages=build_initial_messages("q"), seen_doc_ids=set(),
    )

    assert result["answer"] == "final answer, no tools needed"
    assert result["seen_doc_ids"] == set()


@pytest.mark.asyncio
async def test_loop_dispatches_tool_call_tracks_doc_ids_and_continues_until_final_answer(monkeypatch):
    import agents.loop as loop_module

    calls = {"n": 0}

    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {
                        "name": "search_es", "arguments": '{"query": "gst"}',
                    }}],
                    "reasoning": None,
                }
            return {"content": "answer citing [d1]", "tool_calls": None, "reasoning": None}

    async def fake_dispatch_tool_call(name, arguments, *, gateway, es_client, milvus_client):
        assert name == "search_es"
        assert arguments == {"query": "gst"}
        return {"rows": [{"doc_id": "d1", "score": 1.0}]}

    monkeypatch.setattr(loop_module, "dispatch_tool_call", fake_dispatch_tool_call)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await run_agent_loop(
        FakeGateway(), es_client=object(), milvus_client=object(),
        messages=build_initial_messages("q"), seen_doc_ids=set(), on_step=on_step,
    )

    assert result["answer"] == "answer citing [d1]"
    assert result["seen_doc_ids"] == {"d1"}
    assert [s for s, _ in steps] == ["agent_tool_call", "agent_tool_result"]
    assert steps[0][1] == {"name": "search_es", "arguments": {"query": "gst"}}
    assert steps[1][1] == {"name": "search_es", "result": {"rows": [{"doc_id": "d1", "score": 1.0}]}}


@pytest.mark.asyncio
async def test_loop_records_lookup_doc_citation_doc_id_and_survives_tool_error(monkeypatch):
    import agents.loop as loop_module

    calls = {"n": 0}

    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"x\"}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "lookup_doc", "arguments": "{\"doc_id\": \"d2\"}"}},
                    ],
                    "reasoning": None,
                }
            return {"content": "done", "tool_calls": None, "reasoning": None}

    async def fake_dispatch_tool_call(name, arguments, *, gateway, es_client, milvus_client):
        if name == "search_es":
            raise RuntimeError("ES timed out")
        return {"citation": {"court": "SC"}}

    monkeypatch.setattr(loop_module, "dispatch_tool_call", fake_dispatch_tool_call)

    result = await run_agent_loop(
        FakeGateway(), es_client=object(), milvus_client=object(),
        messages=build_initial_messages("q"), seen_doc_ids=set(),
    )

    assert result["answer"] == "done"
    assert result["seen_doc_ids"] == {"d2"}
    tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
    assert "RuntimeError: ES timed out" in tool_messages[0]["content"]
