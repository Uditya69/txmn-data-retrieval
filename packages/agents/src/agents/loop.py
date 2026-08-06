import json
from typing import Awaitable, Callable

from agents.tools import TOOL_SCHEMAS, dispatch_tool_call

OnStep = Callable[[str, dict], Awaitable[None]]

SYSTEM_PROMPT = (
    "You are a legal research assistant over Indian case-law. Use the available "
    "search tools to find evidence before answering. Every claim in your final "
    "answer must be backed by a doc_id you retrieved via a tool call this "
    "session. Cite doc_ids inline in square brackets, e.g. [12345]. Never cite "
    "a doc_id you did not actually retrieve."
)


def build_initial_messages(query: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def _collect_doc_ids(tool_name: str, arguments: dict, result: dict) -> set[str]:
    doc_ids: set[str] = set()
    for row in result.get("rows", []):
        if "doc_id" in row:
            doc_ids.add(str(row["doc_id"]))
    if tool_name == "lookup_doc" and result.get("citation") is not None:
        doc_ids.add(str(arguments.get("doc_id")))
    return doc_ids


async def run_agent_loop(
    gateway, es_client, milvus_client, messages: list[dict], seen_doc_ids: set[str], on_step: OnStep | None = None,
) -> dict:
    messages = list(messages)
    seen_doc_ids = set(seen_doc_ids)

    while True:
        response = await gateway.chat_with_tools(role="agent_chat", messages=messages, tools=TOOL_SCHEMAS)
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return {"answer": response.get("content") or "", "seen_doc_ids": seen_doc_ids, "messages": messages}

        messages.append({"role": "assistant", "content": response.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            if on_step:
                await on_step("agent_tool_call", {"name": name, "arguments": arguments})
            try:
                result = await dispatch_tool_call(
                    name, arguments, gateway=gateway, es_client=es_client, milvus_client=milvus_client,
                )
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            seen_doc_ids |= _collect_doc_ids(name, arguments, result)
            if on_step:
                await on_step("agent_tool_result", {"name": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
