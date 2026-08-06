from agents.citations import validate_citations
from agents.loop import build_initial_messages, run_agent_loop

MAX_CITATION_RETRIES = 3


async def run_agentic_search(gateway, es_client, milvus_client, query: str, on_step=None) -> dict:
    messages = build_initial_messages(query)
    seen_doc_ids: set[str] = set()

    for attempt in range(1, MAX_CITATION_RETRIES + 1):
        loop_result = await run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=on_step)
        messages = loop_result["messages"]
        seen_doc_ids = loop_result["seen_doc_ids"]
        invalid = validate_citations(loop_result["answer"], seen_doc_ids)

        if not invalid:
            if on_step:
                await on_step("agent_answer", {"answer": loop_result["answer"], "doc_ids": sorted(seen_doc_ids)})
            return {"ok": True, "answer": loop_result["answer"], "doc_ids": sorted(seen_doc_ids)}

        if on_step:
            await on_step("agent_citation_rejected", {"invalid_doc_ids": invalid, "attempt": attempt})

        if attempt == MAX_CITATION_RETRIES:
            return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": invalid}

        messages = messages + [{
            "role": "user",
            "content": (
                f"Your answer cited doc_id(s) not retrieved by any tool call this session: {invalid}. "
                "Revise your answer to cite only doc_ids you actually retrieved."
            ),
        }]
