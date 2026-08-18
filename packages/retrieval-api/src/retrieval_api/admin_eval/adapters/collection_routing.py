from pathlib import Path
from typing import AsyncIterator

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.collection_routing_eval import check_routing_case, load_routing_cases
from retrieval_api.gateway_client import GatewayClient

DATASET_PATH = Path("evals/collection_routing_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_routing_cases(DATASET_PATH)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0

    for i, case in enumerate(cases, start=1):
        try:
            result = await extract_intent(gateway, case["query"])
        except Exception as exc:
            yield {"type": "case", "id": case["id"], "query": case["query"], "status": "error", "detail": {"error": str(exc)}}
        else:
            outcome = check_routing_case(case["expected_categories"], result["intent"])
            case_ok = outcome != "wrong"
            passed += case_ok
            yield {
                "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                "detail": {
                    "outcome": outcome, "expect": case["expect"],
                    "expected": case["expected_categories"], "actual": result["intent"],
                },
            }
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
