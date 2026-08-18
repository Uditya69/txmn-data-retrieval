from pathlib import Path
from typing import AsyncIterator

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.intent_eval import check_intent_case, load_intent_cases

DATASET_PATH = Path("evals/intent_filter_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_intent_cases(DATASET_PATH)
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
            filters_ok, categories_ok = check_intent_case(
                case["expected_filters"], result["filters"], case["expected_categories"], result["intent"],
            )
            case_ok = filters_ok and categories_ok
            passed += case_ok
            yield {
                "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                "detail": {
                    "filters": {"ok": filters_ok, "expected": case["expected_filters"], "actual": result["filters"]},
                    "categories": {"ok": categories_ok, "expected": case["expected_categories"], "actual": result["intent"]},
                },
            }
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
