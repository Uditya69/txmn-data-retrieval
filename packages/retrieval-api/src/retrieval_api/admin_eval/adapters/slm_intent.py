from pathlib import Path
from typing import AsyncIterator

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.slm_intent_eval import check_categories, check_filters, check_rewrite, load_cases

DATASET_PATH = Path("evals/slm_intent_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_cases(DATASET_PATH)
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
            cat_status = check_categories(case["expected_categories"], result["intent"])
            rewrite_ok, rewrite_reasons = check_rewrite(case, result["search_query"])
            filters_ok = check_filters(case["expected_filters"], result["filters"])
            case_ok = cat_status != "wrong" and rewrite_ok and filters_ok
            passed += case_ok
            yield {
                "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                "detail": {
                    "rewrite": result["search_query"], "rewrite_ok": rewrite_ok, "rewrite_reasons": rewrite_reasons,
                    "categories": {"status": cat_status, "expected": case["expected_categories"], "actual": result["intent"]},
                    "filters": {"ok": filters_ok, "expected": case["expected_filters"], "actual": result["filters"]},
                },
            }
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
