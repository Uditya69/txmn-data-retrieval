from pathlib import Path
from typing import AsyncIterator

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.retrieval_eval import evaluate_case, load_cases

DATASET_PATH = Path("evals/retrieval_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_cases(DATASET_PATH)
    if limit:
        cases = cases[:limit]  # first-N slice - NOT the same as evaluate_case's own `limit` kwarg
    total = len(cases)

    settings = get_settings()
    es_client = get_es_client(settings)
    milvus_client = get_milvus_client(settings)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0

    try:
        for i, case in enumerate(cases, start=1):
            try:
                result = await evaluate_case(
                    case, gateway, es_client, milvus_client,
                    langfuse_enabled=False, skip_agentic=True, skip_synthesis=True,
                )
            except Exception as exc:
                yield {"type": "case", "id": case["id"], "query": case["query"], "status": "error", "detail": {"error": str(exc)}}
            else:
                # "reranker" is the final retrieval-pipeline stage a synthesis call
                # actually consumes - used as the single pass/fail headline signal;
                # every stage's own rank is still exposed in detail for full context.
                rank = result["ranks"]["reranker"]
                case_ok = rank is not None and rank <= case["pass_at"]
                passed += case_ok
                yield {
                    "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                    "detail": {
                        "ranks": result["ranks"], "pass_at": case["pass_at"],
                        "errors": result["errors"], "timings_ms": result["timings_ms"],
                    },
                }
            yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}
    finally:
        await es_client.close()
        milvus_client.close()

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
