import argparse
import asyncio
import json
from pathlib import Path

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient


def load_intent_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("intent filter eval dataset must be a non-empty JSON array")
    seen: set[str] = set()
    for case in cases:
        required = {"id", "query", "expected_filters", "expected_categories"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
    return cases


def check_intent_case(
    expected_filters: dict, actual_filters: dict,
    expected_categories: list[str], actual_categories: list[str],
) -> tuple[bool, bool]:
    filters_ok = expected_filters == actual_filters
    categories_ok = set(expected_categories) == set(actual_categories)
    return filters_ok, categories_ok


async def run(gateway_url: str, model: str | None, dataset_path: str | Path) -> None:
    cases = load_intent_cases(dataset_path)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0
    for case in cases:
        try:
            result = await extract_intent(gateway, case["query"], model=model)
        except Exception as exception:
            print(f"ERROR {case['id']}: {exception}")
            continue
        filters_ok, categories_ok = check_intent_case(
            case["expected_filters"], result["filters"],
            case["expected_categories"], result["intent"],
        )
        ok = filters_ok and categories_ok
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {case['id']}: "
            f"filters(expected={case['expected_filters']} actual={result['filters']} ok={filters_ok}) "
            f"categories(expected={case['expected_categories']} actual={result['intent']} ok={categories_ok})"
        )
    print(f"\n{passed}/{len(cases)} passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-only intent/filter/category extraction accuracy check")
    parser.add_argument("--gateway-url", default="http://localhost:8011")
    parser.add_argument("--model", default=None, help="Override the slm role's model")
    parser.add_argument("--dataset", default="evals/intent_filter_cases.json")
    args = parser.parse_args()
    asyncio.run(run(args.gateway_url, args.model, args.dataset))


if __name__ == "__main__":
    main()
