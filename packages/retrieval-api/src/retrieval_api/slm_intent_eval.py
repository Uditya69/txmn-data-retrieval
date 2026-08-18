"""Combined accuracy check for the chat:slm role's extract_intent call - covers all
three things it produces in one pass: the rewritten search_query (chat:slm rewrite
safety), the intent category tags (collection routing), and the extracted filters.

Reuses the same real extract_intent() call the AI Mode pipeline runs (via
GatewayClient against a live model-gateway) - no mocking. Distinct from
intent_eval.py (filters+categories only, exact-match) and collection_routing_eval.py
(categories only, safe-empty tolerant): this script also grades the search_query
rewrite itself (must_contain / rewrite_must_not_contain substrings), and supports
--limit to run only the first N cases of the dataset for a fast smoke pass.
"""

import argparse
import asyncio
import json
from pathlib import Path

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient

_VALID_EXPECT = {"confident", "vague"}


def load_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("slm/intent eval dataset must be a non-empty JSON array")
    seen: set[str] = set()
    for case in cases:
        required = {"id", "query", "expect", "expected_categories", "expected_filters"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["expect"] not in _VALID_EXPECT:
            raise ValueError(f"{case['id']}: expect must be one of {sorted(_VALID_EXPECT)}, got {case['expect']!r}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
    return cases


def check_categories(expected_categories: list[str], actual_categories: list[str]) -> str:
    """Returns 'exact', 'safe-empty', or 'wrong' - same rule as collection_routing_eval:
    an empty actual result is always a pass (search-all is never a defect); only a
    non-empty, mismatched result fails."""
    if not actual_categories:
        return "safe-empty"
    if set(actual_categories) == set(expected_categories):
        return "exact"
    return "wrong"


def check_rewrite(case: dict, search_query: str) -> tuple[bool, list[str]]:
    """Grades the chat:slm rewrite against must_contain / rewrite_must_not_contain
    substring lists (case-insensitive). Returns (ok, reasons-for-failure)."""
    rewritten_lower = search_query.casefold()
    reasons = []
    for token in case.get("rewrite_must_contain", []):
        if token.casefold() not in rewritten_lower:
            reasons.append(f"missing required token {token!r}")
    for token in case.get("rewrite_must_not_contain", []):
        if token.casefold() in rewritten_lower:
            reasons.append(f"forbidden token {token!r} leaked into rewrite")
    return not reasons, reasons


def check_filters(expected_filters: dict, actual_filters: dict) -> bool:
    return expected_filters == actual_filters


async def run(gateway_url: str, model: str | None, dataset_path: str | Path, limit: int | None) -> None:
    cases = load_cases(dataset_path)
    if limit is not None:
        cases = cases[:limit]
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)

    cat_tally = {"exact": 0, "safe-empty": 0, "wrong": 0}
    rewrite_pass = 0
    filters_pass = 0
    all_pass = 0
    errors = 0

    for case in cases:
        try:
            result = await extract_intent(gateway, case["query"], model=model)
        except Exception as exception:
            print(f"ERROR {case['id']}: {exception}")
            errors += 1
            continue

        cat_outcome = check_categories(case["expected_categories"], result["intent"])
        cat_tally[cat_outcome] += 1
        rewrite_ok, rewrite_reasons = check_rewrite(case, result["search_query"])
        filters_ok = check_filters(case["expected_filters"], result["filters"])

        rewrite_pass += rewrite_ok
        filters_pass += filters_ok
        case_ok = cat_outcome != "wrong" and rewrite_ok and filters_ok
        all_pass += case_ok

        status = "PASS" if case_ok else "FAIL"
        print(
            f"{status} {case['id']} [{case['expect']}]\n"
            f"  categories: {cat_outcome} (expected={case['expected_categories']} actual={result['intent']})\n"
            f"  rewrite: {'ok' if rewrite_ok else 'FAIL ' + '; '.join(rewrite_reasons)} "
            f"(original={case['query']!r} -> rewritten={result['search_query']!r})\n"
            f"  filters: {'ok' if filters_ok else 'FAIL'} (expected={case['expected_filters']} actual={result['filters']})"
        )

    total = len(cases)
    ran = total - errors
    cat_passed = cat_tally["exact"] + cat_tally["safe-empty"]
    print("\n--- summary ---")
    print(f"cases run: {ran}/{total} (errors={errors})")
    if ran:
        print(
            f"categories: {cat_passed}/{ran} passed "
            f"(exact={cat_tally['exact']} safe-empty={cat_tally['safe-empty']} wrong={cat_tally['wrong']})"
        )
        print(f"rewrite:    {rewrite_pass}/{ran} passed")
        print(f"filters:    {filters_pass}/{ran} passed")
        print(f"overall:    {all_pass}/{ran} passed (all three checks)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accuracy check for the chat:slm extract_intent call: rewritten "
        "search_query safety, intent category routing, and filter extraction - all "
        "three graded together per case."
    )
    parser.add_argument("--gateway-url", default="http://localhost:8001")
    parser.add_argument("--model", default=None, help="Override the slm role's model")
    parser.add_argument("--dataset", default="evals/slm_intent_cases.json")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N cases of the dataset (e.g. --limit 10 out of 50 total). "
        "Default: run all cases.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    asyncio.run(run(args.gateway_url, args.model, args.dataset, args.limit))


if __name__ == "__main__":
    main()
