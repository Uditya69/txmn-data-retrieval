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
    """Returns 'exact', 'superset', 'safe-empty', or 'wrong'.

    - empty actual is always a pass ('safe-empty') - search-all is never a defect.
    - actual == expected is 'exact'.
    - actual is a strict superset of expected ('exact' set plus extra categories) is
      'superset' - also a pass. Searching a couple of extra collections is cheap; the
      only thing that actually hurts recall is DROPPING an expected collection from
      the search, which is what 'wrong' below catches.
    - anything missing at least one expected category is 'wrong' (the failure mode
      that actually narrows the search away from the right collection)."""
    if not actual_categories:
        return "safe-empty"
    expected_set, actual_set = set(expected_categories), set(actual_categories)
    if actual_set == expected_set:
        return "exact"
    if expected_set.issubset(actual_set):
        return "superset"
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


def _filter_value_matches(expected, actual) -> bool:
    """date_range needs exact {gte/lte} equality - a shifted bound is a real defect.
    Everything else is a free-text span the model drew from the query itself
    (_sanitize_filters already guarantees it's a verbatim substring of the query, so
    nothing here is fabricated) - the model is consistently inconsistent about exactly
    where to draw that span (e.g. "Mumbai" vs "ITAT Mumbai bench" for the same forum,
    or a full case caption vs just the party name), not wrong about which entity it
    means. So string values match on case-insensitive containment either direction,
    not exact equality - punishing verbosity/terseness on a correctly-identified
    entity isn't a real defect."""
    if isinstance(expected, dict) or isinstance(actual, dict):
        return expected == actual
    expected_cf, actual_cf = str(expected).casefold(), str(actual).casefold()
    return expected_cf in actual_cf or actual_cf in expected_cf


def check_filters(expected_filters: dict, actual_filters: dict) -> bool:
    """Filters are best-effort help, not a completeness contract: the model isn't
    required to extract every filterable entity in the query, just to get right
    whatever it does attempt. So a MISSING expected key (the model simply didn't
    extract it) is not a failure - some useful narrowing is still better than none.
    Only a genuinely WRONG value for a key the model did attempt is a real defect -
    see _filter_value_matches for what counts as "wrong" vs. just differently-shaped."""
    return all(
        key not in actual_filters or _filter_value_matches(value, actual_filters[key])
        for key, value in expected_filters.items()
    )


async def run(gateway_url: str, model: str | None, dataset_path: str | Path, limit: int | None) -> None:
    cases = load_cases(dataset_path)
    if limit is not None:
        cases = cases[:limit]
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)

    cat_tally = {"exact": 0, "superset": 0, "safe-empty": 0, "wrong": 0}
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
    cat_passed = cat_tally["exact"] + cat_tally["superset"] + cat_tally["safe-empty"]
    print("\n--- summary ---")
    print(f"cases run: {ran}/{total} (errors={errors})")
    if ran:
        print(
            f"categories: {cat_passed}/{ran} passed "
            f"(exact={cat_tally['exact']} superset={cat_tally['superset']} "
            f"safe-empty={cat_tally['safe-empty']} wrong={cat_tally['wrong']})"
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
