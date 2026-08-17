import argparse
import asyncio
import json
from pathlib import Path

from common.schemas import collections_for_intent
from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient

_VALID_EXPECT = {"confident", "vague"}


def load_routing_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("collection routing eval dataset must be a non-empty JSON array")
    seen: set[str] = set()
    for case in cases:
        required = {"id", "query", "expect", "expected_categories"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["expect"] not in _VALID_EXPECT:
            raise ValueError(f"{case['id']}: expect must be one of {sorted(_VALID_EXPECT)}, got {case['expect']!r}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
    return cases


def check_routing_case(expected_categories: list[str], actual_categories: list[str]) -> str:
    """Returns 'exact', 'safe-empty', or 'wrong'. An empty actual result is always a pass
    (search-all is never a defect, whether the query was genuinely vague or one we expected
    to classify confidently) - only a non-empty, mismatched result is a failure: a confidently
    wrong category tag, which silently narrows the search to the wrong collections."""
    if not actual_categories:
        return "safe-empty"
    if set(actual_categories) == set(expected_categories):
        return "exact"
    return "wrong"


async def run(gateway_url: str, model: str | None, dataset_path: str | Path) -> None:
    cases = load_routing_cases(dataset_path)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    tally = {"exact": 0, "safe-empty": 0, "wrong": 0}
    by_expect = {"confident": {"exact": 0, "safe-empty": 0, "wrong": 0}, "vague": {"exact": 0, "safe-empty": 0, "wrong": 0}}

    for case in cases:
        try:
            result = await extract_intent(gateway, case["query"], model=model)
        except Exception as exception:
            print(f"ERROR {case['id']}: {exception}")
            continue
        actual = result["intent"]
        outcome = check_routing_case(case["expected_categories"], actual)
        tally[outcome] += 1
        by_expect[case["expect"]][outcome] += 1
        status = "PASS" if outcome != "wrong" else "FAIL"
        searched = collections_for_intent(actual)
        print(
            f"{status} {case['id']} ({outcome}) [{case['expect']}]: "
            f"expected={case['expected_categories']} actual={actual} "
            f"searched_collections={len(searched)}"
        )

    passed = tally["exact"] + tally["safe-empty"]
    total = sum(tally.values())
    print(f"\n{passed}/{total} passed  (exact={tally['exact']} safe-empty={tally['safe-empty']} wrong={tally['wrong']})")
    for expect_label, counts in by_expect.items():
        expect_total = sum(counts.values())
        if expect_total:
            expect_passed = counts["exact"] + counts["safe-empty"]
            print(
                f"  {expect_label}: {expect_passed}/{expect_total} passed "
                f"(exact={counts['exact']} safe-empty={counts['safe-empty']} wrong={counts['wrong']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collection-routing accuracy check: intent must either exactly match the "
        "expected category tag, or come back empty (safe search-all fallback) - never a "
        "confidently wrong non-empty tag."
    )
    parser.add_argument("--gateway-url", default="http://localhost:8011")
    parser.add_argument("--model", default=None, help="Override the slm role's model")
    parser.add_argument("--dataset", default="evals/collection_routing_cases.json")
    args = parser.parse_args()
    asyncio.run(run(args.gateway_url, args.model, args.dataset))


if __name__ == "__main__":
    main()
