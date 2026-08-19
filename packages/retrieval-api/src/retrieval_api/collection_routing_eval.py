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
    """Returns 'exact', 'superset', 'safe-empty', or 'wrong'.

    - empty actual is always a pass ('safe-empty') - search-all is never a defect,
      whether the query was genuinely vague or one we expected to classify confidently.
    - actual == expected is 'exact'.
    - actual is a strict superset of expected (all expected categories present, plus
      extra) is 'superset' - also a pass. Searching a couple of extra collections is
      cheap; the only thing that actually hurts recall is DROPPING an expected
      collection from the search, which is what 'wrong' below catches.
    - anything missing at least one expected category is 'wrong' - a confidently
      wrong/incomplete category tag, which silently narrows the search away from a
      collection it should have covered."""
    if not actual_categories:
        return "safe-empty"
    expected_set, actual_set = set(expected_categories), set(actual_categories)
    if actual_set == expected_set:
        return "exact"
    # expected_set must be non-empty here too - an empty expected_set is trivially a
    # subset of any actual_set, which would otherwise call a confidently-wrong tag on
    # a genuinely vague query (expected_categories == []) a "superset" pass instead of
    # the "wrong" this eval exists to catch.
    if expected_set and expected_set.issubset(actual_set):
        return "superset"
    return "wrong"


async def run(gateway_url: str, model: str | None, dataset_path: str | Path) -> None:
    cases = load_routing_cases(dataset_path)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    tally = {"exact": 0, "superset": 0, "safe-empty": 0, "wrong": 0}
    by_expect = {
        "confident": {"exact": 0, "superset": 0, "safe-empty": 0, "wrong": 0},
        "vague": {"exact": 0, "superset": 0, "safe-empty": 0, "wrong": 0},
    }

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

    passed = tally["exact"] + tally["superset"] + tally["safe-empty"]
    total = sum(tally.values())
    print(
        f"\n{passed}/{total} passed  (exact={tally['exact']} superset={tally['superset']} "
        f"safe-empty={tally['safe-empty']} wrong={tally['wrong']})"
    )
    for expect_label, counts in by_expect.items():
        expect_total = sum(counts.values())
        if expect_total:
            expect_passed = counts["exact"] + counts["superset"] + counts["safe-empty"]
            print(
                f"  {expect_label}: {expect_passed}/{expect_total} passed "
                f"(exact={counts['exact']} superset={counts['superset']} "
                f"safe-empty={counts['safe-empty']} wrong={counts['wrong']})"
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
