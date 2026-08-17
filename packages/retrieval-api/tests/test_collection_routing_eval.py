import json
from pathlib import Path

import pytest

from retrieval_api.collection_routing_eval import check_routing_case, load_routing_cases


def test_repository_routing_dataset_has_cases_and_unique_ids():
    root = Path(__file__).parents[3]
    cases = load_routing_cases(root / "evals" / "collection_routing_cases.json")

    assert len(cases) >= 12
    assert len({case["id"] for case in cases}) == len(cases)


def test_repository_routing_dataset_has_both_confident_and_vague_cases():
    root = Path(__file__).parents[3]
    cases = load_routing_cases(root / "evals" / "collection_routing_cases.json")

    expects = {case["expect"] for case in cases}
    assert expects == {"confident", "vague"}


def test_repository_routing_dataset_vague_cases_expect_empty_categories():
    root = Path(__file__).parents[3]
    cases = load_routing_cases(root / "evals" / "collection_routing_cases.json")

    for case in cases:
        if case["expect"] == "vague":
            assert case["expected_categories"] == []


def test_load_routing_cases_validates_required_keys(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "R1", "query": "q", "expect": "vague"}]))

    with pytest.raises(ValueError, match="missing"):
        load_routing_cases(path)


def test_load_routing_cases_rejects_invalid_expect_value(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "R1", "query": "q", "expect": "sort-of", "expected_categories": []},
    ]))

    with pytest.raises(ValueError, match="expect must be one of"):
        load_routing_cases(path)


def test_load_routing_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "R1", "query": "q1", "expect": "vague", "expected_categories": []},
        {"id": "R1", "query": "q2", "expect": "vague", "expected_categories": []},
    ]))

    with pytest.raises(ValueError, match="duplicate"):
        load_routing_cases(path)


def test_check_routing_case_exact_match_on_confident_case():
    assert check_routing_case(["caselaws"], ["caselaws"]) == "exact"


def test_check_routing_case_order_independent_exact_match():
    assert check_routing_case(["acts", "caselaws"], ["caselaws", "acts"]) == "exact"


def test_check_routing_case_empty_actual_is_always_safe_empty():
    # Whether the case was expected to classify confidently or not, an empty result
    # (search-all fallback) is never a defect.
    assert check_routing_case(["caselaws"], []) == "safe-empty"
    assert check_routing_case([], []) == "safe-empty"


def test_check_routing_case_wrong_nonempty_result_is_wrong():
    assert check_routing_case(["caselaws"], ["tariff"]) == "wrong"


def test_check_routing_case_nonempty_result_on_vague_case_is_wrong():
    # A confidently-tagged result on a genuinely vague query is the exact failure mode
    # this eval exists to catch - never treated as a pass just because it's "plausible."
    assert check_routing_case([], ["caselaws"]) == "wrong"
