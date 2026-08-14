import json
from pathlib import Path

import pytest

from retrieval_api.intent_eval import check_intent_case, load_intent_cases


def test_repository_intent_filter_dataset_has_cases_and_unique_ids():
    root = Path(__file__).parents[3]
    cases = load_intent_cases(root / "evals" / "intent_filter_cases.json")

    assert len(cases) >= 12
    assert len({case["id"] for case in cases}) == len(cases)


def test_repository_intent_filter_dataset_covers_every_category():
    root = Path(__file__).parents[3]
    cases = load_intent_cases(root / "evals" / "intent_filter_cases.json")

    covered = {category for case in cases for category in case["expected_categories"]}
    assert covered == {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}


def test_load_intent_cases_validates_required_keys(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "F1", "query": "q"}]))

    with pytest.raises(ValueError, match="missing"):
        load_intent_cases(path)


def test_load_intent_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "F1", "query": "q1", "expected_filters": {}, "expected_categories": []},
        {"id": "F1", "query": "q2", "expected_filters": {}, "expected_categories": []},
    ]))

    with pytest.raises(ValueError, match="duplicate"):
        load_intent_cases(path)


def test_check_intent_case_matches_exact_filters_and_categories():
    assert check_intent_case(
        {"court": "Bombay High Court"}, {"court": "Bombay High Court"}, ["caselaws"], ["caselaws"],
    ) == (True, True)


def test_check_intent_case_flags_filter_mismatch_independently_of_category():
    filters_ok, categories_ok = check_intent_case(
        {"court": "Bombay High Court"}, {}, ["caselaws"], ["caselaws"],
    )
    assert filters_ok is False
    assert categories_ok is True


def test_check_intent_case_flags_category_mismatch_independently_of_filters():
    filters_ok, categories_ok = check_intent_case(
        {}, {}, ["acts"], ["caselaws"],
    )
    assert filters_ok is True
    assert categories_ok is False


def test_check_intent_case_category_match_is_order_independent():
    filters_ok, categories_ok = check_intent_case(
        {}, {}, ["acts", "caselaws"], ["caselaws", "acts"],
    )
    assert categories_ok is True
