import json
from pathlib import Path

import pytest

from retrieval_api.intent_eval import check_intent_case, load_intent_cases


def test_repository_intent_filter_dataset_has_twelve_cases_and_unique_ids():
    root = Path(__file__).parents[3]
    cases = load_intent_cases(root / "evals" / "intent_filter_cases.json")

    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12


def test_load_intent_cases_validates_required_keys(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "F1", "query": "q"}]))

    with pytest.raises(ValueError, match="missing"):
        load_intent_cases(path)


def test_load_intent_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "F1", "query": "q1", "expected_filters": {}},
        {"id": "F1", "query": "q2", "expected_filters": {}},
    ]))

    with pytest.raises(ValueError, match="duplicate"):
        load_intent_cases(path)


def test_check_intent_case_matches_exact_filters():
    assert check_intent_case({"court": "Bombay High Court"}, {"court": "Bombay High Court"}) is True


def test_check_intent_case_flags_mismatch():
    assert check_intent_case({"court": "Bombay High Court"}, {"court": "Delhi High Court"}) is False
    assert check_intent_case({"court": "Bombay High Court"}, {}) is False
    assert check_intent_case({}, {"court": "Bombay High Court"}) is False
