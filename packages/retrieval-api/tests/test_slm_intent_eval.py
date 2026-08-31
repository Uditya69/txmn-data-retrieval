from retrieval_api.slm_intent_eval import tally_slm_records


def test_tally_slm_records_counts_category_outcomes_rewrite_filters_and_all_pass():
    records = [
        {"id": "S1", "category_outcome": "exact", "rewrite_ok": True, "filters_ok": True, "case_ok": True, "error": None},
        {"id": "S2", "category_outcome": "superset", "rewrite_ok": True, "filters_ok": False, "case_ok": False, "error": None},
        {"id": "S3", "category_outcome": "safe-empty", "rewrite_ok": False, "filters_ok": True, "case_ok": False, "error": None},
        {"id": "S4", "category_outcome": "wrong", "rewrite_ok": True, "filters_ok": True, "case_ok": False, "error": None},
        {"id": "S5", "category_outcome": None, "rewrite_ok": None, "filters_ok": None, "case_ok": None, "error": "boom"},
    ]

    result = tally_slm_records(records)

    assert result["total"] == 5
    assert result["errors"] == 1
    assert result["ran"] == 4
    assert result["cat_tally"] == {"exact": 1, "superset": 1, "safe-empty": 1, "wrong": 1}
    assert result["cat_passed"] == 3
    assert result["rewrite_pass"] == 3
    assert result["filters_pass"] == 3
    assert result["all_pass"] == 1
