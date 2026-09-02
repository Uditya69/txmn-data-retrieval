from common.repotaxmannapi_scoring import build_function_score_functions


def test_recency_ladder_matches_real_8_tier_formula():
    functions = build_function_score_functions(group_id="0")
    recency = [
        fn for fn in functions
        if "filter" in fn and "range" in fn.get("filter", {})
        and "formatteddocumentdate" in fn["filter"]["range"]
    ]
    assert len(recency) == 8
    weights = sorted((fn["weight"] for fn in recency), reverse=True)
    assert weights == [18, 15, 13, 10, 8, 5, 3.5, 1.5]


def test_recency_ladder_windows_match_real_date_math():
    functions = build_function_score_functions(group_id="0")
    recency = {
        fn["weight"]: fn["filter"]["range"]["formatteddocumentdate"]
        for fn in functions
        if "filter" in fn and "range" in fn.get("filter", {})
        and "formatteddocumentdate" in fn["filter"]["range"]
    }
    assert recency[18] == {"gte": "now-1d", "lte": "now"}
    assert recency[15] == {"gte": "now-7d", "lte": "now-1d"}
    assert recency[13] == {"gte": "now-1M", "lte": "now-7d"}
    assert recency[10] == {"gte": "now-3M", "lte": "now-1M"}
    assert recency[8] == {"gte": "now-1y", "lte": "now-3M"}
    assert recency[5] == {"gte": "now-2y", "lte": "now-1y"}
    assert recency[3.5] == {"gte": "now-5y", "lte": "now-2y"}
    assert recency[1.5] == {"gte": "now-150y", "lte": "now-5y"}


def test_field_value_factor_stack_matches_real_formula():
    functions = build_function_score_functions(group_id="0")
    by_field = {
        fn["field_value_factor"]["field"]: fn["field_value_factor"]
        for fn in functions if "field_value_factor" in fn
    }
    assert by_field["documenttypeboost"] == {"field": "documenttypeboost"}
    assert by_field["viewcount"] == {"field": "viewcount", "factor": 0.0000018, "modifier": "log2p"}
    assert by_field["court_boost"] == {
        "field": "court_boost", "factor": 0.0000018, "modifier": "log2p", "missing": 0,
    }
    assert by_field["total_score"] == {
        "field": "total_score", "factor": 0.0000018, "modifier": "log2p", "missing": 0,
    }
    assert by_field["landmarkruling"] == {
        "field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0,
    }


def test_landmarkruling_is_filtered_to_exclude_sentinel_value():
    functions = build_function_score_functions(group_id="0")
    landmark_fn = next(
        fn for fn in functions
        if fn.get("field_value_factor", {}).get("field") == "landmarkruling"
    )
    assert landmark_fn["filter"] == {"bool": {"must_not": [{"term": {"landmarkruling": -10}}]}}


def test_returns_exactly_the_5_static_field_value_factors_and_8_recency_tiers():
    functions = build_function_score_functions(group_id="0")
    fvf_fields = [fn["field_value_factor"]["field"] for fn in functions if "field_value_factor" in fn]
    assert sorted(fvf_fields) == sorted(
        ["documenttypeboost", "viewcount", "court_boost", "total_score", "landmarkruling"]
    )
    assert len(functions) == 8 + 5
