from common.repotaxmannapi_scoring import build_function_score_functions


def test_recency_ladder_matches_real_8_tier_formula():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    recency = [
        fn for fn in functions
        if "filter" in fn and "range" in fn.get("filter", {})
        and "formatteddocumentdate" in fn["filter"]["range"]
    ]
    assert len(recency) == 8
    weights = sorted((fn["weight"] for fn in recency), reverse=True)
    assert weights == [18, 15, 13, 10, 8, 5, 3.5, 1.5]


def test_recency_ladder_windows_match_real_date_math():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
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
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
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
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    landmark_fn = next(
        fn for fn in functions
        if fn.get("field_value_factor", {}).get("field") == "landmarkruling"
    )
    assert landmark_fn["filter"] == {"bool": {"must_not": [{"term": {"landmarkruling": -10}}]}}


def test_returns_exactly_the_5_static_field_value_factors_and_8_recency_tiers():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    fvf_fields = [fn["field_value_factor"]["field"] for fn in functions if "field_value_factor" in fn]
    assert sorted(fvf_fields) == sorted(
        ["documenttypeboost", "viewcount", "court_boost", "total_score", "landmarkruling"]
    )
    # For a group_id that is neither the Act nor Rule group, the two unconditional
    # group.id/subgroup.id Weight functions are still present (10000000 each), but neither
    # edition-subgroup Weight function is (group_id doesn't resolve to an edition id).
    # Plus the two unconditional stateGst/financeact penalty Weight functions (0.03/0.02)
    # added in Task 8 - present for every group_id, not just Act/Rule groups.
    assert len(functions) == 2 + 8 + 5 + 2


def test_group_boost_defaults_to_ten_million_for_unrelated_group():
    functions = build_function_score_functions(group_id="999999999", latest_finance_act_year="2025")
    group_id_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.id", {}).get("query") == "999999999"
    )
    assert group_id_fn["weight"] == 10000000


def test_group_boost_is_two_for_act_group():
    functions = build_function_score_functions(group_id="111050000000000064", latest_finance_act_year="2025")
    group_id_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.id", {}).get("query") == "111050000000000064"
    )
    assert group_id_fn["weight"] == 2


def test_group_boost_is_four_for_rule_group():
    functions = build_function_score_functions(group_id="111050000000000026", latest_finance_act_year="2025")
    group_id_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.id", {}).get("query") == "111050000000000026"
    )
    assert group_id_fn["weight"] == 4


def test_subgroup_id_weight_function_mirrors_group_id_weight_function():
    functions = build_function_score_functions(group_id="111050000000000064", latest_finance_act_year="2025")
    subgroup_fn = next(
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.subgroup.id", {}).get("query")
        == "111050000000000064"
    )
    assert subgroup_fn["weight"] == 2


def test_act_group_gets_old_and_new_income_tax_act_edition_subgroup_boosts():
    functions = build_function_score_functions(group_id="111050000000000064", latest_finance_act_year="2025")
    edition_fns = {
        fn["filter"]["match"]["groups.group.subgroup.id"]["query"]: fn["weight"]
        for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.subgroup.id", {}).get("query")
        in ("111050000000010687", "111050000000020042")
    }
    assert edition_fns == {"111050000000010687": 2, "111050000000020042": 3}


def test_rule_group_gets_old_and_new_income_tax_rules_edition_subgroup_boosts():
    functions = build_function_score_functions(group_id="111050000000000026", latest_finance_act_year="2025")
    edition_fns = {
        fn["filter"]["match"]["groups.group.subgroup.id"]["query"]: fn["weight"]
        for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.subgroup.id", {}).get("query")
        in ("111050000000010121", "111050000000020129")
    }
    assert edition_fns == {"111050000000010121": 2, "111050000000020129": 3}


def test_unrelated_group_gets_no_edition_subgroup_boosts():
    functions = build_function_score_functions(group_id="999999999", latest_finance_act_year="2025")
    edition_ids = {
        "111050000000010687", "111050000000020042",
        "111050000000010121", "111050000000020129",
    }
    matched = [
        fn for fn in functions
        if fn.get("filter", {}).get("match", {}).get("groups.group.subgroup.id", {}).get("query") in edition_ids
    ]
    assert matched == []


def test_state_gst_non_caselaws_penalty_present():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    penalty = next(
        fn for fn in functions
        if fn.get("weight") == 0.03
    )
    assert penalty["filter"] == {
        "bool": {
            "filter": [
                {"terms": {"categories.subcategory.id": ["111050000000017095"]}},
            ],
            "must_not": [
                {"term": {"groups.group.url.keyword": "caselaws"}},
            ],
        }
    }


def test_finance_act_old_year_penalty_present():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2025")
    penalty = next(
        fn for fn in functions
        if fn.get("weight") == 0.02
    )
    assert penalty["filter"] == {
        "bool": {
            "filter": [
                {"terms": {"groups.group.subgroup.id": ["111050000000010567"]}},
            ],
            "must_not": [
                {"terms": {"year.name.keyword": ["2025"]}},
            ],
        }
    }


def test_finance_act_penalty_uses_the_passed_latest_finance_act_year_not_a_hardcoded_one():
    functions = build_function_score_functions(group_id="0", latest_finance_act_year="2026")
    penalty = next(fn for fn in functions if fn.get("weight") == 0.02)
    assert penalty["filter"]["bool"]["must_not"] == [{"terms": {"year.name.keyword": ["2026"]}}]
