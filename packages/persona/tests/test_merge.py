from persona.merge import KNOWN_CATEGORIES, merge_category_affinity, merge_expertise_patch


def test_merge_category_affinity_starts_from_empty():
    result = merge_category_affinity({}, existing_count=0, categories=["caselaws"])
    assert result["caselaws"] == 1.0
    assert result["acts"] == 0.0
    assert set(result.keys()) == set(KNOWN_CATEGORIES)


def test_merge_category_affinity_averages_across_rounds():
    first = merge_category_affinity({}, existing_count=0, categories=["caselaws"])
    second = merge_category_affinity(first, existing_count=1, categories=["acts"])
    assert second["caselaws"] == 0.5
    assert second["acts"] == 0.5
    assert second["commentary"] == 0.0


def test_merge_category_affinity_handles_multiple_tags_in_one_round():
    result = merge_category_affinity({}, existing_count=0, categories=["acts", "rules"])
    assert result["acts"] == 1.0
    assert result["rules"] == 1.0
    assert result["caselaws"] == 0.0


def test_merge_expertise_patch_returns_existing_unchanged_when_patch_none():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    assert merge_expertise_patch(existing, None) == existing


def test_merge_expertise_patch_overwrites_only_provided_keys():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    result = merge_expertise_patch(existing, {"expertise_level": "student"})
    assert result == {"expertise_level": "student", "query_style": "precise-citation"}


def test_merge_expertise_patch_handles_empty_existing():
    result = merge_expertise_patch({}, {"expertise_level": "student"})
    assert result == {"expertise_level": "student"}


def test_merge_expertise_patch_drops_invalid_expertise_level():
    existing = {"expertise_level": "practitioner"}
    result = merge_expertise_patch(existing, {"expertise_level": "omniscient"})
    assert result == {"expertise_level": "practitioner"}


def test_merge_expertise_patch_drops_invalid_query_style():
    existing = {"query_style": "broad"}
    result = merge_expertise_patch(existing, {"query_style": "essay-length"})
    assert result == {"query_style": "broad"}


def test_merge_expertise_patch_strips_extraneous_keys():
    result = merge_expertise_patch({}, {"expertise_level": "student", "injected": "malicious"})
    assert result == {"expertise_level": "student"}
    assert "injected" not in result


def test_merge_expertise_patch_returns_existing_unchanged_when_all_values_invalid():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    result = merge_expertise_patch(existing, {"expertise_level": "bogus", "query_style": "bogus"})
    assert result == existing


def test_merge_expertise_patch_valid_values_still_merge_normally():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    result = merge_expertise_patch(existing, {"expertise_level": "expert", "query_style": "broad"})
    assert result == {"expertise_level": "expert", "query_style": "broad"}
