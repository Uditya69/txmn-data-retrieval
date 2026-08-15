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
