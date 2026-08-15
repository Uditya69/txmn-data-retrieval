KNOWN_CATEGORIES = ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]


def merge_category_affinity(
    existing_affinity: dict[str, float], existing_count: int, categories: list[str],
) -> dict[str, float]:
    result = {}
    for category in KNOWN_CATEGORIES:
        prior = existing_affinity.get(category, 0.0)
        indicator = 1.0 if category in categories else 0.0
        result[category] = (prior * existing_count + indicator) / (existing_count + 1)
    return result


def merge_expertise_patch(existing: dict, patch: dict | None) -> dict:
    if not patch:
        return existing
    return {**existing, **patch}
