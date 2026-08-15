KNOWN_CATEGORIES = ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]

VALID_EXPERTISE_LEVELS = {"student", "practitioner", "expert"}
VALID_QUERY_STYLES = {"broad", "precise-citation"}


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

    # Never merge unvalidated SLM output verbatim - render_persona_context
    # interpolates expertise_level/query_style straight into the synthesis
    # system prompt on every future request for this user, so an
    # out-of-enum or extraneous key here is a same-account prompt-injection
    # vector. Drop anything that isn't one of the design spec's enumerated
    # values, and drop any key other than expertise_level/query_style entirely.
    filtered = {}
    expertise_level = patch.get("expertise_level")
    if expertise_level in VALID_EXPERTISE_LEVELS:
        filtered["expertise_level"] = expertise_level
    query_style = patch.get("query_style")
    if query_style in VALID_QUERY_STYLES:
        filtered["query_style"] = query_style

    if not filtered:
        return existing
    return {**existing, **filtered}
