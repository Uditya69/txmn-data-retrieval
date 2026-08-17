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


def _resolve_mode(votes: dict[str, int], previous: str | None) -> str:
    max_count = max(votes.values())
    leaders = sorted(value for value, count in votes.items() if count == max_count)
    # Tie: keep the previous mode rather than churn to an arbitrary leader -
    # see docs/superpowers/specs/2026-08-17-persona-context-trust-gating-design.md §1.
    return previous if previous in leaders else leaders[0]


def _merge_vote_field(existing: dict, new_value: str, votes_key: str, value_key: str) -> tuple[dict, str]:
    votes = existing.get(votes_key)
    if votes is None:
        # Migration: an old doc has the plain string field but no tally yet -
        # seed the tally with that value as one vote before adding this one.
        old_value = existing.get(value_key)
        votes = {old_value: 1} if old_value else {}
    else:
        votes = dict(votes)
    votes[new_value] = votes.get(new_value, 0) + 1
    return votes, _resolve_mode(votes, existing.get(value_key))


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

    result = dict(existing)
    if "expertise_level" in filtered:
        votes, mode = _merge_vote_field(existing, filtered["expertise_level"], "expertise_votes", "expertise_level")
        result["expertise_votes"] = votes
        result["expertise_level"] = mode
    if "query_style" in filtered:
        votes, mode = _merge_vote_field(existing, filtered["query_style"], "query_style_votes", "query_style")
        result["query_style_votes"] = votes
        result["query_style"] = mode
    return result
