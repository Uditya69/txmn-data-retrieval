MAX_LIST_ITEMS = 8
MAX_ITEM_LEN = 80


def _clean_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip()[:MAX_ITEM_LEN])
        if len(cleaned) >= MAX_LIST_ITEMS:
            break
    return cleaned


def _clean_score(value) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def validate_understanding(raw: dict | None) -> dict | None:
    """Constrains a raw Query Understanding Record (typically straight off an
    SLM's JSON output) to its expected shape before it is ever stored or later
    interpolated into a prompt. Drops anything that doesn't fit rather than
    merging unvalidated free text - mirrors merge.py::merge_expertise_patch's
    precedent, and is required by persona-query-understanding's injection-
    safety requirement: these fields eventually feed render_persona_context's
    output, which lands verbatim in a future synthesis system prompt for the
    same account.

    Returns None if nothing usable survives validation.
    """
    if not isinstance(raw, dict):
        return None

    concepts = _clean_string_list(raw.get("concepts"))
    legal_entities = _clean_string_list(raw.get("legal_entities"))
    research_objective = _clean_string_list(raw.get("research_objective"))
    specificity = _clean_score(raw.get("specificity"))
    confidence = _clean_score(raw.get("confidence"))

    if not any([concepts, legal_entities, research_objective]):
        return None

    return {
        "concepts": concepts,
        "legal_entities": legal_entities,
        "research_objective": research_objective,
        # Explicit, never omitted (persona-query-understanding: "confidence is
        # bounded and explicit") - default to the low end when the model gave
        # no usable score, rather than pretending full confidence.
        "specificity": specificity if specificity is not None else 0.0,
        "confidence": confidence if confidence is not None else 0.0,
    }
