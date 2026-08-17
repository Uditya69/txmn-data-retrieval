from persona.merge import KNOWN_CATEGORIES

_TRUST_THRESHOLD = 20

RELEVANCE_INSTRUCTION = (
    "The note above is a prior about this user's typical usage, not a "
    "fact about this query. Use it only if this query is genuinely "
    "ambiguous on its own. If the query's own content conflicts with or "
    "is unrelated to the note, ignore the note and rely on the query "
    "alone."
)


def render_persona_context(persona: dict | None) -> str:
    if not persona or persona.get("query_count", 0) < _TRUST_THRESHOLD:
        return ""

    affinity = persona.get("category_affinity") or {}
    top_categories = [c for c in KNOWN_CATEGORIES if affinity.get(c, 0.0) > 0.0]
    top_categories.sort(key=lambda c: affinity.get(c, 0.0), reverse=True)

    expertise_level = persona.get("expertise_level")
    query_style = persona.get("query_style")

    if not top_categories and not expertise_level and not query_style:
        return ""

    parts = []
    if top_categories:
        parts.append(f"frequently asks about {', '.join(top_categories[:2])}")
    if expertise_level:
        parts.append(f"expertise level: {expertise_level}")
    if query_style:
        parts.append(f"query style: {query_style}")

    return "This user " + "; ".join(parts) + "."
