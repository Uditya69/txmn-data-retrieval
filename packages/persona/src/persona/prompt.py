from persona.merge import KNOWN_CATEGORIES


def render_persona_context(persona: dict | None) -> str:
    if not persona:
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
