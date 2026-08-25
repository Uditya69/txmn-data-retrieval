import re

from persona.clustering import cosine_similarity

_ACTIVE_STATES = ("active", "reactive")

_HAS_DIGIT_RE = re.compile(r"\d")
_ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")


def _entity_specificity_score(entity: str) -> int:
    """Ranks a legal_entities string by how useful it is as a human/model-
    facing topic label. repository.py merges entities into a topic via a
    plain `set()` (arbitrary iteration order), so without this, a topic's
    first two stored entities can just as easily be generic descriptive
    phrases ("eligibility conditions", "time limit") as the actual
    recognizable anchor ("GST", "Section 17(5)") - confirmed live: this
    directly gated whether extract_intent's persona-aware search_query
    expansion (intent.py) had anything concrete enough to expand into.
    """
    score = 0
    if _HAS_DIGIT_RE.search(entity):
        score += 3  # "Section 7", "Section 17(5)" - a citation-shaped anchor
    words = entity.split()
    if len(words) <= 2 and any(_ACRONYM_RE.match(w.strip("()")) for w in words):
        score += 3  # "GST", "IBC", "TDS" - a short recognizable acronym
    if entity[:1].isupper() and not entity.isupper():
        score += 1  # Title Case - reads as a proper noun, not a generic phrase
    if entity.islower():
        score -= 1  # all-lowercase generic phrase - deprioritize
    return score


def _best_entities(entities: list[str], n: int) -> list[str]:
    ranked = sorted(entities, key=_entity_specificity_score, reverse=True)
    return ranked[:n]

RELEVANCE_INSTRUCTION = (
    "The note above is a prior about this user's typical usage, not a "
    "fact about this query. Use it only if this query is genuinely "
    "ambiguous on its own. If the query's own content conflicts with or "
    "is unrelated to the note, ignore the note and rely on the query "
    "alone."
)

_MAX_RENDERED_TOPICS = 3


def _topic_label(topic: dict) -> str:
    entities = topic.get("legal_entities") or []
    if entities:
        return ", ".join(_best_entities(entities, 2))
    categories = topic.get("categories") or []
    if categories:
        return ", ".join(categories[:2])
    return "a recent topic"


def render_persona_context(snapshot: list[dict], settings) -> str:
    """Renders the user's CURRENT focus (active/reactive topics above the
    confidence floor), not a lifetime average - persona-context-rendering's
    "reflects current, not lifetime-averaged, interest" requirement. Empty
    snapshot or no topic clearing the floor renders empty context, matching
    a guest (persona-context-rendering's low-evidence requirement).
    """
    relevant = [
        topic for topic in snapshot
        if topic.get("state") in _ACTIVE_STATES and topic.get("score", 0.0) >= settings.persona_context_confidence_floor
    ]
    if not relevant:
        return ""

    relevant.sort(key=lambda topic: topic.get("score", 0.0), reverse=True)
    top = relevant[:_MAX_RENDERED_TOPICS]
    parts = [f"{_topic_label(topic)} ({topic['state']})" for topic in top]
    return "This user is currently focused on: " + "; ".join(parts) + "."


def render_topic_hypotheses(query_embedding: list[float], snapshot: list[dict], settings, top_k: int = 3) -> str:
    """For a current query whose terms could plausibly match more than one of
    the user's known topics, renders weighted candidate interpretations
    rather than asserting one as certain (persona-context-rendering:
    "ambiguous current-query terms may surface topic-hypothesis
    confidences"). Renders nothing when fewer than two candidates clear the
    confidence floor - a clear-cut query gets no hypothesis text at all.
    """
    scored = [
        (topic, cosine_similarity(topic.get("representative_embedding") or [], query_embedding))
        for topic in snapshot
    ]
    scored = [(topic, sim) for topic, sim in scored if sim > 0.0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    candidates = scored[:top_k]
    above_floor = [(topic, sim) for topic, sim in candidates if sim >= settings.persona_context_confidence_floor]

    if len(above_floor) < 2:
        return ""

    total = sum(sim for _, sim in above_floor)
    lines = [f"{_topic_label(topic)} {sim / total:.2f}" for topic, sim in above_floor]
    return "Possible topic interpretations for this query, by confidence: " + "; ".join(lines) + "."
