import math
import re

# Weights for combining the four similarity signals into one score used
# against persona_cluster_similarity_threshold. See design.md decision #2.
# Entities carries more weight than embedding: a shared legal citation/Act/
# section reference is a far more reliable same-topic signal in caselaw
# research than raw sentence-embedding cosine similarity turned out to be in
# practice (two genuinely related but differently-worded queries scored as
# low as ~0.32 cosine on Voyage's query_embed, which is tuned for
# query->document retrieval matching, not query->query similarity) - found
# during real end-to-end persona testing, see design.md's clustering risk.
_WEIGHTS = {"embedding": 0.35, "entities": 0.40, "temporal": 0.15, "categories": 0.10}

# Temporal proximity decays to ~0 by this many hours apart - two queries a
# week apart contribute almost no temporal-continuity signal on their own.
_TEMPORAL_HALF_LIFE_HOURS = 48.0

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "for", "under", "in", "on", "by",
    "with", "is", "are", "was", "were", "be", "been", "at", "as", "vs", "v",
}
_PAREN_RE = re.compile(r"\(([^)]+)\)")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_entity_tokens(entities: list[str]) -> set[str]:
    """Turns free-text entity/concept strings (typically SLM-generated, so
    the same real-world entity gets phrased slightly differently call to
    call - "Insolvency and Bankruptcy Code (IBC)" vs "Insolvency and
    Bankruptcy Code" vs bare "IBC") into a normalized token set so overlap
    is measured on substance, not exact phrasing. Parenthetical content
    (commonly an acronym) is pulled out as its own token in addition to the
    main phrase, and the phrase itself contributes both as a whole (credits
    exact multi-word matches) and as its individual significant words
    (credits partial matches) - all lowercased, punctuation-stripped,
    stopwords dropped.
    """
    tokens: set[str] = set()
    for raw in entities:
        for paren in _PAREN_RE.findall(raw):
            tokens.add(paren.strip().lower())
        main = _PAREN_RE.sub("", raw)
        main = _PUNCT_RE.sub(" ", main).lower().strip()
        if not main:
            continue
        words = [w for w in main.split() if w and w not in _STOPWORDS]
        if not words:
            continue
        tokens.add(" ".join(words))
        tokens.update(words)
    return tokens


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def jaccard_overlap(a: list[str], b: list[str]) -> float:
    set_a, set_b = {x.lower() for x in a}, {x.lower() for x in b}
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def entity_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard overlap over normalized entity tokens (see
    _normalize_entity_tokens) rather than exact whole-phrase matching -
    tolerates the SLM phrasing the same legal entity differently across
    separate extraction calls.
    """
    tokens_a, tokens_b = _normalize_entity_tokens(a), _normalize_entity_tokens(b)
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def temporal_proximity(hours_apart: float) -> float:
    if hours_apart < 0:
        hours_apart = 0.0
    return math.exp(-hours_apart / _TEMPORAL_HALF_LIFE_HOURS)


def _hours_between(t1, t2) -> float:
    return abs((t2 - t1).total_seconds()) / 3600.0


def combined_similarity(topic: dict, embedding: list[float], entities: list[str], categories: list[str], timestamp) -> tuple[float, dict]:
    """Scores one candidate existing topic against a new event's features.
    Returns (score, signals) where signals names which components fired -
    persona-topic-clustering's explainability requirement.
    """
    embedding_sim = cosine_similarity(topic.get("representative_embedding") or [], embedding)
    entity_sim = entity_similarity(topic.get("legal_entities") or [], entities)
    category_sim = jaccard_overlap(topic.get("categories") or [], categories)

    last_event_at = topic.get("last_event_at")
    temporal_sim = temporal_proximity(_hours_between(last_event_at, timestamp)) if last_event_at else 0.0

    score = (
        _WEIGHTS["embedding"] * embedding_sim
        + _WEIGHTS["entities"] * entity_sim
        + _WEIGHTS["temporal"] * temporal_sim
        + _WEIGHTS["categories"] * category_sim
    )
    signals = {
        "embedding_similarity": embedding_sim,
        "entity_overlap": entity_sim,
        "temporal_proximity": temporal_sim,
        "category_overlap": category_sim,
    }
    return score, signals


def find_best_topic(
    existing_topics: list[dict], embedding: list[float], entities: list[str], categories: list[str],
    timestamp, threshold: float,
) -> tuple[dict | None, float, dict]:
    """Picks the closest existing topic for a new event, or None if nothing
    clears `threshold` - persona-topic-clustering's "below-threshold creates a
    new topic" requirement. Never force-assigns to the nearest-but-still-weak
    match.
    """
    best_topic, best_score, best_signals = None, 0.0, {}
    for topic in existing_topics:
        score, signals = combined_similarity(topic, embedding, entities, categories, timestamp)
        if score > best_score:
            best_topic, best_score, best_signals = topic, score, signals
    if best_topic is not None and best_score >= threshold:
        return best_topic, best_score, best_signals
    return None, best_score, best_signals


def should_start_new_episode(last_event_at, timestamp, episode_gap_hours: float) -> bool:
    if last_event_at is None:
        return True
    return _hours_between(last_event_at, timestamp) > episode_gap_hours
