STATES = ("discovered", "emerging", "active", "fading", "dormant", "reactive")

# Fixed transition graph - deliberately has NO active -> dormant edge, so a
# declining topic always passes through "fading" first
# (persona-interest-state-machine: "gradual decline" requirement).
STATE_GRAPH = {
    "discovered": {"discovered", "emerging"},
    "emerging": {"emerging", "active"},
    "active": {"active", "fading"},
    "fading": {"fading", "dormant", "active"},
    "dormant": {"dormant", "reactive"},
    "reactive": {"reactive", "active", "dormant"},
}

EMERGING_THRESHOLD = 0.10
ACTIVE_THRESHOLD = 0.35
DORMANT_THRESHOLD = 0.05


def next_state(current_state: str, score: float, sessions_since_transition: int, min_sessions: int) -> str:
    """Advances one topic's state given its freshly recomputed interest score.

    Upward transitions (toward more engaged states) require BOTH a score
    above threshold AND at least `min_sessions` distinct corroborating
    sessions since the last transition - a single session's spike cannot
    promote a topic on its own (persona-interest-state-machine's hysteresis
    requirement). Downward transitions need only the score decline; the
    graph shape itself (no active->dormant edge) enforces passing through
    "fading" first.
    """
    corroborated = sessions_since_transition >= min_sessions

    if current_state == "discovered":
        return "emerging" if score >= EMERGING_THRESHOLD and corroborated else "discovered"
    if current_state == "emerging":
        return "active" if score >= ACTIVE_THRESHOLD and corroborated else "emerging"
    if current_state == "active":
        return "fading" if score < ACTIVE_THRESHOLD else "active"
    if current_state == "fading":
        if score >= ACTIVE_THRESHOLD and corroborated:
            return "active"
        if score < DORMANT_THRESHOLD:
            return "dormant"
        return "fading"
    if current_state == "dormant":
        return "dormant"
    if current_state == "reactive":
        return "active" if score >= ACTIVE_THRESHOLD and corroborated else "reactive"
    return current_state


def reactivate_if_dormant(current_state: str, has_new_event: bool) -> str:
    """A dormant topic with fresh activity moves to "reactive" (distinct from a
    brand-new "discovered" topic), preserving prior identity/history rather
    than resetting - persona-interest-state-machine's dormant->reactive
    requirement. Call this before `next_state` on the same tick a new event
    lands.
    """
    if current_state == "dormant" and has_new_event:
        return "reactive"
    return current_state


def detect_pivot(declining_topic: dict, rising_topic: dict) -> bool:
    """Reports a pivot only when the rising topic's own state already reflects
    corroborated evidence (state machine's hysteresis already gates
    active/reactive) and the declining topic has moved past "active" into
    "fading"/"dormant" - never from a single day's score crossover alone.
    """
    declining = declining_topic.get("state") in ("fading", "dormant")
    rising = rising_topic.get("state") in ("active", "reactive")
    return declining and rising
