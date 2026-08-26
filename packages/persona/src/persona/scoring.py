import math

# Saturating transform constant: raw_score / _SATURATION_K controls how many
# accumulated (decayed) evidence-weight units it takes to approach the
# asymptote. The transform 1 - exp(-raw/K) never reaches 1.0 for any finite
# input, so no single event (max weight ~29, see evidence.SIGNAL_WEIGHTS) can
# push a fresh topic straight to the maximum representable score -
# persona-interest-scoring's "single event cannot dominate" requirement.
_SATURATION_K = 50.0


def interest_score(events: list[dict], at_time, decay_lambda: float) -> float:
    """Computes a topic's interest score at `at_time` from its event history.

    Each event is `{"evidence_weight": float, "timestamp": datetime}`. Older
    events contribute exponentially less (persona-interest-scoring's recency-
    discount requirement); the function is pure and deterministic - the same
    `events`/`at_time`/`decay_lambda` always produce the same score
    (persona-interest-scoring's determinism requirement).
    """
    raw = 0.0
    for event in events:
        age_days = (at_time - event["timestamp"]).total_seconds() / 86400.0
        age_days = max(0.0, age_days)
        raw += event["evidence_weight"] * math.exp(-decay_lambda * age_days)
    return 1.0 - math.exp(-raw / _SATURATION_K)
