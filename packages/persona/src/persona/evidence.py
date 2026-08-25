SIGNAL_WEIGHTS = {
    "submitted": 1.0,
    "clicked": 2.0,
    "opened": 3.0,
    "read_deeply": 4.0,
    "saved": 6.0,
    "repeated_related_query": 5.0,
    "returned_later": 8.0,
}


def evidence_weight(signals: dict[str, bool] | None) -> float:
    """Combines interaction signals for one query event into a single evidence
    weight. Always includes "submitted" - a query event with no other signal
    still carries the minimum non-zero weight for having been submitted at all
    (persona-interest-evidence: "every query event carries an evidence weight").

    Additive, not averaged, so upgrading a signal (e.g. clicked -> saved) can
    only raise the weight, never lower it (persona-interest-evidence's
    monotonicity requirement).
    """
    signals = signals or {}
    total = SIGNAL_WEIGHTS["submitted"]
    for name, present in signals.items():
        if present and name in SIGNAL_WEIGHTS and name != "submitted":
            total += SIGNAL_WEIGHTS[name]
    return total
