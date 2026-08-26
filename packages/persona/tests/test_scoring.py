from datetime import datetime, timedelta, timezone

from persona.scoring import interest_score

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_interest_score_zero_for_no_events():
    assert interest_score([], T0, decay_lambda=0.03) == 0.0


def test_interest_score_deterministic_for_same_history():
    events = [{"evidence_weight": 5.0, "timestamp": T0}]
    first = interest_score(events, T0 + timedelta(days=3), decay_lambda=0.03)
    second = interest_score(events, T0 + timedelta(days=3), decay_lambda=0.03)
    assert first == second


def test_interest_score_recency_discount():
    events = [{"evidence_weight": 5.0, "timestamp": T0}]
    fresh = interest_score(events, T0, decay_lambda=0.03)
    stale = interest_score(events, T0 + timedelta(days=90), decay_lambda=0.03)
    assert stale < fresh


def test_interest_score_single_event_cannot_reach_maximum():
    max_weight_event = [{"evidence_weight": 29.0, "timestamp": T0}]
    score = interest_score(max_weight_event, T0, decay_lambda=0.03)
    assert score < 0.99


def test_interest_score_increases_with_more_corroborating_events():
    one_event = [{"evidence_weight": 7.0, "timestamp": T0}]
    many_events = [{"evidence_weight": 7.0, "timestamp": T0 + timedelta(days=i)} for i in range(5)]
    at = T0 + timedelta(days=5)
    assert interest_score(many_events, at, decay_lambda=0.03) > interest_score(one_event, at, decay_lambda=0.03)
