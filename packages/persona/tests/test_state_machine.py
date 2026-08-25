from persona.state_machine import ACTIVE_THRESHOLD, detect_pivot, next_state, reactivate_if_dormant


def test_next_state_discovered_requires_score_and_corroboration():
    assert next_state("discovered", score=0.5, sessions_since_transition=1, min_sessions=2) == "discovered"
    assert next_state("discovered", score=0.5, sessions_since_transition=2, min_sessions=2) == "emerging"


def test_next_state_single_session_spike_does_not_promote_to_active():
    assert next_state("emerging", score=ACTIVE_THRESHOLD + 0.1, sessions_since_transition=1, min_sessions=2) == "emerging"


def test_next_state_sustained_evidence_promotes_to_active():
    assert next_state("emerging", score=ACTIVE_THRESHOLD + 0.1, sessions_since_transition=2, min_sessions=2) == "active"


def test_next_state_active_declines_to_fading_not_dormant():
    assert next_state("active", score=0.05, sessions_since_transition=0, min_sessions=2) == "fading"


def test_next_state_fading_requires_further_decline_to_reach_dormant():
    assert next_state("fading", score=0.2, sessions_since_transition=0, min_sessions=2) == "fading"
    assert next_state("fading", score=0.01, sessions_since_transition=0, min_sessions=2) == "dormant"


def test_next_state_dormant_never_transitions_on_score_alone():
    assert next_state("dormant", score=0.9, sessions_since_transition=5, min_sessions=2) == "dormant"


def test_reactivate_if_dormant_moves_to_reactive_on_new_event():
    assert reactivate_if_dormant("dormant", has_new_event=True) == "reactive"
    assert reactivate_if_dormant("dormant", has_new_event=False) == "dormant"
    assert reactivate_if_dormant("active", has_new_event=True) == "active"


def test_detect_pivot_requires_declining_and_corroborated_rising_topic():
    declining = {"state": "fading"}
    rising = {"state": "active"}
    assert detect_pivot(declining, rising) is True


def test_detect_pivot_false_when_rising_topic_not_yet_corroborated():
    declining = {"state": "fading"}
    rising_uncorroborated = {"state": "emerging"}
    assert detect_pivot(declining, rising_uncorroborated) is False


def test_detect_pivot_false_when_declining_topic_still_active():
    still_active = {"state": "active"}
    rising = {"state": "active"}
    assert detect_pivot(still_active, rising) is False
