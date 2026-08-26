from persona.evidence import evidence_weight


def test_evidence_weight_submitted_only():
    assert evidence_weight({}) == 1.0
    assert evidence_weight(None) == 1.0


def test_evidence_weight_adds_interaction_signals():
    weight_click = evidence_weight({"clicked": True})
    weight_save = evidence_weight({"clicked": True, "saved": True})
    assert weight_click > 1.0
    assert weight_save > weight_click


def test_evidence_weight_upgrading_signal_never_lowers_weight():
    before = evidence_weight({"clicked": True})
    after = evidence_weight({"clicked": True, "saved": True})
    assert after >= before


def test_evidence_weight_ignores_false_signals():
    assert evidence_weight({"clicked": False, "saved": False}) == 1.0
