from persona.prompt import RELEVANCE_INSTRUCTION, render_persona_context, render_topic_hypotheses


def _topic(topic_id, state, score, entities=None, embedding=None):
    return {
        "topic_id": topic_id,
        "state": state,
        "score": score,
        "legal_entities": entities or [],
        "categories": [],
        "representative_embedding": embedding or [],
    }


def test_render_persona_context_returns_empty_for_empty_snapshot(persona_settings):
    assert render_persona_context([], persona_settings) == ""


def test_render_persona_context_returns_empty_below_confidence_floor(persona_settings):
    snapshot = [_topic("t1", "active", 0.01, entities=["IBC"])]
    assert render_persona_context(snapshot, persona_settings) == ""


def test_render_persona_context_ignores_discovered_and_fading_topics(persona_settings):
    snapshot = [
        _topic("t1", "discovered", 0.9, entities=["IBC"]),
        _topic("t2", "fading", 0.9, entities=["GST"]),
    ]
    assert render_persona_context(snapshot, persona_settings) == ""


def test_render_persona_context_names_current_active_topic(persona_settings):
    snapshot = [_topic("t1", "active", 0.6, entities=["GST"])]
    context = render_persona_context(snapshot, persona_settings)
    assert "GST" in context
    assert "active" in context


def test_render_persona_context_prefers_current_focus_over_higher_lifetime_history(persona_settings):
    # A pivoted user: IBC was dominant historically but has faded; GST is
    # the current active topic. Only GST should appear.
    snapshot = [
        _topic("ibc", "fading", 0.9, entities=["IBC"]),
        _topic("gst", "active", 0.4, entities=["GST"]),
    ]
    context = render_persona_context(snapshot, persona_settings)
    assert "GST" in context
    assert "IBC" not in context


def test_render_topic_hypotheses_empty_when_only_one_candidate_clears_floor(persona_settings):
    snapshot = [_topic("t1", "active", 0.5, embedding=[1.0, 0.0])]
    assert render_topic_hypotheses([1.0, 0.0], snapshot, persona_settings) == ""


def test_render_topic_hypotheses_renders_weighted_candidates_for_ambiguous_query(persona_settings):
    snapshot = [
        _topic("ibc", "active", 0.8, entities=["IBC"], embedding=[1.0, 0.0]),
        _topic("gst", "active", 0.3, entities=["GST"], embedding=[0.6, 0.8]),
    ]
    result = render_topic_hypotheses([0.9, 0.4], snapshot, persona_settings)
    assert "IBC" in result
    assert "GST" in result


def test_relevance_instruction_is_a_nonempty_stable_string():
    assert isinstance(RELEVANCE_INSTRUCTION, str)
    assert "ignore" in RELEVANCE_INSTRUCTION.lower()
    assert len(RELEVANCE_INSTRUCTION) > 0
