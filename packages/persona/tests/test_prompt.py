from persona.prompt import render_persona_context


def test_render_persona_context_returns_empty_string_for_none():
    assert render_persona_context(None) == ""


def test_render_persona_context_returns_empty_string_for_no_signal_yet():
    assert render_persona_context({"user_id": "u1", "query_count": 0, "category_affinity": {}}) == ""


def test_render_persona_context_names_top_category_and_expertise():
    persona = {
        "category_affinity": {"acts": 0.1, "caselaws": 0.8, "commentary": 0.1, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "expertise_level": "practitioner",
    }
    context = render_persona_context(persona)
    assert "caselaws" in context
    assert "practitioner" in context


def test_render_persona_context_includes_query_style_when_present():
    persona = {
        "category_affinity": {"acts": 0.9, "caselaws": 0.0, "commentary": 0.0, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "query_style": "precise-citation",
    }
    context = render_persona_context(persona)
    assert "precise-citation" in context
