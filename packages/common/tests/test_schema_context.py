from common.schema_context import build_schema_context


def test_build_schema_context_mentions_every_collection():
    context = build_schema_context()

    for collection in ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"]:
        assert collection in context


def test_build_schema_context_lists_filter_fields_and_sample_values():
    context = build_schema_context()

    assert "court" in context
    assert "act" in context
    assert "section" in context
    assert "Supreme Court" in context
    assert "Income-tax Act, 1961" in context
