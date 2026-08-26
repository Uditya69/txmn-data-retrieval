import json

import pytest

from common.instant_classifier_llm_dataset import (
    LABELS,
    build_labeling_messages,
    chunk,
    format_jsonl_row,
    load_checkpoint,
    normalize_query,
    parse_labels_response,
    save_checkpoint,
)


def test_normalize_query_decodes_percent_encoding():
    raw = "gst%20on%20resturant%20service%20through%20zomato%2C%20swiggy"
    assert normalize_query(raw) == "gst on resturant service through zomato, swiggy"


def test_normalize_query_strips_whitespace():
    assert normalize_query("  section%2054  ") == "section 54"


def test_normalize_query_leaves_literal_plus_untouched():
    # unquote (not unquote_plus) - a literal '+' in a query (e.g. "c++ tax") isn't
    # this DB's space encoding (%20 is), so it must not be turned into a space.
    assert normalize_query("c%2B%2B%20tax") == "c++ tax"


def test_chunk_splits_into_fixed_size_groups():
    assert chunk(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_chunk_empty_list():
    assert chunk([], 20) == []


def test_build_labeling_messages_includes_every_query_and_all_labels():
    queries = ["Section 54", "gst on restaurant service via zomato"]
    messages = build_labeling_messages(queries)
    joined = json.dumps(messages)
    for q in queries:
        assert q in joined
    for label in LABELS:
        assert label in joined


def test_parse_labels_response_happy_path():
    content = json.dumps({"labels": ["KEYWORD", "INTENT", "HYBRID"]})
    assert parse_labels_response(content, expected_count=3) == [
        "KEYWORD",
        "INTENT",
        "HYBRID",
    ]


def test_parse_labels_response_rejects_wrong_count():
    content = json.dumps({"labels": ["KEYWORD"]})
    with pytest.raises(ValueError, match="expected 2 labels, got 1"):
        parse_labels_response(content, expected_count=2)


def test_parse_labels_response_rejects_unknown_label():
    content = json.dumps({"labels": ["KEYWORD", "NOT_A_LABEL"]})
    with pytest.raises(ValueError, match="unknown label"):
        parse_labels_response(content, expected_count=2)


def test_parse_labels_response_rejects_non_json():
    with pytest.raises(ValueError):
        parse_labels_response("not json", expected_count=1)


def test_format_jsonl_row_shape():
    line = format_jsonl_row(
        query_text="section 54", label="KEYWORD", source="llm:model-x", date_added="2026-08-26"
    )
    row = json.loads(line)
    assert row == {
        "query_text": "section 54",
        "label": "KEYWORD",
        "source": "llm:model-x",
        "date_added": "2026-08-26",
    }


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, {"TAXMANNUSERSEARCHHISTORY": 100})
    assert load_checkpoint(path) == {"TAXMANNUSERSEARCHHISTORY": 100}


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert load_checkpoint(tmp_path / "missing.json") == {}
