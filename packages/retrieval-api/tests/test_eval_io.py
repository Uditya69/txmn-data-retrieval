import json

from retrieval_api.eval_io import append_result, filter_pending, load_completed_ids, read_records


def test_append_result_writes_one_json_line(tmp_path):
    path = tmp_path / "out.jsonl"

    append_result(path, {"id": "A1", "status": "PASS"})

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"id": "A1", "status": "PASS"}


def test_append_result_appends_subsequent_calls(tmp_path):
    path = tmp_path / "out.jsonl"

    append_result(path, {"id": "A1"})
    append_result(path, {"id": "A2"})

    lines = path.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["A1", "A2"]


def test_load_completed_ids_returns_empty_set_for_missing_file(tmp_path):
    path = tmp_path / "missing.jsonl"

    assert load_completed_ids(path) == set()


def test_load_completed_ids_returns_ids_from_existing_file(tmp_path):
    path = tmp_path / "out.jsonl"
    append_result(path, {"id": "A1"})
    append_result(path, {"id": "A2"})

    assert load_completed_ids(path) == {"A1", "A2"}


def test_load_completed_ids_drops_truncated_trailing_line(tmp_path):
    path = tmp_path / "out.jsonl"
    append_result(path, {"id": "A1"})
    with path.open("a") as handle:
        handle.write('{"id": "A2", "status": "PA')  # simulated crash mid-write

    assert load_completed_ids(path) == {"A1"}


def test_read_records_returns_full_parsed_objects(tmp_path):
    path = tmp_path / "out.jsonl"
    append_result(path, {"id": "A1", "status": "PASS"})
    append_result(path, {"id": "A2", "status": "FAIL"})

    assert read_records(path) == [
        {"id": "A1", "status": "PASS"},
        {"id": "A2", "status": "FAIL"},
    ]


def test_read_records_drops_truncated_trailing_line(tmp_path):
    path = tmp_path / "out.jsonl"
    append_result(path, {"id": "A1"})
    with path.open("a") as handle:
        handle.write('{"id": "A2"')

    assert read_records(path) == [{"id": "A1"}]


def test_filter_pending_drops_cases_whose_id_is_completed():
    cases = [{"id": "A1"}, {"id": "A2"}, {"id": "A3"}]

    assert filter_pending(cases, {"A2"}) == [{"id": "A1"}, {"id": "A3"}]


def test_filter_pending_returns_all_cases_for_empty_completed_set():
    cases = [{"id": "A1"}, {"id": "A2"}]

    assert filter_pending(cases, set()) == cases
