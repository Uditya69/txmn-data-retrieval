"""Shared JSONL result persistence for eval scripts (slm_intent_eval, intent_eval,
collection_routing_eval, retrieval_eval) - lets a long unattended run against a slow
local LLM survive a crash/kill partway through. JSONL, not a growing JSON array, so a
kill mid-write corrupts at most the one in-flight trailing line rather than the whole
file - both load_completed_ids and read_records drop an unparseable trailing line
instead of raising."""

import json
from pathlib import Path


def append_result(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def _parsed_lines(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_completed_ids(path: str | Path) -> set[str]:
    return {record["id"] for record in _parsed_lines(path)}


def read_records(path: str | Path) -> list[dict]:
    return _parsed_lines(path)


def filter_pending(cases: list[dict], completed_ids: set[str]) -> list[dict]:
    return [case for case in cases if case["id"] not in completed_ids]
