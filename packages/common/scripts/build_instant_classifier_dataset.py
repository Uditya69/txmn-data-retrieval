"""Pulls real user search queries from TaxmannDW, has DeepInfra label each one
KEYWORD / INTENT / HYBRID (the existing Instant-mode classifier's labels - see
common/instant_classifier/labels.py), and appends the result to
packages/common/data/instant_classifier/train.jsonl.

Run: uv run --package common python packages/common/scripts/build_instant_classifier_dataset.py

Env required (see .env.example): DB_IP, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
DEEPINFRA_API_KEY, DEEPINFRA_ML_CLASSIFIER_MODEL.

Resumable: llm_dataset_checkpoint.json tracks the max ID already pulled per source
table, so a re-run only picks up rows newer than the last run.
"""

import datetime
import json
import os
from pathlib import Path

import httpx
import pymysql
from dotenv import load_dotenv

from common.instant_classifier_llm_dataset import (
    build_labeling_messages,
    chunk,
    format_jsonl_row,
    load_checkpoint,
    normalize_query,
    parse_labels_response,
    save_checkpoint,
)

_DATA_DIR = Path(__file__).parent.parent / "data" / "instant_classifier"
_OUTPUT_PATH = _DATA_DIR / "train.jsonl"
_CHECKPOINT_PATH = _DATA_DIR / "llm_dataset_checkpoint.json"

_TABLES = ["TAXMANNUSERSEARCHHISTORY"]
_RESEARCH_PATH_PREFIX = "https://www.taxmann.com/research/%"
_BATCH_SIZE = 10
_TARGET_COUNT = 4  # tops the LLM-labeled total up to 200 (196 written so far)
_DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"


def _fetch_new_rows(checkpoint: dict[str, int]) -> list[tuple[str, int, str, str]]:
    """Returns (table, id, search_keyword, created_at) rows newer than checkpoint,
    across both tables, oldest first, filtered to the /research path."""
    conn = pymysql.connect(
        host=os.environ["DB_IP"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        connect_timeout=10,
    )
    try:
        cur = conn.cursor()
        rows = []
        for table in _TABLES:
            last_id = checkpoint.get(table, 0)
            cur.execute(
                f"""
                SELECT ID, SEARCHKEYWORD, CREATEDAT FROM {table}
                WHERE ID > %s AND COMPONENTURL LIKE %s
                ORDER BY ID ASC
                LIMIT %s
                """,
                (last_id, _RESEARCH_PATH_PREFIX, _TARGET_COUNT),
            )
            rows.extend((table, row_id, keyword, str(created_at)) for row_id, keyword, created_at in cur.fetchall())
        cur.close()
    finally:
        conn.close()
    rows.sort(key=lambda r: r[3])
    return rows


def _existing_query_texts() -> set[str]:
    if not _OUTPUT_PATH.exists():
        return set()
    texts = set()
    for line in _OUTPUT_PATH.read_text().splitlines():
        if line.strip():
            texts.add(json.loads(line)["query_text"])
    return texts


def _select_next_batch(
    rows: list[tuple[str, int, str, str]], already_labeled: set[str]
) -> list[tuple[str, int, str]]:
    """Normalizes, dedupes against already_labeled and within this pull (keeping
    first occurrence), caps at _TARGET_COUNT. Returns (table, id, normalized_query)
    triples so the caller can advance the checkpoint only past rows that actually
    make it into a written batch."""
    seen_text = set(already_labeled)
    selected: list[tuple[str, int, str]] = []
    for table, row_id, raw_keyword, _created_at in rows:
        if len(selected) >= _TARGET_COUNT:
            break
        text = normalize_query(raw_keyword)
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        selected.append((table, row_id, text))
    return selected


_MAX_LABEL_ATTEMPTS = 3


def _label_batch(client: httpx.Client, model: str, api_key: str, queries: list[str]) -> list[str]:
    """The classifier model occasionally drops/adds a label and returns the wrong
    count - retry a few times before giving up on this batch."""
    last_error: ValueError | None = None
    for _attempt in range(_MAX_LABEL_ATTEMPTS):
        response = client.post(
            _DEEPINFRA_URL,
            json={
                "model": model,
                "messages": build_labeling_messages(queries),
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            return parse_labels_response(content, expected_count=len(queries))
        except ValueError as exc:
            last_error = exc
    raise last_error


def main() -> None:
    load_dotenv()
    checkpoint = load_checkpoint(_CHECKPOINT_PATH)
    rows = _fetch_new_rows(checkpoint)
    selected = _select_next_batch(rows, _existing_query_texts())

    if not selected:
        print("No new queries under the /research path since the last checkpoint.")
        return

    api_key = os.environ["DEEPINFRA_API_KEY"]
    model = os.environ["DEEPINFRA_ML_CLASSIFIER_MODEL"]
    today = datetime.date.today().isoformat()
    source = f"llm:{model}"

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with httpx.Client(timeout=60.0) as client, _OUTPUT_PATH.open("a") as out:
        for batch in chunk(selected, _BATCH_SIZE):
            queries = [q for _table, _id, q in batch]
            try:
                labels = _label_batch(client, model, api_key, queries)
            except ValueError as exc:
                print(f"skipping batch after {_MAX_LABEL_ATTEMPTS} bad responses: {exc}")
                continue
            for query_text, label in zip(queries, labels):
                out.write(format_jsonl_row(query_text=query_text, label=label, source=source, date_added=today))
                out.write("\n")
            out.flush()

            for table, row_id, _q in batch:
                checkpoint[table] = max(checkpoint.get(table, 0), row_id)
            save_checkpoint(_CHECKPOINT_PATH, checkpoint)

            written += len(batch)
            print(f"labeled {written}/{len(selected)}")

    print(f"Wrote {written} labeled queries to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
