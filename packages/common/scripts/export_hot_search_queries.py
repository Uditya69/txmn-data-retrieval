"""Pulls all /research search queries from TaxmannDW (both TAXMANNUSERSEARCHHISTORY
and TAXMANNUSERSEARCHHISTORY_2023), normalizes them, counts how often each exact
normalized query recurs, and writes a frequency-sorted CSV to Hot Queries/.

Run: uv run --package common python packages/common/scripts/export_hot_search_queries.py

Env required (see .env.example): DB_IP, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME.
"""

import csv
import os
from collections import Counter
from pathlib import Path

import pymysql
from dotenv import load_dotenv

from common.instant_classifier_llm_dataset import normalize_query

_TABLES = ["TAXMANNUSERSEARCHHISTORY", "TAXMANNUSERSEARCHHISTORY_2023"]
_RESEARCH_PATH_PREFIX = "https://www.taxmann.com/research%"
_OUTPUT_PATH = Path(__file__).parent.parent.parent.parent / "Hot Queries" / "hot_search_queries.csv"


def _fetch_all_queries() -> list[str]:
    print(f"Connecting to {os.environ['DB_NAME']} at {os.environ['DB_IP']}:{os.environ['DB_PORT']} ...")
    conn = pymysql.connect(
        host=os.environ["DB_IP"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        connect_timeout=10,
    )
    print("Connected.")
    try:
        cur = conn.cursor()
        raw_keywords: list[str] = []
        for table in _TABLES:
            print(f"Fetching from {table} ...")
            cur.execute(
                f"""
                SELECT SEARCHKEYWORD FROM {table}
                WHERE COMPONENTURL LIKE %s
                """,
                (_RESEARCH_PATH_PREFIX,),
            )
            table_rows = [row[0] for row in cur.fetchall() if row[0]]
            print(f"  {table}: {len(table_rows)} rows")
            raw_keywords.extend(table_rows)
        cur.close()
    finally:
        conn.close()
    return raw_keywords


def main() -> None:
    load_dotenv()
    raw_keywords = _fetch_all_queries()

    print(f"Normalizing and counting {len(raw_keywords)} rows ...")
    counts: Counter[str] = Counter()
    for raw in raw_keywords:
        text = normalize_query(raw)
        if text:
            counts[text] += 1

    print(f"Writing {len(counts)} unique queries to {_OUTPUT_PATH} ...")
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "count"])
        for query, count in counts.most_common():
            writer.writerow([query, count])

    print(f"Done. Pulled {len(raw_keywords)} rows, {len(counts)} unique normalized queries.")
    print(f"Wrote {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
