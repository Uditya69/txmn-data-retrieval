"""Pure helpers for building more training data for the existing Instant-mode
classifier (packages/common/src/common/instant_classifier/labels.py: KEYWORD = ES
only, INTENT = Milvus dense only, HYBRID = both) from real user search queries.
See packages/common/scripts/build_instant_classifier_dataset.py for the DB + LLM
orchestration that uses these.
"""

import json
from pathlib import Path
from urllib.parse import unquote

LABELS = ("KEYWORD", "INTENT", "HYBRID")

_SYSTEM_PROMPT = f"""You label search queries for a legal/tax research system with one of \
three retrieval routes:

- KEYWORD: exact lookups a lexical (Elasticsearch) index resolves well - section/rule/article \
numbers, citations, case names, circular/notification numbers, short keyword phrases.
- INTENT: conceptual or natural-language questions with no exact term to match, resolved via \
dense/semantic vector search - paraphrasable questions about a legal situation or how \
something works.
- HYBRID: queries mixing an exact reference with a conceptual question, or ambiguous \
queries that benefit from both a lexical and a semantic pass.

Given a JSON array of queries, respond with ONLY a JSON object: \
{{"labels": [<label for query 1>, <label for query 2>, ...]}}. The labels array must have \
exactly the same length and order as the input queries. Every label must be one of: \
{", ".join(LABELS)}."""


def normalize_query(raw: str) -> str:
    return unquote(raw).strip()


def chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_labeling_messages(queries: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(queries)},
    ]


def parse_labels_response(content: str, expected_count: int) -> list[str]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {content!r}") from exc

    labels = parsed.get("labels") if isinstance(parsed, dict) else None
    if not isinstance(labels, list):
        raise ValueError(f"LLM response missing 'labels' array: {content!r}")
    if len(labels) != expected_count:
        raise ValueError(f"expected {expected_count} labels, got {len(labels)}: {content!r}")
    for label in labels:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r} (expected one of {LABELS})")
    return labels


def format_jsonl_row(*, query_text: str, label: str, source: str, date_added: str) -> str:
    return json.dumps(
        {
            "query_text": query_text,
            "label": label,
            "source": source,
            "date_added": date_added,
        }
    )


def load_checkpoint(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_checkpoint(path: Path, checkpoint: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2))
