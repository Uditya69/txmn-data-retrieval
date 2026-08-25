"""Probe the trained instant-mode query classifier against a hand-built case set.

Prints per-case predicted label, confidence, routing plan, and (where
expected_label is set) a pass/fail mark. Cases with no expected_label are
"probe only" - useful for tricky/ambiguous queries where you want to see the
model's behavior without asserting a ground truth.

Usage: see README at bottom of file / CLAUDE.md running-things section.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from common.instant_classifier import classify, confidence_threshold
from common.instant_classifier.labels import resolve_routing, routing_plan

_DEFAULT_CASES_PATH = Path(__file__).parent.parent.parent.parent / "evals" / "instant_classifier_probe_cases.json"


def _load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Path to probe cases JSON")
    parser.add_argument("--category", default=None, help="Only run cases whose category matches this value")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]

    threshold = confidence_threshold()
    rows = []
    scored_total = 0
    scored_correct = 0
    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "scored": 0})

    for case in cases:
        query = case["query"]
        expected = case.get("expected_label") or None
        result = classify(query)
        effective = resolve_routing(result, threshold)
        plan = routing_plan(effective)

        row = {
            "id": case.get("id"),
            "category": case.get("category"),
            "query": query,
            "raw_label": result.label,
            "confidence": round(result.confidence, 4),
            "effective_label": effective,
            "routing": plan,
            "expected_label": expected,
        }

        cat = case.get("category", "uncategorized")
        by_category[cat]["total"] += 1
        if expected:
            match = effective == expected
            row["match"] = match
            scored_total += 1
            by_category[cat]["scored"] += 1
            if match:
                scored_correct += 1
                by_category[cat]["correct"] += 1
        rows.append(row)

    if args.json:
        print(json.dumps({"rows": rows, "scored_total": scored_total, "scored_correct": scored_correct}, indent=2))
        return

    print(f"confidence_threshold={threshold}\n")
    header = f"{'id':<5} {'category':<24} {'query':<45} {'raw':<8} {'conf':<6} {'effective':<9} {'expected':<9} {'match'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        match_str = "" if row.get("expected_label") is None else ("PASS" if row["match"] else "FAIL")
        print(
            f"{row['id']:<5} {row['category']:<24} {row['query'][:45]:<45} "
            f"{row['raw_label']:<8} {row['confidence']:<6} {row['effective_label']:<9} "
            f"{(row['expected_label'] or '-'):<9} {match_str}"
        )

    print()
    if scored_total:
        print(f"Overall (scored cases only): {scored_correct}/{scored_total} = {scored_correct / scored_total:.2%}")
    else:
        print("No cases had an expected_label - nothing scored, probe-only run.")

    print("\nBy category:")
    for cat, stats in by_category.items():
        scored_note = f"{stats['correct']}/{stats['scored']} scored" if stats["scored"] else "probe only"
        print(f"  {cat:<24} total={stats['total']:<3} {scored_note}")


if __name__ == "__main__":
    main()
