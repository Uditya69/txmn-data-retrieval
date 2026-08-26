"""Evaluate the trained instant-mode query classifier against its own training data.

This runs every query in train.jsonl (and, optionally, eval_frozen.jsonl) back through
the committed classifier artifact and checks the predicted label against the label the
query was trained with. It answers "did the model actually learn the training set" -
a fit check, not a generalization check. train_instant_classifier.py's held-out
eval_frozen.jsonl score (see accuracy_at_threshold in instant_classifier_model_meta.json)
is the number that matters for judging real-world accuracy; a low score here on
training data itself points at underfitting (bad features, too few training examples for
a label, mislabeled rows) rather than a model that merely fails to generalize.

Usage:
  uv run python packages/common/scripts/eval_instant_classifier_on_training_data.py
  uv run python packages/common/scripts/eval_instant_classifier_on_training_data.py --include-eval
  uv run python packages/common/scripts/eval_instant_classifier_on_training_data.py --json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common.instant_classifier.labels import LABELS, ClassifierResult, resolve_routing
from common.instant_classifier.pipeline import load_artifact

_DATA_DIR = Path(__file__).parent.parent / "data" / "instant_classifier"
_TRAIN_PATH = _DATA_DIR / "train.jsonl"
_EVAL_PATH = _DATA_DIR / "eval_frozen.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--include-eval", action="store_true",
        help="Also score eval_frozen.jsonl (held-out set) alongside train.jsonl, reported separately",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    parser.add_argument("--show-misses", type=int, default=20, help="Max misclassified rows to print (0 = none)")
    args = parser.parse_args()

    pipeline, meta = load_artifact()
    threshold = meta["confidence_threshold"]

    datasets = [("train", _TRAIN_PATH)]
    if args.include_eval:
        datasets.append(("eval_frozen", _EVAL_PATH))

    report = {}
    for name, path in datasets:
        rows = _load_jsonl(path)
        texts = [r["query_text"] for r in rows]
        true_labels = [r["label"] for r in rows]

        proba = pipeline.predict_proba(texts)
        classes = pipeline.classes_
        raw_labels = [classes[p.argmax()] for p in proba]
        confidences = [float(p.max()) for p in proba]
        effective_labels = [
            resolve_routing(ClassifierResult(label=rl, confidence=c), threshold)
            for rl, c in zip(raw_labels, confidences)
        ]

        confusion = defaultdict(Counter)  # true -> Counter(predicted -> count)
        misses = []
        raw_correct = 0
        effective_correct = 0
        for text, true, raw, eff, conf in zip(texts, true_labels, raw_labels, effective_labels, confidences):
            confusion[true][eff] += 1
            if raw == true:
                raw_correct += 1
            if eff == true:
                effective_correct += 1
            else:
                misses.append({"query": text, "true": true, "raw": raw, "effective": eff, "confidence": round(conf, 4)})

        total = len(rows)
        per_label = {}
        for label in LABELS:
            label_total = sum(1 for t in true_labels if t == label)
            label_correct = confusion[label][label] if label_total else 0
            per_label[label] = {
                "total": label_total,
                "correct": label_correct,
                "accuracy": (label_correct / label_total) if label_total else None,
            }

        report[name] = {
            "total": total,
            "raw_accuracy": raw_correct / total if total else None,
            "effective_accuracy": effective_correct / total if total else None,
            "per_label": per_label,
            "confusion": {t: dict(preds) for t, preds in confusion.items()},
            "misses": misses,
        }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"confidence_threshold={threshold}  (predictions below this confidence fall back to HYBRID)\n")
    for name, r in report.items():
        print(f"=== {name} ({r['total']} queries) ===")
        print(f"raw label accuracy:       {r['raw_accuracy']:.2%}")
        print(f"effective (post-threshold) accuracy: {r['effective_accuracy']:.2%}")
        print()
        print(f"{'label':<10} {'total':<7} {'correct':<8} {'accuracy'}")
        for label, stats in r["per_label"].items():
            acc = f"{stats['accuracy']:.2%}" if stats["accuracy"] is not None else "-"
            print(f"{label:<10} {stats['total']:<7} {stats['correct']:<8} {acc}")
        print()
        print("confusion (rows = true label, cols = predicted effective label):")
        label_col = "true \\ pred"
        header = f"{label_col:<12}" + "".join(f"{l:<10}" for l in LABELS)
        print(header)
        for true_label in LABELS:
            row = report[name]["confusion"].get(true_label, {})
            print(f"{true_label:<12}" + "".join(f"{row.get(l, 0):<10}" for l in LABELS))
        print()

        if args.show_misses and r["misses"]:
            shown = r["misses"][: args.show_misses]
            print(f"misclassified ({len(r['misses'])} total, showing {len(shown)}):")
            for m in shown:
                print(
                    f"  [{m['true']} -> {m['effective']}] (raw={m['raw']}, conf={m['confidence']}) {m['query']!r}"
                )
        print()


if __name__ == "__main__":
    main()
