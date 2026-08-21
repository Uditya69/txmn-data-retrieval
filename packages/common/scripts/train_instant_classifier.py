import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from common.instant_classifier.pipeline import build_pipeline, save_artifact

_DATA_DIR = Path(__file__).parent.parent / "data" / "instant_classifier"
_TRAIN_PATH = _DATA_DIR / "train.jsonl"
_EVAL_PATH = _DATA_DIR / "eval_frozen.jsonl"


def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["query_text"])
        labels.append(row["label"])
    return texts, labels


def _sweep_threshold(pipeline, eval_texts: list[str], eval_labels: list[str]) -> tuple[float, float]:
    """Picks the confidence cutoff (0.1-0.9) that determines when resolve_routing()'s
    fallback branch fires (confidence < threshold -> FALLBACK).

    Accuracy at each threshold is computed over KEPT predictions only
    (kept_correct / kept_total, NOT the fixed eval-set size) - otherwise raising the
    threshold can only remove examples from the numerator, never add any, which makes
    the metric non-increasing in threshold and guarantees the sweep always "wins" at
    the lowest threshold tried regardless of the data. That also matters because a
    3-class softmax's predict_proba().max() is never below ~1/3 - a threshold near 0.1
    would then never actually gate anything, leaving resolve_routing()'s fallback branch
    dead code.

    Among thresholds whose kept-set accuracy is at least as good as the unfiltered
    (overall) accuracy, picks the HIGHEST one - being more selective should never come
    at the cost of the kept predictions being less reliable than just trusting every
    prediction. If no threshold clears that bar (possible on a small/noisy eval set),
    falls back to whichever threshold has the best kept-set accuracy, ties broken
    toward the lower threshold so more queries stay on the automatic-routing path
    instead of falling back unnecessarily. Thresholds that would keep zero predictions
    are skipped (undefined accuracy)."""
    proba = pipeline.predict_proba(eval_texts)
    classes = pipeline.classes_
    predicted = [classes[row.argmax()] for row in proba]
    confidences = [row.max() for row in proba]
    overall_accuracy = accuracy_score(eval_labels, predicted)

    candidates: list[tuple[float, float]] = []  # (threshold, kept_accuracy)
    for threshold in [i / 10 for i in range(1, 10)]:
        kept_pairs = [
            (p, t) for p, t, c in zip(predicted, eval_labels, confidences) if c >= threshold
        ]
        if not kept_pairs:
            continue  # nothing survives this threshold - accuracy undefined, skip it
        kept_correct = sum(p == t for p, t in kept_pairs)
        candidates.append((threshold, kept_correct / len(kept_pairs)))

    qualifying = [c for c in candidates if c[1] >= overall_accuracy]
    if qualifying:
        return max(qualifying, key=lambda c: c[0])
    return max(candidates, key=lambda c: (c[1], -c[0]))


def main() -> None:
    train_texts, train_labels = _load_jsonl(_TRAIN_PATH)
    eval_texts, eval_labels = _load_jsonl(_EVAL_PATH)

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(eval_texts)
    overall_accuracy = accuracy_score(eval_labels, predictions)
    threshold, accuracy_at_threshold = _sweep_threshold(pipeline, eval_texts, eval_labels)

    meta = {
        "version": 1,
        "train_examples": len(train_texts),
        "eval_examples": len(eval_texts),
        "overall_eval_accuracy": overall_accuracy,
        "confidence_threshold": threshold,
        "accuracy_at_threshold": accuracy_at_threshold,
        "labels": sorted(set(train_labels)),
    }
    save_artifact(pipeline, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
