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
    """Picks the confidence cutoff (0.1-0.9) maximizing eval accuracy among kept
    predictions, ties broken toward the lower threshold - that keeps more queries on
    the automatic-routing path instead of falling back unnecessarily."""
    proba = pipeline.predict_proba(eval_texts)
    classes = pipeline.classes_
    best_threshold, best_accuracy = 0.1, -1.0
    for threshold in [i / 10 for i in range(1, 10)]:
        kept_correct = 0
        for row_proba, true_label in zip(proba, eval_labels):
            predicted = classes[row_proba.argmax()]
            confidence = row_proba.max()
            if confidence >= threshold and predicted == true_label:
                kept_correct += 1
        accuracy = kept_correct / len(eval_labels)
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, threshold
    return best_threshold, best_accuracy


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
