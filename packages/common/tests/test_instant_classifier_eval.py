import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from common.instant_classifier.pipeline import load_artifact

_EVAL_PATH = Path(__file__).parent.parent / "data" / "instant_classifier" / "eval_frozen.jsonl"

# Floor set from the accuracy actually measured when this artifact was trained (see
# Step 3's printed overall_eval_accuracy = 0.875), minus a ~0.1 safety margin, rounded
# down to one decimal (0.775 -> 0.7) - catches a bad retrain-and-commit regressing
# accuracy, not a specific target number.
_ACCURACY_FLOOR = 0.7


def _load_jsonl(path):
    texts, labels = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["query_text"])
        labels.append(row["label"])
    return texts, labels


def test_committed_artifact_meets_eval_accuracy_floor():
    pipeline, meta = load_artifact()
    texts, labels = _load_jsonl(_EVAL_PATH)
    predictions = pipeline.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    assert accuracy >= _ACCURACY_FLOOR, f"eval accuracy {accuracy} fell below floor {_ACCURACY_FLOOR}"
    assert meta["overall_eval_accuracy"] >= _ACCURACY_FLOOR
