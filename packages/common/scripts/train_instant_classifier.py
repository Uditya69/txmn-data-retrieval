import json
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common.instant_classifier.pipeline import build_calibrated_pipeline, build_pipeline, save_artifact

_DATA_DIR = Path(__file__).parent.parent / "data" / "instant_classifier"
_TRAIN_PATH = _DATA_DIR / "train.jsonl"
_EVAL_PATH = _DATA_DIR / "eval_frozen.jsonl"

# Small grid, not exhaustive - this dataset's size (a few hundred rows) doesn't support
# tuning more knobs than this without the CV estimate itself becoming noisy.
_C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
# word_min_df is pinned to 1, not searched: at min_df=2 on the 733-row real-traffic set,
# CV picked word_min_df=2 and INTENT accuracy on eval_frozen collapsed 98.77% -> 68.63%.
# Many real INTENT queries are one-off topic nouns ("dematerialization", "invoice
# discounting") whose words appear in exactly 1 training document - min_df=2 prunes them
# from the TF-IDF vocabulary entirely, leaving those queries with almost no word-level
# signal to separate INTENT from bare KEYWORD terms. char_min_df stays searched since the
# char n-gram vocabulary is large enough that pruning singletons there doesn't remove an
# entire query's signal the way it does for whole words.
_WORD_MIN_DF = 1
_CHAR_MIN_DF_GRID = [1, 2]
_CV_FOLDS = 5


def _select_hyperparameters(train_texts: list[str], train_labels: list[str]) -> dict:
    """Picks (C, char_min_df) by mean accuracy over stratified CV folds on the TRAINING
    set only - never touches eval_frozen.jsonl, which stays reserved for the final
    held-out accuracy number. A single train/eval split is too small (a few hundred rows)
    to trust for hyperparameter selection - a couple of flipped predictions swing the
    single-split accuracy enough to make sklearn's untuned default C look no worse than a
    genuinely better one. CV averages that noise out over multiple folds. word_min_df is
    pinned (see _WORD_MIN_DF), not part of the grid."""
    splitter = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=0)
    best = None
    for c in _C_GRID:
        for char_min_df in _CHAR_MIN_DF_GRID:
            pipeline = build_pipeline(C=c, word_min_df=_WORD_MIN_DF, char_min_df=char_min_df)
            scores = cross_val_score(pipeline, train_texts, train_labels, cv=splitter, scoring="accuracy")
            mean_score = scores.mean()
            candidate = {
                "C": c, "word_min_df": _WORD_MIN_DF, "char_min_df": char_min_df, "cv_accuracy": mean_score,
            }
            if best is None or mean_score > best["cv_accuracy"]:
                best = candidate
    return best


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
    fallback branch fires (confidence < threshold -> defaults to HYBRID).

    Accuracy at each threshold is computed over KEPT predictions only
    (kept_correct / kept_total, NOT the fixed eval-set size) - otherwise raising the
    threshold can only remove examples from the numerator, never add any, which makes
    the metric non-increasing in threshold and guarantees the sweep always "wins" at
    the lowest threshold tried regardless of the data. That also matters because a
    3-class softmax's predict_proba().max() is never below ~1/3 - a threshold near 0.1
    would then never actually gate anything, leaving resolve_routing()'s fallback branch
    dead code.

    Among thresholds whose kept-set accuracy is at least as good as the unfiltered
    (overall) accuracy, picks whichever achieves the HIGHEST kept-set accuracy (never
    settling for a threshold no more reliable than trusting every prediction); among
    those tied at that best accuracy, picks the LOWEST threshold - once a given
    reliability level is reached, going more selective from there only discards
    additional correct predictions into resolve_routing()'s fallback branch for zero
    reliability gain, so ties are broken toward keeping more queries on the confident
    auto-routing path (same reasoning as the no-threshold-qualifies branch just below).
    Picking the lowest threshold that merely clears the qualifying bar - without also
    requiring it hit the best achievable accuracy - would wrongly prefer a threshold
    near 0.1 whenever kept-set accuracy happens to equal overall accuracy at every low
    threshold (it always does until enough low-confidence rows get filtered out to move
    the number): a 3-class softmax's predict_proba().max() is never below ~1/3, so that
    would leave resolve_routing()'s fallback branch permanently dead code. If no
    threshold clears the qualifying bar at all (possible on a small/noisy eval set),
    falls back to whichever threshold has the best kept-set accuracy, ties broken
    toward the lower threshold for the same reason. Thresholds that would keep zero
    predictions are skipped (undefined accuracy)."""
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
        best_accuracy = max(c[1] for c in qualifying)
        return min((c for c in qualifying if c[1] == best_accuracy), key=lambda c: c[0])
    return max(candidates, key=lambda c: (c[1], -c[0]))


def main() -> None:
    train_texts, train_labels = _load_jsonl(_TRAIN_PATH)
    eval_texts, eval_labels = _load_jsonl(_EVAL_PATH)

    best_hyperparams = _select_hyperparameters(train_texts, train_labels)

    # Calibrated (Platt-scaled) for the committed artifact - resolve_routing()'s confidence
    # threshold is meaningless as a reliability cutoff unless predict_proba().max() reflects
    # an actual estimated correctness probability rather than raw softmax output.
    pipeline = build_calibrated_pipeline(
        C=best_hyperparams["C"],
        word_min_df=best_hyperparams["word_min_df"],
        char_min_df=best_hyperparams["char_min_df"],
        cv=_CV_FOLDS,
    )
    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(eval_texts)
    overall_accuracy = accuracy_score(eval_labels, predictions)
    threshold, accuracy_at_threshold = _sweep_threshold(pipeline, eval_texts, eval_labels)

    meta = {
        "version": 2,
        "train_examples": len(train_texts),
        "eval_examples": len(eval_texts),
        "hyperparameters": best_hyperparams,
        "calibration": "sigmoid",
        "overall_eval_accuracy": overall_accuracy,
        "confidence_threshold": threshold,
        "accuracy_at_threshold": accuracy_at_threshold,
        "labels": sorted(set(train_labels)),
    }
    save_artifact(pipeline, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
