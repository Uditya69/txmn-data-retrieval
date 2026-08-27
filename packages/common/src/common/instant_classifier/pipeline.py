import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common.instant_classifier.features import build_feature_union

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_MODEL_PATH = _DATA_DIR / "instant_classifier_model.joblib"
_DEFAULT_META_PATH = _DATA_DIR / "instant_classifier_model_meta.json"


def build_pipeline(C: float = 1.0, word_min_df: int = 1, char_min_df: int = 1) -> Pipeline:
    return Pipeline([
        ("features", build_feature_union(word_min_df=word_min_df, char_min_df=char_min_df)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=C)),
    ])


def build_calibrated_pipeline(
    C: float = 1.0, word_min_df: int = 1, char_min_df: int = 1, cv: int = 5,
) -> CalibratedClassifierCV:
    """Wraps build_pipeline() (features + LogisticRegression) in Platt-scaling
    calibration so predict_proba().max() reflects an actual estimated probability of
    correctness, not just raw softmax output - resolve_routing()'s confidence_threshold
    was empirically swept against raw softmax values, which don't need to correspond to
    real likelihoods. Sigmoid (Platt), not isotonic, because isotonic needs more calibration
    samples per class than this dataset's size to avoid overfitting the calibration curve
    itself. Wrapping the whole feature+classifier Pipeline (not just the classifier) means
    each CV fold refits the TF-IDF vocabulary independently - no leakage from eval-fold
    queries into the vocabulary used to score them."""
    return CalibratedClassifierCV(
        build_pipeline(C=C, word_min_df=word_min_df, char_min_df=char_min_df), method="sigmoid", cv=cv,
    )


def save_artifact(pipeline: Pipeline, meta: dict, model_path: Path = None, meta_path: Path = None) -> None:
    model_path = model_path or _DEFAULT_MODEL_PATH
    meta_path = meta_path or _DEFAULT_META_PATH
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(meta, indent=2))


def load_artifact(model_path: Path = None, meta_path: Path = None) -> tuple[Pipeline, dict]:
    model_path = model_path or _DEFAULT_MODEL_PATH
    meta_path = meta_path or _DEFAULT_META_PATH
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Instant classifier artifact missing ({model_path}, {meta_path}) - run "
            "packages/common/scripts/train_instant_classifier.py before starting the service."
        )
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return pipeline, meta
