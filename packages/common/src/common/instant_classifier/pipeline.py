import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common.instant_classifier.features import build_feature_union

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_MODEL_PATH = _DATA_DIR / "instant_classifier_model.joblib"
_DEFAULT_META_PATH = _DATA_DIR / "instant_classifier_model_meta.json"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", build_feature_union()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


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
