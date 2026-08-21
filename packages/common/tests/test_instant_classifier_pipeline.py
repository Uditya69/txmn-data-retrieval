import json

import pytest

from common.instant_classifier.pipeline import build_pipeline, load_artifact, save_artifact

_TRAIN_TEXTS = ["Section 52", "Section 80C", "How do I evade tax", "What can I do about GST", "Rule 6DD", "Article 21"]
_TRAIN_LABELS = ["KEYWORD", "KEYWORD", "INTENT", "INTENT", "KEYWORD", "KEYWORD"]


def test_build_pipeline_fits_and_predicts():
    pipeline = build_pipeline()
    pipeline.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    predictions = pipeline.predict(["Section 100"])
    assert predictions[0] in {"KEYWORD", "INTENT"}


def test_save_and_load_artifact_roundtrips(tmp_path):
    pipeline = build_pipeline()
    pipeline.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    meta = {"version": 1, "confidence_threshold": 0.5}

    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    save_artifact(pipeline, meta, model_path=model_path, meta_path=meta_path)

    loaded_pipeline, loaded_meta = load_artifact(model_path=model_path, meta_path=meta_path)
    assert loaded_meta == meta
    assert loaded_pipeline.predict(["Section 52"])[0] == pipeline.predict(["Section 52"])[0]
    assert json.loads(meta_path.read_text()) == meta


def test_load_artifact_raises_when_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(model_path=tmp_path / "missing.joblib", meta_path=tmp_path / "missing.json")
