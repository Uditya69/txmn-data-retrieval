# Instant Mode Query Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-only sklearn classifier that labels every Instant Mode query `KEYWORD | HYBRID | INTENT`, use that label to pick ES boost profiles (replacing the existing rule-based `classify_query_shape()`), and — behind a new opt-in `auto_route` toggle — use it to decide whether to skip ES, skip Milvus, or query both with RRF fusion, falling back to today's always-both behavior when the model isn't confident or the toggle is off.

**Architecture:** A new `packages/common/src/common/instant_classifier/` submodule holds a `FeatureUnion` (citation regex, gazetteer, structural, intent-language, TF-IDF word/char n-grams) + `LogisticRegression` pipeline, trained offline from a committed JSONL dataset and loaded once at `retrieval-api` startup. `labels.py` is the single place that maps a label to an ES boost-profile key and a backend-routing plan; `es_client.py`, `rerank.py`, and `search.py` each consume it for their own concern.

**Tech Stack:** scikit-learn (`LogisticRegression`, `TfidfVectorizer`, `FeatureUnion`, `Pipeline`), `joblib` for artifact serialization, existing `common.legal_lexicon`/`common.query_tokenizer` primitives, FastAPI lifespan for eager model loading, React/TypeScript for the new toggle.

**Spec:** `docs/superpowers/specs/2026-08-20-instant-mode-query-classifier-design.md`

## Global Constraints

- Python 3.11 only (repo-wide constraint — `pymilvus`'s `grpcio` has no 3.14 wheel).
- No raw-score blending between ES and Milvus results anywhere (CLAUDE.md hard rule 3) — RRF fusion is rank-based only, unchanged from today's `rrf_merge_by_doc_id`.
- No LLM/SLM call anywhere in this classifier's inference path — CPU-only, in-process, <5ms target.
- `uv sync --all-packages` (not bare `uv sync`) after any `pyproject.toml` dependency change, or editable installs break.
- Model artifact + training data live in `packages/common` (not a new package) — this repo is trending toward fewer, lighter-weight packages.
- Every new/changed public function needs a test; run affected package's test suite (`uv run pytest packages/common/tests` / `packages/retrieval-api/tests`) before each commit.

---

### Task 1: Fold `KNOWN_COURTS` / `_LEGAL_MARKERS` into the shared lexicon

Two small hand-maintained lists duplicate what `legal_lexicon.json` is supposed to be the single source of truth for (flagged in `docs/superpowers/specs/2026-08-11-instant-mode-es-retrieval-redesign-design.md`): `schema_context.KNOWN_COURTS` (9 full court names, used in AI Mode's LLM prompt and in `intent.py`'s rewrite-safety check) and `intent._LEGAL_MARKERS` (10 Act names, same rewrite-safety check). Fold both into `legal_lexicon.json` as new top-level keys — not into the existing `courts`/`synonyms` keys, which store differently-shaped data (short abbreviations/city tokens for single-token lookup, not full multi-word names) — so this is a pure refactor with zero behavior change.

**Files:**
- Modify: `packages/common/src/common/data/legal_lexicon.json`
- Modify: `packages/common/src/common/legal_lexicon.py`
- Modify: `packages/common/src/common/schema_context.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
- Test: `packages/common/tests/test_legal_lexicon.py`
- Test: `packages/common/tests/test_schema_context.py`

**Interfaces:**
- Produces: `common.legal_lexicon.KNOWN_COURT_FULL_NAMES: list[str]`, `common.legal_lexicon.KNOWN_ACT_NAMES: set[str]` (lowercase, matching `_LEGAL_MARKERS`'s existing casefold-compare usage) — later tasks don't consume these, but `schema_context.py`/`intent.py` do.

- [ ] **Step 1: Write the failing test for the new lexicon keys**

```python
# packages/common/tests/test_legal_lexicon.py — add at end of file
from common.legal_lexicon import KNOWN_ACT_NAMES, KNOWN_COURT_FULL_NAMES


def test_known_court_full_names_includes_original_nine():
    for name in [
        "Supreme Court", "Delhi High Court", "Bombay High Court", "Madras High Court",
        "Calcutta High Court", "Karnataka High Court", "Gujarat High Court",
        "Income Tax Appellate Tribunal", "Customs Excise and Service Tax Appellate Tribunal",
    ]:
        assert name in KNOWN_COURT_FULL_NAMES


def test_known_act_names_includes_original_ten():
    for name in [
        "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
        "bharatiya sakshya adhiniyam", "indian penal code", "income-tax act",
        "income tax act", "cgst act", "igst act", "customs act",
        "code of criminal procedure", "indian evidence act",
    ]:
        assert name in KNOWN_ACT_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_legal_lexicon.py -k "court_full_names or act_names" -v`
Expected: FAIL with `ImportError: cannot import name 'KNOWN_ACT_NAMES'`

- [ ] **Step 3: Add the two new keys to legal_lexicon.json and load them in legal_lexicon.py**

Edit `packages/common/src/common/data/legal_lexicon.json`: add two new top-level keys alongside `courts`/`journals`/`stopwords`/`synonyms`/`normalizations`:

```json
"court_full_names": [
    "Supreme Court", "Delhi High Court", "Bombay High Court", "Madras High Court",
    "Calcutta High Court", "Karnataka High Court", "Gujarat High Court",
    "Income Tax Appellate Tribunal", "Customs Excise and Service Tax Appellate Tribunal"
],
"act_names": [
    "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
    "bharatiya sakshya adhiniyam", "indian penal code", "income-tax act",
    "income tax act", "cgst act", "igst act", "customs act",
    "code of criminal procedure", "indian evidence act"
]
```

Add to `packages/common/src/common/legal_lexicon.py`, after the existing `_NORMALIZATIONS = _LEXICON["normalizations"]` line:

```python
KNOWN_COURT_FULL_NAMES: list[str] = _LEXICON["court_full_names"]
KNOWN_ACT_NAMES: set[str] = set(_LEXICON["act_names"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_legal_lexicon.py -k "court_full_names or act_names" -v`
Expected: PASS

- [ ] **Step 5: Point schema_context.py and intent.py at the shared lexicon**

Edit `packages/common/src/common/schema_context.py` — replace the hardcoded `KNOWN_COURTS` list:

```python
from common.legal_lexicon import KNOWN_COURT_FULL_NAMES
from common.schemas import MILVUS_COLLECTIONS

COLLECTION_DESCRIPTIONS = { ... }  # unchanged

KNOWN_FILTER_FIELDS = ["court", "act", "section", "party", "date_range", "bench", "judge"]

KNOWN_COURTS = KNOWN_COURT_FULL_NAMES
```

(Keeping the `KNOWN_COURTS` name as an alias avoids touching every import site in this step — `intent.py` already does `from common.schema_context import KNOWN_COURTS, build_schema_context`.)

Edit `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py` — replace the hardcoded `_LEGAL_MARKERS` set (around line 270) with:

```python
from common.legal_lexicon import KNOWN_ACT_NAMES

_LEGAL_MARKERS = KNOWN_ACT_NAMES
```

Remove the old literal `_LEGAL_MARKERS = {...}` block entirely — `_safe_rewrite` (line 287-301) and every other reference keep using the name `_LEGAL_MARKERS` unchanged, so no other line in that file needs to change.

- [ ] **Step 6: Run the full existing test suites for both files to confirm no regression**

Run: `uv run pytest packages/common/tests/test_schema_context.py packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: PASS (identical behavior — this step only changed where the data comes from)

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/data/legal_lexicon.json packages/common/src/common/legal_lexicon.py packages/common/src/common/schema_context.py packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/common/tests/test_legal_lexicon.py
git commit -m "refactor(lexicon): fold KNOWN_COURTS/_LEGAL_MARKERS into shared legal_lexicon.json"
```

---

### Task 2: `instant_classifier` package skeleton + label taxonomy

Add the sklearn/joblib dependency and create the taxonomy module that every later task (feature extraction, pipeline, ES boost selection, backend routing) reads from. This is the one file that knows what `KEYWORD`/`HYBRID`/`INTENT`/`FALLBACK` mean downstream.

**Files:**
- Modify: `packages/common/pyproject.toml`
- Create: `packages/common/src/common/instant_classifier/__init__.py` (empty for now — Task 6 fills in the public API)
- Create: `packages/common/src/common/instant_classifier/labels.py`
- Test: `packages/common/tests/test_instant_classifier_labels.py`

**Interfaces:**
- Produces: `ClassifierResult(label: str, confidence: float)` (frozen dataclass), constants `KEYWORD="KEYWORD"`, `HYBRID="HYBRID"`, `INTENT="INTENT"`, `FALLBACK="FALLBACK"`, `boost_profile_key(label: str) -> str`, `resolve_routing(result: ClassifierResult, threshold: float) -> str`, `routing_plan(effective_label: str) -> dict` (keys `"es"`, `"milvus"`, `"fuse"`, all `bool`).

- [ ] **Step 1: Add scikit-learn/joblib/numpy to common's dependencies**

Edit `packages/common/pyproject.toml`:

```toml
dependencies = [
  "pydantic-settings>=2.5",
  "pymilvus>=2.4",
  "elasticsearch>=8.15,<9",
  "aiohttp>=3.9",
  "tiktoken>=0.7",
  "scikit-learn>=1.5,<2",
  "joblib>=1.4,<2",
  "numpy>=1.26,<2",
]
```

Run: `uv sync --all-packages`
Expected: resolves and installs scikit-learn/joblib/numpy into the shared venv without error.

- [ ] **Step 2: Write the failing test for labels.py**

```python
# packages/common/tests/test_instant_classifier_labels.py
from common.instant_classifier.labels import (
    FALLBACK, HYBRID, INTENT, KEYWORD,
    ClassifierResult, boost_profile_key, resolve_routing, routing_plan,
)


def test_boost_profile_key_maps_each_label_to_itself():
    assert boost_profile_key(KEYWORD) == "KEYWORD"
    assert boost_profile_key(HYBRID) == "HYBRID"
    assert boost_profile_key(INTENT) == "INTENT"


def test_boost_profile_key_maps_fallback_to_hybrid():
    # Balanced weighting is the safest default when confidence is too low to trust
    # a KEYWORD-only or INTENT-only boost profile.
    assert boost_profile_key(FALLBACK) == "HYBRID"


def test_resolve_routing_keeps_label_when_confident():
    result = ClassifierResult(label=KEYWORD, confidence=0.95)
    assert resolve_routing(result, threshold=0.5) == KEYWORD


def test_resolve_routing_falls_back_when_below_threshold():
    result = ClassifierResult(label=KEYWORD, confidence=0.4)
    assert resolve_routing(result, threshold=0.5) == FALLBACK


def test_routing_plan_keyword_skips_milvus_no_fusion():
    assert routing_plan(KEYWORD) == {"es": True, "milvus": False, "fuse": False}


def test_routing_plan_intent_skips_es_no_fusion():
    assert routing_plan(INTENT) == {"es": False, "milvus": True, "fuse": False}


def test_routing_plan_hybrid_queries_both_and_fuses():
    assert routing_plan(HYBRID) == {"es": True, "milvus": True, "fuse": True}


def test_routing_plan_fallback_queries_both_no_fusion():
    # Matches today's default Instant Mode behavior exactly.
    assert routing_plan(FALLBACK) == {"es": True, "milvus": True, "fuse": False}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_instant_classifier_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.instant_classifier'`

- [ ] **Step 4: Create the package and labels.py**

Create `packages/common/src/common/instant_classifier/__init__.py` (empty file for now).

Create `packages/common/src/common/instant_classifier/labels.py`:

```python
from dataclasses import dataclass

KEYWORD = "KEYWORD"
HYBRID = "HYBRID"
INTENT = "INTENT"
FALLBACK = "FALLBACK"

LABELS = (KEYWORD, HYBRID, INTENT)

_BOOST_PROFILE_KEY = {KEYWORD: KEYWORD, HYBRID: HYBRID, INTENT: INTENT, FALLBACK: HYBRID}

_ROUTING = {
    KEYWORD: {"es": True, "milvus": False, "fuse": False},
    HYBRID: {"es": True, "milvus": True, "fuse": True},
    INTENT: {"es": False, "milvus": True, "fuse": False},
    FALLBACK: {"es": True, "milvus": True, "fuse": False},
}


@dataclass(frozen=True)
class ClassifierResult:
    label: str
    confidence: float


def boost_profile_key(label: str) -> str:
    return _BOOST_PROFILE_KEY[label]


def resolve_routing(result: ClassifierResult, threshold: float) -> str:
    """Below threshold: trust neither the label nor its confidence for routing
    purposes, fall back to querying every backend (today's default Instant Mode
    behavior) rather than risk skipping a backend the model was unsure about."""
    if result.confidence < threshold:
        return FALLBACK
    return result.label


def routing_plan(effective_label: str) -> dict:
    return _ROUTING[effective_label]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_instant_classifier_labels.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/common/pyproject.toml packages/common/uv.lock packages/common/src/common/instant_classifier/ packages/common/tests/test_instant_classifier_labels.py
git commit -m "feat(instant-classifier): add label taxonomy and routing/boost-profile mapping"
```

---

### Task 3: Feature extractors

Five feature groups (regex, gazetteer, structural, intent-language, TF-IDF), each a standalone sklearn-compatible transformer taking raw query text and returning a numeric vector — none of them know about the `KEYWORD/HYBRID/INTENT` taxonomy, so they're reusable if a future model needs a different label set.

**Files:**
- Create: `packages/common/src/common/instant_classifier/features.py`
- Test: `packages/common/tests/test_instant_classifier_features.py`

**Interfaces:**
- Consumes: `common.legal_lexicon.SECTION_PATTERN/CITATION_PATTERN/PARTY_PATTERN/is_known_court/is_known_journal/expand_synonyms` (all pre-existing).
- Produces: `RegexFeaturizer`, `GazetteerFeaturizer`, `StructuralFeaturizer`, `IntentLanguageFeaturizer` (each `fit(X, y=None) -> self`, `transform(X: list[str]) -> np.ndarray` of shape `(len(X), n_features)`), `build_feature_union() -> sklearn.pipeline.FeatureUnion`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/common/tests/test_instant_classifier_features.py
import numpy as np

from common.instant_classifier.features import (
    GazetteerFeaturizer, IntentLanguageFeaturizer, RegexFeaturizer, StructuralFeaturizer, build_feature_union,
)


def test_regex_featurizer_detects_section_citation_party():
    rows = RegexFeaturizer().transform(["Section 52", "2024 ITR 123", "CIT vs Infosys", "plain text query"])
    assert rows[0].tolist() == [1.0, 0.0, 0.0]
    assert rows[1].tolist() == [0.0, 1.0, 0.0]
    assert rows[2][2] == 1.0
    assert rows[3].tolist() == [0.0, 0.0, 0.0]


def test_gazetteer_featurizer_detects_court_journal_legal_term():
    rows = GazetteerFeaturizer().transform(["ITAT order", "TAXMAN reference", "CBDT circular", "random words here"])
    assert rows[0][0] == 1.0  # ITAT is a known court
    assert rows[1][1] == 1.0  # TAXMAN is a known journal
    assert rows[2][2] == 1.0  # CBDT has a synonym expansion
    assert rows[3].tolist() == [0.0, 0.0, 0.0]


def test_structural_featurizer_counts_tokens_and_trailing_question_mark():
    rows = StructuralFeaturizer().transform(["Section 52", "How do I file tax returns?", '"quoted phrase" query'])
    assert rows[0][0] == 2.0 and rows[0][1] == 0.0 and rows[0][2] == 0.0
    assert rows[1][1] == 1.0
    assert rows[2][2] == 1.0


def test_intent_language_featurizer_detects_question_words_and_first_person():
    rows = IntentLanguageFeaturizer().transform(["How do I evade tax", "Section 52", "We should appeal if possible"])
    assert rows[0][0] == 1.0 and rows[0][1] == 1.0  # "how", "i"
    assert rows[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert rows[2][2] == 1.0 and rows[2][3] == 1.0  # "should", "if"


def test_build_feature_union_transforms_a_batch_of_queries_to_a_2d_array():
    union = build_feature_union()
    matrix = union.fit_transform(["Section 52", "How do I evade tax", "Where is Section 52 applicable"])
    assert matrix.shape[0] == 3
    assert isinstance(matrix, np.ndarray) or hasattr(matrix, "toarray")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_instant_classifier_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.instant_classifier.features'`

- [ ] **Step 3: Implement features.py**

```python
# packages/common/src/common/instant_classifier/features.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from common.legal_lexicon import (
    CITATION_PATTERN, PARTY_PATTERN, SECTION_PATTERN, expand_synonyms, is_known_court, is_known_journal,
)

_QUESTION_WORDS = {"what", "where", "when", "why", "who", "how", "which", "whose"}
_FIRST_PERSON = {"i", "me", "my", "we", "our", "us"}
_MODALS = {"can", "could", "should", "would", "may", "might", "must", "shall", "will"}
_CONDITIONALS = {"if", "unless", "provided", "assuming"}


class RegexFeaturizer(BaseEstimator, TransformerMixin):
    """Binary hits for the three structural regexes already used elsewhere in Instant
    mode (query_tokenizer/es_client) to detect citations, section refs, and party names."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            rows.append([
                1.0 if SECTION_PATTERN.search(text) else 0.0,
                1.0 if CITATION_PATTERN.search(text) else 0.0,
                1.0 if PARTY_PATTERN.search(text) else 0.0,
            ])
        return np.array(rows)


def _has_legal_term(text: str) -> bool:
    return any(expand_synonyms(token) != [token] for token in text.split())


class GazetteerFeaturizer(BaseEstimator, TransformerMixin):
    """Binary per-category hits (court/journal/legal-term mention) - category
    granularity only, no raw counts, per the PRD's v1 feature design."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            tokens = text.split()
            rows.append([
                1.0 if any(is_known_court(t) for t in tokens) else 0.0,
                1.0 if any(is_known_journal(t) for t in tokens) else 0.0,
                1.0 if _has_legal_term(text) else 0.0,
            ])
        return np.array(rows)


class StructuralFeaturizer(BaseEstimator, TransformerMixin):
    """Token count, trailing '?', quote presence. Runs on text that's already been
    through query_tokenizer's normalization upstream, so a stray leftover symbol and
    a genuine question mark were already disambiguated before this ever sees the text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            stripped = text.strip()
            rows.append([
                float(len(stripped.split())),
                1.0 if stripped.endswith("?") else 0.0,
                1.0 if '"' in stripped else 0.0,
            ])
        return np.array(rows)


class IntentLanguageFeaturizer(BaseEstimator, TransformerMixin):
    """Question words, first-person pronouns, modal verbs, conditional markers - general
    English function words, not legal-domain, so kept separate from legal_lexicon.json."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            tokens = {t.strip("?.,!\"'").lower() for t in text.split()}
            rows.append([
                1.0 if tokens & _QUESTION_WORDS else 0.0,
                1.0 if tokens & _FIRST_PERSON else 0.0,
                1.0 if tokens & _MODALS else 0.0,
                1.0 if tokens & _CONDITIONALS else 0.0,
            ])
        return np.array(rows)


def build_feature_union() -> FeatureUnion:
    return FeatureUnion([
        ("regex", RegexFeaturizer()),
        ("gazetteer", GazetteerFeaturizer()),
        ("structural", StructuralFeaturizer()),
        ("intent_language", IntentLanguageFeaturizer()),
        ("tfidf_word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
        ("tfidf_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_instant_classifier_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/instant_classifier/features.py packages/common/tests/test_instant_classifier_features.py
git commit -m "feat(instant-classifier): add feature extractors for regex/gazetteer/structural/intent-language/tfidf"
```

---

### Task 4: Pipeline assembly + artifact save/load

**Files:**
- Create: `packages/common/src/common/instant_classifier/pipeline.py`
- Test: `packages/common/tests/test_instant_classifier_pipeline.py`

**Interfaces:**
- Consumes: `build_feature_union()` from Task 3.
- Produces: `build_pipeline() -> sklearn.pipeline.Pipeline`, `save_artifact(pipeline, meta: dict, model_path=None, meta_path=None) -> None`, `load_artifact(model_path=None, meta_path=None) -> tuple[Pipeline, dict]` (raises `FileNotFoundError` if either file is missing).

- [ ] **Step 1: Write the failing tests**

```python
# packages/common/tests/test_instant_classifier_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_instant_classifier_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.instant_classifier.pipeline'`

- [ ] **Step 3: Implement pipeline.py**

```python
# packages/common/src/common/instant_classifier/pipeline.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_instant_classifier_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/instant_classifier/pipeline.py packages/common/tests/test_instant_classifier_pipeline.py
git commit -m "feat(instant-classifier): add pipeline assembly and artifact save/load"
```

---

### Task 5: Seed training data + training script + committed artifact

Hand-authored seed dataset (60 train / 24 eval examples across the three classes) to unblock implementation now. **This is a starting point, not the final dataset** — replace/extend `train.jsonl` with real historical-query labels once available, and rerun the script.

**Files:**
- Create: `packages/common/data/instant_classifier/train.jsonl`
- Create: `packages/common/data/instant_classifier/eval_frozen.jsonl`
- Create: `packages/common/scripts/train_instant_classifier.py`
- Create (generated by running the script in Step 4): `packages/common/src/common/data/instant_classifier_model.joblib`, `packages/common/src/common/data/instant_classifier_model_meta.json`
- Test: `packages/common/tests/test_instant_classifier_eval.py`

**Interfaces:**
- Consumes: `build_pipeline`/`save_artifact` (Task 4).
- Produces: the committed model artifact + meta file that Task 6's `classify()` loads.

- [ ] **Step 1: Create the seed training data**

Create `packages/common/data/instant_classifier/train.jsonl` (one JSON object per line, `query_text`/`label`/`source`/`date_added`):

```jsonl
{"query_text": "Section 52", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Section 80C", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Rule 6DD", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Article 21", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "u/s 54F", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "2024 ITR 123", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "133 taxmann.com 196", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "CIT vs Infosys Ltd", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "ITAT Mumbai order", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "CBDT circular 2023", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Section 194O TDS", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "AAR ruling GST", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Rule 37CA", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Section 68 unexplained cash credit", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "CESTAT Chennai", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "GSTL 2022 45", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Section 143(3) assessment", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Delhi High Court Section 271", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "80-IA deduction", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Krishana Goel vs Principal Commissioner", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Where is Section 52 applicable", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "When does Section 80C deduction apply", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How is Rule 6DD used in practice", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What happens under Section 68 for unexplained credits", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can Section 194O apply to online sellers", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Is Article 21 relevant for tax search cases", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How does ITAT interpret Section 143(3)", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What if CBDT circular 2023 conflicts with Section 54F", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Does Section 271 penalty apply after Delhi High Court ruling", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How should I claim 80-IA deduction", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What is the procedure under Rule 37CA", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can AAR ruling on GST be challenged", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How to interpret CESTAT Chennai order on Section 65", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What does Section 143(1) intimation mean for me", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Is Section 12AA registration mandatory for trusts", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do courts apply Section 68 for cash credits", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What's the scope of Section 194O beyond ecommerce", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can I appeal ITAT Mumbai's Section 271 order", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How does Section 80C interact with 80CCD", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What triggers reassessment under Section 148", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do I evade tax", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What can I do to reduce my tax liability", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How should a company claim depreciation on goodwill", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can a startup claim tax exemption", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What happens if I don't file my tax return on time", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How is capital gains tax calculated for property sale", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What are the penalties for late GST filing", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can I claim HRA and home loan interest together", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do I respond to an income tax notice", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What is the process for GST registration", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Should I file taxes as an individual or HUF", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How does TDS work for freelancers", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What deductions can salaried employees claim", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do I calculate advance tax", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can I carry forward business losses", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What is the tax treatment of cryptocurrency gains", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do NRIs pay tax on Indian income", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What happens during a tax audit", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How can I save tax through investments", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Is gift from a relative taxable", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
```

Create `packages/common/data/instant_classifier/eval_frozen.jsonl` (never trained on — used only to gate every retrain):

```jsonl
{"query_text": "Section 54EC", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Rule 8D", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Article 14", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "u/s 269SS", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "2023 GSTL 88", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "NCLT Mumbai order", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Section 40A(3) disallowance", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "TAXMAN 2021 456", "label": "KEYWORD", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Where does Section 54EC exemption apply", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How does Rule 8D disallowance work in assessments", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can Article 14 be invoked in tax discrimination cases", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What does u/s 269SS restrict", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How should NCLT Mumbai's order be applied", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Is Section 40A(3) applicable to cash payments", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What if 2023 GSTL 88 conflicts with a later ruling", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can TAXMAN 2021 456 be relied upon for appeals", "label": "HYBRID", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do I set up a tax-saving investment plan", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What is the best way to file GST returns", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can a partnership firm claim depreciation", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How do I dispute an income tax demand", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What is presumptive taxation for small businesses", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "How is rental income taxed", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "Can I claim tax benefit on education loan", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
{"query_text": "What is the due date for filing tax audit report", "label": "INTENT", "source": "manual", "date_added": "2026-08-20"}
```

- [ ] **Step 2: Write the training script**

```python
# packages/common/scripts/train_instant_classifier.py
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
```

- [ ] **Step 3: Run the script to produce the committed artifact**

Run: `cd packages/common && uv run python scripts/train_instant_classifier.py`
Expected: prints a JSON meta blob with `overall_eval_accuracy` — note the printed value, it's used as the test floor in the next step (expect roughly 0.85-1.0 given the gazetteer feature recognizing NCLT/TAXMAN in the eval set; use `overall_eval_accuracy - 0.1` rounded down to one decimal as a safety margin, not a hardcoded guess).

- [ ] **Step 4: Write the eval-floor regression test using the actual printed accuracy**

```python
# packages/common/tests/test_instant_classifier_eval.py
import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from common.instant_classifier.pipeline import load_artifact

_EVAL_PATH = Path(__file__).parent.parent / "data" / "instant_classifier" / "eval_frozen.jsonl"

# Floor set from the accuracy actually measured when this artifact was trained (see
# Step 3's printed overall_eval_accuracy), minus a safety margin - catches a bad
# retrain-and-commit regressing accuracy, not a specific target number.
_ACCURACY_FLOOR = 0.75


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_instant_classifier_eval.py -v`
Expected: PASS. If it fails, the seed data/features need adjustment before proceeding — do not lower the floor to make it pass.

- [ ] **Step 6: Commit**

```bash
git add packages/common/data/instant_classifier/ packages/common/scripts/train_instant_classifier.py packages/common/src/common/data/instant_classifier_model.joblib packages/common/src/common/data/instant_classifier_model_meta.json packages/common/tests/test_instant_classifier_eval.py
git commit -m "feat(instant-classifier): add seed training data, training script, and committed model artifact"
```

---

### Task 6: Public `classify()` API with confidence-gated routing

**Files:**
- Modify: `packages/common/src/common/instant_classifier/__init__.py`
- Test: `packages/common/tests/test_instant_classifier_api.py`

**Interfaces:**
- Consumes: `load_artifact` (Task 4), `ClassifierResult`/`resolve_routing`/`routing_plan`/`boost_profile_key` (Task 2).
- Produces: `classify(query: str) -> ClassifierResult`, `confidence_threshold() -> float`, `effective_label(query: str) -> str` (applies the confidence gate, returns `KEYWORD|HYBRID|INTENT|FALLBACK`) — this is what Task 7/8/9 call.

- [ ] **Step 1: Write the failing tests**

```python
# packages/common/tests/test_instant_classifier_api.py
from common.instant_classifier import classify, confidence_threshold, effective_label
from common.instant_classifier.labels import FALLBACK, HYBRID, INTENT, KEYWORD


def test_classify_returns_confident_keyword_for_bare_section_ref():
    result = classify("Section 52")
    assert result.label == KEYWORD
    assert result.confidence > 0.5


def test_classify_returns_confident_intent_for_pure_question():
    result = classify("How do I evade tax")
    assert result.label == INTENT
    assert result.confidence > 0.5


def test_classify_returns_hybrid_for_anchor_plus_question():
    result = classify("Where is Section 52 applicable")
    assert result.label == HYBRID


def test_confidence_threshold_is_a_float_between_zero_and_one():
    threshold = confidence_threshold()
    assert 0.0 <= threshold <= 1.0


def test_effective_label_matches_classify_when_confident():
    assert effective_label("Section 52") == KEYWORD


def test_effective_label_falls_back_below_threshold(monkeypatch):
    import common.instant_classifier as module

    monkeypatch.setattr(module, "classify", lambda query: module.ClassifierResult(label=KEYWORD, confidence=0.0))
    assert module.effective_label("anything") == FALLBACK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_instant_classifier_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify'`

- [ ] **Step 3: Implement `__init__.py`**

```python
# packages/common/src/common/instant_classifier/__init__.py
from functools import lru_cache

from common.instant_classifier.labels import ClassifierResult, resolve_routing
from common.instant_classifier.pipeline import load_artifact

__all__ = ["classify", "confidence_threshold", "effective_label", "ClassifierResult"]


@lru_cache
def _load():
    return load_artifact()


def classify(query: str) -> ClassifierResult:
    pipeline, _ = _load()
    proba = pipeline.predict_proba([query])[0]
    label = pipeline.classes_[proba.argmax()]
    confidence = float(proba.max())
    return ClassifierResult(label=label, confidence=confidence)


def confidence_threshold() -> float:
    _, meta = _load()
    return meta["confidence_threshold"]


def effective_label(query: str) -> str:
    return resolve_routing(classify(query), confidence_threshold())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_instant_classifier_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/instant_classifier/__init__.py packages/common/tests/test_instant_classifier_api.py
git commit -m "feat(instant-classifier): add public classify()/effective_label() API"
```

---

### Task 7: Retire `classify_query_shape()`; wire ES boost-profile selection to the new classifier

**Files:**
- Modify: `packages/common/src/common/query_tokenizer.py`
- Modify: `packages/common/src/common/es_client.py`
- Test: `packages/common/tests/test_query_tokenizer.py`
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Consumes: `common.instant_classifier.effective_label` (Task 6).
- Produces: `_BOOST_PROFILES` now keyed by `"KEYWORD"/"HYBRID"/"INTENT"` (was `"citation"/"provision"/"plain"`); `build_query_preview(query)["shape"]` now holds one of those four new labels (including `"FALLBACK"`) instead of the old three.

- [ ] **Step 1: Remove `classify_query_shape()` and its tests**

Delete the `classify_query_shape` function (lines 17-26) from `packages/common/src/common/query_tokenizer.py`. It's no longer imported by anything after this task — `CITATION_PATTERN`/`PARTY_PATTERN`/`SECTION_PATTERN` imports at the top of the file stay (still used by `_classify_merged_chunk` indirectly via other functions)... actually check: only `classify_query_shape` used `CITATION_PATTERN`/`PARTY_PATTERN`/`SECTION_PATTERN` directly in this file. Since they're unused elsewhere in `query_tokenizer.py` after removal, drop them from the import on line 3-5 too, keeping only `expand_synonyms, is_known_journal, is_stopword`.

Delete these five tests from `packages/common/tests/test_query_tokenizer.py`: `test_classify_query_shape_detects_citation`, `test_classify_query_shape_detects_party_citation`, `test_classify_query_shape_detects_provision`, `test_classify_query_shape_defaults_to_plain`, `test_classify_query_shape_prefers_citation_when_both_patterns_present`. Remove `classify_query_shape` from the `from common.query_tokenizer import (...)` line at the top of the test file.

- [ ] **Step 2: Run test to verify the rest of query_tokenizer's suite still passes**

Run: `uv run pytest packages/common/tests/test_query_tokenizer.py -v`
Expected: PASS (only the five deleted tests are gone; nothing else referenced `classify_query_shape`)

- [ ] **Step 3: Write the failing es_client test for the new taxonomy**

Edit `packages/common/tests/test_es_client.py` — replace the `test_build_query_preview_matches_what_raw_search_actually_sends` test's shape assertion. Since the exact label the trained model assigns to `"Section 6 of Income Tax Act"` depends on the real committed artifact from Task 5, determine it empirically first:

Run: `uv run python -c "from common.instant_classifier import effective_label; print(effective_label('Section 6 of Income Tax Act'))"`

Use whatever it prints (e.g. `KEYWORD`) as the literal expected value below — do not guess:

```python
def test_build_query_preview_matches_what_raw_search_actually_sends():
    preview = build_query_preview("Section 6 of Income Tax Act")

    assert preview["query"] == "Section 6 of Income Tax Act"
    assert preview["shape"] == "<paste the printed label here>"
    assert any(c["type"] == "section" and c["text"] == "Section 6" for c in preview["chunks"])
    assert "bool" in preview["es_query"]
```

Also add a new test confirming the boost-profile keys accept the new taxonomy:

```python
def test_build_field_query_accepts_new_taxonomy_labels():
    for label in ("KEYWORD", "HYBRID", "INTENT", "FALLBACK"):
        query = _build_field_query("test query", label)
        assert "bool" in query
```

(`_build_field_query` is already imported/used elsewhere in this test file - confirm the import line at the top includes it, add it if not.)

- [ ] **Step 4: Run to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k "build_query_preview or build_field_query_accepts" -v`
Expected: FAIL — `preview["shape"]` is still the old value (`"provision"`), and `_BOOST_PROFILES` doesn't have a `"FALLBACK"` key yet.

- [ ] **Step 5: Update es_client.py**

Rename the `_BOOST_PROFILES` keys (same values, new keys) around line 146:

```python
_BOOST_PROFILES = {
    "KEYWORD": {"heading": 5.0, "subheading": 3.0, "fullcontent": 1.0,
                "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.5},
    "HYBRID": {"heading": 2.0, "subheading": 3.0, "fullcontent": 1.0,
               "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 2.5},
    "INTENT": {"heading": 2.0, "subheading": 2.0, "fullcontent": 1.5,
               "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 1.0},
    "FALLBACK": {"heading": 2.0, "subheading": 3.0, "fullcontent": 1.0,
                 "facts_text": 1.0, "held_text": 1.0, "headnotes_text": 2.5},
}
```

(`FALLBACK` reuses the `HYBRID` profile — same balanced-weighting rationale as `labels.boost_profile_key`.)

Update `build_query_preview` (around line 265-280) to call the classifier instead of `classify_query_shape`:

```python
from common.instant_classifier import effective_label
# remove: from common.query_tokenizer import chunk_query, classify_query_shape, expand_query_synonyms
from common.query_tokenizer import chunk_query, expand_query_synonyms


def build_query_preview(query: str) -> dict:
    shape = effective_label(query)
    expanded_query = expand_query_synonyms(query)
    chunks = chunk_query(query)
    return {
        "query": query,
        "shape": shape,
        "expanded_query": expanded_query if expanded_query != query else None,
        "chunks": chunks,
        "es_query": _build_field_query(expanded_query, shape, chunks=chunks),
    }
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -v`
Expected: PASS (full file — confirm no other test in this file still references the old `"citation"/"provision"/"plain"` keys; update any that do the same way)

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/query_tokenizer.py packages/common/src/common/es_client.py packages/common/tests/test_query_tokenizer.py packages/common/tests/test_es_client.py
git commit -m "refactor(es-client): retire classify_query_shape, wire ES boost profiles to instant_classifier"
```

---

### Task 8: Wire RRF fusion weights to the new taxonomy

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/instant/rerank.py`
- Test: `packages/retrieval-api/tests/test_instant_rerank.py`

**Interfaces:**
- Modifies existing `rerank_instant_results(..., shape: str, ...)` parameter — renamed to `label: str`, values now `KEYWORD/HYBRID/INTENT/FALLBACK` instead of `citation/provision/plain`.

- [ ] **Step 1: Update the failing test cases first**

In `packages/retrieval-api/tests/test_instant_rerank.py`, every call passing `shape="plain"` (there are several — search the file for `shape="plain"` and `shape="citation"`/`shape="provision"` if present) becomes `label="INTENT"` (closest analog to old `"plain"` — Milvus-favored weighting) unless the test's own comment specifies a different shape's weighting; update the comment referencing "shape=\"plain\" weights milvus_dense..." to "label=\"INTENT\" weights milvus_dense...".

Run: `uv run pytest packages/retrieval-api/tests/test_instant_rerank.py -v`
Expected: FAIL — `rerank_instant_results()`/`_SHAPE_RRF_WEIGHTS` still take `shape`, not `label`, so this doesn't error yet but sets up Step 3's real failure once the param is renamed. (If it passes unchanged because `shape`/`label` are both just keyword args with the same runtime meaning, that's fine — proceed to Step 2, which is the step that actually changes behavior.)

- [ ] **Step 2: Rename `_SHAPE_RRF_WEIGHTS` keys and the `shape` parameter**

In `packages/retrieval-api/src/retrieval_api/instant/rerank.py`:

```python
_LABEL_RRF_WEIGHTS: dict[str, dict[str, float]] = {
    "KEYWORD": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "HYBRID": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
    "INTENT": {"es": 1.0, "milvus_dense": 1.5, "milvus_sparse": 0.5},
    "FALLBACK": {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5},
}
```

Rename every `shape` parameter/reference in `rerank_instant_results` to `label`, and `_SHAPE_RRF_WEIGHTS` to `_LABEL_RRF_WEIGHTS`:

```python
async def rerank_instant_results(
    gateway: GatewayClient,
    es_client,
    query: str,
    label: str,
    es_result: list[dict],
    milvus_dense: dict[str, list[dict]],
    milvus_sparse: dict[str, list[dict]],
    rrf: bool = True,
    rerank: bool = True,
    on_step: OnStep | None = None,
) -> list[dict]:
    ...
    if rrf:
        weights = _LABEL_RRF_WEIGHTS.get(label, {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0})
        ...
```

(Only the parameter name and the dict name change — the body logic is untouched.)

- [ ] **Step 3: Run to verify tests pass**

Run: `uv run pytest packages/retrieval-api/tests/test_instant_rerank.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/instant/rerank.py packages/retrieval-api/tests/test_instant_rerank.py
git commit -m "refactor(instant-rerank): rename shape param/weights to the new KEYWORD/HYBRID/INTENT taxonomy"
```

---

### Task 9: Automatic backend routing in `run_instant()` behind an `auto_route` toggle

Adds a new `auto_route: bool = False` parameter (default off, matching every other Instant-mode toggle's safe default) that, when true, uses `effective_label()` to decide whether to skip ES, skip Milvus, or query both with forced RRF fusion — replacing the manual `rrf` boolean's effect only when `auto_route` is on. When `auto_route` is false, behavior is byte-identical to today.

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/instant/search.py`
- Test: `packages/retrieval-api/tests/test_instant_search.py`

**Interfaces:**
- Consumes: `common.instant_classifier.effective_label`, `common.instant_classifier.labels.routing_plan` (Task 6/2).
- Produces: `run_instant(..., auto_route: bool = False)` — new parameter; existing `rrf`/`rerank` parameters unchanged in meaning when `auto_route=False`.

- [ ] **Step 1: Write the failing tests**

```python
# add to packages/retrieval-api/tests/test_instant_search.py
@pytest.mark.asyncio
async def test_run_instant_auto_route_keyword_skips_milvus(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2}]

    milvus_called = False

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        nonlocal milvus_called
        milvus_called = True
        return {}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label", lambda query: "KEYWORD")

    gateway = AsyncMock()
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="Section 52", auto_route=True,
    )

    assert result["es"] == [{"doc_id": "d1", "score": 4.2}]
    assert result["milvus"] is None
    assert not milvus_called
    gateway.embed.assert_not_called()


@pytest.mark.asyncio
async def test_run_instant_auto_route_intent_skips_es(monkeypatch):
    import retrieval_api.instant.search as search_module

    es_called = False

    async def fake_raw_search(client, query, limit=20):
        nonlocal es_called
        es_called = True
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label", lambda query: "INTENT")

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="how do I evade tax", auto_route=True,
    )

    assert result["es"] is None
    assert not es_called
    assert result["milvus"] is not None


@pytest.mark.asyncio
async def test_run_instant_auto_route_hybrid_forces_rrf_fusion(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label", lambda query: "HYBRID")

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="where is section 52 applicable",
        auto_route=True, rrf=False,  # auto_route overrides the manual rrf=False when it's on
    )

    assert "reranked" in result
    assert any(row["doc_id"] == "d2" for row in result["reranked"])  # fused in from Milvus


@pytest.mark.asyncio
async def test_run_instant_auto_route_false_preserves_today_behavior(monkeypatch):
    """auto_route defaults to False - identical to the existing always-both, manual-rrf
    behavior every other test in this file already exercises."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["es"] is not None
    assert result["milvus"] is not None
    assert "reranked" not in result
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_instant_search.py -k auto_route -v`
Expected: FAIL with `TypeError: run_instant() got an unexpected keyword argument 'auto_route'`

- [ ] **Step 3: Implement routing in search.py**

Edit `packages/retrieval-api/src/retrieval_api/instant/search.py`:

```python
from common.instant_classifier import effective_label
from common.instant_classifier.labels import routing_plan
```

Replace the `classify_query_shape` import (it no longer exists) and every call site that used it. `_run_es`'s internal `shape = classify_query_shape(query)` (used only for the langfuse trace input) becomes:

```python
async def _run_es(es_client, query: str, on_step: OnStep | None) -> tuple[list[dict] | None, str | None]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="retriever", name="search-es",
        input={"query": query, "limit": _ES_LIMIT},
    ) as span:
        ...  # unchanged body, just drop query_shape from the trace input dict
```

Update `run_instant`:

```python
async def run_instant(
    gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None,
    rrf: bool = False, rerank: bool = False, auto_route: bool = False,
) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="instant-search", input={"query": query}):
        if on_step is not None:
            await on_step("query_analysis", build_query_preview(query))

        label = effective_label(query) if auto_route else None
        plan = routing_plan(label) if auto_route else {"es": True, "milvus": True, "fuse": False}

        es_task = _run_es(es_client, query, on_step) if plan["es"] else None
        milvus_task = _run_milvus(gateway, milvus_client, query, on_step) if plan["milvus"] else None

        if es_task is not None and milvus_task is not None:
            (es_result, es_error), (milvus_dense, milvus_sparse, milvus_error) = await asyncio.gather(es_task, milvus_task)
        elif es_task is not None:
            es_result, es_error = await es_task
            milvus_dense, milvus_sparse, milvus_error = None, None, None
        else:
            es_result, es_error = None, None
            milvus_dense, milvus_sparse, milvus_error = await milvus_task

        result = {
            "es": es_result,
            "es_error": es_error,
            "milvus": milvus_dense,
            "milvus_sparse": milvus_sparse,
            "milvus_error": milvus_error,
        }

        effective_rrf = plan["fuse"] if auto_route else rrf
        if not effective_rrf and not rerank:
            return result

        reranked_error = es_error or (milvus_error if effective_rrf else None)
        reranked = []
        if reranked_error is None:
            with langfuse.start_as_current_observation(
                as_type="chain", name="rerank", input={"query": query, "rrf": effective_rrf, "rerank": rerank},
            ) as rerank_span:
                try:
                    reranked = await rerank_instant_results(
                        gateway, es_client, query, label or "FALLBACK",
                        es_result or [], milvus_dense or {}, milvus_sparse or {},
                        rrf=effective_rrf, rerank=rerank, on_step=on_step,
                    )
                    rerank_span.update(output={"num_reranked": len(reranked)})
                    if on_step is not None:
                        await on_step("instant_reranked", {"hits": reranked})
                except Exception as exc:  # noqa: BLE001 - branch isolation is the point
                    reranked_error = str(exc)
                    rerank_span.update(level="ERROR", status_message=reranked_error)
        result["reranked"] = reranked
        result["reranked_error"] = reranked_error
    return result
```

- [ ] **Step 4: Run to verify tests pass**

Run: `uv run pytest packages/retrieval-api/tests/test_instant_search.py -v`
Expected: PASS (all tests, including the pre-existing always-both ones — `auto_route` defaults to `False`)

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/instant/search.py packages/retrieval-api/tests/test_instant_search.py
git commit -m "feat(instant-search): add auto_route param for classifier-driven ES/Milvus backend routing"
```

---

### Task 10: Wire the `auto_route` toggle through `/ws/search`

**Files:**
- Modify: `packages/common/src/common/config.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py`
- Test: `packages/retrieval-api/tests/test_ws_integration.py` (or wherever the existing `rrf`/`rerank` message-field tests live — locate via `grep -rn "message.get(\"rrf\"" packages/retrieval-api/tests`)

**Interfaces:**
- Produces: `Settings.instant_mode_auto_route_enabled: bool = True` (server-side kill switch, same pattern as `instant_mode_rerank_enabled`); `/ws/search` now reads `message.get("auto_route", False)`.

- [ ] **Step 1: Locate the existing rrf/rerank message-field test to extend**

Run: `grep -rn "message.get(\"rrf\"\|\"rrf\": " packages/retrieval-api/tests/*.py`

Read the matching test file's existing test for how `rrf`/`rerank` message fields are asserted, and add an analogous test asserting `auto_route` defaults to `False` when absent from the message and is forwarded to `run_instant` when present and `instant_mode_auto_route_enabled` is `True`. Follow that file's existing mocking pattern exactly (it already mocks `run_instant` or the websocket at some layer — mirror the existing `rrf`/`rerank` assertions with `auto_route` added).

- [ ] **Step 2: Run to verify the new/extended test fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -k auto_route -v`
Expected: FAIL — `ws.py` doesn't read or forward `auto_route` yet.

- [ ] **Step 3: Add the settings kill switch**

Edit `packages/common/src/common/config.py`, after `instant_mode_rerank_enabled`:

```python
    # Kill switch for Instant mode's classifier-driven automatic backend routing
    # (the `auto_route` field on the /ws/search message) - same pattern as
    # instant_mode_rerank_enabled above. False here forces every request onto
    # today's always-both-backends behavior regardless of what the client asks for.
    instant_mode_auto_route_enabled: bool = True
```

- [ ] **Step 4: Wire it through ws.py**

Edit `packages/retrieval-api/src/retrieval_api/ws.py` around line 111-112:

```python
    rrf = message.get("rrf", False)
    rerank = message.get("rerank", False) and settings.instant_mode_rerank_enabled
    auto_route = message.get("auto_route", False) and settings.instant_mode_auto_route_enabled
```

Update the cache-key logic (around line 148-156) to include `auto_route`, since it changes result content the same way `rrf`/`rerank` do:

```python
    instant_cache_key = (
        f"instant_auto_route_{auto_route}_rrf_{rrf}_rerank_{rerank}"
    )
```

(Replaces the four-branch if/elif chain with one deterministic key covering all three toggles — simpler than adding an 8th branch for the new dimension.)

Update the `run_instant(...)` call site (around line 214-216) to pass `auto_route=auto_route`.

- [ ] **Step 5: Run to verify tests pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/config.py packages/retrieval-api/src/retrieval_api/ws.py packages/retrieval-api/tests/test_ws_integration.py
git commit -m "feat(ws): add auto_route toggle and server-side kill switch for classifier routing"
```

---

### Task 11: Fail-loud eager model loading at startup

Per the spec: a missing/corrupt classifier artifact must crash startup, not silently degrade every request to fallback routing.

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/main.py`
- Test: `packages/retrieval-api/tests/test_main_startup.py` (new file)

**Interfaces:**
- No new public interface — this task calls the existing `common.instant_classifier.classify` inside the FastAPI `lifespan`.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_main_startup.py
import pytest


def test_lifespan_raises_if_classifier_artifact_missing(monkeypatch):
    import common.instant_classifier as classifier_module
    from retrieval_api.main import app

    def _raise(*args, **kwargs):
        raise FileNotFoundError("artifact missing")

    monkeypatch.setattr(classifier_module, "_load", _raise)

    with pytest.raises(FileNotFoundError):
        with pytest.warns(None):
            import asyncio

            async def _run():
                async with app.router.lifespan_context(app):
                    pass

            asyncio.run(_run())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_main_startup.py -v`
Expected: FAIL — nothing in `lifespan` calls the classifier yet, so no exception is raised.

- [ ] **Step 3: Add eager loading to lifespan**

Edit `packages/retrieval-api/src/retrieval_api/main.py`:

```python
from common.instant_classifier import classify

...

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loud: a missing/corrupt Instant-mode classifier artifact must crash
    # startup, not silently degrade every request to fallback routing forever -
    # unlike the Mongo/Milvus/persona degrade-don't-crash patterns below, a broken
    # classifier is a deployment bug, not a transient dependency outage.
    classify("startup warmup query")
    try:
        settings = get_auth_settings()
        client = get_mongo_client(settings)
        await ensure_refresh_token_indexes(get_refresh_tokens_collection(client, settings))
    except Exception:
        logger.exception("Failed to ensure refresh-token indexes at startup - continuing without them")
    yield
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_main_startup.py -v`
Expected: PASS

- [ ] **Step 5: Run the full retrieval-api suite to confirm nothing else broke**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/main.py packages/retrieval-api/tests/test_main_startup.py
git commit -m "feat(main): eagerly load instant classifier artifact at startup, fail loud on missing artifact"
```

---

### Task 12: Frontend `AutoRoute` toggle

Mirrors the existing `RerankToggle` component/wiring exactly (same component, same `useState`/payload pattern as `rrf`/`rerank`).

**Files:**
- Modify: `packages/web/src/api/useSearch.ts`
- Modify: `packages/web/src/App.tsx`
- Test: `packages/web/src/api/useSearch.test.ts` (or wherever the existing rrf/rerank payload test lives — locate via `grep -rln "rrf" packages/web/src/**/*.test.ts*`)

**Interfaces:**
- Modifies `useSearch`'s returned `search()` signature: adds `autoRoute?: boolean` after `rrf`, before `conversationId`.

- [ ] **Step 1: Locate and extend the existing rrf/rerank payload test**

Run: `grep -rln "rrf" packages/web/src --include="*.test.ts*"`

Read that test's existing assertion on the WS payload (it currently checks `{ query, mode, trace, rerank, rrf }` gets sent). Add a new test case asserting the payload includes `autoRoute: false` by default and `autoRoute: true` when passed, following the exact same pattern as the existing `rrf` assertion in that file.

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/web && npm test -- useSearch` (or the project's actual test command — check `package.json`'s `scripts.test`)
Expected: FAIL — `search()` doesn't accept/forward `autoRoute` yet.

- [ ] **Step 3: Update useSearch.ts**

Edit `packages/web/src/api/useSearch.ts`:

```typescript
export function useSearch(
  wsUrl: string,
  accessToken?: string | null,
  onSessionExpired?: () => void,
): SearchState & {
  search: (
    query: string, trace: boolean, mode?: SearchMode, rerank?: boolean, rrf?: boolean, autoRoute?: boolean,
    conversationId?: string,
  ) => void
} {
  const [state, setState] = useState<SearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (
      query: string, trace: boolean, mode: SearchMode = 'both', rerank: boolean = false, rrf: boolean = false,
      autoRoute: boolean = false, conversationId?: string,
    ) => {
      socketRef.current?.close()
      setState({ loading: true, instant: null, aiMode: null, traceSteps: [], wsError: null })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, wsError: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        const payload: Record<string, unknown> = { query, mode, trace, rerank, rrf, auto_route: autoRoute }
        if (accessToken) payload.access_token = accessToken
        if (conversationId) payload.conversation_id = conversationId
        socket.send(JSON.stringify(payload))
      })
      // ... rest of the function unchanged
```

- [ ] **Step 4: Update App.tsx**

Add the state and toggle, mirroring `rrf`/`rerank` exactly (near line 49-50):

```typescript
  const [rerank, setRerank] = useState(false)
  const [rrf, setRrf] = useState(false)
  const [autoRoute, setAutoRoute] = useState(false)
```

Update the search call (near line 139):

```typescript
      classicSearch.search(question, true, 'both', rerank, rrf, autoRoute, auth.token ? conversationId : undefined)
```

Add the toggle in the UI (near line 277-279):

```tsx
              <RerankToggle label="RRF" checked={rrf} onToggle={setRrf} />
              <RerankToggle label="Rerank" checked={rerank} onToggle={setRerank} />
              <RerankToggle label="Auto-Route" checked={autoRoute} onToggle={setAutoRoute} />
              <RerankToggle label="Reasoning" checked={showReasoning} onToggle={setShowReasoning} />
```

- [ ] **Step 5: Run to verify tests pass**

Run: `cd packages/web && npm test -- useSearch`
Expected: PASS

- [ ] **Step 6: Manually verify in a browser**

Start the dev server (`npm run dev` in `packages/web`, backend running via `docker compose up -d --build` or local `uv run`), open the Instant/classic search UI, confirm the new "Auto-Route" checkbox appears next to RRF/Rerank/Reasoning and toggling it changes the WS payload (check via browser devtools network tab) without breaking the existing RRF/Rerank toggles.

- [ ] **Step 7: Commit**

```bash
git add packages/web/src/api/useSearch.ts packages/web/src/App.tsx packages/web/src/api/useSearch.test.ts
git commit -m "feat(web): add Auto-Route toggle for classifier-driven Instant mode backend routing"
```

---

## Post-plan follow-up (not part of this plan's tasks)

- Replace the hand-authored seed `train.jsonl`/`eval_frozen.jsonl` with real historical-query labels once the user's log export is available, then rerun `train_instant_classifier.py` and recommit the artifact.
- Once `auto_route` has live accuracy data, revisit whether it should default to `True`.
- A future spec can evaluate reusing `instant_classifier/features.py`'s taxonomy-agnostic transformers for an AI-Mode-facing non-LLM classifier — explicitly out of scope here (see design doc's Non-goals).
