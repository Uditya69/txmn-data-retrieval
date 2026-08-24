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
