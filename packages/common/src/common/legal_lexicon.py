import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

_DATA_PATH = Path(__file__).parent / "data" / "legal_lexicon.json"
_LEXICON = json.loads(_DATA_PATH.read_text())

_COURTS = set(_LEXICON["courts"])
_JOURNALS = set(_LEXICON["journals"])
_STOPWORDS = set(_LEXICON["stopwords"])
_SYNONYMS = _LEXICON["synonyms"]
_NORMALIZATIONS = _LEXICON["normalizations"]

KNOWN_COURT_FULL_NAMES: list[str] = _LEXICON["court_full_names"]
KNOWN_ACT_NAMES: set[str] = set(_LEXICON["act_names"])
_GENERAL_LEGAL_TERMS = set(_LEXICON["general_legal_terms"])


def is_known_court(token: str) -> bool:
    return token.upper() in _COURTS


def is_known_journal(token: str) -> bool:
    # journals are stored space-separated ("TAXMANN COM"); a query token spells the
    # same abbreviation with punctuation ("taxmann.com", "S.T.R.") - normalizing
    # non-alnum characters to spaces before lookup lets one lexicon entry match every
    # punctuation variant instead of needing a duplicate entry per variant.
    normalized = re.sub(r"[^A-Z0-9]+", " ", token.upper()).strip()
    return normalized in _JOURNALS


def is_stopword(token: str) -> bool:
    return token.upper() in _STOPWORDS


def expand_synonyms(token: str) -> list[str]:
    return _SYNONYMS.get(token.upper(), [token])


def normalize(token: str) -> str:
    return _NORMALIZATIONS.get(token.upper(), token)


# Layer B: structural regex, hand-written (grammar, not data) - see design doc section 3.
SECTION_PATTERN = re.compile(
    r"\b(?:section|sec\.?|u/s|rule|article)\s*\d+[A-Za-z]*(?:\(\d+\))?", re.IGNORECASE
)
CITATION_PATTERN = re.compile(
    r"\(?\d{4}\)?\s*\(?\d*\)?\s*[A-Z]{2,10}\s*\d+", re.IGNORECASE
)
PARTY_PATTERN = re.compile(r"\bv(?:s\.?|ersus)?\.?\s+\w", re.IGNORECASE)

# Single-token court/journal/general-legal-term names only - act_names are multi-word
# phrases, and fuzzy-matching a single query token against a multi-word target always
# scores low, so they'd never be the closest match anyway.
_FUZZY_TERMS = [t for t in (_COURTS | _JOURNALS | _GENERAL_LEGAL_TERMS) if " " not in t]
_FUZZY_SCORE_CUTOFF = 82.0
_FUZZY_MIN_TOKEN_LEN = 4


def fuzzy_correct_query(query: str) -> tuple[str, list[dict]]:
    """Corrects misspelled court/journal/general-legal-vocabulary terms by fuzzy-matching
    each token against the known-term lists that back is_known_court()/is_known_journal()
    plus general_legal_terms (section, assessee, tribunal, etc.). Tokens already an exact
    match, or shorter than _FUZZY_MIN_TOKEN_LEN, or containing a digit (section numbers,
    citations) are left untouched - all three are cases where a short/legit token can
    score deceptively high against an unrelated known term (e.g. "GST" -> "GSTL" at 85.7,
    above threshold)."""
    corrections = []
    corrected_tokens = []
    for token in query.split():
        stripped = token.strip(".,;:\"'()")
        upper = stripped.upper()
        if (
            len(stripped) < _FUZZY_MIN_TOKEN_LEN
            or not stripped.isalpha()
            or upper in _COURTS
            or upper in _GENERAL_LEGAL_TERMS
            or is_known_journal(stripped)
        ):
            corrected_tokens.append(token)
            continue
        match = process.extractOne(upper, _FUZZY_TERMS, scorer=fuzz.ratio, score_cutoff=_FUZZY_SCORE_CUTOFF)
        if match is None:
            corrected_tokens.append(token)
            continue
        matched_term, score, _ = match
        corrections.append({"original": token, "corrected": matched_term, "score": score})
        corrected_tokens.append(matched_term)
    return " ".join(corrected_tokens), corrections
