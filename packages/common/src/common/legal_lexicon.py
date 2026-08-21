import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "legal_lexicon.json"
_LEXICON = json.loads(_DATA_PATH.read_text())

_COURTS = set(_LEXICON["courts"])
_JOURNALS = set(_LEXICON["journals"])
_STOPWORDS = set(_LEXICON["stopwords"])
_SYNONYMS = _LEXICON["synonyms"]
_NORMALIZATIONS = _LEXICON["normalizations"]

KNOWN_COURT_FULL_NAMES: list[str] = _LEXICON["court_full_names"]
KNOWN_ACT_NAMES: set[str] = set(_LEXICON["act_names"])


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
