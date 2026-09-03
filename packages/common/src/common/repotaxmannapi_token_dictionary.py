"""Loads the extracted repotaxmannapi token dictionary (see
scripts/extract_repotaxmannapi_token_dictionary.py for how data/
repotaxmannapi_token_dictionary.json is generated from repotaxmannapi's
TokenParserElastic.resx - a separate, read-only .NET codebase). Cached at module level:
the dictionary is ~2600 entries, small enough to load once and reuse across every call in a
process, same pattern common.legal_lexicon already uses for its own JSON-backed lexicon."""
import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

_DATA_PATH = Path(__file__).parent / "data" / "repotaxmannapi_token_dictionary.json"


class TokenDictEntry(TypedDict):
    element_type: str
    tag_no: str
    proximity: int
    boost_factor: int
    group_id: str
    search_text: str | None


@lru_cache(maxsize=1)
def load_repotaxmannapi_token_dictionary() -> dict[str, TokenDictEntry]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))
