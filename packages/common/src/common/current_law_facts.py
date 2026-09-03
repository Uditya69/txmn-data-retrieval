"""Loads the hand-maintained "which Act text is currently in force" fact table (see
data/current_law_facts.json's own top-level "_comment"). Cached at module level, same
pattern common.repotaxmannapi_boost_config/common.legal_lexicon already use for their own
JSON-backed data.

Deliberately a plain JSON file, not env vars/Settings fields: these facts change by hand a
few times a year (an Act supersession, a new edition) - same rationale
repotaxmannapi_boost_config.py already gives for its own data."""
import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

_DATA_PATH = Path(__file__).parent / "data" / "current_law_facts.json"


class ActFact(TypedDict):
    name: str
    es_act_field: str
    status: str
    effective_from: str | None
    last_verified: str


class CurrentLawFacts(TypedDict):
    acts: list[ActFact]
    default_act_when_unspecified: str


def _strip_comments(node):
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if k != "_comment"}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


@lru_cache(maxsize=1)
def load_current_law_facts() -> CurrentLawFacts:
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _strip_comments(raw)


def acts_relevant_to_text(text: str) -> list[ActFact]:
    """Filters the full Act table down to entries whose es_act_field substring actually
    appears in `text` (a turn's retrieved excerpt block) - keeps the synthesis prompt from
    growing with facts about Acts that have nothing to do with this query."""
    facts = load_current_law_facts()
    return [act for act in facts["acts"] if act["es_act_field"] in text]
