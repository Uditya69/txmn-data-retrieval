"""Loads the operationally-variable years/current-editions/tuned-boost-weights behind
Instant mode's repotaxmannapi-parity logic (see data/repotaxmannapi_boost_config.json's own
top-level "_comment" for exactly what belongs here vs stays a code constant). Cached at
module level, same pattern common.repotaxmannapi_token_dictionary/common.legal_lexicon
already use for their own JSON-backed data - small enough to load once per process.

This is deliberately a plain JSON file, not env vars/Settings fields: these values change
by hand a few times a year (a new tariff edition, a new financial year), same rationale
common/es_client.py's own comments already give for keeping them as literals rather than
introducing Settings plumbing - a config FILE (checked into git, diffable, reviewable) is
the natural middle ground between "buried in code" and "full Settings/env machinery" for
values like this. See docs/pending-data-followups.md for which of these are confirmed real
(cross-checked against repotaxmannapi/TaxmannAPI/Web.config or this repo's own live ES
index) versus still-unverified placeholders."""
import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

_DATA_PATH = Path(__file__).parent / "data" / "repotaxmannapi_boost_config.json"


class GstTariffConfig(TypedDict):
    goods_subgroup_id: str
    goods_latest_subsubgroup_id: str
    services_subgroup_id: str
    services_latest_subsubgroup_id: str
    cgst_sgst_subgroup_id: str
    cgst_sgst_latest_subsubgroup_id: str


class FormsConfig(TypedDict):
    formtype_id: str
    latest_year: str


class AccountStandardConfig(TypedDict):
    subgroup_id: str
    excluded_year: str


class OecdModelCommentariesConfig(TypedDict):
    subsubgroup_id: str
    latest_year: str


class FinanceActGeneralConfig(TypedDict):
    subgroup_id: str
    current_year: str
    boost: float


class RepotaxmannapiBoostConfig(TypedDict):
    latest_edition_years: dict[str, str]
    gst_tariff: GstTariffConfig
    forms: FormsConfig
    account_standard: AccountStandardConfig
    oecd_model_commentaries: OecdModelCommentariesConfig
    edition_boosts_by_instrument_kind: dict[str, list[list]]
    static_group_membership_boosts: dict
    finance_act_general: FinanceActGeneralConfig
    group_signal_should_boosts: dict[str, float]
    static_taxonomy_boosts: dict
    latest_finance_act_year_for_multiply_formula: dict[str, str]


def _strip_comments(node):
    """Recursively drops "_comment" keys from dicts - present throughout the JSON file
    purely as inline documentation, never a real config value any caller should see."""
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if k != "_comment"}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


@lru_cache(maxsize=1)
def load_repotaxmannapi_boost_config() -> RepotaxmannapiBoostConfig:
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _strip_comments(raw)
