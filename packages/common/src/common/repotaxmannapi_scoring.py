"""Python port of repotaxmannapi/TaxmannAPI/Elastic/GlobalSearchResearch.cs's FunctionScore
groupBoost resolution, edition-subgroup boosts, recency ladder, and static field_value_factor
stack (a separate, read-only .NET codebase).

Every constant here is copied verbatim from GlobalSearchResearch.cs, independently
re-verified 2026-09-02 against the real current file (the plan's own line citations for
this file have been found stale/wrong in earlier tasks of this replica plan, so they were
not trusted blind):

- groupBoost default + Act/Rule overrides: lines 613-621 (`int groupBoost = 10000000;`
  through the `if (groupid == Constants_GetIdByName.RuleFormId) { groupBoost = 4; }`
  block). Task 7's own brief cited 607-615; the real block is 613-621 (off by ~6 lines).
- The four groupid/groupBoost/edition-subgroup Weight(...) functions in scope for this
  task: lines 625-629, inside the `.Functions(f => ...)` builder of the top-level
  FunctionScore query (which starts at line 623 and closes with
  `.BoostMode(FunctionBoostMode.Multiply)` at line 657). The brief cited 620-625; the
  real block is 625-629 (off by ~5 lines). In source order:
    - `w`  (625-626): filter groups.group.id == groupid,             weight groupBoost
    - `w0` (626-627): filter groups.group.subgroup.id == groupid,    weight groupBoost
    - `w1` (627-628): filter groups.group.subgroup.id == old-edition-subgroup-id (only
      set when groupid is ActGroupId or RuleFormId, else null/never-matches), weight 2
    - `wc` (628-629): same but new-edition-subgroup-id,               weight 3
  NOTE (found on this independent re-read, not previously flagged by earlier tasks):
  immediately after `wc` the real source has TWO MORE Weight(...) functions at lines
  629-631 - `wcc` (groups.group.id == ComparativeGroupId when groupid is Act/Rule group,
  weight 3) and `wc1` (categories.*.subcategory.id == CentralGST when groupid ==
  Constants_GetIdByName.CirNot, weight 100). These are NOT among "the four Weight(...)
  functions using groupid/groupBoost/edition-subgroup ids" this task's brief scopes in,
  and are not part of Task 8's scope either (Task 8 covers the stateGst/caseLaws and
  financeact Weight(0.03)/Weight(0.02) functions at lines 640-652, i.e. `w8`/`w10` in the
  real source). `wcc`/`wc1` are therefore left unimplemented here - out of scope for this
  task per its brief, flagged for a future task rather than silently added.
- Recency ladder (8 date-range Weight functions): lines 632-639 (unchanged from Task 6's
  citation, still accurate).
- field_value_factor stack (5 functions): lines 653-657 (unchanged from Task 6's
  citation, still accurate).

All weight/factor/modifier/missing values matched the brief's stated numbers exactly on
independent re-read - no value corrections were needed, only the line-number citations
above.

groupBoost/edition-subgroup id constants (verified against
repotaxmannapi/TaxmannAPI/BL/Constants.cs):
- ActGroupId = "111050000000000064" (Constants.cs:313), RuleFormId =
  "111050000000000026" (Constants.cs:292) - both match the brief exactly.
- IncomeTaxAct1961SGroupId = "111050000000010687", IncomeTaxAct2025SGroupId =
  "111050000000020042" (Constants.cs:253-254) - both match the brief exactly, and (per
  the brief, already live-verified against this repo's own ES index in an earlier
  session - see common.es_client._EDITION_BOOSTS_BY_INSTRUMENT_KIND for the sum-mode
  equivalent lookup) correctly share the `111050...` id namespace that
  groups.group.subgroup.id actually uses.
- Constants.cs's own IncomeTaxRule1962/IncomeTaxRule2026 constants (Constants.cs:342-343)
  are "103010000000000171"/"103010000000002191" - a DIFFERENT id namespace
  (`103010...`) than groups.group.subgroup.id uses, confirmed by this independent
  re-read of Constants.cs. A `Match` query against those values would never match a real
  document under groups.group.subgroup.id. Per the brief (already live-verified against
  this repo's own ES index in an earlier session, not re-verifiable here), the real
  subgroup ids for these two editions are "111050000000010121" ("Income-tax Rules,
  1962") and "111050000000020129" ("Income-tax Rules, 2026"). This module uses the
  brief's live-verified `111050...` ids, NOT Constants.cs's `103010...` ones - this is a
  genuine bug/inconsistency in the source being replicated, being deliberately worked
  around rather than reproduced.

This module intentionally uses different constants than common.es_client's own
_apply_boost (this repo's existing sum-mode formula) for the same-named fields
(court_boost, documenttypeboost) - the two are independently-tuned formulas for two
different combination modes (multiply here via BoostMode.Multiply, sum there), not a
discrepancy to reconcile.
"""

_ACT_GROUP_ID = "111050000000000064"
_RULE_FORM_ID = "111050000000000026"
# Verified live against this repo's own ES index in an earlier session (see
# common.es_client._EDITION_BOOSTS_BY_INSTRUMENT_KIND for the sum-mode equivalent lookup) -
# NOT the ids in repotaxmannapi's own BL/Constants.cs IncomeTaxRule1962/2026 constants
# ("103010000000000171"/"103010000000002191"), which use a different id namespace than
# groups.group.subgroup.id actually uses and would never match a real document under that
# field. See module docstring for the full explanation of this discrepancy.
_INCOME_TAX_ACT_1961_SUBGROUP_ID = "111050000000010687"
_INCOME_TAX_ACT_2025_SUBGROUP_ID = "111050000000020042"
_INCOME_TAX_RULES_1962_SUBGROUP_ID = "111050000000010121"
_INCOME_TAX_RULES_2026_SUBGROUP_ID = "111050000000020129"

# 8-tier recency ladder. GlobalSearchResearch.cs:632-639 (DateMath.Now.Subtract(...)
# GreaterThanOrEquals/LessThanOrEquals pairs, each with its own .Weight(...)).
_RECENCY_TIERS = [
    ("now-1d", "now", 18),
    ("now-7d", "now-1d", 15),
    ("now-1M", "now-7d", 13),
    ("now-3M", "now-1M", 10),
    ("now-1y", "now-3M", 8),
    ("now-2y", "now-1y", 5),
    ("now-5y", "now-2y", 3.5),
    ("now-150y", "now-5y", 1.5),
]


def _resolve_group_boost(group_id: str) -> int:
    """GlobalSearchResearch.cs:613-621. Default 10000000, overridden to 2 for the Act
    group and 4 for the Rule group (each comment reads "due to 'rural agriculture land
    section 56'" in the real source)."""
    if group_id == _ACT_GROUP_ID:
        return 2
    if group_id == _RULE_FORM_ID:
        return 4
    return 10000000


def _resolve_edition_subgroup_id(group_id: str, *, current: bool) -> str | None:
    """GlobalSearchResearch.cs:627-629 (`w1`/`wc`) - the edition-subgroup id to match
    against groups.group.subgroup.id, or None when group_id is neither the Act nor Rule
    group (matching the real source's `null` in that ternary, which the ES `Match` query
    below would never match)."""
    if group_id == _ACT_GROUP_ID:
        return _INCOME_TAX_ACT_2025_SUBGROUP_ID if current else _INCOME_TAX_ACT_1961_SUBGROUP_ID
    if group_id == _RULE_FORM_ID:
        return _INCOME_TAX_RULES_2026_SUBGROUP_ID if current else _INCOME_TAX_RULES_1962_SUBGROUP_ID
    return None


def build_function_score_functions(group_id: str) -> list[dict]:
    """Build the ES `functions` array for the groupBoost/edition-subgroup boosts,
    recency ladder, and 5 static field_value_factor boosts - a verbatim port of
    GlobalSearchResearch.cs's FunctionScore stack (see module docstring for exact line
    citations and what's deliberately excluded, i.e. the `wcc`/`wc1` comparative-group
    and CirNot/CentralGST Weight functions, and Task 8's stateGst/financeact penalty
    functions).
    """
    group_boost = _resolve_group_boost(group_id)
    functions: list[dict] = [
        {"filter": {"match": {"groups.group.id": {"query": group_id}}}, "weight": group_boost},
        {"filter": {"match": {"groups.group.subgroup.id": {"query": group_id}}}, "weight": group_boost},
    ]

    old_edition_id = _resolve_edition_subgroup_id(group_id, current=False)
    if old_edition_id is not None:
        functions.append({
            "filter": {"match": {"groups.group.subgroup.id": {"query": old_edition_id}}}, "weight": 2,
        })
    current_edition_id = _resolve_edition_subgroup_id(group_id, current=True)
    if current_edition_id is not None:
        functions.append({
            "filter": {"match": {"groups.group.subgroup.id": {"query": current_edition_id}}}, "weight": 3,
        })

    functions.extend(
        {"filter": {"range": {"formatteddocumentdate": {"gte": gte, "lte": lte}}}, "weight": weight}
        for gte, lte, weight in _RECENCY_TIERS
    )

    # field_value_factor stack. GlobalSearchResearch.cs:653-657.
    # documenttypeboost: .FieldValueFactor(fl => fl.Field(fl1 => fl1.documenttypeboost)) -
    # no .Factor()/.Modifier()/.Missing() call, so NEST defaults apply (factor 1, no modifier).
    functions.append({"field_value_factor": {"field": "documenttypeboost"}})
    # viewcount: .Factor(0.0000018).Modifier(FieldValueFactorModifier.Log2P) - no .Missing() call.
    functions.append({"field_value_factor": {"field": "viewcount", "factor": 0.0000018, "modifier": "log2p"}})
    # court_boost: .Missing(0).Factor(0.0000018).Modifier(Log2P).Missing(0) (Missing set twice,
    # both times to 0).
    functions.append({
        "field_value_factor": {"field": "court_boost", "factor": 0.0000018, "modifier": "log2p", "missing": 0},
    })
    # total_score: identical shape to court_boost - .Missing(0).Factor(0.0000018).Modifier(Log2P).Missing(0).
    functions.append({
        "field_value_factor": {"field": "total_score", "factor": 0.0000018, "modifier": "log2p", "missing": 0},
    })
    # landmarkruling: filtered via .Filter(... MustNot Term(landmarkruling == -10) ...), then
    # .Field(landmarkruling).Missing(0).Factor(1.2).Modifier(Log2P).Missing(0).
    functions.append({
        "filter": {"bool": {"must_not": [{"term": {"landmarkruling": -10}}]}},
        "field_value_factor": {"field": "landmarkruling", "factor": 1.2, "modifier": "log2p", "missing": 0},
    })

    return functions
