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
  Constants_GetIdByName.CirNot, weight 100). These were NOT among "the four Weight(...)
  functions using groupid/groupBoost/edition-subgroup ids" this task's original brief
  scoped in, so were left unimplemented at the time - flagged, not silently dropped.
  IMPLEMENTED 2026-09-03 (`wcc`/`wc1` below): `wcc`'s ComparativeGroupId has 0 live docs
  in this repo's index today (confirmed 2026-09-02 investigation, same as
  common.es_client's Account Standard/AAA Model Report/OECD Model Commentary), and `wc1`'s
  CirNot/CentralGST combination hasn't been checked live either - both ported anyway
  rather than skipped for a present-day data gap: this repo's own logic should match
  production's regardless of what's indexed today, since either could gain matching docs
  later without a code change to re-enable them.
- Recency ladder (8 date-range Weight functions): lines 632-639 (unchanged from Task 6's
  citation, still accurate).
- field_value_factor stack (5 functions): lines 653-657 (unchanged from Task 6's
  citation, still accurate).
- Task 8's two multiply-mode penalty functions (`w8`/`w10`): real lines 639-646 (`w8`,
  stateGst-non-caselaws, weight 0.03) and 646-652 (`w10`, finance-act-old-year, weight
  0.02). Task 8's own brief cited "636-648" for this combined block (and Task 7's report,
  written while scoping Task 8 prospectively, had cited "640-652"); the real combined
  block, re-verified independently on this task's own read of the current file, is
  639-652 - `w8` starts mid-line 639 (immediately after the recency ladder's last
  `.Weight(1.5))`), not line 640.
  - `stateGstCatFilter` = `categories.FirstOrDefault().subcategory.id` `Terms` match
    against `Constants_GetIdByName.StateGSTCatID` = `"111050000000017095"`
    (BL/Constants.cs:245).
  - `caseLawsFilter` = `groups.group.url.keyword` `MustNot Term` == `Constants_GroupUrl.Caselaws`
    = `"caselaws"` (BL/Constants.cs:62).
  - `financeactBoostNewFilter` = `groups.group.subgroup.id` `Terms` match against
    `Constants_GetIdByName.FinanceActsSGroupId` = `"111050000000010567"`
    (BL/Constants.cs:266).
  - `financeactBoostNewYearFilter` = `year.name.keyword` `Terms` match against the runtime
    `ConfigurationManager.AppSettings["LattestFinanceActYearID"]` (GlobalSearchResearch.cs:538)
    - not a compile-time constant, so `build_function_score_functions` takes it as the
    required `latest_finance_act_year: str` parameter instead of hardcoding a year.

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

# Task 8's two multiply-mode penalty functions (`w8`/`w10` in the real source, real lines
# 640-652 - see module docstring's independent re-verification note; confirmed again on
# this task's own read of the current file). Verified against
# repotaxmannapi/TaxmannAPI/BL/Constants.cs:
# - StateGSTCatID = "111050000000017095" (Constants.cs:245)
# - FinanceActsSGroupId = "111050000000010567" (Constants.cs:266)
# - Constants_GroupUrl.Caselaws = "caselaws" (Constants.cs:62)
_STATE_GST_CAT_ID = "111050000000017095"
_FINANCE_ACTS_SGROUP_ID = "111050000000010567"
_CASELAWS_GROUP_URL = "caselaws"

# `wcc`/`wc1` (GlobalSearchResearch.cs:630-631). Verified against
# repotaxmannapi/TaxmannAPI/BL/Constants.cs:
# - ComparativeGroupId = "111050000000020048" (Constants.cs:337)
# - CirNot (groups.group.id) = "111050000000000057" (Constants.cs:316)
# - CentralGST (categories.subcategory.id) = "111050000000017093" (Constants.cs:326)
_COMPARATIVE_GROUP_ID = "111050000000020048"
_CIRNOT_GROUP_ID = "111050000000000057"
_CENTRAL_GST_CAT_ID = "111050000000017093"

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


def build_function_score_functions(group_id: str, latest_finance_act_year: str) -> list[dict]:
    """Build the ES `functions` array for the groupBoost/edition-subgroup boosts, the
    comparative-group (`wcc`) and CirNot/CentralGST (`wc1`) Weight functions, the recency
    ladder, 5 static field_value_factor boosts, and the two multiply-mode penalty functions
    (stateGst-non-caselaws, finance-act-old-year) - a verbatim port of
    GlobalSearchResearch.cs's FunctionScore stack (see module docstring for exact line
    citations).

    `latest_finance_act_year` corresponds to the real source's
    `ConfigurationManager.AppSettings["LattestFinanceActYearID"]` (GlobalSearchResearch.cs:538)
    - a runtime app-setting in repotaxmannapi, not a compile-time constant, so it must be
    supplied by the caller rather than hardcoded here.
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

    # wcc (GlobalSearchResearch.cs:629-630): groups.group.id == ComparativeGroupId, but
    # only when group_id is the Act or Rule group (else the real source's ternary yields
    # null, matching nothing) - weight 3.
    if group_id in (_ACT_GROUP_ID, _RULE_FORM_ID):
        functions.append({
            "filter": {"match": {"groups.group.id": {"query": _COMPARATIVE_GROUP_ID}}}, "weight": 3,
        })

    # wc1 (GlobalSearchResearch.cs:630-631): categories.subcategory.id == CentralGST, but
    # only when group_id is CirNot (circulars/notifications) - weight 100, proportionally
    # much larger than every other function here.
    if group_id == _CIRNOT_GROUP_ID:
        functions.append({
            "filter": {"match": {"categories.subcategory.id": {"query": _CENTRAL_GST_CAT_ID}}}, "weight": 100,
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

    # w8 (GlobalSearchResearch.cs:639-646): penalize non-caselaws documents in the StateGST
    # subcategory - `stateGstCatFilter` (categories.subcategory.id in StateGSTCatID) AND
    # `caseLawsFilter` (groups.group.url != "caselaws"), weight 0.03.
    functions.append({
        "filter": {
            "bool": {
                "filter": [
                    {"terms": {"categories.subcategory.id": [_STATE_GST_CAT_ID]}},
                ],
                "must_not": [
                    {"term": {"groups.group.url.keyword": _CASELAWS_GROUP_URL}},
                ],
            }
        },
        "weight": 0.03,
    })

    # w10 (GlobalSearchResearch.cs:646-652): penalize Finance-Act-subgroup documents whose
    # year is not the latest finance-act year - `financeactBoostNewFilter`
    # (groups.group.subgroup.id in FinanceActsSGroupId) AND NOT `financeactBoostNewYearFilter`
    # (year.name in the current LattestFinanceActYearID app setting), weight 0.02.
    functions.append({
        "filter": {
            "bool": {
                "filter": [
                    {"terms": {"groups.group.subgroup.id": [_FINANCE_ACTS_SGROUP_ID]}},
                ],
                "must_not": [
                    {"terms": {"year.name.keyword": [latest_finance_act_year]}},
                ],
            }
        },
        "weight": 0.02,
    })

    return functions
