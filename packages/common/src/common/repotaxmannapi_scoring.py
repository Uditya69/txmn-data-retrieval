"""Python port of repotaxmannapi/TaxmannAPI/Elastic/GlobalSearchResearch.cs's FunctionScore
recency ladder and static field_value_factor stack (a separate, read-only .NET codebase).

Every constant here is copied verbatim from GlobalSearchResearch.cs, independently
re-verified 2026-09-02 against the real current file (the plan's own line citations for
this file have been found stale/wrong in earlier tasks of this replica plan, so they were
not trusted blind):

- Recency ladder (8 date-range Weight functions): lines 632-639, inside the
  `.Functions(f => ...)` builder of the top-level FunctionScore query (which starts at
  line 623 and closes with `.BoostMode(FunctionBoostMode.Multiply)` at line 657). The
  plan brief cited 627-634 for this block; the real block is 632-639 (off by ~5 lines,
  because lines 625-631 are groupBoost/subgroup Weight functions that are explicitly
  out of scope for this task - see Task 7).
- field_value_factor stack (5 functions): lines 653-657. The plan brief cited 649-653;
  the real block is 653-657 (off by ~4 lines, same reason).

All 8 recency weights and all 5 field_value_factor factor/modifier/missing values matched
the brief's stated numbers exactly on independent re-read - no value corrections were
needed, only the line-number citations above.

Explicitly OUT OF SCOPE for this module (belongs to a later task):
- groupBoost resolution (lines 613-621, 625-631: group/subgroup id match weights,
  varying by groupid - ActGroupId/RuleFormId/etc.) and edition subgroup boosts.
- The stateGstCatFilter/caseLawsFilter Weight(0.03) function (line 640-646) and the
  financeactBoostNewFilter Weight(0.02) function (line 646-652).
Both are part of Task 7's groupBoost scope, not this task's recency-ladder/static-boost
scope, and are intentionally not built here.

This module intentionally uses different constants than common.es_client's own
_apply_boost (this repo's existing sum-mode formula) for the same-named fields
(court_boost, documenttypeboost) - the two are independently-tuned formulas for two
different combination modes (multiply here via BoostMode.Multiply, sum there), not a
discrepancy to reconcile.
"""

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


def build_function_score_functions(group_id: str) -> list[dict]:
    """Build the ES `functions` array for the recency ladder + 5 static
    field_value_factor boosts, as a verbatim port of GlobalSearchResearch.cs's
    FunctionScore stack (recency and field_value_factor portions only - see module
    docstring for what's deliberately excluded).

    `group_id` is accepted now to match the eventual signature once Task 7 adds
    groupBoost resolution on top of this function; it is not used by this task's
    scope (recency ladder and static field_value_factor functions carry no group_id
    dependency in the real source).
    """
    functions: list[dict] = [
        {"filter": {"range": {"formatteddocumentdate": {"gte": gte, "lte": lte}}}, "weight": weight}
        for gte, lte, weight in _RECENCY_TIERS
    ]

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
