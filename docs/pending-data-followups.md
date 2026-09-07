## Config file for operationally-variable values (2026-09-03)

Every year/current-edition-id/tuned-boost-weight this document talks about updating by
hand now lives in `packages/common/src/common/data/repotaxmannapi_boost_config.json`
(loaded via `common/repotaxmannapi_boost_config.py`), not scattered as Python literals in
`common/es_client.py`. **When any value below needs updating, edit that JSON file, not the
code** - `es_client.py`'s constants (`_LATEST_EDITION_ONLY_YEARS`, `_GST_TARIFF_*`,
`_FORMS_LATEST_YEAR`, `_ACCOUNT_STANDARD_*`, `_OECD_MODEL_COMMENTARIES_*`,
`_STATIC_GROUP_MEMBERSHIP_BOOSTS`, `_FINANCE_ACT_GENERAL_*`, `_EDITION_BOOSTS_BY_
INSTRUMENT_KIND`, `_GROUP_SIGNAL_SHOULD_BOOSTS`, `_STATIC_TAXONOMY_BOOSTS`, and
`repotaxmannapi_scoring.build_function_score_functions`'s `latest_finance_act_year` call
site) all read from this file at import time. Structural taxonomy ids that don't drift the
same way (e.g. `wcc`/`wc1`'s ComparativeGroupId/CirNot/CentralGST in
`repotaxmannapi_scoring.py`, `_AAA_MODEL_REPORT_GROUP_ID`) stay as code constants,
matching how the real source itself keeps them compile-time rather than app-settings.

# Pending follow-ups: logic ported ahead of data

Repotaxmannapi-parity logic in this repo that was implemented (2026-09-03 session) even
though the ES field(s) it depends on are **currently unpopulated / zero-doc** in this
repo's live index (`researchindex_aic_test`). Each was implemented deliberately, not
skipped, on the reasoning that a doc matching it later shouldn't need a code change to
start working correctly — but "implemented against zero live docs" also means **none of
this has ever been exercised against a real matching document**. If any of the
content types/fields below get indexed, re-verify the corresponding logic against real
data before trusting it in production, same way every other live-verified id/year/field
in this codebase was checked before being trusted (see `common/es_client.py`'s own
comments for that verification pattern).

Check this file whenever new content types or fields get indexed - if what landed
matches an item below, go re-verify it, don't assume the code path is already correct
just because it exists.

## `common/es_client.py`

### `_additional_exclusion_filters()` - 4 of its 7 filters currently match zero docs

- **`MinusHasChildFilter`** (`parentheadings.hasfile == "no"`) - the whole `parentheadings`
  object is mapped but 0% populated (0/410,427). If `parentheadings`/`hasfile` ever gets
  indexed: verify the filter actually excludes the intended stub/placeholder-heading docs
  and doesn't accidentally catch real content (e.g. if `hasfile` turns out multi-valued and
  ES's own array-flattening semantics don't match the "first element" the real .NET
  `FirstOrDefault()` reads).
- **AAA Model Report** (`groups.group.id == "111050000000017485"`) - 0 docs. If indexed:
  confirm the id is still correct (constants can be reused across content revisions) and
  that this content type doesn't need a should-boost re-admission the way GST Tariff/Forms
  got instead of a flat exclusion.
- **Account Standard year-2015 exclusion** (`groups.group.subgroup.id ==
  "111050000000011681"` AND `year.id == "2015"`) - 0 docs. If indexed: verify non-2015
  editions actually pass through (this is an exclude-one-year filter, not a
  keep-only-latest one like the others - easy to misread if re-touched later) and that
  `year.id` (not `year.name`) is genuinely the right field once real docs exist to check.
- **OECD Model Commentaries** (`groups.group.subgroup.subsubgroup.id ==
  "111050000000011106"`, admitted only for `year.id == "2017"`) - 0 docs. If indexed:
  **"2017" may be stale by then** - re-check what the real current OECD Model Commentary
  edition year actually is at that point, this was never re-derived from live data, just
  ported from the .NET constant as-is.

### GST Tariff latest-edition filters - real data exists, but the "current edition" ids will drift

`_GST_TARIFF_{GOODS,SERVICES,CGST_SGST}_LATEST_SUBSUBGROUP_ID` (in
`data/repotaxmannapi_boost_config.json`) are hardcoded to "Edition 42 Goods" / "Edition 35
Services (IGST)" / "Edition 35 Services (CGST + SGST)" as observed live on 2026-09-03.
Unlike the zero-doc items above, this WILL go stale on its own as new tariff editions get
published (production resolves this via a live `GstTariffBL` dataset lookup, confirmed by
the team (Slack, 2026-09-03) to be backed by a SQL server table, not just config - this
repo has no equivalent of either) - there is no code-side trigger that notices a new
edition exists. **Periodically re-check** (e.g. whenever GST tariff results look
wrong/stale) via a `groups.group.subgroup.subsubgroup` terms aggregation scoped to each
subgroup id and bump these values in the JSON config file to the new highest "Edition N" id.

**The CGST+SGST subgroup (`111050000000018741`, "GST Tariff for Services (CGST + SGST)")
was found missing entirely in a later audit pass** (2026-09-03, same day) - a real
correctness bug, not a zero-doc future-proofing gap: 413 live docs had NO latest-edition
filtering applied at all (every edition of every CGST+SGST doc passed through
unconditionally) before this was added, because the original 2 filters (Goods/Services)
only matched their own subgroup ids and let everything else - including this 3rd, real,
populated subgroup - through unconditionally. Lesson: when the real source excludes a
whole top-level group and re-admits per-subgroup (here: the "Tariff" group,
`groups.group.id` `111050000000017179`, containing all 3 of these subgroups), enumerate
every subgroup under that group id, don't assume the ones already found are the only ones.

Note: `repotaxmannapi/TaxmannAPI/Web.config` also has (commented-out, i.e. abandoned)
static `EditionGoodsLatest`/`EditionServicesLatest` app-settings from before the live
dataset lookup existed - checked 2026-09-03 and confirmed stale even as a fallback
(resolve to "Edition 37 Goods"/"Edition 32 Services (IGST)", both already behind the live
index's actual latest at the time). Don't use them as a substitute if this ever needs
re-deriving without live ES access - they're not being kept in sync with anything.

### Forms latest-edition filter - genuinely zero data, but the year value is confirmed real

`_forms_latest_edition_filter()` excludes `masterinfo.info.formtype.id == "frmtyp002"`
docs unless `year.name == _FORMS_LATEST_YEAR`. **2026-09-03 update:** this was originally a
guessed placeholder; confirmed real by reading `repotaxmannapi/TaxmannAPI/Web.config`
directly (checked into this checkout) - `LattestFormYear` = `"2026"`, matching the guess
exactly. Team confirmed (Slack, 2026-09-03) these year/group ids live in both the dotnet
repo's config files and the SQL server database - Web.config was sufficient here, no need
to query the DB. `masterinfo.info.formtype` is still 0% populated (0/410,427) - there is no
live Forms document of any kind in this index today, so the year value is confirmed
against config but never against a real matching document. If Forms content ever gets
indexed:
1. Confirm `masterinfo.info.formtype.id == "frmtyp002"` is still the right id for the
   Forms content type on real data.
2. Re-check `_FORMS_LATEST_YEAR` against Web.config again (or the SQL table, if Web.config
   ever falls out of sync with it) in case the year has moved on since 2026-09-03.

### `fetch_doc_categories` - fields exposed but never seen populated

- **`is_unreported`** (from `isuro`) - 0% populated (0/410,427). If it ever gets indexed:
  the field will just start showing up on caselaw cards automatically (no code change
  needed for that part) - but note the real product also has an **include/exclude UI
  toggle** driven by this field ("Include Unreported Case Laws", default on) that was
  deliberately NOT ported here (out of scope - UI/filter feature, not a query-logic gap).
  Revisit whether that toggle is wanted once the underlying data actually exists to filter
  on.
- **`commentary_topic`** falls back from `groups.group.subgroup.subsubgroup.name` (76%
  populated, 20,768/27,291 as of 2026-09-03) to `groups.group.subgroup.name` (100%
  populated, but coarser - e.g. generic "Commentaries") when the subsubgroup is missing. If
  that 76% ever climbs toward 100% (data backfill), the fallback branch will naturally stop
  firing on its own - no action needed unless the coarse fallback text itself is showing up
  somewhere it shouldn't.

## `common/repotaxmannapi_scoring.py`

- **`wcc`** (Comparative-group weight-3 boost, `groups.group.id ==
  "111050000000020048"`, only when `group_id` is Act/Rule) - `ComparativeGroupId` has 0
  live docs. If a "Comparative" content type is ever indexed: also revisit
  `common/es_client.py`'s `_GROUPED_SECTION_PRIORITY` list (currently
  `["ACT", "RULE", "CASELAWS", "COMMENTARY", "Experts Opinion", "Tariff"]`, missing
  "Comparative" and "Article" - see that constant's own comment, this predates the
  2026-09-03 session and was already flagged there, not newly found) - and see the
  Comparative heading-dedup item below, found the same day as a related but distinct gap.

## `masterinfo`/`masterinfo.info` - NOT genuinely dead, empty on OUR index only (corrected 2026-09-07)

**This section's original 2026-09-03 conclusion ("confirmed genuinely dead - no substitute
exists... completely empty objects on every sampled document") is WRONG and superseded -
do not trust it, kept below struck through for the record only.** It was checked only
against this repo's own index (`researchindex_aic_test`); it was never cross-checked
against prod's real live data until now.

**What's actually confirmed (2026-09-07, direct pull of prod's live ES `_source` for 3
`documenttype=="act"` ids - `102120000000099179`/`...099180`/`...099184`, Income-tax Act
2025 sections 148/149/153, alongside the same 3 ids from our own index for a byte-level
diff):** `masterinfo`/`masterinfo.info` genuinely is `{}` in our index for all 3 - that
part holds. But prod's real live document for the SAME ids has it fully populated:
`masterinfo.info.act[]` (id/name/url, e.g. `"Income-Tax Act, 2025"`), `.section[]`
(id/`actsectionyearid`/name/url, e.g. `"section-153"`), plus `actno`/`rule`/`state`/
`classification` arrays (empty on these particular docs, but present as real arrays, not
absent structure). **The field is real and populated in the actual production data model
- it's simply missing from this repo's own index**, an ingestion gap in whatever
populates `researchindex_aic_test`'s `masterinfo`, not a dead/unpopulated field in the
underlying data itself.

Narrower, now-open re-checks, not yet done:
- The 3 sampled ids are all `documenttype=="act"` - confirms `masterinfo.info.act[]`/
  `.section[]` are real and populated in prod for Act docs specifically. The original
  claim was about 3 CASELAWS-specific card fields instead
  (`InfavourOf`/`masterinfo.info.infavourof[].name`,
  `CourtName`/`masterinfo.info.court[].shortName`,
  `AuthorName`/`masterinfo.info.authors[].name`) - whether THOSE specific subfields are
  similarly populated on a real prod CASELAWS document hasn't been checked yet. Same
  structure, different content type, different sample needed before assuming they're
  populated too.
- `otherinfo.*` (`judge`/`partyname`/`fullcitation`/`counselname`/`appealno`/`asstyr`)
  was checked as `masterinfo`'s substitute under the old (wrong) "genuinely dead"
  framing - worth re-confirming whether `otherinfo.*` is itself real/populated in prod
  too, or was its own separate live-data confirmation.

This also means: this repo's own `masterinfo`-dependent should-clauses/citation joins
that assumed dead data and were skipped/simplified on that basis are worth another look -
check `common/es_client.py`'s own `masterinfo`-related comments (several reference this
now-superseded "confirmed 0% populated" conclusion) before trusting any of them as still
accurate. Also re-check whether this same our-index-vs-prod gap applies to any other field
flagged "confirmed dead" elsewhere in this doc - none of those were cross-checked against
prod's real data either, same blind spot as this one had.

## Consciously out of scope, not overlooked - a whole facet/filter/aggregation system in `GlobalSearchResearch.cs`

An audit pass (2026-09-03) found `GlobalSearchResearch.cs`'s `GetSearchResult`/
`GetSearchResultFilter` methods implement an entire result-page facet system this repo has
NOT ported, matching the user's own explicit instruction that session ("we don't need
filters/UI, only query logic") - listed here so it isn't mistaken for something missed by
accident if anyone revisits scope later:

- `CatUrlFilter`/`GroupUrlFilter`/`BenchFilter`/`CourtFilter`/`ActFilter`/`RuleFilter`/
  `SectionFilter`/`YearFilter`/`SubjectFilter` - term filters applied when a user has
  selected result-page facets, plus 9 matching `by_X` sub-aggregations that populate the
  facet-count lists themselves.
- `isUroFilter` - a real hard **query-level** exclusion (`MustNot(isuro==true)` unless
  `isUroIncluded`), distinct from the `is_unreported` **card-display** field already
  ported above (display-only, never filters).
- `isItactExclude`/`ITAct1961Filter` - an Income-tax-Act-1961-specific exclude toggle,
  gated by a query parameter (`filterSearch.filter.exclude[0]=="true"`) - NOT the same
  mechanism as the already-ported Income-tax Act 1961 edition-year exclusion (that one is
  unconditional and subgroup+year keyed; this one is act-id keyed and only fires when a
  user has explicitly toggled it).
- `actSectionYearFormat` - cross-references a selected Act+Section combination across
  editions/associate docs, only relevant when both an Act filter and a Section filter are
  simultaneously selected.

Also out of scope, found the same audit pass, same reasoning (UI-driven advanced-search
feature, not global-search query logic): `GetAnyOfSearchGlobalSearchQuery`/
`GetNotIncludeGlobalSearchQuery`/`GetDateRangeGlobalSearchQuery` (the Google-style
all-of/exact/any-of/exclude + date-range fields behind the real product's "Advance Search"
modal) and `GetSubjectQuery` (a separate subject-search query shape). If category-scoped
browsing or advanced search is ever wanted as a product feature, start from
`GlobalSearchResearch.cs`'s methods listed here, not from scratch.

## Comparative-group heading-based dedup - flagged, not implemented

`GlobalSearchIndexController.cs:337-347`: for docs in the (0-live-doc) Comparative group,
the heading is split on `"v."`, regex-normalized, and checked against every already-added
card's heading via `StartsWith` - a duplicate is skipped from the result set entirely. This
is a card-assembly-time dedup step, distinct from `wcc`'s scoring-side Comparative handling
above. Not implemented (2026-09-03) - deliberately, not an oversight: with 0 live
Comparative docs there's no way to write a meaningful test against real behavior, and it's
a more speculative regex-based text-processing step than the exclusion filters/should-boosts
implemented elsewhere this session. Implement alongside `wcc`'s data landing, not before.

## Group URLs with zero docs at all - card-text reshaping IMPLEMENTED but entirely unverified

Live-checked 2026-09-03 via a `groups.group.url` terms aggregation: only `caselaws`,
`act`, `rule`, `commentary`, `experts-opinion`, `tariff` exist in this index. The remaining
7 group urls from `GlobalSearchIndexController.cs:244-297`'s `switch (c.groups.group.url)`
(9 cases total, StandardGuidanceNotes/FinancialsAndDisclosures share one) were **all
implemented in `fetch_doc_categories` on 2026-09-03** despite zero live docs, per this
session's future-proofing precedent - but unlike every other zero-doc item in this
document, these were never checked against a single real document of these types at all
(not even the field PATHS, let alone value population). Re-read
`GlobalSearchIndexController.cs:244-297` directly and cross-check every field path listed
below against a real document once any of these land - do not assume this port is correct
just because it exists and has tests (the tests only confirm the code does what the .NET
source says, not that the .NET source's field paths match this repo's actual ES mapping):

- **`form`** → `form_name` from `masterinfo.info.form[0].name`.
- **`dta`** (tax treaty) → `dta_name` from `masterinfo.iltinfoes[0].country1/2.name`;
  `heading_override`/`subheading_override` rebuild heading as `"<subheading> : <heading>"`
  and subheading as `groups.group.subgroup.subsubgroup.subsubsubgroup.name`.
- **`cbdt`** → `cbdt_name` from `parentheadings[0].name`/`.pname`; `heading_override` falls
  back to old subheading if heading is blank; `subheading_override` becomes old
  `shortcontent`; `shortcontent_override` is always `""`.
- **`news`** → `subheading_override` = `shortcontent`.
- **`bill`/`ordinances`/`report`/`listinginformal-report`/`practice-procedure`** (note the
  hyphens on the last two - the real `Constants_GroupUrl` string values, verified against
  `repotaxmannapi/TaxmannAPI/BL/Constants.cs` directly after an initial guess without
  hyphens was wrong) → `heading_override` appends `" - " + parentheadings[0].name`;
  `subheading_override` = `shortcontent`.
- **`cirnot`** → `heading_override` appends `" - Dated dd-MM-yyyy"`, parsed from
  `displaydocumentdatestring` (confirmed dead corpus-wide as of 2026-09-02 - fetched again
  here anyway since CirNot itself has 0 docs to independently re-check that finding
  against).
- **`standard-guidance-notes`/`financials-and-disclosures`** (also hyphenated, same
  Constants.cs check) → `heading_override` fully rebuilt as
  `"<masterinfo.info.company[0].name> <subheading> : <year.name>"`.

`heading_override`/`subheading_override`/`shortcontent_override` are NOT wired into what
Instant mode's cards actually display (`raw_search`'s own `heading`/`subheading` return
values, or the frontend's `card.heading`) - they're computed and exposed on
`fetch_doc_categories`'s entry dict only. Wiring them into the actually-displayed title/
snippet is a separate, deferred step - do that once real data exists to verify the
transform is even correct, not before.

## `formatteddocumentdate` frozen for `act`/`rule`/most `commentary` - shared upstream, NOT a replica-only staleness bug (corrected 2026-09-07)

Confirmed live 2026-09-07, three ways - our primary index (`researchindex_aic_test`), the
raw source data this repo's pipeline ingests from (`tm-dp/data/statutory/**/*.json`,
sampled at scale), and (**new**, see correction below) a direct pull of prod's own live ES
document for the same ids:

- `documenttype=="act"`: 100% frozen at `2012-01-01` (83,309/83,309 in our index; same in
  a 3,000-file source sample).
- `documenttype=="rule"`: also 100% frozen, but at a DIFFERENT constant, `1900-01-01`
  (3,000-file source sample) - a second, independently-broken content type, not
  previously checked.
- `commentary`: 66% frozen at `1900-01-01` (3,000-file source sample), remaining ~34% has
  real varying dates - a partial version of the same bug.
- `articles`, `tariff`, `caselaws`: clean, real varying dates (spot-checked, not
  exhaustive).

A real, varying, genuinely-usable date DOES exist in the very same source records under a
different key, `lastpublished_date` (e.g. `2026-06-07`, `2024-03-15`, ... - real per-
document last-publish dates) - unused for this purpose. `documentdate` (a second date-ish
field, string-typed) carries the identical frozen values as `formatteddocumentdate`, not
an independent signal.

**CORRECTION (2026-09-07, same day - do not trust the superseded version of this note
below the original session's `git blame`):** the original write-up of this item
speculated "prod's live index almost certainly has real, varying dates for these
documents" as the explanation for prod outranking our index on recency-sensitive queries.
**That speculation was checked against real data and is wrong** - a direct pull of prod's
own live ES document for 3 of these ids (`102120000000099179`/`...099180`/`...099184`,
Income-tax Act 2025 sections 148/149/153) came back **byte-identical** to our own index's
values: `formatteddocumentdate: 2012-01-01T00:00:00`, `documentdate:
20120101^01-01-2012`, `lastpublished_date: 2026-06-06T00:00:00`, `created_date:
1900-01-01T00:00:00` - all four fields, exact match. Prod is working with the identical
frozen date on this doc and still gets no recency credit for it (`×1`, same as ours) -
that is NOT what's producing prod's better ranking for it.

So: this is confirmed as a genuine, shared upstream data-quality issue (present in
`tm-dp/data/statutory`'s source records, present in prod's real live index, present in
ours - not specific to this repo's replica or its ingestion), but it does **not** explain
observed ranking gaps between our index and prod for these queries - something else
(likely the *competing* case-law documents' `viewcount`/`court_boost`/`total_score`
differing between corpora, not this doc's own date) accounts for that, still
unidentified. Don't re-cite the disproven "prod probably has real dates" framing anywhere
else in this repo's docs if it shows up - this is the correction of record.

This remains worth fixing on its own merits regardless (a genuinely broken recency signal
for the majority of statutory content, `act`+`rule`+`commentary`), just not framed as
"why we lose to prod" anymore. Not something fixable in `common/es_client.py`/
`repotaxmannapi_scoring.py` - the field choice there is correct and byte-matches
`GlobalSearchResearch.cs:632-639`; the fix, if any, belongs wherever
`formatteddocumentdate` gets populated for these content types (upstream of
`tm-dp/data/statutory` - outside this repo entirely, not `data-extraction-pipeline`
either, since the placeholder is already present in the source JSON that pipeline
consumes).

## ES2 (`researchindex2024final_dev`) - separate index, not currently used for anything

Confirmed live 2026-09-03 (via `ES2_URI`/`ES2_INDEX` in `.env`, not the primary
`ES_URI`/`ES_INDEX`): `viewcount == 10` on 41.0% of the whole index (236,721/578,061) vs
6.9% on the primary index (`researchindex_aic_test`) - a real, severe placeholder-data
skew that breaks the multiply-formula's popularity signal for that index specifically.
This is a data problem on ES2 itself, not something any query-logic change here can fix.
**If `ES_INDEX`/`ES_URI` in `.env` (or `Settings`) is ever pointed at ES2 instead of the
primary index, re-run this same check first** - don't assume the skew has been fixed just
because time has passed; re-verify against live data at that point.
