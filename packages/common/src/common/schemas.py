MILVUS_COLLECTIONS = [
    "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    "act_section", "rule_section", "article_section", "commentary_section",
]

CHUNKED_COLLECTIONS = {
    "digest", "facts", "held", "ruling",
    "act_section", "rule_section", "article_section", "commentary_section",
}

BM25_SOURCE_FIELD = {name: "text" for name in MILVUS_COLLECTIONS}
BM25_SOURCE_FIELD["metadata"] = "heading_subheading_text"

# "ruling" and the act/rule/article/commentary section collections have no
# sparse_vector field in Milvus - the ingestion pipeline dropped their BM25
# Function. Sparse search must skip them rather than query a field that
# doesn't exist.
SPARSE_VECTOR_COLLECTIONS = set(MILVUS_COLLECTIONS) - {
    "ruling", "act_section", "rule_section", "article_section", "commentary_section",
}

# ES's groups.group.name field maps 1:1 onto 4 of these 5 by name (ACT/RULE/COMMENTARY/
# CASELAWS). article_section is the exception - a naming mismatch, not a data problem:
# verified live against 20 doc_ids spanning the full id range (20/20 consistent), and
# independently confirmed by a teammate familiar with the ingestion side. See
# docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md.
ES_GROUP_FOR_COLLECTION: dict[str, str] = {
    "ruling": "CASELAWS",
    "act_section": "ACT",
    "rule_section": "RULE",
    "article_section": "Experts Opinion",
    "commentary_section": "COMMENTARY",
}

# Display labels for ES's `categories.name` and `groups.group.name` values, used to
# build the "category | group" badge shown on Instant mode result cards. Ported
# verbatim from the data team's own catList/groupList lookup tables (the same ones
# the reference product renders that badge from) - not independently derived, so a
# raw value with no entry here is passed through unchanged rather than guessed at.
CATEGORY_DISPLAY_LABELS: dict[str, str] = {
    "Account & Audit": "Account & Audit",
    "Bare Act": "Indian Acts & Rules",
    "BILLS": "Bills",
    "COMPANY AND SEBI": "Company Law",
    "DIRECT TAX LAWS": "Income Tax",
    "FEMA BANKING INSURANCE": "Fema & Banking",
    "GOODS & SERVICES TAX": "Excise/ST/VAT",
    "GST New": "GST",
    "International Taxation": "International Tax",
    "IBC": "Insolvency & Bankruptcy Code",
    "SERVICE TAX LAWS": "Excise/ST/VAT",
    "Competition Law": "Competition Law",
    "Transfer Pricing": "Transfer Pricing",
    "COMPETITION ACT": "Competition Law",
    "CORPORATE LAWS": "Corporate Laws",
    "International Tax": "International Tax",
    "Labour Laws": "Labour Laws",
}

GROUP_DISPLAY_LABELS: dict[str, str] = {
    "ACT": "Acts",
    "Advance Ruling": "Advance Ruling",
    "BILL": "Bills",
    "CASELAWS": "Case Laws",
    "CBDT": "CBDT on Finance Acts",
    "CIRNOT": "Circulars & Notifications",
    "COMMENTARY": "Commentaries",
    "DTA": "Treaties",
    "DTC": "Direct Tax Code",
    "Experts Opinion": "Articles",
    "Featured story": "Analysis",
    "Foreign Companies": "Foreign Companies",
    "Guidelines": "Guidelines",
    "News": "News",
    "Non Resident": "Non Resident",
    "PRACTICE & PROCEDURE": "Practice & Procedure",
    "REPORT": "Reports",
    "RULE": "Rules",
    "Transfer Pricing": "Transfer Pricing",
    "Witholding Tax": "Witholding Tax",
    "AAA Model Report": "AAA Model Report",
    "AAA Group": "Account Standard",
    "Act of Parliament/Amendment Act": "Act of Parliament",
    "FAQs": "FAQs",
    "Video & Presentation": "Video & Presentation",
    "Companies Act Topics": "Companies Act Topics",
    "LISTING/INFORMAL REPORT": "Informal Guidelines",
    "STANDARDS": "Standards",
    "Ordinances": "Ordinance",
    "COMPANY ALLIED LAWS": "Company Allied Laws",
    "Indian Constitution": "Indian Constitution",
    "form": "Forms",
    "CIRCULAR": "Notification",
    "NOTIFICATION": "Forms",
    "Standard & Guidance notes": "Standard",
    "ITFG Opinions": "ITFG Opinions",
    "IND AS Simplified": "IND AS Simplified",
    "Queries": "Queries",
    "AAAOther": "AAA Other",
    "Financials and Disclosures": "Financials and Disclosures",
    "Tariff": "Tariff",
}

MASTERINFO_CITATION_FIELDS = [
    "heading",
    "subheading",
    "masterinfo.citations",
    "masterinfo.info.court",
    "masterinfo.info.bench",
    "otherinfo.judge",
    "otherinfo.partyname",
]

# Maps each intent category tag to the Milvus collection(s) it routes to. "tariff"
# has no entry - tariff_section isn't in MILVUS_COLLECTIONS yet (parked in the
# ingestion pipeline's _disabled_collections, not live) - a tariff-only intent tag
# falls through collections_for_intent's fallback instead. "caselaws" maps to the
# original 7 collections including metadata - its fields (landmark_ruling, doc-level
# heading/subheading) are case-doc-specific, not a generic cross-category collection.
CATEGORY_COLLECTIONS: dict[str, list[str]] = {
    "caselaws": ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"],
    "acts": ["act_section"],
    "rules": ["rule_section"],
    "articles": ["article_section"],
    "commentary": ["commentary_section"],
}


def collections_for_intent(intent: list[str]) -> list[str]:
    """Which Milvus collections to search for a given intent category list.
    Empty/unrecognized-only intent (nothing confidently tagged, a tariff-only
    tag, or a value CATEGORY_COLLECTIONS has no entry for) falls back to
    searching every collection - never worse than the old always-search-
    everything behavior. Multi-category intent unions its groups. Return
    order follows MILVUS_COLLECTIONS's own order, not intent's tag order, so
    trace/log output stays collection-order-stable regardless of tag order.
    """
    if not intent:
        return list(MILVUS_COLLECTIONS)
    routed = {collection for tag in intent for collection in CATEGORY_COLLECTIONS.get(tag, [])}
    return [c for c in MILVUS_COLLECTIONS if c in routed] or list(MILVUS_COLLECTIONS)
