from common.schemas import MILVUS_COLLECTIONS

COLLECTION_DESCRIPTIONS = {
    "case_summary": "one-paragraph summary of the whole case",
    "digest": "short digest points extracted from the judgment",
    "headnotes": "editorial headnotes summarizing the legal question and outcome",
    "facts": "factual narrative of the case",
    "held": "the court's holding/ratio - what was decided",
    "ruling": "the final operative ruling/order text",
    "metadata": "document-level heading and subheading text",
}

KNOWN_FILTER_FIELDS = ["court", "act", "section", "party", "date_range"]

KNOWN_COURTS = [
    "Supreme Court", "Delhi High Court", "Bombay High Court", "Madras High Court",
    "Calcutta High Court", "Karnataka High Court", "Gujarat High Court",
    "Income Tax Appellate Tribunal", "Customs Excise and Service Tax Appellate Tribunal",
]

KNOWN_ACTS = [
    "Income-tax Act, 1961", "CGST Act", "IGST Act", "Customs Act, 1962",
    "Bharatiya Nyaya Sanhita", "Bharatiya Nagarik Suraksha Sanhita",
    "Bharatiya Sakshya Adhiniyam", "Indian Penal Code", "Code of Criminal Procedure",
    "Indian Evidence Act",
]


def build_schema_context() -> str:
    collection_lines = "\n".join(
        f"- {name}: {COLLECTION_DESCRIPTIONS[name]}" for name in MILVUS_COLLECTIONS
    )
    return (
        "Searchable collections (all are searched together, phrase rewritten_query "
        "to read naturally against each of them):\n"
        f"{collection_lines}\n\n"
        f"Recognized filter fields: {', '.join(KNOWN_FILTER_FIELDS)}\n"
        f"Common courts: {', '.join(KNOWN_COURTS)}\n"
        f"Common acts: {', '.join(KNOWN_ACTS)}\n"
        "These lists are not exhaustive. If the query names a court/act not listed, "
        "preserve the user's exact wording; never guess or canonicalize its value."
    )
