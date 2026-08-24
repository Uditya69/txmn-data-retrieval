from common.legal_lexicon import KNOWN_COURT_FULL_NAMES
from common.schemas import MILVUS_COLLECTIONS

COLLECTION_DESCRIPTIONS = {
    "case_summary": "one-paragraph summary of the whole case",
    "digest": "longer case digest - repeats the headnote text then adds a HELD-reasoning section; typically 3-5x longer than headnotes, not a short excerpt",
    "headnotes": "editorial headnotes summarizing the legal question and outcome",
    "facts": "factual narrative of the case",
    "held": "the court's holding/ratio - what was decided",
    "ruling": "full judgment opinion text (verbatim, chunked) - judge's fact recap, framing of legal questions, and reasoning, not just a final order",
    "metadata": "document-level heading and subheading text",
    "act_section": "statute section text from Acts (e.g. Income-tax Act, CGST Act)",
    "rule_section": "statute section text from Rules made under an Act",
    "article_section": "editorial article/commentary text on tax topics",
    "commentary_section": "editorial commentary text explaining provisions",
}

KNOWN_FILTER_FIELDS = ["court", "act", "section", "party", "date_range", "bench", "judge"]

KNOWN_COURTS = KNOWN_COURT_FULL_NAMES

def build_schema_context() -> str:
    collection_lines = "\n".join(
        f"- {name}: {COLLECTION_DESCRIPTIONS[name]}" for name in MILVUS_COLLECTIONS
    )
    return (
        "Searchable collections (searched together unless routed by category, "
        "phrase search_query to read naturally against each of them):\n"
        f"{collection_lines}\n\n"
        f"Recognized filter fields: {', '.join(KNOWN_FILTER_FIELDS)}\n"
        f"Common courts: {', '.join(KNOWN_COURTS[:3])}, etc.\n"
        "These lists are not exhaustive. If the query names a court/act not listed, "
        "preserve the user's exact wording; never guess or canonicalize its value."
    )
