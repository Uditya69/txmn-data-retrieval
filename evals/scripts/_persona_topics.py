"""Shared topic definitions for the rigorous keyword-expansion persona eval.

Each topic gets a synthetic user_id (prefixed eval-persona- so it's trivially
identifiable/cleanable from the shared dev Mongo) and a sequence of realistic
queries fed through the REAL persona pipeline (extract_query_understanding ->
embed -> record_query_event) so the resulting rendered persona_context is
genuinely pipeline-derived, not hand-authored to match what the eval expects.

Each query names the same specific anchor (Act/section, or "India"/
"Indian constitution" for art14) so persona.clustering's entity-overlap-
weighted similarity reliably merges all 4 events into ONE growing topic
(confirmed empirically necessary: first draft varied phrasing enough that
several topics fragmented into multiple single-event "discovered" topics
instead of accumulating toward "active" - see build_persona_test_snapshots.py
run log). Phrasing still varies naturally around that fixed anchor - these
are not copies of the bare test query or the gold hint used later in the
eval, and are never used as the persona_context text itself (that comes from
the real pipeline's own render_persona_context output).
"""

TOPICS = [
    {
        "key": "sec54",
        "user_id": "eval-persona-sec54",
        "categories": ["acts"],
        "queries": [
            "Section 54 Income-tax Act capital gains exemption when I sell my house and buy another",
            "Section 54 Income-tax Act - how long do I have to reinvest sale proceeds",
            "Section 54 Income-tax Act exemption if the new house is still under construction",
            "Section 54 Income-tax Act capital gains account scheme deadline",
        ],
    },
    {
        "key": "sec43b",
        "user_id": "eval-persona-sec43b",
        "categories": ["acts"],
        "queries": [
            "Section 43B Income-tax Act disallowance for unpaid GST liability as business expense",
            "Section 43B Income-tax Act - deduction for bonus not paid before year end",
            "Section 43B Income-tax Act deduction if statutory dues paid after due date",
            "Section 43B Income-tax Act disallowance of unpaid dues at year end",
        ],
    },
    {
        "key": "gst_rule6",
        "user_id": "eval-persona-gstrule6",
        "categories": ["rules"],
        "queries": [
            "CGST Rules Rule 6 conditions for claiming input tax credit",
            "CGST Rules Rule 6 - when does ITC need to be reversed",
            "CGST Rules Rule 6 credit on capital goods used partly for exempt supplies",
            "CGST Rules Rule 6 reversing input credit on non-payment to supplier",
        ],
    },
    {
        "key": "sec80hh",
        "user_id": "eval-persona-sec80hh",
        "categories": ["acts"],
        "queries": [
            "Section 80HH Income-tax Act deduction for setting up a new factory in a backward area",
            "Section 80HH Income-tax Act deduction for profits from a newly established industrial undertaking",
            "Section 80HH Income-tax Act eligibility conditions for backward area industrial deduction",
            "Section 80HH Income-tax Act - how many years can I claim the deduction",
        ],
    },
    {
        "key": "sec61",
        "user_id": "eval-persona-sec61",
        "categories": ["acts"],
        "queries": [
            "Section 61 Income-tax Act clubbing of income for a revocable transfer of assets",
            "Section 61 Income-tax Act - if I can take the asset back later, who pays tax on the income",
            "Section 61 Income-tax Act revocable trust income taxed to whom",
            "Section 61 Income-tax Act income from asset transferred with a right to reclaim it",
        ],
    },
    {
        "key": "art14",
        "user_id": "eval-persona-art14",
        "categories": ["articles"],
        "queries": [
            "Article 14 Constitution of India right to equality case law",
            "Article 14 Constitution of India - what counts as discrimination",
            "Article 14 Constitution of India equal protection of laws landmark judgments",
            "Article 14 Constitution of India - can the state treat similarly placed people differently",
        ],
    },
    {
        "key": "trademark",
        "user_id": "eval-persona-trademark",
        "categories": [],
        "queries": [
            "trademark infringement remedies under the Trade Marks Act India",
            "trademark passing off action India - how is it different from infringement",
            "trademark registration process and opposition period India",
            "trademark infringement - how to stop a competitor using a similar brand name",
        ],
    },
]
