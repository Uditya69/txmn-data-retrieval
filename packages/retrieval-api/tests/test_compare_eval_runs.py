import json
from pathlib import Path

import pytest

from retrieval_api.compare_eval_runs import (
    build_comparison_table, citation_pass_rate, load_run, stage_pass_rate,
)


def _run(name, ranks_list, citation_valids):
    return {
        "run_name": name,
        "parameters": {"slm_model": None, "reranker_model": None, "synthesis_model": None},
        "results": [
            {"id": f"Q{i}", "pass_at": 5, "ranks": ranks, "citation_valid": valid}
            for i, (ranks, valid) in enumerate(zip(ranks_list, citation_valids))
        ],
    }


def test_load_run_reads_json_file(tmp_path):
    path = tmp_path / "run.json"
    payload = _run("baseline", [{"es": 1}], [True])
    path.write_text(json.dumps(payload))

    assert load_run(path) == payload


def test_stage_pass_rate_counts_ranks_within_pass_at():
    run = _run("baseline", [{"es": 1}, {"es": 10}, {"es": None}], [True, True, True])

    passed, total = stage_pass_rate(run, "es")

    assert (passed, total) == (1, 3)


def test_citation_pass_rate_counts_valid_flags():
    run = _run("baseline", [{}, {}, {}], [True, False, True])

    passed, total = citation_pass_rate(run)

    assert (passed, total) == (2, 3)


def test_build_comparison_table_reports_delta_vs_baseline():
    baseline = _run("baseline", [{"es": 1}], [True])
    candidate = _run("candidate", [{"es": 10}], [False])

    table = build_comparison_table(baseline, [candidate])

    assert table == [{
        "run_name": "candidate",
        "stage_deltas": {"es": -1, "raw_dense": 0, "raw_sparse": 0, "rewritten_dense": 0, "rewritten_sparse": 0, "rrf": 0, "reranker": 0, "agentic": 0},
        "citation_pass_delta": -1,
    }]
