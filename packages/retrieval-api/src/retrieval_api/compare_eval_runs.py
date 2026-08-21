import argparse
import json
from pathlib import Path

_STAGES = [
    "es", "raw_dense", "raw_sparse", "rewritten_dense", "rewritten_sparse",
    "rrf", "reranker",
]


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def stage_pass_rate(run: dict, stage: str) -> tuple[int, int]:
    results = run["results"]
    passed = sum(
        1 for r in results
        if r["ranks"].get(stage) is not None and r["ranks"][stage] <= r["pass_at"]
    )
    return passed, len(results)


def citation_pass_rate(run: dict) -> tuple[int, int]:
    results = run["results"]
    passed = sum(1 for r in results if r.get("citation_valid"))
    return passed, len(results)


def build_comparison_table(baseline: dict, candidates: list[dict]) -> list[dict]:
    baseline_stage_passed = {stage: stage_pass_rate(baseline, stage)[0] for stage in _STAGES}
    baseline_citation_passed = citation_pass_rate(baseline)[0]
    table = []
    for candidate in candidates:
        stage_deltas = {
            stage: stage_pass_rate(candidate, stage)[0] - baseline_stage_passed[stage]
            for stage in _STAGES
        }
        citation_delta = citation_pass_rate(candidate)[0] - baseline_citation_passed
        table.append({
            "run_name": candidate["run_name"],
            "stage_deltas": stage_deltas,
            "citation_pass_delta": citation_delta,
        })
    return table


def _print_table(baseline: dict, table: list[dict]) -> None:
    total = len(baseline["results"])
    print(f"Baseline: {baseline['run_name']} ({total} queries)")
    header = "run_name".ljust(28) + "".join(s[:10].rjust(12) for s in _STAGES) + "citation".rjust(12)
    print(header)
    for row in table:
        cells = "".join(f"{row['stage_deltas'][s]:+d}".rjust(12) for s in _STAGES)
        print(row["run_name"].ljust(28) + cells + f"{row['citation_pass_delta']:+d}".rjust(12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval runs against a baseline")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    candidates = [load_run(path) for path in args.candidates]
    table = build_comparison_table(baseline, candidates)
    _print_table(baseline, table)


if __name__ == "__main__":
    main()
