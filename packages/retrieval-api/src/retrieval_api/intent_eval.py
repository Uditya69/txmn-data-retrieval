import argparse
import asyncio
import json
from pathlib import Path

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.eval_io import append_result, filter_pending, load_completed_ids, read_records
from model_gateway.client import GatewayClient


def load_intent_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("intent filter eval dataset must be a non-empty JSON array")
    seen: set[str] = set()
    for case in cases:
        required = {"id", "query", "expected_filters", "expected_categories"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
    return cases


def check_intent_case(
    expected_filters: dict, actual_filters: dict,
    expected_categories: list[str], actual_categories: list[str],
) -> tuple[bool, bool]:
    filters_ok = expected_filters == actual_filters
    categories_ok = set(expected_categories) == set(actual_categories)
    return filters_ok, categories_ok


def tally_intent_records(records: list[dict]) -> dict:
    errors = sum(1 for record in records if record.get("error"))
    passed = sum(1 for record in records if record.get("ok"))
    return {"total": len(records), "errors": errors, "passed": passed, "ran": len(records) - errors}


def print_intent_summary(records: list[dict]) -> None:
    summary = tally_intent_records(records)
    print(f"\n{summary['passed']}/{summary['ran']} passed (errors={summary['errors']})")


async def run(
    model: str | None, dataset_path: str | Path,
    output: Path | None = None, resume: bool = False,
) -> None:
    cases = load_intent_cases(dataset_path)
    records: list[dict] = read_records(output) if (output and resume) else []
    if output and resume:
        cases = filter_pending(cases, load_completed_ids(output))
    gateway = GatewayClient(trace_enabled=False)
    for case in cases:
        try:
            result = await extract_intent(gateway, case["query"], model=model)
        except Exception as exception:
            record = {"id": case["id"], "query": case["query"], "ok": None, "error": f"{exception}"}
            print(f"ERROR {case['id']}: {exception}")
            records.append(record)
            if output:
                append_result(output, record)
            continue
        filters_ok, categories_ok = check_intent_case(
            case["expected_filters"], result["filters"],
            case["expected_categories"], result["intent"],
        )
        ok = filters_ok and categories_ok
        record = {
            "id": case["id"], "query": case["query"], "ok": ok, "error": None,
            "expected_filters": case["expected_filters"], "actual_filters": result["filters"],
            "expected_categories": case["expected_categories"], "actual_categories": result["intent"],
            "reasoning": result.get("reasoning"),
        }
        records.append(record)
        if output:
            append_result(output, record)
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {case['id']}: "
            f"filters(expected={case['expected_filters']} actual={result['filters']} ok={filters_ok}) "
            f"categories(expected={case['expected_categories']} actual={result['intent']} ok={categories_ok})"
        )
    print_intent_summary(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-only intent/filter/category extraction accuracy check")
    parser.add_argument("--model", default=None, help="Override the slm role's model")
    parser.add_argument("--dataset", default="evals/datasets/intent_filter_cases.json")
    parser.add_argument("--output", type=Path, help="append per-case results to this JSONL file as they complete")
    parser.add_argument("--resume", action="store_true", help="skip case IDs already present in --output")
    parser.add_argument("--summarize", type=Path, help="print the summary for an existing JSONL results file and exit - no gateway calls")
    args = parser.parse_args()
    if args.summarize:
        print_intent_summary(read_records(args.summarize))
        return
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    asyncio.run(run(args.model, args.dataset, output=args.output, resume=args.resume))


if __name__ == "__main__":
    main()
