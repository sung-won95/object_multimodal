from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from oarag.evaluation.remaining_miss_audit import (  # noqa: E402
    build_remaining_miss_audit,
    load_jsonl,
    synthetic_remaining_miss_rows,
    write_remaining_miss_audit_outputs,
)


DEFAULT_QUERY_RESULTS = (
    REPO_ROOT
    / "reports"
    / "mit_deep_learning_eval"
    / "validation_4suite_20260610"
    / "query_results.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "issue_256_remaining_miss_audit"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    query_results_path = args.query_results.expanduser()
    depth_diagnostics_path = args.depth_diagnostics.expanduser() if args.depth_diagnostics else None
    input_status = "loaded"
    unrun_reason = None
    if query_results_path.exists():
        query_rows = load_jsonl(query_results_path)
        depth_rows = load_jsonl(depth_diagnostics_path) if depth_diagnostics_path else []
    else:
        query_rows, depth_rows = synthetic_remaining_miss_rows()
        input_status = "synthetic_fixture"
        unrun_reason = (
            "1-4 lecture benchmark query_results were not available in this isolated "
            "worktree; wrote a synthetic fixture smoke report instead."
        )

    audit = build_remaining_miss_audit(
        query_rows=query_rows,
        depth_rows=depth_rows,
        run_label=args.run_label,
        input_status=input_status,
        unrun_reason=unrun_reason,
    )
    artifacts = write_remaining_miss_audit_outputs(
        output_dir=args.output_dir.expanduser(),
        audit=audit,
    )
    print(
        "Wrote public-safe remaining miss audit: "
        f"{artifacts['audit']}, {artifacts['metrics']}, {artifacts['summary']}"
    )
    if unrun_reason:
        print(f"Note: {unrun_reason}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit evidence_unit_candidate not_found benchmark seeds with public-safe "
            "rank, match, quality, search-field, and Meili-depth buckets."
        )
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        default=DEFAULT_QUERY_RESULTS,
        help=(
            "Benchmark query_results.jsonl from a 1-4 lecture retrieval_answer_matrix run. "
            "If absent, a synthetic fixture smoke report is written."
        ),
    )
    parser.add_argument(
        "--depth-diagnostics",
        type=Path,
        help=(
            "Optional public-safe JSONL keyed by query_id with meili_depth_found, "
            "target_candidate_quality_bucket, and target_search_field_coverage buckets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit.json, metrics.json, and summary.md.",
    )
    parser.add_argument(
        "--run-label",
        default="issue_256_remaining_miss_audit",
        help="Public-safe run label stored in the audit payload.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
