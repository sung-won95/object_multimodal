from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES_PATH = REPO_ROOT / "eval" / "mit_deep_learning_stt" / "queries.csv"
DEFAULT_README_PATH = REPO_ROOT / "eval" / "mit_deep_learning_stt" / "README.md"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "eval" / "mit_deep_learning_stt" / "benchmark_matrix_manifest.json"
SCHEMA_VERSION = "mit-deep-learning-eval-dataset-validation-v1"

REQUIRED_QUESTION_TYPES = {
    "concept_explanation",
    "definition_lookup",
    "multimodal_grounded",
}
README_AUDIT_KEYWORDS = (
    "human audit",
    "audited label",
    "timestamp correction",
    "mean timestamp correction",
)
FORBIDDEN_OUTPUT_FIELDS = (
    "query_text",
    "query_text_ko",
    "reference_answer",
    "transcript",
    "transcript_text",
    "transcript_excerpt",
)
REQUIRED_TIMESTAMP_FIELDS = (
    "gold_start_time",
    "gold_end_time",
    "gold_timestamp_center",
)
OPTIONAL_TIMESTAMP_FIELDS = (
    "audited_start_time",
    "audited_end_time",
    "audited_timestamp_center",
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = validate_dataset(
        queries_path=args.queries,
        readme_path=args.readme,
        expected_video_ids=read_expected_video_ids(args.manifest),
        min_queries=args.min_queries,
        min_videos=args.min_videos,
        min_per_video=args.min_per_video,
        min_human_audit_ratio=args.min_human_audit_ratio,
        require_readme_audit_summary=args.require_readme_audit_summary,
    )
    assert_public_safe(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the MIT Deep Learning evaluation query CSV before closing "
            "the human-audit completion gate."
        )
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Benchmark matrix manifest whose suite video_id values define the required "
            "24 MIT lecture videos."
        ),
    )
    parser.add_argument("--min-queries", type=int, default=240)
    parser.add_argument("--min-videos", type=int, default=24)
    parser.add_argument("--min-per-video", type=int, default=10)
    parser.add_argument("--min-human-audit-ratio", type=float, default=0.2)
    parser.add_argument(
        "--require-readme-audit-summary",
        action="store_true",
        help="Require README.md to document human audit results and timestamp corrections.",
    )
    return parser


def validate_dataset(
    *,
    queries_path: Path,
    readme_path: Path,
    expected_video_ids: set[str] | None = None,
    min_queries: int = 240,
    min_videos: int = 24,
    min_per_video: int = 10,
    min_human_audit_ratio: float = 0.2,
    require_readme_audit_summary: bool = False,
) -> dict[str, Any]:
    resolved_queries_path = queries_path.expanduser().resolve()
    resolved_readme_path = readme_path.expanduser().resolve()
    rows = read_csv_rows(resolved_queries_path)
    failures: list[dict[str, Any]] = []

    row_count = len(rows)
    video_counts = Counter(normalize(row.get("video_id")) for row in rows)
    video_counts.pop("", None)
    observed_video_ids = set(video_counts)
    expected_video_ids = set(expected_video_ids or set())
    split_counts = Counter(normalize(row.get("split")) for row in rows)
    split_counts.pop("", None)
    question_type_counts = Counter(normalize(row.get("question_type")) for row in rows)
    question_type_counts.pop("", None)
    annotator_stats = summarize_annotators(rows)
    timestamp_stats = validate_timestamps(rows)
    readme_stats = validate_readme_audit_summary(
        readme_path=resolved_readme_path,
        required=require_readme_audit_summary,
    )

    if row_count < min_queries:
        failures.append(
            {
                "code": "min_queries_not_met",
                "message": "query row count is below the completion threshold",
                "observed": row_count,
                "required": min_queries,
            }
        )

    unique_video_count = len(video_counts)
    if unique_video_count < min_videos:
        failures.append(
            {
                "code": "min_videos_not_met",
                "message": "not all required lecture videos are represented",
                "observed": unique_video_count,
                "required": min_videos,
            }
        )
    missing_expected_video_ids = sorted(expected_video_ids.difference(observed_video_ids))
    unexpected_video_ids = sorted(observed_video_ids.difference(expected_video_ids))
    if missing_expected_video_ids:
        failures.append(
            {
                "code": "expected_videos_missing",
                "message": "one or more manifest-required lecture video IDs are missing",
                "missing_video_ids": missing_expected_video_ids,
                "missing_count": len(missing_expected_video_ids),
            }
        )
    underfilled_video_count = sum(1 for count in video_counts.values() if count < min_per_video)
    if underfilled_video_count:
        failures.append(
            {
                "code": "min_per_video_not_met",
                "message": "one or more videos have too few query rows",
                "underfilled_video_count": underfilled_video_count,
                "required_per_video": min_per_video,
                "minimum_observed_per_video": min(video_counts.values()) if video_counts else 0,
            }
        )

    if annotator_stats["human_audit_ratio"] < min_human_audit_ratio:
        failures.append(
            {
                "code": "min_human_audit_ratio_not_met",
                "message": "human-audited row ratio is below the completion threshold",
                "observed": annotator_stats["human_audit_ratio"],
                "required": min_human_audit_ratio,
            }
        )

    if annotator_stats["all_non_empty_annotators_codex"]:
        failures.append(
            {
                "code": "all_annotators_codex_generated",
                "message": "all non-empty annotator_id values are codex-generated labels",
            }
        )

    if timestamp_stats["invalid_count"]:
        failures.append(
            {
                "code": "invalid_timestamps",
                "message": "timestamp fields must be numeric and end times must be >= start times",
                "invalid_count": timestamp_stats["invalid_count"],
                "sample_issue_codes": timestamp_stats["sample_issue_codes"],
                "sample_query_ids": timestamp_stats["sample_query_ids"],
            }
        )

    split_result = validate_split_balance(split_counts=split_counts, row_count=row_count)
    if not split_result["ok"]:
        failures.append(split_result["failure"])

    missing_question_types = sorted(REQUIRED_QUESTION_TYPES.difference(question_type_counts))
    if missing_question_types:
        failures.append(
            {
                "code": "required_question_types_missing",
                "message": "required question_type categories are missing",
                "missing_question_types": missing_question_types,
            }
        )

    if require_readme_audit_summary and not readme_stats["ok"]:
        failures.append(readme_stats["failure"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": not failures,
        "inputs": {
            "queries": resolved_queries_path.name,
            "readme": resolved_readme_path.name,
        },
        "thresholds": {
            "min_queries": min_queries,
            "min_videos": min_videos,
            "min_per_video": min_per_video,
            "min_human_audit_ratio": min_human_audit_ratio,
            "split_min_ratio": 0.35,
            "required_question_types": sorted(REQUIRED_QUESTION_TYPES),
            "require_readme_audit_summary": require_readme_audit_summary,
        },
        "summary": {
            "row_count": row_count,
            "unique_video_count": unique_video_count,
            "expected_video_count": len(expected_video_ids),
            "missing_expected_video_count": len(missing_expected_video_ids),
            "unexpected_video_count": len(unexpected_video_ids),
            "min_rows_per_video": min(video_counts.values()) if video_counts else 0,
            "max_rows_per_video": max(video_counts.values()) if video_counts else 0,
            "underfilled_video_count": underfilled_video_count,
            "split_counts": dict(sorted(split_counts.items())),
            "question_type_counts": dict(sorted(question_type_counts.items())),
            "timestamp_invalid_count": timestamp_stats["invalid_count"],
            "human_audited_count": annotator_stats["human_audited_count"],
            "codex_annotated_count": annotator_stats["codex_annotated_count"],
            "empty_annotator_count": annotator_stats["empty_annotator_count"],
            "human_audit_ratio": annotator_stats["human_audit_ratio"],
            "all_non_empty_annotators_codex": annotator_stats[
                "all_non_empty_annotators_codex"
            ],
            "readme_audit_summary": readme_stats["summary"],
        },
        "privacy": {
            "question_and_answer_text": "excluded",
            "lecture_text": "excluded",
            "local_paths": "excluded",
        },
        "failures": failures,
    }
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"query CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_expected_video_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {resolved_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    suites = payload.get("suites")
    if not isinstance(suites, list):
        raise ValueError(f"benchmark manifest missing suites list: {resolved_path}")
    video_ids = {
        normalize(suite.get("video_id"))
        for suite in suites
        if isinstance(suite, dict) and normalize(suite.get("video_id"))
    }
    if not video_ids:
        raise ValueError(f"benchmark manifest has no suite video_id values: {resolved_path}")
    return video_ids


def summarize_annotators(rows: list[dict[str, str]]) -> dict[str, Any]:
    annotator_ids = [normalize(row.get("annotator_id")) for row in rows]
    non_empty = [annotator_id for annotator_id in annotator_ids if annotator_id]
    codex_count = sum(1 for annotator_id in non_empty if annotator_id.startswith("codex_"))
    human_count = sum(
        1 for annotator_id in non_empty if not annotator_id.startswith("codex_")
    )
    return {
        "human_audited_count": human_count,
        "codex_annotated_count": codex_count,
        "empty_annotator_count": len(annotator_ids) - len(non_empty),
        "human_audit_ratio": human_count / len(rows) if rows else 0.0,
        "all_non_empty_annotators_codex": bool(non_empty) and codex_count == len(non_empty),
    }


def validate_timestamps(rows: list[dict[str, str]]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        values: dict[str, float] = {}
        for field in REQUIRED_TIMESTAMP_FIELDS:
            raw_value = normalize(row.get(field))
            value = parse_float(raw_value)
            if value is None:
                issues.append(
                    {
                        "query_id": public_query_id(row, index),
                        "issue_code": f"{field}_not_numeric",
                    }
                )
            else:
                values[field] = value
        for field in OPTIONAL_TIMESTAMP_FIELDS:
            raw_value = normalize(row.get(field))
            if raw_value:
                value = parse_float(raw_value)
                if value is None:
                    issues.append(
                        {
                            "query_id": public_query_id(row, index),
                            "issue_code": f"{field}_not_numeric",
                        }
                    )
                else:
                    values[field] = value
        if values.get("gold_start_time") is not None and values.get("gold_end_time") is not None:
            if values["gold_end_time"] < values["gold_start_time"]:
                issues.append(
                    {
                        "query_id": public_query_id(row, index),
                        "issue_code": "gold_end_before_start",
                    }
                )
        if values.get("audited_start_time") is not None and values.get("audited_end_time") is not None:
            if values["audited_end_time"] < values["audited_start_time"]:
                issues.append(
                    {
                        "query_id": public_query_id(row, index),
                        "issue_code": "audited_end_before_start",
                    }
                )
    return {
        "invalid_count": len(issues),
        "sample_issue_codes": [issue["issue_code"] for issue in issues[:5]],
        "sample_query_ids": [issue["query_id"] for issue in issues[:5]],
    }


def validate_split_balance(*, split_counts: Counter[str], row_count: int) -> dict[str, Any]:
    dev_count = split_counts.get("dev", 0)
    test_count = split_counts.get("test", 0)
    dev_ratio = dev_count / row_count if row_count else 0.0
    test_ratio = test_count / row_count if row_count else 0.0
    if dev_ratio >= 0.35 and test_ratio >= 0.35:
        return {"ok": True}
    return {
        "ok": False,
        "failure": {
            "code": "dev_test_split_unbalanced",
            "message": "dev and test splits must both exist and each cover at least 35% of rows",
            "dev_ratio": dev_ratio,
            "test_ratio": test_ratio,
            "dev_count": dev_count,
            "test_count": test_count,
        },
    }


def validate_readme_audit_summary(*, readme_path: Path, required: bool) -> dict[str, Any]:
    if not required:
        return {
            "ok": True,
            "summary": {
                "required": False,
                "present": None,
                "section_present": None,
                "matched_keywords": [],
            },
        }
    if not readme_path.exists():
        return {
            "ok": False,
            "summary": {
                "required": True,
                "present": False,
                "section_present": False,
                "matched_keywords": [],
            },
            "failure": {
                "code": "readme_audit_summary_missing",
                "message": "README audit summary is required but README file was not found",
            },
        }
    text = readme_path.read_text(encoding="utf-8")
    section = extract_human_audit_results_section(text)
    section_present = section is not None
    section_text = (section or "").lower()
    matched = [keyword for keyword in README_AUDIT_KEYWORDS if keyword in section_text]
    ok = section_present and "human audit" in matched and any(
        keyword in matched
        for keyword in ("audited label", "timestamp correction", "mean timestamp correction")
    )
    return {
        "ok": ok,
        "summary": {
            "required": True,
            "present": ok,
            "section_present": section_present,
            "matched_keywords": matched,
        },
        "failure": {
            "code": "readme_audit_summary_missing",
            "message": (
                "README must include a Human Audit Results section summarizing audited labels "
                "and timestamp correction status before this gate can close"
            ),
            "matched_keywords": matched,
        },
    }


def extract_human_audit_results_section(text: str) -> str | None:
    lines = text.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        normalized = line.strip().lower()
        if normalized in {"## human audit result", "## human audit results"}:
            start_index = index
            break
    if start_index is None:
        return None
    section_lines: list[str] = []
    for line in lines[start_index + 1 :]:
        if line.startswith("## "):
            break
        section_lines.append(line)
    return "\n".join(section_lines)


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def public_query_id(row: dict[str, str], index: int) -> str:
    return normalize(row.get("query_id")) or f"row_{index}"


def normalize(value: object) -> str:
    return str(value or "").strip()


def assert_public_safe(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    lowered = serialized.lower()
    for forbidden in FORBIDDEN_OUTPUT_FIELDS:
        if forbidden in lowered:
            raise AssertionError(f"validator output contains forbidden token: {forbidden}")


if __name__ == "__main__":
    sys.exit(main())
