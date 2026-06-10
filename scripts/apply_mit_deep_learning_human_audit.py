from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = Path(
    "eval/mit_deep_learning_stt/candidates/expanded_seed_v1/queries_candidate.csv"
)
DEFAULT_AUDIT_SAMPLE = Path(
    "eval/mit_deep_learning_stt/candidates/expanded_seed_v1/human_audit_sample.csv"
)
DEFAULT_OUTPUT_DIR = Path("eval/mit_deep_learning_stt/candidates/expanded_seed_v1/audited")
VALID_APPLY_STATUSES = {"approved", "corrected"}

FIELDNAMES = [
    "query_id",
    "split",
    "video_id",
    "lecture_title",
    "project_id",
    "project_dir",
    "query_text",
    "query_text_ko",
    "reference_answer",
    "expected_topic",
    "expected_time_hint",
    "expected_visual_hint",
    "gold_start_time",
    "gold_end_time",
    "gold_timestamp_center",
    "gold_modality",
    "gold_visual_entity",
    "gold_entity_link_note",
    "question_type",
    "gold_segment_id",
    "gold_frame_ids",
    "gold_frame_timestamps",
    "gold_visual_entity_ids",
    "gold_visual_entity_texts",
    "linked_entity_count",
    "frame_count",
    "annotator_id",
    "confidence",
    "source",
    "license",
    "notes",
]

AUDITED_FIELD_TO_CANDIDATE_FIELD = {
    "audited_start_time": "gold_start_time",
    "audited_end_time": "gold_end_time",
    "audited_timestamp_center": "gold_timestamp_center",
    "audited_modality": "gold_modality",
    "audited_question_type": "question_type",
    "audited_query_text": "query_text",
    "audited_reference_answer": "reference_answer",
    "audited_confidence": "confidence",
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    result = apply_human_audit(
        candidates_path=resolve_path(args.candidates, repo_root),
        audit_sample_path=resolve_path(args.audit_sample, repo_root),
        output_dir=resolve_path(args.output_dir, repo_root),
        audited_annotator_prefix=args.audited_annotator_prefix,
        min_audited_ratio=args.min_audited_ratio,
    )
    print(json.dumps(public_cli_payload(result), indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply human audit decisions to MIT Deep Learning expanded eval candidates."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--audit-sample", type=Path, default=DEFAULT_AUDIT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audited-annotator-prefix", default="human_")
    parser.add_argument("--min-audited-ratio", type=float, default=0.2)
    return parser


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (repo_root / path).resolve()


def apply_human_audit(
    *,
    candidates_path: Path,
    audit_sample_path: Path,
    output_dir: Path,
    audited_annotator_prefix: str = "human_",
    min_audited_ratio: float = 0.2,
) -> dict[str, Any]:
    candidate_rows = read_csv_rows(candidates_path)
    audit_rows = read_csv_rows(audit_sample_path)
    candidate_by_query_id = {normalize(row.get("query_id")): row for row in candidate_rows}

    duplicate_candidate_ids = duplicate_values(normalize(row.get("query_id")) for row in candidate_rows)
    duplicate_audit_ids = duplicate_values(normalize(row.get("query_id")) for row in audit_rows)
    correction_seconds: list[float] = []
    applied_query_ids: set[str] = set()
    corrected_count = 0
    rejected_count = 0
    invalid_auditor_count = 0
    missing_candidate_count = 0

    for audit_row in audit_rows:
        status = normalize(audit_row.get("audit_status")).lower()
        query_id = normalize(audit_row.get("query_id"))
        if status == "rejected":
            rejected_count += 1
            continue
        if status not in VALID_APPLY_STATUSES:
            continue
        if not query_id or query_id not in candidate_by_query_id:
            missing_candidate_count += 1
            continue
        auditor_id = normalize(audit_row.get("auditor_id"))
        if not is_human_auditor(auditor_id):
            invalid_auditor_count += 1
            continue

        candidate_row = candidate_by_query_id[query_id]
        correction_seconds.append(timestamp_correction_seconds(candidate_row, audit_row))
        apply_audit_row(candidate_row, audit_row, auditor_id, audited_annotator_prefix)
        applied_query_ids.add(query_id)
        if status == "corrected":
            corrected_count += 1

    audited_count = len(applied_query_ids)
    candidate_count = len(candidate_rows)
    audited_ratio = audited_count / candidate_count if candidate_count else 0.0
    mean_correction = (
        sum(correction_seconds) / len(correction_seconds) if correction_seconds else 0.0
    )
    failures = build_failures(
        audited_ratio=audited_ratio,
        min_audited_ratio=min_audited_ratio,
        duplicate_candidate_ids=duplicate_candidate_ids,
        duplicate_audit_ids=duplicate_audit_ids,
        invalid_auditor_count=invalid_auditor_count,
        missing_candidate_count=missing_candidate_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_output = output_dir / "queries_audited_candidate.csv"
    results_json = output_dir / "human_audit_results.json"
    results_md = output_dir / "human_audit_results.md"
    write_csv(queries_output, candidate_rows, FIELDNAMES)

    summary = {
        "ok": not failures,
        "audited_count": audited_count,
        "candidate_count": candidate_count,
        "audited_ratio": audited_ratio,
        "corrected_count": corrected_count,
        "rejected_count": rejected_count,
        "mean_timestamp_correction_seconds": mean_correction,
        "min_audited_ratio": min_audited_ratio,
        "invalid_auditor_count": invalid_auditor_count,
        "missing_candidate_count": missing_candidate_count,
        "public_safe_note": (
            "Human audit results are aggregate-only; query text, reference answers, "
            "local absolute paths, embedding payloads, and credentials are excluded."
        ),
        "failures": failures,
    }
    results_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_md.write_text(human_audit_results_markdown(summary), encoding="utf-8")
    assert_public_safe_summary(summary)
    return {
        **summary,
        "outputs": {
            "queries": queries_output.name,
            "results_json": results_json.name,
            "results_md": results_md.name,
        },
    }


def apply_audit_row(
    candidate_row: dict[str, str],
    audit_row: dict[str, str],
    auditor_id: str,
    audited_annotator_prefix: str,
) -> None:
    for audit_field, candidate_field in AUDITED_FIELD_TO_CANDIDATE_FIELD.items():
        value = normalize(audit_row.get(audit_field))
        if value:
            candidate_row[candidate_field] = value
    candidate_row["expected_time_hint"] = expected_time_hint(candidate_row)
    candidate_row["annotator_id"] = normalized_auditor_id(
        auditor_id,
        audited_annotator_prefix=audited_annotator_prefix,
    )
    candidate_row["notes"] = "human audit applied to expanded evaluation candidate"


def normalized_auditor_id(auditor_id: str, *, audited_annotator_prefix: str) -> str:
    if not audited_annotator_prefix:
        return auditor_id
    if auditor_id.startswith(audited_annotator_prefix):
        return auditor_id
    return f"{audited_annotator_prefix}{auditor_id}"


def timestamp_correction_seconds(
    candidate_row: dict[str, str],
    audit_row: dict[str, str],
) -> float:
    explicit = parse_float(normalize(audit_row.get("timestamp_correction_seconds")))
    if explicit is not None:
        return abs(explicit)
    original_center = parse_float(normalize(candidate_row.get("gold_timestamp_center")))
    audited_center = parse_float(normalize(audit_row.get("audited_timestamp_center")))
    if original_center is None or audited_center is None:
        return 0.0
    return abs(audited_center - original_center)


def expected_time_hint(row: dict[str, str]) -> str:
    start = normalize(row.get("gold_start_time"))
    end = normalize(row.get("gold_end_time"))
    if start and end:
        return f"{start}-{end}s"
    return normalize(row.get("expected_time_hint"))


def build_failures(
    *,
    audited_ratio: float,
    min_audited_ratio: float,
    duplicate_candidate_ids: list[str],
    duplicate_audit_ids: list[str],
    invalid_auditor_count: int,
    missing_candidate_count: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if audited_ratio < min_audited_ratio:
        failures.append(
            {
                "code": "min_audited_ratio_not_met",
                "message": "human-audited row ratio is below the required threshold",
                "observed": audited_ratio,
                "required": min_audited_ratio,
            }
        )
    if duplicate_candidate_ids:
        failures.append(
            {
                "code": "duplicate_candidate_query_id",
                "message": "candidate CSV contains duplicate query_id values",
                "duplicate_count": len(duplicate_candidate_ids),
            }
        )
    if duplicate_audit_ids:
        failures.append(
            {
                "code": "duplicate_audit_query_id",
                "message": "audit sample contains duplicate query_id values",
                "duplicate_count": len(duplicate_audit_ids),
            }
        )
    if invalid_auditor_count:
        failures.append(
            {
                "code": "invalid_auditor_id",
                "message": "approved or corrected audit rows must have non-codex auditor_id values",
                "invalid_count": invalid_auditor_count,
            }
        )
    if missing_candidate_count:
        failures.append(
            {
                "code": "audit_query_id_missing_from_candidates",
                "message": "approved or corrected audit rows referenced missing candidate query_id values",
                "missing_count": missing_candidate_count,
            }
        )
    return failures


def human_audit_results_markdown(summary: dict[str, Any]) -> str:
    mean_correction = summary["mean_timestamp_correction_seconds"]
    return "\n".join(
        [
            "## Human Audit Results",
            "",
            (
                f"Human audit applied {summary['audited_count']} audited label rows "
                f"out of {summary['candidate_count']} expanded candidates "
                f"({summary['audited_ratio']:.3f} audited ratio)."
            ),
            "",
            f"- Corrected audited labels: {summary['corrected_count']}",
            f"- Rejected audit rows excluded from the public dataset: {summary['rejected_count']}",
            (
                "- Timestamp correction policy: prefer numeric "
                "`timestamp_correction_seconds` when supplied; otherwise use the absolute "
                "delta between audited and candidate timestamp centers."
            ),
            f"- Mean timestamp correction seconds: {mean_correction:.3f}",
            "- Public-safe note: aggregate counts only; query and reference text are excluded.",
            "",
        ]
    )


def public_cli_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result["ok"],
        "audited_count": result["audited_count"],
        "candidate_count": result["candidate_count"],
        "audited_ratio": result["audited_ratio"],
        "corrected_count": result["corrected_count"],
        "rejected_count": result["rejected_count"],
        "mean_timestamp_correction_seconds": result[
            "mean_timestamp_correction_seconds"
        ],
        "outputs": result["outputs"],
        "failures": result["failures"],
        "public_safe_note": result["public_safe_note"],
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def duplicate_values(values: Any) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def is_human_auditor(auditor_id: str) -> bool:
    return bool(auditor_id) and not auditor_id.startswith("codex_")


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def normalize(value: object) -> str:
    return str(value or "").strip()


def assert_public_safe_summary(summary: dict[str, Any]) -> None:
    serialized = json.dumps(summary, sort_keys=True).lower()
    forbidden = (
        "/users/",
        "/private/",
        "bearer ",
        "auth token",
        "raw vector",
        "query_text",
        "reference_answer",
    )
    for token in forbidden:
        if token in serialized:
            raise AssertionError(f"summary contains forbidden public output token: {token}")


if __name__ == "__main__":
    sys.exit(main())
