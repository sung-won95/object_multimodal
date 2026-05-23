from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oarag.core.io import write_json


REPORT_SCHEMA_VERSION = "evaluation-report-v1"
PAPER_TABLE_COLUMNS = [
    "suite_id",
    "suite_type",
    "domain",
    "condition",
    "query_count",
    "hit_at_5s",
    "hit_at_10s",
    "hit_at_15s",
    "mrr_at_max_delta",
    "top1_mean_abs_error",
    "frame_backed_ratio",
    "linked_entity_backed_ratio",
    "grounded_answer_ratio",
    "citation_coverage_ratio",
    "answer_citation_precision",
    "answer_citation_recall",
    "expected_citation_hit_ratio",
    "mean_unsupported_claim_count",
    "unsupported_claim_ratio",
    "mean_processing_time_ms",
]


@dataclass(frozen=True)
class EvaluationReport:
    output_dir: Path
    paper_table_csv_path: Path
    paper_table_markdown_path: Path
    reproducibility_json_path: Path
    reproducibility_markdown_path: Path
    paper_table_rows: list[dict[str, Any]]
    reproducibility: dict[str, Any]


def generate_evaluation_report(
    *,
    metrics_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    command: list[str] | None = None,
    commit_sha: str | None = None,
) -> EvaluationReport:
    resolved_metrics_path = metrics_path.expanduser().resolve()
    metrics = _read_json(resolved_metrics_path)
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else resolved_metrics_path.parent / "paper_report"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = paper_table_rows(metrics)
    paper_table_csv_path = resolved_output_dir / "paper_table.csv"
    paper_table_markdown_path = resolved_output_dir / "paper_table.md"
    reproducibility_json_path = resolved_output_dir / "reproducibility.json"
    reproducibility_markdown_path = resolved_output_dir / "reproducibility.md"

    _write_paper_table_csv(paper_table_csv_path, table_rows)
    paper_table_markdown_path.write_text(
        paper_table_markdown(metrics=metrics, rows=table_rows),
        encoding="utf-8",
    )
    reproducibility = reproducibility_payload(
        metrics=metrics,
        metrics_artifact=resolved_metrics_path.name,
        table_artifacts=[paper_table_csv_path.name, paper_table_markdown_path.name],
        command=command or [],
        commit_sha=commit_sha or _git_commit(repo_root),
    )
    write_json(reproducibility_json_path, reproducibility)
    reproducibility_markdown_path.write_text(
        reproducibility_markdown(reproducibility),
        encoding="utf-8",
    )

    return EvaluationReport(
        output_dir=resolved_output_dir,
        paper_table_csv_path=paper_table_csv_path,
        paper_table_markdown_path=paper_table_markdown_path,
        reproducibility_json_path=reproducibility_json_path,
        reproducibility_markdown_path=reproducibility_markdown_path,
        paper_table_rows=table_rows,
        reproducibility=reproducibility,
    )


def paper_table_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in metrics.get("suites", []):
        if not isinstance(suite, dict):
            continue
        variants = [item for item in suite.get("variants", []) if isinstance(item, dict)]
        modes = [item for item in suite.get("modes", []) if isinstance(item, dict)]
        if variants:
            for variant in variants:
                rows.append(
                    _paper_table_row(
                        suite=suite,
                        item=variant,
                        condition=str(variant.get("variant_id") or variant.get("label") or ""),
                    )
                )
        elif modes:
            for mode in modes:
                rows.append(
                    _paper_table_row(
                        suite=suite,
                        item=mode,
                        condition=str(mode.get("mode") or ""),
                    )
                )
        else:
            rows.append(_paper_table_row(suite=suite, item=suite, condition="suite"))
    return rows


def paper_table_markdown(*, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Paper Evaluation Table: {metrics.get('run_id')}",
        "",
        "| suite | condition | domain | queries | Hit@10s | MRR | top1 err | frame | linked | grounded | cite P | cite R | expected hit | unsupported | latency ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {suite_id} | {condition} | {domain} | {query_count} | {hit10} | {mrr} | "
            "{top1} | {frame} | {linked} | {grounded} | {precision} | {recall} | {hit} | "
            "{unsupported} | {latency} |".format(
                suite_id=row.get("suite_id"),
                condition=row.get("condition"),
                domain=row.get("domain"),
                query_count=row.get("query_count"),
                hit10=_format_metric(row.get("hit_at_10s")),
                mrr=_format_metric(row.get("mrr_at_max_delta")),
                top1=_format_metric(row.get("top1_mean_abs_error")),
                frame=_format_metric(row.get("frame_backed_ratio")),
                linked=_format_metric(row.get("linked_entity_backed_ratio")),
                grounded=_format_metric(row.get("grounded_answer_ratio")),
                precision=_format_metric(row.get("answer_citation_precision")),
                recall=_format_metric(row.get("answer_citation_recall")),
                hit=_format_metric(row.get("expected_citation_hit_ratio")),
                unsupported=_format_metric(row.get("mean_unsupported_claim_count")),
                latency=_format_metric(row.get("mean_processing_time_ms")),
            )
        )
    lines.extend(
        [
            "",
            "Raw queries, answer text, transcript excerpts, candidate evidence, local paths, "
            "and raw candidate IDs are intentionally excluded.",
            "Citation grounding metrics are deterministic expected-hint overlap proxies for "
            "regression tracking, not a replacement for full LLM answer-quality judgment.",
            "",
        ]
    )
    return "\n".join(lines)


def reproducibility_payload(
    *,
    metrics: dict[str, Any],
    metrics_artifact: str,
    table_artifacts: list[str],
    command: list[str],
    commit_sha: str | None,
) -> dict[str, Any]:
    suites = [suite for suite in metrics.get("suites", []) if isinstance(suite, dict)]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": metrics.get("run_id"),
        "commit": {
            "sha": commit_sha,
        },
        "command": {
            "argv": _sanitize_command(command),
        },
        "artifacts": {
            "metrics": metrics_artifact,
            "paper_tables": table_artifacts,
        },
        "dataset_descriptors": [_suite_descriptor(suite) for suite in suites],
        "benchmark": {
            "suite_count": metrics.get("suite_count"),
            "query_count": metrics.get("query_count"),
            "deltas": metrics.get("deltas"),
            "anti_overfit": metrics.get("anti_overfit"),
        },
        "privacy": {
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_excerpts": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
            "raw_candidate_ids": "excluded_or_hashed",
            "grounding_proxy_metrics": (
                "deterministic_expected_hint_overlap_not_full_llm_quality"
            ),
        },
    }


def reproducibility_markdown(payload: dict[str, Any]) -> str:
    commit_sha = (payload.get("commit") or {}).get("sha") or "unknown"
    lines = [
        f"# Reproducibility Report: {payload.get('run_id')}",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Commit: `{commit_sha}`",
        f"- Metrics artifact: `{(payload.get('artifacts') or {}).get('metrics')}`",
        "",
        "## Dataset Descriptors",
        "",
    ]
    for descriptor in payload.get("dataset_descriptors", []):
        lines.append(
            "- `{suite_id}` ({suite_type}, {domain}): queries={query_count}, privacy={privacy}".format(
                suite_id=descriptor.get("suite_id"),
                suite_type=descriptor.get("suite_type"),
                domain=descriptor.get("domain"),
                query_count=descriptor.get("query_count"),
                privacy=descriptor.get("privacy"),
            )
        )
    lines.extend(
        [
            "",
            "## Public Output Policy",
            "",
            "This report contains aggregate metrics, configuration descriptors, commit metadata, "
            "and artifact names only. Raw queries, transcripts, answer text, candidate evidence, "
            "local filesystem paths, and raw candidate identifiers are excluded.",
            "Grounding and citation metrics are deterministic expected-hint overlap proxies for "
            "regression tracking; they do not replace full LLM answer-quality review.",
            "",
        ]
    )
    return "\n".join(lines)


def _paper_table_row(
    *,
    suite: dict[str, Any],
    item: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    row = {
        "suite_id": suite.get("suite_id"),
        "suite_type": suite.get("suite_type"),
        "domain": suite.get("domain"),
        "condition": condition,
        "query_count": item.get("query_count") or item.get("count") or 0,
        "hit_at_5s": item.get("hit_at_5s"),
        "hit_at_10s": item.get("hit_at_10s"),
        "hit_at_15s": item.get("hit_at_15s"),
        "mrr_at_max_delta": item.get("mrr_at_max_delta"),
        "top1_mean_abs_error": item.get("top1_mean_abs_error"),
        "frame_backed_ratio": item.get("frame_backed_ratio"),
        "linked_entity_backed_ratio": item.get(
            "linked_entity_backed_ratio",
            item.get("linked_entity_ratio"),
        ),
        "grounded_answer_ratio": item.get("grounded_answer_ratio"),
        "citation_coverage_ratio": item.get("citation_coverage_ratio"),
        "answer_citation_precision": item.get("answer_citation_precision"),
        "answer_citation_recall": item.get("answer_citation_recall"),
        "expected_citation_hit_ratio": item.get("expected_citation_hit_ratio"),
        "mean_unsupported_claim_count": item.get("mean_unsupported_claim_count"),
        "unsupported_claim_ratio": item.get("unsupported_claim_ratio"),
        "mean_processing_time_ms": item.get(
            "mean_processing_time_ms",
            item.get("mean_elapsed_time_ms"),
        ),
    }
    return {column: row.get(column) for column in PAPER_TABLE_COLUMNS}


def _suite_descriptor(suite: dict[str, Any]) -> dict[str, Any]:
    descriptor = (
        suite.get("dataset_descriptor")
        if isinstance(suite.get("dataset_descriptor"), dict)
        else {}
    )
    return {
        "suite_id": suite.get("suite_id"),
        "suite_type": suite.get("suite_type"),
        "domain": suite.get("domain"),
        "query_count": suite.get("query_count") or suite.get("count") or 0,
        "descriptor_id": descriptor.get("descriptor_id") or descriptor.get("dataset_id"),
        "split": descriptor.get("split"),
        "version": descriptor.get("version"),
        "privacy": descriptor.get("privacy", "aggregate_only"),
    }


def _write_paper_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in PAPER_TABLE_COLUMNS})


def _sanitize_command(command: list[str]) -> list[str]:
    sanitized: list[str] = []
    for token in command:
        text = str(token)
        if "/" in text or "\\" in text or text.startswith("~"):
            name = Path(text).name
            sanitized.append(name or "<path>")
        else:
            sanitized.append(text)
    return sanitized


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root.expanduser().resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit or None


def _format_metric(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.4f}".rstrip("0").rstrip(".")
