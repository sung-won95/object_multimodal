from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "mit_deep_learning_eval" / "seed_v1"
DEFAULT_METRICS_PATH = DEFAULT_REPORT_DIR / "metrics.json"
SCHEMA_VERSION = "mit-failure-analysis-v1"

FORBIDDEN_OUTPUT_TOKENS = (
    "/Users/",
    "/private/tmp",
    "query_text",
    "transcript_text",
    "transcript_excerpt",
    "transcript_window_text",
    "frame_path",
)

BUCKET_DESCRIPTIONS = {
    "no_hit_at_30s": {
        "interpretation": "The expected timestamp was not retrieved within the 30 second tolerance.",
        "next_action": "Treat this as the primary recall gap before making paper claims.",
    },
    "miss_at_10s": {
        "interpretation": "The stricter 10 second window still misses most expected timestamps.",
        "next_action": "Use this as the headline precision-localization risk for seed results.",
    },
    "high_top1_error": {
        "interpretation": "The top ranked candidate is far from the expected timestamp.",
        "next_action": "Audit ranking signals and candidate scoring before relying on top-1 evidence.",
    },
    "low_frame_backed": {
        "interpretation": "A material share of results is not backed by frame evidence.",
        "next_action": "Improve frame coverage before comparing visual grounding claims.",
    },
    "low_linked_entity_backed": {
        "interpretation": "A material share of results lacks linked visual-entity evidence.",
        "next_action": "Audit entity linking and visual label coverage before grounding claims.",
    },
    "candidate_label_audit_needed": {
        "interpretation": "Misses coincide with weak frame or linked-entity backing.",
        "next_action": "Inspect visual candidate labeling and entity links with private inputs only.",
    },
    "rerank_candidate_needed": {
        "interpretation": "Reranking is disabled while top-1 localization error is large.",
        "next_action": "Run a reranker or candidate-generation ablation before paper-level reporting.",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    metrics_path = args.metrics.expanduser()
    query_results_path = args.query_results.expanduser() if args.query_results else None
    output_json = args.output_json.expanduser() if args.output_json else None
    output_md = args.output_md.expanduser() if args.output_md else None

    result = analyze_failures(
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        output_json_path=output_json,
        output_markdown_path=output_md,
        high_top1_error_seconds=args.high_top1_error_seconds,
        rerank_top1_error_seconds=args.rerank_top1_error_seconds,
        low_backed_ratio=args.low_backed_ratio,
    )
    print(
        "Wrote public-safe failure analysis: "
        f"{result['artifacts']['json']}, {result['artifacts']['markdown']}"
    )
    if result["inputs"]["query_results"]["status"] != "loaded":
        print(
            "Note: query_results.jsonl was not loaded; query-level buckets use "
            "metrics-derived estimates."
        )
    return 0


def analyze_failures(
    *,
    metrics_path: Path,
    query_results_path: Path | None = None,
    output_json_path: Path | None = None,
    output_markdown_path: Path | None = None,
    high_top1_error_seconds: float = 600.0,
    rerank_top1_error_seconds: float = 300.0,
    low_backed_ratio: float = 0.75,
) -> dict[str, Any]:
    resolved_metrics_path = metrics_path.expanduser().resolve()
    metrics = _read_json(resolved_metrics_path)
    report_dir = resolved_metrics_path.parent
    resolved_output_json_path = output_json_path or report_dir / "failure_analysis.json"
    resolved_output_markdown_path = output_markdown_path or report_dir / "failure_analysis.md"
    resolved_query_results_path = (
        query_results_path.expanduser().resolve()
        if query_results_path is not None
        else report_dir / "query_results.jsonl"
    )

    suite_rows = _suite_variant_rows(metrics)
    query_records, query_status = _load_query_records(resolved_query_results_path)
    thresholds = {
        "high_top1_error_seconds": high_top1_error_seconds,
        "rerank_top1_error_seconds": rerank_top1_error_seconds,
        "low_backed_ratio": low_backed_ratio,
        "max_delta_seconds": _max_delta(metrics),
    }
    suite_rows = [
        _with_failure_flags(row, thresholds=thresholds)
        for row in suite_rows
    ]
    query_bucket_counts = _query_bucket_counts(query_records, thresholds=thresholds)
    suite_bucket_counts = _suite_bucket_counts(suite_rows, thresholds=thresholds)
    bucket_summaries = _bucket_summaries(
        rows=suite_rows,
        query_bucket_counts=query_bucket_counts,
        suite_bucket_counts=suite_bucket_counts,
        query_records_loaded=query_status["status"] == "loaded",
        total_queries=_number(metrics.get("query_count")) or 0.0,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": metrics.get("run_id"),
        "inputs": {
            "metrics": {
                "artifact": resolved_metrics_path.name,
                "status": "loaded",
            },
            "query_results": query_status,
        },
        "artifacts": {
            "json": resolved_output_json_path.name,
            "markdown": resolved_output_markdown_path.name,
        },
        "privacy": {
            "raw_query_strings": "excluded",
            "raw_transcripts": "excluded",
            "candidate_evidence_text": "excluded",
            "local_filesystem_paths": "excluded",
            "frame_file_references": "excluded",
            "raw_candidate_identifiers": "excluded",
        },
        "thresholds": thresholds,
        "summary": _summary(metrics, suite_rows),
        "failure_buckets": bucket_summaries,
        "by_domain": _domain_summaries(suite_rows),
        "by_suite_variant": _public_suite_rows(suite_rows),
        "limitations": _limitations(query_status["status"] == "loaded"),
    }
    _assert_public_safe(payload)

    resolved_output_json_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(payload)
    _assert_public_safe(markdown)
    resolved_output_markdown_path.write_text(markdown, encoding="utf-8")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    query_status = payload["inputs"]["query_results"]
    lines = [
        f"# MIT Seed Retrieval Failure Analysis: {payload.get('run_id')}",
        "",
        "## Headline",
        "",
        (
            "- Baseline aggregate: "
            f"mean Hit@10s={_fmt(summary.get('mean_hit_at_10s'))}, "
            f"mean MRR={_fmt(summary.get('mean_mrr_at_max_delta'))}, "
            f"mean Hit@30s={_fmt(summary.get('mean_hit_at_30s'))}."
        ),
        (
            "- Coverage: "
            f"{summary.get('suite_count')} suites, {summary.get('query_count')} queries, "
            f"{summary.get('domain_count')} domain."
        ),
        (
            "- Query-level detail: "
            f"{query_status.get('status')}"
            + (
                f" ({query_status.get('record_count')} records)."
                if query_status.get("status") == "loaded"
                else "; estimates are derived from suite metrics."
            )
        ),
        "",
        "## Failure Buckets",
        "",
        "| bucket | level | affected queries | query share | suite variants | interpretation | next action |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for bucket in payload["failure_buckets"]:
        affected = bucket.get("affected_query_count")
        if affected is None:
            affected = bucket.get("affected_query_estimate")
        lines.append(
            "| {bucket} | {level} | {affected} | {share} | {suite_count} | {why} | {action} |".format(
                bucket=bucket["bucket"],
                level=bucket["level"],
                affected=_fmt_count(affected),
                share=_fmt(bucket.get("affected_query_ratio")),
                suite_count=bucket.get("affected_suite_variant_count", 0),
                why=bucket["interpretation"],
                action=bucket["next_action"],
            )
        )

    lines.extend(
        [
            "",
            "## Domain Aggregate",
            "",
            "| domain | suites | queries | Hit@10s | Hit@30s | MRR | top1 err s | frame-backed | linked-backed | top buckets |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["by_domain"]:
        top_buckets = _top_domain_buckets(row.get("failure_buckets", {}))
        lines.append(
            "| {domain} | {suite_count} | {query_count} | {hit10} | {hit30} | {mrr} | {top1} | {frame} | {linked} | {buckets} |".format(
                domain=row["domain"],
                suite_count=row["suite_variant_count"],
                query_count=_fmt_count(row["query_count"]),
                hit10=_fmt(row.get("mean_hit_at_10s")),
                hit30=_fmt(row.get("mean_hit_at_30s")),
                mrr=_fmt(row.get("mean_mrr_at_max_delta")),
                top1=_fmt(row.get("mean_top1_abs_error_seconds")),
                frame=_fmt(row.get("mean_frame_backed_ratio")),
                linked=_fmt(row.get("mean_linked_entity_backed_ratio")),
                buckets=", ".join(top_buckets) or "-",
            )
        )

    lines.extend(
        [
            "",
            "## Suite Variant Aggregate",
            "",
            "| suite | variant | queries | Hit@10s | Hit@30s | MRR | top1 err s | frame | linked | buckets |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["by_suite_variant"]:
        lines.append(
            "| {suite_id} | {variant_id} | {query_count} | {hit10} | {hit30} | {mrr} | {top1} | {frame} | {linked} | {buckets} |".format(
                suite_id=row["suite_id"],
                variant_id=row["variant_id"],
                query_count=_fmt_count(row["query_count"]),
                hit10=_fmt(row.get("hit_at_10s")),
                hit30=_fmt(row.get("hit_at_30s")),
                mrr=_fmt(row.get("mrr_at_max_delta")),
                top1=_fmt(row.get("top1_mean_abs_error_seconds")),
                frame=_fmt(row.get("frame_backed_ratio")),
                linked=_fmt(row.get("linked_entity_backed_ratio")),
                buckets=", ".join(row.get("failure_flags", [])) or "-",
            )
        )

    lines.extend(
        [
            "",
            "## Public Output Policy",
            "",
            "This artifact is aggregate-only. It excludes raw query strings, raw transcripts, "
            "candidate evidence text, local filesystem paths, frame file references, and raw "
            "candidate identifiers.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a public-safe aggregate failure analysis for the MIT deep "
            "learning seed retrieval baseline."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to metrics.json. Defaults to the MIT seed_v1 report.",
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        default=None,
        help=(
            "Optional query_results.jsonl. When omitted, the script looks next to "
            "metrics.json and falls back to suite-level estimates if absent."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to failure_analysis.json next to metrics.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to failure_analysis.md next to metrics.json.",
    )
    parser.add_argument(
        "--high-top1-error-seconds",
        type=float,
        default=600.0,
        help="Top-1 absolute error threshold for high_top1_error.",
    )
    parser.add_argument(
        "--rerank-top1-error-seconds",
        type=float,
        default=300.0,
        help="Top-1 absolute error threshold for rerank_candidate_needed.",
    )
    parser.add_argument(
        "--low-backed-ratio",
        type=float,
        default=0.75,
        help="Frame/entity-backed ratio threshold for low-backed buckets.",
    )
    return parser


def _suite_variant_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in metrics.get("suites", []):
        if not isinstance(suite, dict):
            continue
        variants = [item for item in suite.get("variants", []) if isinstance(item, dict)]
        modes = [item for item in suite.get("modes", []) if isinstance(item, dict)]
        if variants:
            for variant in variants:
                rows.append(_suite_row(suite, variant, _condition_id(variant, "variant")))
        elif modes:
            for mode in modes:
                rows.append(_suite_row(suite, mode, _condition_id(mode, "mode")))
        else:
            rows.append(_suite_row(suite, suite, "suite"))
    return rows


def _suite_row(
    suite: dict[str, Any],
    item: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    rerank = item.get("rerank") if isinstance(item.get("rerank"), dict) else suite.get("rerank")
    lexicon = (
        item.get("domain_lexicon")
        if isinstance(item.get("domain_lexicon"), dict)
        else suite.get("domain_lexicon")
    )
    if not isinstance(rerank, dict):
        rerank = {}
    if not isinstance(lexicon, dict):
        lexicon = {}
    return {
        "suite_id": _safe_label(suite.get("suite_id")),
        "suite_type": _safe_label(suite.get("suite_type")),
        "domain": _safe_label(suite.get("domain")),
        "variant_id": _safe_label(variant_id),
        "query_count": _number(item.get("query_count") or suite.get("query_count")) or 0.0,
        "hit_at_5s": _number(item.get("hit_at_5s", suite.get("hit_at_5s"))),
        "hit_at_10s": _number(item.get("hit_at_10s", suite.get("hit_at_10s"))),
        "hit_at_15s": _number(item.get("hit_at_15s", suite.get("hit_at_15s"))),
        "hit_at_30s": _number(item.get("hit_at_30s", suite.get("hit_at_30s"))),
        "mrr_at_max_delta": _number(
            item.get("mrr_at_max_delta", suite.get("mrr_at_max_delta"))
        ),
        "mean_abs_error_seconds": _number(
            item.get("mean_abs_error", suite.get("mean_abs_error"))
        ),
        "top1_mean_abs_error_seconds": _number(
            item.get("top1_mean_abs_error", suite.get("top1_mean_abs_error"))
        ),
        "frame_backed_ratio": _number(
            item.get("frame_backed_ratio", suite.get("frame_backed_ratio"))
        ),
        "linked_entity_backed_ratio": _number(
            item.get(
                "linked_entity_backed_ratio",
                suite.get("linked_entity_backed_ratio"),
            )
        ),
        "mean_processing_time_ms": _number(
            item.get("mean_processing_time_ms", suite.get("mean_processing_time_ms"))
        ),
        "rerank_enabled": bool(rerank.get("enabled", False)),
        "domain_lexicon_enabled": bool(lexicon.get("enabled", False)),
    }


def _with_failure_flags(row: dict[str, Any], *, thresholds: dict[str, float]) -> dict[str, Any]:
    hit30 = row.get("hit_at_30s")
    hit10 = row.get("hit_at_10s")
    top1 = row.get("top1_mean_abs_error_seconds")
    frame_ratio = row.get("frame_backed_ratio")
    linked_ratio = row.get("linked_entity_backed_ratio")
    flags: list[str] = []
    if hit30 is not None and hit30 <= 0:
        flags.append("no_hit_at_30s")
    if hit10 is not None and hit10 < 1:
        flags.append("miss_at_10s")
    if top1 is not None and top1 >= thresholds["high_top1_error_seconds"]:
        flags.append("high_top1_error")
    if frame_ratio is not None and frame_ratio < thresholds["low_backed_ratio"]:
        flags.append("low_frame_backed")
    if linked_ratio is not None and linked_ratio < thresholds["low_backed_ratio"]:
        flags.append("low_linked_entity_backed")
    if (
        hit30 is not None
        and hit30 < 0.5
        and (
            (frame_ratio is not None and frame_ratio < thresholds["low_backed_ratio"])
            or (linked_ratio is not None and linked_ratio < thresholds["low_backed_ratio"])
        )
    ):
        flags.append("candidate_label_audit_needed")
    if (
        not row.get("rerank_enabled")
        and hit30 is not None
        and hit30 < 1
        and top1 is not None
        and top1 >= thresholds["rerank_top1_error_seconds"]
    ):
        flags.append("rerank_candidate_needed")
    updated = dict(row)
    updated["failure_flags"] = flags
    return updated


def _query_bucket_counts(
    records: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
) -> dict[str, int]:
    counts = {bucket: 0 for bucket in BUCKET_DESCRIPTIONS}
    for record in records:
        hit_by_delta = record.get("hit_by_delta")
        if not isinstance(hit_by_delta, dict):
            hit_by_delta = {}
        hit30 = bool(hit_by_delta.get("30", hit_by_delta.get(30, False)))
        hit10 = bool(hit_by_delta.get("10", hit_by_delta.get(10, False)))
        top1 = _number(record.get("top1_abs_error"))
        frame_backed = _bool_or_none(record.get("frame_backed"))
        linked_backed = _bool_or_none(record.get("linked_entity_backed"))
        segment_hits = _number(record.get("segment_search_hits")) or _number(record.get("search_hits")) or 0
        visual_hits = _number(record.get("visual_entity_search_hits")) or 0
        rerank = record.get("rerank") if isinstance(record.get("rerank"), dict) else {}
        rerank_enabled = bool(rerank.get("enabled", False))

        if not hit30:
            counts["no_hit_at_30s"] += 1
        if not hit10:
            counts["miss_at_10s"] += 1
        if top1 is not None and top1 >= thresholds["high_top1_error_seconds"]:
            counts["high_top1_error"] += 1
        if frame_backed is False:
            counts["low_frame_backed"] += 1
        if linked_backed is False:
            counts["low_linked_entity_backed"] += 1
        if (
            not hit30
            and (
                frame_backed is False
                or linked_backed is False
                or visual_hits <= 0
            )
        ):
            counts["candidate_label_audit_needed"] += 1
        if (
            not rerank_enabled
            and not hit30
            and top1 is not None
            and top1 >= thresholds["rerank_top1_error_seconds"]
            and segment_hits > 0
        ):
            counts["rerank_candidate_needed"] += 1
    return counts


def _suite_bucket_counts(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {
        bucket: {"suite_variant_count": 0.0, "query_estimate": 0.0}
        for bucket in BUCKET_DESCRIPTIONS
    }
    for row in rows:
        q = row.get("query_count") or 0.0
        flags = set(row.get("failure_flags", []))
        for bucket in BUCKET_DESCRIPTIONS:
            if bucket in flags:
                counts[bucket]["suite_variant_count"] += 1

        counts["no_hit_at_30s"]["query_estimate"] += _miss_estimate(
            q,
            row.get("hit_at_30s"),
        )
        counts["miss_at_10s"]["query_estimate"] += _miss_estimate(q, row.get("hit_at_10s"))
        if row.get("top1_mean_abs_error_seconds") is not None and row[
            "top1_mean_abs_error_seconds"
        ] >= thresholds["high_top1_error_seconds"]:
            counts["high_top1_error"]["query_estimate"] += q
        counts["low_frame_backed"]["query_estimate"] += _miss_estimate(
            q,
            row.get("frame_backed_ratio"),
        )
        counts["low_linked_entity_backed"]["query_estimate"] += _miss_estimate(
            q,
            row.get("linked_entity_backed_ratio"),
        )
        if "candidate_label_audit_needed" in flags:
            counts["candidate_label_audit_needed"]["query_estimate"] += _miss_estimate(
                q,
                row.get("hit_at_30s"),
            )
        if "rerank_candidate_needed" in flags:
            counts["rerank_candidate_needed"]["query_estimate"] += _miss_estimate(
                q,
                row.get("hit_at_30s"),
            )
    return counts


def _bucket_summaries(
    *,
    rows: list[dict[str, Any]],
    query_bucket_counts: dict[str, int],
    suite_bucket_counts: dict[str, dict[str, float]],
    query_records_loaded: bool,
    total_queries: float,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for bucket, description in BUCKET_DESCRIPTIONS.items():
        suite_count = int(suite_bucket_counts[bucket]["suite_variant_count"])
        query_estimate = suite_bucket_counts[bucket]["query_estimate"]
        affected_query_count = query_bucket_counts[bucket] if query_records_loaded else None
        affected_value = affected_query_count if affected_query_count is not None else query_estimate
        bucket_rows = [row for row in rows if bucket in row.get("failure_flags", [])]
        summary = {
            "bucket": bucket,
            "level": "query" if query_records_loaded else "suite_variant_estimate",
            "affected_suite_variant_count": suite_count,
            "affected_query_ratio": _ratio(affected_value, total_queries),
            "representative_suite_variants": _representative_suite_variants(bucket_rows),
            "interpretation": description["interpretation"],
            "next_action": description["next_action"],
        }
        if query_records_loaded:
            summary["affected_query_count"] = affected_query_count
            summary["metrics_query_estimate"] = _clean_number(query_estimate)
        else:
            summary["affected_query_estimate"] = _clean_number(query_estimate)
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda item: (
            item.get("affected_query_count", item.get("affected_query_estimate", 0)) or 0
        ),
        reverse=True,
    )


def _domain_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("domain") or "unknown")].append(row)
    summaries: list[dict[str, Any]] = []
    for domain, domain_rows in sorted(grouped.items()):
        summaries.append(
            {
                "domain": domain,
                "suite_variant_count": len(domain_rows),
                "query_count": _clean_number(sum(row.get("query_count") or 0 for row in domain_rows)),
                "mean_hit_at_10s": _weighted_mean(domain_rows, "hit_at_10s"),
                "mean_hit_at_30s": _weighted_mean(domain_rows, "hit_at_30s"),
                "mean_mrr_at_max_delta": _weighted_mean(domain_rows, "mrr_at_max_delta"),
                "mean_top1_abs_error_seconds": _weighted_mean(
                    domain_rows,
                    "top1_mean_abs_error_seconds",
                ),
                "mean_frame_backed_ratio": _weighted_mean(domain_rows, "frame_backed_ratio"),
                "mean_linked_entity_backed_ratio": _weighted_mean(
                    domain_rows,
                    "linked_entity_backed_ratio",
                ),
                "failure_buckets": _domain_bucket_estimates(domain_rows),
            }
        )
    return summaries


def _domain_bucket_estimates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    estimates: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_DESCRIPTIONS:
        bucket_rows = [row for row in rows if bucket in row.get("failure_flags", [])]
        estimates[bucket] = {
            "suite_variant_count": len(bucket_rows),
            "affected_query_estimate": _clean_number(
                _bucket_query_estimate(bucket, rows)
            ),
        }
    return estimates


def _bucket_query_estimate(bucket: str, rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        query_count = row.get("query_count") or 0.0
        if bucket == "no_hit_at_30s":
            total += _miss_estimate(query_count, row.get("hit_at_30s"))
        elif bucket == "miss_at_10s":
            total += _miss_estimate(query_count, row.get("hit_at_10s"))
        elif bucket == "low_frame_backed":
            total += _miss_estimate(query_count, row.get("frame_backed_ratio"))
        elif bucket == "low_linked_entity_backed":
            total += _miss_estimate(query_count, row.get("linked_entity_backed_ratio"))
        elif bucket in row.get("failure_flags", []):
            total += query_count
    return total


def _public_suite_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    keys = [
        "suite_id",
        "suite_type",
        "domain",
        "variant_id",
        "query_count",
        "hit_at_5s",
        "hit_at_10s",
        "hit_at_15s",
        "hit_at_30s",
        "mrr_at_max_delta",
        "mean_abs_error_seconds",
        "top1_mean_abs_error_seconds",
        "frame_backed_ratio",
        "linked_entity_backed_ratio",
        "mean_processing_time_ms",
        "rerank_enabled",
        "domain_lexicon_enabled",
        "failure_flags",
    ]
    for row in rows:
        public_rows.append({key: row.get(key) for key in keys})
    return public_rows


def _summary(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_count = len({row.get("domain") for row in rows if row.get("domain")})
    summary = {
        "suite_count": metrics.get("suite_count") or len(rows),
        "suite_variant_count": len(rows),
        "query_count": metrics.get("query_count") or sum(row.get("query_count") or 0 for row in rows),
        "domain_count": domain_count,
        "mean_hit_at_10s": _weighted_mean(rows, "hit_at_10s"),
        "mean_hit_at_30s": _weighted_mean(rows, "hit_at_30s"),
        "mean_mrr_at_max_delta": _weighted_mean(rows, "mrr_at_max_delta"),
        "mean_top1_abs_error_seconds": _weighted_mean(rows, "top1_mean_abs_error_seconds"),
        "mean_frame_backed_ratio": _weighted_mean(rows, "frame_backed_ratio"),
        "mean_linked_entity_backed_ratio": _weighted_mean(
            rows,
            "linked_entity_backed_ratio",
        ),
    }
    anti_overfit = metrics.get("anti_overfit")
    if isinstance(anti_overfit, dict):
        domains = anti_overfit.get("domains")
        if isinstance(domains, dict) and len(domains) == 1:
            domain_summary = next(iter(domains.values()))
            if isinstance(domain_summary, dict):
                summary["mean_hit_at_10s"] = _number(
                    domain_summary.get("mean_hit_at_10s")
                ) or summary["mean_hit_at_10s"]
                summary["mean_mrr_at_max_delta"] = _number(
                    domain_summary.get("mean_mrr_at_max_delta")
                ) or summary["mean_mrr_at_max_delta"]
    return {key: _clean_number(value) for key, value in summary.items()}


def _limitations(query_records_loaded: bool) -> list[str]:
    limitations = [
        "This is an aggregate diagnostic, not a claim that the seed baseline is paper-ready.",
        "Buckets can overlap, so counts should not be summed into a unique failure total.",
        "Top-1 error thresholds are diagnostic cutoffs for triage, not statistical significance tests.",
    ]
    if query_records_loaded:
        limitations.append(
            "Query-level records were used only for counts; raw text, evidence snippets, paths, and candidate identifiers were discarded."
        )
    else:
        limitations.append(
            "Query-level records were unavailable, so affected-query counts are estimated from suite ratios."
        )
    return limitations


def _load_query_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {
            "artifact": path.name,
            "status": "not_found",
            "record_count": 0,
            "message": (
                "Optional query-level results were absent; suite metrics were used "
                "for aggregate estimates."
            ),
        }
    records = _read_jsonl(path)
    return records, {
        "artifact": path.name,
        "status": "loaded",
        "record_count": len(records),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path.name}:{line_number}")
            rows.append(payload)
    return rows


def _assert_public_safe(value: Any) -> None:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in text:
            raise ValueError(f"Unsafe token reached public output: {token}")


def _representative_suite_variants(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    representatives = []
    for row in rows[:limit]:
        representatives.append(
            {
                "suite_id": str(row.get("suite_id") or "unknown"),
                "variant_id": str(row.get("variant_id") or "suite"),
                "domain": str(row.get("domain") or "unknown"),
            }
        )
    return representatives


def _top_domain_buckets(failure_buckets: Any, limit: int = 3) -> list[str]:
    if not isinstance(failure_buckets, dict):
        return []
    ranked = []
    for bucket, summary in failure_buckets.items():
        if not isinstance(summary, dict):
            continue
        count = _number(summary.get("affected_query_estimate")) or 0.0
        ranked.append((bucket, count))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [bucket for bucket, count in ranked[:limit] if count > 0]


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = _number(row.get(key))
        if value is None:
            continue
        weight = _number(row.get("query_count")) or 1.0
        weighted_sum += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return _clean_number(weighted_sum / total_weight)


def _miss_estimate(query_count: float, ratio: Any) -> float:
    parsed_ratio = _number(ratio)
    if parsed_ratio is None:
        return 0.0
    return max(0.0, query_count * (1.0 - parsed_ratio))


def _ratio(value: float | int | None, total: float) -> float | None:
    if value is None or total <= 0:
        return None
    return _clean_number(float(value) / total)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _clean_number(value: Any) -> Any:
    parsed = _number(value)
    if parsed is None:
        return value
    rounded = round(parsed, 4)
    if float(rounded).is_integer():
        return int(rounded)
    return rounded


def _fmt(value: Any) -> str:
    parsed = _number(value)
    if parsed is None:
        return "-"
    if parsed == 0:
        return "0"
    if parsed == 1:
        return "1"
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _fmt_count(value: Any) -> str:
    parsed = _number(value)
    if parsed is None:
        return "-"
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.2f}".rstrip("0").rstrip(".")


def _condition_id(item: dict[str, Any], fallback: str) -> str:
    return _safe_label(
        item.get("variant_id")
        or item.get("label")
        or item.get("mode")
        or item.get("condition")
        or fallback
    )


def _safe_label(value: Any) -> str:
    label = str(value or "unknown")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        label = label.replace(token, "<redacted>")
    if "/" in label or "\\" in label:
        return Path(label).name or "<redacted>"
    return label


def _max_delta(metrics: dict[str, Any]) -> float | None:
    deltas = [_number(delta) for delta in metrics.get("deltas", [])]
    parsed = [delta for delta in deltas if delta is not None]
    return max(parsed) if parsed else None


if __name__ == "__main__":
    raise SystemExit(main())
