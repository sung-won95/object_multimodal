from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from oarag.core.io import write_json


PAPER_METRIC_INTERVALS_SCHEMA_VERSION = "paper-metric-intervals-v1"
DEFAULT_PAPER_METRICS = [
    "hit_at_10s",
    "mrr",
    "top1_abs_error",
    "grounded_answer",
    "expected_citation_hit",
    "unsupported_claim_count",
]
DEFAULT_BOOTSTRAP_SAMPLE_COUNT = 1000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CONFIDENCE_LEVEL = 0.95
PAPER_METRIC_INTERVALS_JSON = "paper_metric_intervals.json"
PAPER_METRIC_INTERVALS_CSV = "paper_metric_intervals.csv"
PAPER_METRIC_INTERVALS_MARKDOWN = "paper_metric_intervals.md"

LOWER_IS_BETTER_METRICS = {
    "top1_abs_error",
    "unsupported_claim_count",
    "warning_count",
    "processing_time_ms",
    "elapsed_time_ms",
}


@dataclass(frozen=True)
class PaperMetricIntervalsRun:
    output_dir: Path
    json_path: Path
    csv_path: Path
    markdown_path: Path
    payload: dict[str, Any]


def generate_metric_intervals(
    *,
    query_results_path: Path,
    output_dir: Path | None = None,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
    baseline_variant_id: str | None = None,
    metrics: Iterable[str] | None = None,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    sample_count: int = DEFAULT_BOOTSTRAP_SAMPLE_COUNT,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PaperMetricIntervalsRun:
    """Generate private-safe bootstrap CIs and paired deltas from query_results.jsonl."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    resolved_query_results_path = query_results_path.expanduser().resolve()
    rows = _read_jsonl(resolved_query_results_path)
    metric_names = _normalize_metrics(metrics)
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else resolved_query_results_path.parent / "paper_metric_intervals"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_metrics_path = metrics_path.expanduser().resolve() if metrics_path is not None else None
    metrics_payload = _read_optional_json(resolved_metrics_path)
    resolved_report_path = report_path.expanduser().resolve() if report_path is not None else None
    run_id = _run_id(rows=rows, metrics=metrics_payload)
    rng = random.Random(seed)

    variants = _variant_order(rows)
    variant_summaries = [
        _variant_summary(
            variant_id=variant_id,
            rows=[row for row in rows if _row_variant_id(row) == variant_id],
            metric_names=metric_names,
            sample_count=sample_count,
            confidence_level=confidence_level,
            rng=rng,
        )
        for variant_id in variants
    ]
    paired_deltas = _paired_delta_summaries(
        rows=rows,
        variants=variants,
        baseline_variant_id=baseline_variant_id,
        metric_names=metric_names,
        sample_count=sample_count,
        confidence_level=confidence_level,
        rng=rng,
    )
    caveats = _global_caveats(
        variant_summaries=variant_summaries,
        paired_deltas=paired_deltas,
        baseline_variant_id=baseline_variant_id,
        variants=variants,
    )
    robustness_status = _robustness_status(
        row_count=len(rows),
        variants=variants,
        paired_deltas=paired_deltas,
        baseline_variant_id=baseline_variant_id,
        caveats=caveats,
    )
    payload = {
        "schema_version": PAPER_METRIC_INTERVALS_SCHEMA_VERSION,
        "run_id": run_id,
        "status": robustness_status,
        "summary": {
            "robustness_status": robustness_status,
            "row_count": len(rows),
            "variant_count": len(variants),
            "metric_count": len(metric_names),
            "paired_delta_count": len(paired_deltas),
            "baseline_variant_configured": baseline_variant_id is not None,
            "baseline_variant_found": baseline_variant_id in variants
            if baseline_variant_id is not None
            else False,
            "caveat_count": len(caveats),
            "interpretation": _robustness_interpretation(robustness_status),
        },
        "input": {
            "query_results_artifact": _artifact_name(resolved_query_results_path),
            "metrics_artifact": _artifact_name(resolved_metrics_path),
            "report_artifact": _artifact_name(resolved_report_path),
        },
        "config": {
            "metrics": metric_names,
            "baseline_variant_id": baseline_variant_id,
            "bootstrap": {
                "sample_count": sample_count,
                "seed": seed,
                "confidence_level": confidence_level,
                "method": "percentile_resampled_mean",
            },
            "paired_delta": {
                "enabled": baseline_variant_id is not None,
                "definition": "variant_minus_baseline",
                "pairing_key_fields": ["suite_id", "query_id"],
            },
        },
        "metric_definitions": {
            metric: {
                "direction": _metric_direction(metric),
                "mean": "arithmetic_mean_of_observed_query_values",
                "paired_delta": "variant_value_minus_baseline_value_for_matching_query_id",
            }
            for metric in metric_names
        },
        "row_count": len(rows),
        "variant_count": len(variants),
        "variants": variant_summaries,
        "paired_deltas": paired_deltas,
        "caveats": caveats,
        "privacy": {
            "payload": "aggregate_statistics_only",
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
            "query_ids": "excluded_from_outputs",
        },
    }

    json_path = resolved_output_dir / PAPER_METRIC_INTERVALS_JSON
    csv_path = resolved_output_dir / PAPER_METRIC_INTERVALS_CSV
    markdown_path = resolved_output_dir / PAPER_METRIC_INTERVALS_MARKDOWN
    write_json(json_path, payload)
    _write_metric_intervals_csv(csv_path, payload)
    markdown_path.write_text(metric_intervals_markdown(payload), encoding="utf-8")

    return PaperMetricIntervalsRun(
        output_dir=resolved_output_dir,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        payload=payload,
    )


def metric_intervals_markdown(payload: Mapping[str, Any]) -> str:
    run_id = payload.get("run_id") or "unknown"
    confidence_level = (
        (payload.get("config") or {}).get("bootstrap") or {}
    ).get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)
    ci_label = f"{float(confidence_level) * 100:.0f}% CI"
    lines = [
        f"# Paper Metric Intervals: {run_id}",
        "",
        "Aggregate-only bootstrap confidence intervals and paired deltas for paper metrics.",
        "Raw queries, answers, transcripts, candidate evidence, query IDs, and local paths are excluded.",
        "",
        "## Robustness Status",
        "",
    ]
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    lines.extend(
        [
            f"- Status: `{payload.get('status') or summary.get('robustness_status') or 'unknown'}`",
            f"- Paired delta rows: {summary.get('paired_delta_count', 0)}",
            f"- Caveat count: {summary.get('caveat_count', 0)}",
            f"- Interpretation: {summary.get('interpretation') or 'unknown'}",
            "",
        ]
    )
    lines.extend(["## Caveats", ""])
    caveats = [str(item) for item in payload.get("caveats", []) if item]
    if caveats:
        for caveat in caveats:
            lines.append(f"- {caveat}")
    else:
        lines.append("- No caveats were produced for the configured summaries.")

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            f"| variant | metric | n | mean | {ci_label} | direction | caveats |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for variant in _list_of_dicts(payload.get("variants")):
        metrics = variant.get("metrics") if isinstance(variant.get("metrics"), Mapping) else {}
        for metric, summary in metrics.items():
            summary = summary if isinstance(summary, Mapping) else {}
            lines.append(
                "| {variant} | {metric} | {n} | {mean} | [{low}, {high}] | {direction} | {caveats} |".format(
                    variant=_markdown_cell(str(variant.get("variant_id") or "")),
                    metric=_markdown_cell(str(metric)),
                    n=summary.get("observed_count"),
                    mean=_format_metric(summary.get("mean")),
                    low=_format_metric(summary.get("ci_low")),
                    high=_format_metric(summary.get("ci_high")),
                    direction=_markdown_cell(str(summary.get("direction") or "")),
                    caveats=_markdown_cell(", ".join(summary.get("caveats") or [])),
                )
            )

    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            f"| baseline | variant | metric | pairs | mean delta | {ci_label} | mean improvement | direction | caveats |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |",
        ]
    )
    paired_deltas = _list_of_dicts(payload.get("paired_deltas"))
    if paired_deltas:
        for delta in paired_deltas:
            lines.append(
                "| {baseline} | {variant} | {metric} | {n} | {mean_delta} | "
                "[{low}, {high}] | {improvement} | {direction} | {caveats} |".format(
                    baseline=_markdown_cell(str(delta.get("baseline_variant_id") or "")),
                    variant=_markdown_cell(str(delta.get("variant_id") or "")),
                    metric=_markdown_cell(str(delta.get("metric") or "")),
                    n=delta.get("paired_query_count"),
                    mean_delta=_format_metric(delta.get("mean_delta")),
                    low=_format_metric(delta.get("ci_low")),
                    high=_format_metric(delta.get("ci_high")),
                    improvement=_format_metric(delta.get("mean_improvement")),
                    direction=_markdown_cell(str(delta.get("direction") or "")),
                    caveats=_markdown_cell(", ".join(delta.get("caveats") or [])),
                )
            )
    else:
        lines.append("| - | - | - | 0 | - | [-, -] | - | - | baseline_not_configured_or_missing |")

    lines.extend(
        [
            "",
            "## Stable Schema",
            "",
            f"- Schema: `{payload.get('schema_version')}`",
            "- Variant aggregate rows live under `variants[].metrics`.",
            "- Robustness comparisons live under `paired_deltas[]` and use `variant_minus_baseline`.",
            "- Privacy policy is recorded under `privacy`.",
            "",
        ]
    )
    return "\n".join(lines)


def _variant_summary(
    *,
    variant_id: str,
    rows: list[dict[str, Any]],
    metric_names: list[str],
    sample_count: int,
    confidence_level: float,
    rng: random.Random,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in metric_names:
        values = [
            value
            for value in (_metric_value(row, metric) for row in rows)
            if value is not None
        ]
        interval = _mean_interval(
            values,
            sample_count=sample_count,
            confidence_level=confidence_level,
            rng=rng,
        )
        metrics[metric] = {
            "direction": _metric_direction(metric),
            "query_count": len(rows),
            "observed_count": len(values),
            "missing_count": len(rows) - len(values),
            "mean": interval["mean"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "min": _round_metric(min(values)) if values else None,
            "max": _round_metric(max(values)) if values else None,
            "bootstrap_sample_count": sample_count if values else 0,
            "confidence_level": confidence_level,
            "caveats": _sample_caveats(len(values)),
        }
    return {
        "variant_id": variant_id,
        "query_count": len(rows),
        "metrics": metrics,
    }


def _robustness_status(
    *,
    row_count: int,
    variants: list[str],
    paired_deltas: list[dict[str, Any]],
    baseline_variant_id: str | None,
    caveats: list[str],
) -> str:
    if row_count <= 0 or not variants:
        return "needs_evidence"
    if baseline_variant_id is None or baseline_variant_id not in variants:
        return "needs_evidence"
    if not paired_deltas:
        return "needs_evidence"
    if caveats:
        return "needs_evidence"
    return "passed"


def _robustness_interpretation(status: str) -> str:
    if status == "passed":
        return "Paired baseline deltas are available without generated caveats."
    if status == "needs_evidence":
        return (
            "Intervals are descriptive or incomplete; do not treat them as confirmatory "
            "robustness evidence."
        )
    return "Robustness status could not be determined from aggregate intervals."


def _paired_delta_summaries(
    *,
    rows: list[dict[str, Any]],
    variants: list[str],
    baseline_variant_id: str | None,
    metric_names: list[str],
    sample_count: int,
    confidence_level: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if baseline_variant_id is None or baseline_variant_id not in variants:
        return []

    baseline_rows = [row for row in rows if _row_variant_id(row) == baseline_variant_id]
    summaries: list[dict[str, Any]] = []
    for variant_id in variants:
        if variant_id == baseline_variant_id:
            continue
        variant_rows = [row for row in rows if _row_variant_id(row) == variant_id]
        for metric in metric_names:
            baseline_values, baseline_duplicates = _paired_values_by_key(baseline_rows, metric)
            variant_values, variant_duplicates = _paired_values_by_key(variant_rows, metric)
            shared_keys = sorted(set(baseline_values).intersection(variant_values))
            deltas = [variant_values[key] - baseline_values[key] for key in shared_keys]
            interval = _mean_interval(
                deltas,
                sample_count=sample_count,
                confidence_level=confidence_level,
                rng=rng,
            )
            sign = 1.0 if _metric_direction(metric) == "higher_is_better" else -1.0
            improvement_ci_low, improvement_ci_high = _signed_interval(
                interval["ci_low"],
                interval["ci_high"],
                sign,
            )
            caveats = _sample_caveats(len(deltas), paired=True)
            if baseline_duplicates or variant_duplicates:
                caveats.append("duplicate_pair_keys_collapsed_first_value")
            summaries.append(
                {
                    "baseline_variant_id": baseline_variant_id,
                    "variant_id": variant_id,
                    "metric": metric,
                    "direction": _metric_direction(metric),
                    "paired_query_count": len(deltas),
                    "mean_delta": interval["mean"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "mean_improvement": _signed_metric(interval["mean"], sign),
                    "improvement_ci_low": improvement_ci_low,
                    "improvement_ci_high": improvement_ci_high,
                    "bootstrap_sample_count": sample_count if deltas else 0,
                    "confidence_level": confidence_level,
                    "caveats": caveats,
                }
            )
    return summaries


def _mean_interval(
    values: list[float],
    *,
    sample_count: int,
    confidence_level: float,
    rng: random.Random,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        rounded = _round_metric(mean)
        return {"mean": rounded, "ci_low": rounded, "ci_high": rounded}
    means: list[float] = []
    for _ in range(sample_count):
        resampled = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(resampled) / len(resampled))
    alpha = 1.0 - confidence_level
    means.sort()
    low = _percentile(means, alpha / 2.0)
    high = _percentile(means, 1.0 - alpha / 2.0)
    return {
        "mean": _round_metric(mean),
        "ci_low": _round_metric(low),
        "ci_high": _round_metric(high),
    }


def _paired_values_by_key(rows: list[dict[str, Any]], metric: str) -> tuple[dict[str, float], int]:
    values: dict[str, float] = {}
    duplicate_count = 0
    for row in rows:
        key = _pair_key(row)
        if key is None:
            continue
        value = _metric_value(row, metric)
        if value is None:
            continue
        if key in values:
            duplicate_count += 1
            continue
        values[key] = value
    return values, duplicate_count


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    direct = _coerce_metric_value(row.get(metric))
    if direct is not None:
        return direct

    dotted = _coerce_metric_value(_dotted_value(row, metric))
    if dotted is not None:
        return dotted

    if metric.startswith("hit_at_") and metric.endswith("s"):
        delta = metric.removeprefix("hit_at_").removesuffix("s")
        hit_by_delta = row.get("hit_by_delta")
        if isinstance(hit_by_delta, Mapping):
            return _coerce_metric_value(hit_by_delta.get(delta))

    if metric == "grounded_answer":
        answer = row.get("answer")
        if isinstance(answer, Mapping):
            if answer.get("enabled") is False:
                return None
            answer_type = answer.get("answer_type")
            if answer_type is None:
                return None
            return 1.0 if str(answer_type) == "grounded_answer" else 0.0

    if metric in {"expected_citation_hit", "unsupported_claim_count"}:
        answer_grounding = row.get("answer_grounding")
        if isinstance(answer_grounding, Mapping):
            return _coerce_metric_value(answer_grounding.get(metric))

    return None


def _coerce_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"true", "yes", "y", "on"}:
            return 1.0
        if text in {"false", "no", "n", "off"}:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _dotted_value(row: Mapping[str, Any], metric: str) -> Any:
    current: Any = row
    for part in metric.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path.name}:{line_number}")
            rows.append(payload)
    return rows


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_metrics(metrics: Iterable[str] | None) -> list[str]:
    if metrics is None:
        return list(DEFAULT_PAPER_METRICS)
    normalized: list[str] = []
    for raw_metric in metrics:
        for item in str(raw_metric).split(","):
            metric = item.strip()
            if metric and metric not in normalized:
                normalized.append(metric)
    if not normalized:
        raise ValueError("at least one metric must be configured")
    return normalized


def _variant_order(rows: list[dict[str, Any]]) -> list[str]:
    variants: list[str] = []
    for row in rows:
        variant_id = _row_variant_id(row)
        if variant_id not in variants:
            variants.append(variant_id)
    return variants


def _row_variant_id(row: Mapping[str, Any]) -> str:
    for key in ("variant_id", "mode", "condition"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "suite"


def _pair_key(row: Mapping[str, Any]) -> str | None:
    query_id = row.get("query_id")
    if query_id is None or not str(query_id).strip():
        return None
    suite_id = row.get("suite_id")
    suite_text = str(suite_id).strip() if suite_id is not None else ""
    query_text = str(query_id).strip()
    return f"{suite_text}\0{query_text}" if suite_text else query_text


def _metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    if metric.endswith("_abs_error") or metric.endswith("_error"):
        return "lower_is_better"
    if metric.endswith("_count") and metric.startswith("unsupported"):
        return "lower_is_better"
    return "higher_is_better"


def _sample_caveats(count: int, *, paired: bool = False) -> list[str]:
    label = "paired_query_count" if paired else "observed_count"
    if count == 0:
        return [f"no_values:{label}=0"]
    if count < 5:
        return [
            f"very_small_sample:{label}={count}",
            "bootstrap_interval_descriptive_only",
        ]
    if count < 30:
        return [
            f"small_sample:{label}={count}",
            "bootstrap_interval_descriptive_only",
        ]
    return []


def _global_caveats(
    *,
    variant_summaries: list[dict[str, Any]],
    paired_deltas: list[dict[str, Any]],
    baseline_variant_id: str | None,
    variants: list[str],
) -> list[str]:
    caveats: list[str] = []
    if not variants:
        caveats.append("query_results_empty")
    if baseline_variant_id is not None and baseline_variant_id not in variants:
        caveats.append("baseline_variant_not_found")
    if baseline_variant_id is None:
        caveats.append("paired_delta_disabled_no_baseline_variant_id")
    for variant in variant_summaries:
        metrics = variant.get("metrics") if isinstance(variant.get("metrics"), Mapping) else {}
        for summary in metrics.values():
            for caveat in summary.get("caveats", []) if isinstance(summary, Mapping) else []:
                if caveat not in caveats:
                    caveats.append(caveat)
    for delta in paired_deltas:
        for caveat in delta.get("caveats", []):
            if caveat not in caveats:
                caveats.append(caveat)
    if any("small_sample" in caveat for caveat in caveats):
        caveats.append("small_sample_intervals_should_not_be_used_as_confirmatory_evidence")
    return caveats


def _write_metric_intervals_csv(path: Path, payload: Mapping[str, Any]) -> None:
    columns = [
        "row_type",
        "baseline_variant_id",
        "variant_id",
        "metric",
        "query_count",
        "observed_count",
        "paired_query_count",
        "mean",
        "ci_low",
        "ci_high",
        "mean_delta",
        "delta_ci_low",
        "delta_ci_high",
        "mean_improvement",
        "improvement_ci_low",
        "improvement_ci_high",
        "direction",
        "caveats",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for variant in _list_of_dicts(payload.get("variants")):
            metrics = variant.get("metrics") if isinstance(variant.get("metrics"), Mapping) else {}
            for metric, summary in metrics.items():
                summary = summary if isinstance(summary, Mapping) else {}
                writer.writerow(
                    {
                        "row_type": "aggregate",
                        "baseline_variant_id": "",
                        "variant_id": variant.get("variant_id"),
                        "metric": metric,
                        "query_count": summary.get("query_count"),
                        "observed_count": summary.get("observed_count"),
                        "paired_query_count": "",
                        "mean": summary.get("mean"),
                        "ci_low": summary.get("ci_low"),
                        "ci_high": summary.get("ci_high"),
                        "mean_delta": "",
                        "delta_ci_low": "",
                        "delta_ci_high": "",
                        "mean_improvement": "",
                        "improvement_ci_low": "",
                        "improvement_ci_high": "",
                        "direction": summary.get("direction"),
                        "caveats": ";".join(summary.get("caveats") or []),
                    }
                )
        for delta in _list_of_dicts(payload.get("paired_deltas")):
            writer.writerow(
                {
                    "row_type": "paired_delta",
                    "baseline_variant_id": delta.get("baseline_variant_id"),
                    "variant_id": delta.get("variant_id"),
                    "metric": delta.get("metric"),
                    "query_count": "",
                    "observed_count": "",
                    "paired_query_count": delta.get("paired_query_count"),
                    "mean": "",
                    "ci_low": "",
                    "ci_high": "",
                    "mean_delta": delta.get("mean_delta"),
                    "delta_ci_low": delta.get("ci_low"),
                    "delta_ci_high": delta.get("ci_high"),
                    "mean_improvement": delta.get("mean_improvement"),
                    "improvement_ci_low": delta.get("improvement_ci_low"),
                    "improvement_ci_high": delta.get("improvement_ci_high"),
                    "direction": delta.get("direction"),
                    "caveats": ";".join(delta.get("caveats") or []),
                }
            )


def _run_id(*, rows: list[dict[str, Any]], metrics: Mapping[str, Any]) -> str | None:
    value = metrics.get("run_id")
    if value is not None and str(value).strip():
        return str(value).strip()
    for row in rows:
        value = row.get("run_id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _artifact_name(path: Path | None) -> str | None:
    return path.name if path is not None else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = percentile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _signed_metric(value: float | None, sign: float) -> float | None:
    return _round_metric(value * sign) if value is not None else None


def _signed_interval(
    low: float | None,
    high: float | None,
    sign: float,
) -> tuple[float | None, float | None]:
    signed_low = _signed_metric(low, sign)
    signed_high = _signed_metric(high, sign)
    if signed_low is None or signed_high is None:
        return signed_low, signed_high
    return min(signed_low, signed_high), max(signed_low, signed_high)


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _format_metric(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
