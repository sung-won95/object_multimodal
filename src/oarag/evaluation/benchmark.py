from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oarag.ingestion.eduvidqa import iter_records
from oarag.evaluation.eval import evaluate_query, summarize
from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.project_query import query_project
from oarag.core.schemas import EduVidQARecord, SearchCandidate


DEFAULT_DELTAS = [5, 10, 15]


class SearchClient(Protocol):
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    output_dir: Path
    metrics_path: Path
    query_results_path: Path
    summary_path: Path
    metrics: dict[str, Any]


def run_benchmark(
    *,
    client: SearchClient,
    manifest_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> BenchmarkRun:
    manifest = _read_json(manifest_path)
    base_dir = manifest_path.expanduser().resolve().parent
    run_id = str(manifest.get("run_id") or f"run_{int(time.time())}")
    resolved_output_dir = _resolve_output_dir(
        output_dir=output_dir,
        manifest=manifest,
        base_dir=base_dir,
        run_id=run_id,
    )
    resolved_repo_root = (repo_root or base_dir).expanduser().resolve()
    deltas = [int(delta) for delta in manifest.get("deltas", DEFAULT_DELTAS)]

    all_results: list[dict[str, Any]] = []
    suite_metrics: list[dict[str, Any]] = []
    for suite in manifest.get("suites", []):
        suite_type = str(suite.get("type", "")).strip()
        if suite_type == "eduvidqa":
            metrics, results = run_eduvidqa_suite(
                client=client,
                suite=suite,
                base_dir=base_dir,
                deltas=deltas,
            )
        elif suite_type == "local_project":
            metrics, results = run_local_project_suite(
                client=client,
                suite=suite,
                base_dir=base_dir,
                repo_root=resolved_repo_root,
                deltas=deltas,
            )
        else:
            raise ValueError(f"Unsupported benchmark suite type: {suite_type}")
        suite_metrics.append(metrics)
        all_results.extend(results)

    metrics_payload = {
        "run_id": run_id,
        "suite_count": len(suite_metrics),
        "query_count": len(all_results),
        "deltas": deltas,
        "suites": suite_metrics,
        "anti_overfit": _anti_overfit_summary(suite_metrics),
    }

    metrics_path = resolved_output_dir / "metrics.json"
    query_results_path = resolved_output_dir / "query_results.jsonl"
    summary_path = resolved_output_dir / "summary.md"
    write_json(metrics_path, metrics_payload)
    write_jsonl(query_results_path, all_results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_markdown(metrics_payload), encoding="utf-8")

    return BenchmarkRun(
        run_id=run_id,
        output_dir=resolved_output_dir,
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        summary_path=summary_path,
        metrics=metrics_payload,
    )


def run_eduvidqa_suite(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    base_dir: Path,
    deltas: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_path = _resolve_path(base_dir, suite["input"])
    index_uid = str(suite["index"])
    limit = int(suite.get("limit", 5))
    record_limit = _optional_int(suite.get("record_limit"))
    suite_id = str(suite.get("suite_id") or input_path.stem)
    domain = str(suite.get("domain") or "eduvidqa")

    evals = []
    rows: list[dict[str, Any]] = []
    latencies = []
    for record in iter_records(input_path, limit=record_limit):
        response = client.search(index_uid, record.question, limit=limit)
        latencies.append(_optional_float(response.get("processingTimeMs")))
        candidates = [
            SearchCandidate.from_hit(rank=rank, hit=hit)
            for rank, hit in enumerate(response.get("hits", []), start=1)
        ]
        query_eval = evaluate_query(record, candidates, deltas=deltas)
        evals.append(query_eval)
        candidate_errors = _candidate_abs_errors(record, candidates)
        rows.append(
            {
                "suite_id": suite_id,
                "suite_type": "eduvidqa",
                "domain": domain,
                "query_id": record.sample_id,
                "index": index_uid,
                "query_text": record.question,
                "best_abs_error": query_eval.best_abs_error,
                "hit_by_delta": {str(key): value for key, value in query_eval.hit_by_delta.items()},
                "mrr": _reciprocal_rank(candidate_errors, deltas=max(deltas)),
                "top_candidate": _candidate_result(candidates[0], candidate_errors[0])
                if candidates
                else None,
                "processing_time_ms": response.get("processingTimeMs"),
            }
        )

    metric = summarize(evals, deltas=deltas)
    metric.update(
        {
            "suite_id": suite_id,
            "suite_type": "eduvidqa",
            "domain": domain,
            "index": index_uid,
            "limit": limit,
            "mrr_at_max_delta": round(_mean(row["mrr"] for row in rows), 4) if rows else None,
            "top1_mean_abs_error": round(
                _mean(
                    row["top_candidate"]["abs_error"]
                    for row in rows
                    if row.get("top_candidate") is not None
                    and row["top_candidate"].get("abs_error") is not None
                ),
                4,
            )
            if rows
            else None,
            "mean_processing_time_ms": round(
                _mean(value for value in latencies if value is not None), 4
            )
            if any(value is not None for value in latencies)
            else None,
            "query_count": len(rows),
        }
    )
    return metric, rows


def run_local_project_suite(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
    deltas: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queries_path = _resolve_path(base_dir, suite["queries"])
    project_dir = _project_dir_from_suite(suite=suite, base_dir=base_dir, repo_root=repo_root)
    index_uid = str(suite["index"])
    visual_index_uid = str(suite["visual_index"]) if suite.get("visual_index") else None
    limit = int(suite.get("limit", 5))
    neighbor_count = int(suite.get("neighbor_count", 1))
    suite_id = str(suite.get("suite_id") or project_dir.name)
    domain = str(suite.get("domain") or "local_project")
    video_filter = str(suite["video_id"]) if suite.get("video_id") is not None else None
    domain_lexicon_path = (
        Path(str(suite["domain_lexicon"])) if suite.get("domain_lexicon") is not None else None
    )
    rerank_enabled = bool(suite.get("rerank", False))
    rerank_time_hint = str(suite["rerank_time_hint"]) if suite.get("rerank_time_hint") else None
    rerank_time_hint_field = (
        str(suite["rerank_time_hint_field"]) if suite.get("rerank_time_hint_field") else None
    )

    rows: list[dict[str, Any]] = []
    latencies = []
    domain_lexicon_metadata: dict[str, Any] | None = None
    rerank_metadata: dict[str, Any] | None = None
    for query_row in _read_query_csv(queries_path):
        if video_filter and str(query_row.get("video_id")) != video_filter:
            continue
        row_rerank_time_hint = rerank_time_hint
        if rerank_time_hint_field and query_row.get(rerank_time_hint_field):
            row_rerank_time_hint = str(query_row[rerank_time_hint_field])
        started = time.perf_counter()
        response = query_project(
            client=client,
            index_uid=index_uid,
            project_dir=project_dir,
            query=str(query_row["query_text"]),
            visual_index_uid=visual_index_uid,
            limit=limit,
            neighbor_count=neighbor_count,
            domain_lexicon_path=domain_lexicon_path,
            rerank=rerank_enabled,
            rerank_time_hint=row_rerank_time_hint,
        )
        if domain_lexicon_metadata is None:
            loaded_metadata = response.get("domain_lexicon")
            domain_lexicon_metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
        if rerank_metadata is None:
            loaded_rerank = (response.get("retrieval_context") or {}).get("rerank")
            rerank_metadata = loaded_rerank if isinstance(loaded_rerank, dict) else None
        elapsed_ms = (time.perf_counter() - started) * 1000
        processing_time_ms = response.get("processing_time_ms")
        latencies.append(_optional_float(processing_time_ms) or elapsed_ms)
        bundles = response.get("bundles", [])
        response_counts = response.get("counts") if isinstance(response.get("counts"), dict) else {}
        top_candidate = bundles[0].get("candidate") if bundles else None
        expected_ranges = parse_time_hint(str(query_row.get("expected_time_hint") or ""))
        candidate_errors = [
            _local_candidate_error(bundle.get("candidate"), expected_ranges) for bundle in bundles
        ]
        top_error = candidate_errors[0] if candidate_errors else None
        best_error = min((error for error in candidate_errors if error is not None), default=None)
        hit_by_delta = {
            str(delta): best_error is not None and best_error <= delta for delta in deltas
        }
        frame_backed = any(
            bundle.get("evidence_window", {}).get("frame_refs") for bundle in bundles
        )
        linked_backed = any(bundle.get("linked_entities") for bundle in bundles)
        rows.append(
            {
                "suite_id": suite_id,
                "suite_type": "local_project",
                "domain": domain,
                "query_id": str(query_row.get("query_id") or query_row.get("query_text")),
                "video_id": query_row.get("video_id"),
                "index": index_uid,
                "visual_index": visual_index_uid,
                "query_text": query_row.get("query_text"),
                "expected_time_hint": query_row.get("expected_time_hint"),
                "best_abs_error": best_error,
                "top1_abs_error": top_error,
                "hit_by_delta": hit_by_delta,
                "mrr": _reciprocal_rank(candidate_errors, deltas=max(deltas)),
                "top_candidate": top_candidate,
                "frame_backed": frame_backed,
                "linked_entity_backed": linked_backed,
                "domain_lexicon": response.get("domain_lexicon"),
                "query_expansion": response.get("query_expansion"),
                "rerank": (response.get("retrieval_context") or {}).get("rerank"),
                "top_rerank": bundles[0].get("rerank") if bundles else None,
                "search_hits": response_counts.get("search_hits"),
                "segment_search_hits": response_counts.get("segment_search_hits"),
                "visual_entity_search_hits": response_counts.get("visual_entity_search_hits"),
                "bundled_visual_entities": response_counts.get("bundled_visual_entities"),
                "bundled_linked_entities": response_counts.get("bundled_linked_entities"),
                "processing_time_ms": processing_time_ms,
                "elapsed_time_ms": round(elapsed_ms, 4),
            }
        )

    metric = {
        "suite_id": suite_id,
        "suite_type": "local_project",
        "domain": domain,
        "index": index_uid,
        "visual_index": visual_index_uid,
        "limit": limit,
        "query_count": len(rows),
        "domain_lexicon": domain_lexicon_metadata or _empty_domain_lexicon_metadata(),
        "rerank": _suite_rerank_metadata(
            enabled=rerank_enabled,
            observed=rerank_metadata,
            time_hint=rerank_time_hint,
            time_hint_field=rerank_time_hint_field,
        ),
        "mean_abs_error": round(
            _mean(row["best_abs_error"] for row in rows if row["best_abs_error"] is not None), 4
        )
        if any(row["best_abs_error"] is not None for row in rows)
        else None,
        "top1_mean_abs_error": round(
            _mean(row["top1_abs_error"] for row in rows if row["top1_abs_error"] is not None),
            4,
        )
        if any(row["top1_abs_error"] is not None for row in rows)
        else None,
        "mrr_at_max_delta": round(_mean(row["mrr"] for row in rows), 4) if rows else None,
        "frame_backed_ratio": round(_mean(1.0 if row["frame_backed"] else 0.0 for row in rows), 4)
        if rows
        else None,
        "linked_entity_backed_ratio": round(
            _mean(1.0 if row["linked_entity_backed"] else 0.0 for row in rows), 4
        )
        if rows
        else None,
        "mean_processing_time_ms": round(
            _mean(value for value in latencies if value is not None), 4
        )
        if any(value is not None for value in latencies)
        else None,
    }
    for delta in deltas:
        metric[f"hit_at_{delta}s"] = (
            round(_mean(1.0 if row["hit_by_delta"][str(delta)] else 0.0 for row in rows), 4)
            if rows
            else None
        )
    return metric, rows


def parse_time_hint(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for start, end in re_time_ranges(value):
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def re_time_ranges(value: str) -> list[tuple[float, float]]:
    import re

    ranges: list[tuple[float, float]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s?", value):
        ranges.append((float(match.group(1)), float(match.group(2))))
    if ranges:
        return ranges
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*s", value):
        point = float(match.group(1))
        ranges.append((point, point))
    return ranges


def _candidate_abs_errors(
    record: EduVidQARecord,
    candidates: list[SearchCandidate],
) -> list[float | None]:
    errors: list[float | None] = []
    for candidate in candidates:
        if candidate.timestamp_center is None or not record.timestamp_points:
            errors.append(None)
            continue
        errors.append(
            min(abs(candidate.timestamp_center - gold) for gold in record.timestamp_points)
        )
    return errors


def _candidate_result(candidate: SearchCandidate, abs_error: float | None) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "segment_id": candidate.segment_id,
        "sample_id": candidate.sample_id,
        "video_id": candidate.video_id,
        "timestamp_center": candidate.timestamp_center,
        "abs_error": abs_error,
    }


def _reciprocal_rank(candidate_errors: list[float | None], *, deltas: int) -> float:
    for index, error in enumerate(candidate_errors, start=1):
        if error is not None and error <= deltas:
            return 1.0 / index
    return 0.0


def _local_candidate_error(
    candidate: dict[str, Any] | None,
    expected_ranges: list[tuple[float, float]],
) -> float | None:
    if not candidate or not expected_ranges:
        return None
    center = _optional_float(candidate.get("timestamp_center"))
    if center is None:
        start = _optional_float(candidate.get("start_time"))
        end = _optional_float(candidate.get("end_time"))
        if start is not None and end is not None:
            center = (start + end) / 2.0
        else:
            center = start if start is not None else end
    if center is None:
        return None
    return min(
        _point_to_range_distance(center, expected_range) for expected_range in expected_ranges
    )


def _point_to_range_distance(point: float, expected_range: tuple[float, float]) -> float:
    start, end = expected_range
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def _read_query_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in benchmark manifest: {path}")
    return payload


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_output_dir(
    *,
    output_dir: Path | None,
    manifest: dict[str, Any],
    base_dir: Path,
    run_id: str,
) -> Path:
    if output_dir is not None:
        path = output_dir.expanduser()
    elif manifest.get("output_dir") is not None:
        path = Path(str(manifest["output_dir"])).expanduser()
    else:
        path = Path("reports") / "perf_runs" / run_id
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _project_dir_from_suite(*, suite: dict[str, Any], base_dir: Path, repo_root: Path) -> Path:
    if suite.get("project_dir") is not None:
        return _resolve_path(base_dir, suite["project_dir"])
    if suite.get("project_id") is not None:
        return (repo_root / "artifacts" / "projects" / str(suite["project_id"])).resolve()
    raise ValueError("local_project suite requires project_dir or project_id")


def _anti_overfit_summary(suites: list[dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, list[dict[str, Any]]] = {}
    for suite in suites:
        domains.setdefault(str(suite.get("domain") or "unknown"), []).append(suite)
    return {
        "domain_count": len(domains),
        "domains": {
            domain: {
                "suite_count": len(items),
                "query_count": sum(
                    int(item.get("query_count") or item.get("count") or 0) for item in items
                ),
                "mean_hit_at_10s": _domain_mean(items, "hit_at_10s"),
                "mean_mrr_at_max_delta": _domain_mean(items, "mrr_at_max_delta"),
            }
            for domain, items in sorted(domains.items())
        },
    }


def _domain_mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [_optional_float(item.get(key)) for item in items]
    values = [value for value in values if value is not None]
    return round(_mean(values), 4) if values else None


def _summary_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        f"# Retrieval Benchmark: {metrics['run_id']}",
        "",
        "## Suites",
        "",
        "| suite | type | domain | lexicon | queries | Hit@10s | MRR | latency ms | "
        "frame-backed | linked-backed | rerank |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for suite in metrics["suites"]:
        lines.append(
            "| {suite_id} | {suite_type} | {domain} | {lexicon} | {query_count} | "
            "{hit10} | {mrr} | {latency} | {frame} | {linked} | {rerank} |".format(
                suite_id=suite.get("suite_id"),
                suite_type=suite.get("suite_type"),
                domain=suite.get("domain"),
                lexicon=_format_domain_lexicon(suite.get("domain_lexicon")),
                query_count=suite.get("query_count") or suite.get("count") or 0,
                hit10=_format_metric(suite.get("hit_at_10s")),
                mrr=_format_metric(suite.get("mrr_at_max_delta")),
                latency=_format_metric(suite.get("mean_processing_time_ms")),
                frame=_format_metric(suite.get("frame_backed_ratio")),
                linked=_format_metric(suite.get("linked_entity_backed_ratio")),
                rerank=_format_rerank(suite.get("rerank")),
            )
        )
    lines.extend(["", "## Anti-Overfit View", ""])
    for domain, payload in metrics["anti_overfit"]["domains"].items():
        lines.append(
            f"- `{domain}`: suites={payload['suite_count']}, queries={payload['query_count']}, "
            f"mean_hit_at_10s={_format_metric(payload['mean_hit_at_10s'])}, "
            f"mean_mrr={_format_metric(payload['mean_mrr_at_max_delta'])}"
        )
    lines.append("")
    return "\n".join(lines)


def _empty_domain_lexicon_metadata() -> dict[str, Any]:
    return {
        "enabled": False,
        "source_path": None,
        "canonical_term_count": 0,
        "alias_count": 0,
        "term_count": 0,
    }


def _suite_rerank_metadata(
    *,
    enabled: bool,
    observed: dict[str, Any] | None,
    time_hint: str | None,
    time_hint_field: str | None,
) -> dict[str, Any]:
    source = "query_text"
    if time_hint_field:
        source = f"csv:{time_hint_field}"
    elif time_hint:
        source = "suite:rerank_time_hint"

    return {
        "enabled": enabled,
        "strategy": observed.get("strategy") if enabled and observed else None,
        "weights": observed.get("weights") if enabled and observed else None,
        "time_hint_source": source,
    }


def _format_domain_lexicon(value: Any) -> str:
    if not isinstance(value, dict) or not value.get("enabled"):
        return "off"
    source_path = value.get("source_path")
    if source_path:
        return f"on ({Path(str(source_path)).name})"
    return "on"


def _format_rerank(value: Any) -> str:
    if not isinstance(value, dict) or not value.get("enabled"):
        return "off"
    strategy = value.get("strategy")
    return f"on ({strategy})" if strategy else "on"


def _format_metric(value: Any) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "-"
    if math.isclose(parsed, round(parsed)):
        return str(int(round(parsed)))
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _mean(values: Any) -> float:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
