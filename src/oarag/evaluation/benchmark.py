from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oarag.ingestion.eduvidqa import iter_records
from oarag.evaluation.eval import (
    candidate_abs_errors as eval_candidate_abs_errors,
    candidate_diagnostics,
    evaluate_query,
    summarize,
)
from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.evidence import (
    frame_id,
    read_jsonl,
    resolve_project_path,
    segment_window,
)
from oarag.retrieval.answer import compose_answer
from oarag.retrieval.project_index import segment_artifact_path
from oarag.retrieval.project_query import (
    DEFAULT_HYBRID_EMBEDDER,
    DEFAULT_HYBRID_SEMANTIC_RATIO,
    SEGMENT_HIT_SOURCE,
    WINDOW_HIT_SOURCE,
    query_project,
)
from oarag.core.schemas import EduVidQARecord, SearchCandidate


DEFAULT_DELTAS = [5, 10, 15]
PUBLIC_ABLATION_SCHEMA_VERSION = "retrieval-ablation-public-v1"
DEFAULT_ABLATION_MODES = [
    "transcript-only",
    "visual-only",
    "time-aligned",
    "object-aligned",
]
MATRIX_SCHEMA_VERSION = "retrieval-answer-ablation-matrix-v1"
DEFAULT_MATRIX_VARIANTS = [
    {
        "variant_id": "segment_lexical",
        "label": "Segment lexical baseline",
        "index_kind": SEGMENT_HIT_SOURCE,
        "use_domain_lexicon": False,
    },
    {
        "variant_id": "domain_lexicon",
        "label": "Domain lexicon expansion",
        "index_kind": SEGMENT_HIT_SOURCE,
        "use_domain_lexicon": True,
    },
    {
        "variant_id": "hybrid",
        "label": "Hybrid retrieval",
        "index_kind": SEGMENT_HIT_SOURCE,
        "hybrid_retrieval": True,
        "use_domain_lexicon": False,
    },
    {
        "variant_id": "window",
        "label": "Window retrieval",
        "index_kind": WINDOW_HIT_SOURCE,
        "use_domain_lexicon": False,
    },
    {
        "variant_id": "window_hybrid",
        "label": "Window + hybrid retrieval",
        "index_kind": WINDOW_HIT_SOURCE,
        "hybrid_retrieval": True,
        "use_domain_lexicon": False,
    },
    {
        "variant_id": "rerank",
        "label": "Deterministic rerank",
        "index_kind": SEGMENT_HIT_SOURCE,
        "rerank": True,
        "use_domain_lexicon": False,
    },
]
ABLATION_MODE_ALIASES = {
    "transcript": "transcript-only",
    "transcript_only": "transcript-only",
    "transcript-only": "transcript-only",
    "visual": "visual-only",
    "visual_only": "visual-only",
    "visual-only": "visual-only",
    "time": "time-aligned",
    "time_aligned": "time-aligned",
    "time-aligned": "time-aligned",
    "object": "object-aligned",
    "object_aligned": "object-aligned",
    "object-aligned": "object-aligned",
}


class SearchClient(Protocol):
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    output_dir: Path
    metrics_path: Path
    metrics_csv_path: Path
    query_results_path: Path
    summary_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class AblationProject:
    project_dir: Path
    segments: list[dict[str, Any]]
    segment_lookup: dict[str, dict[str, Any]]
    frame_lookup: dict[str, dict[str, Any]]
    visual_entities: list[dict[str, Any]]
    entity_lookup: dict[str, dict[str, Any]]
    links_by_segment: dict[str, list[dict[str, Any]]]
    links_by_entity: dict[str, list[dict[str, Any]]]


def run_benchmark(
    *,
    client: SearchClient,
    manifest_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    diagnostic_top_k: int | None = None,
    run_id: str | None = None,
) -> BenchmarkRun:
    manifest = _read_json(manifest_path)
    base_dir = manifest_path.expanduser().resolve().parent
    run_id = str(run_id or manifest.get("run_id") or f"run_{int(time.time())}")
    resolved_output_dir = _resolve_output_dir(
        output_dir=output_dir,
        manifest=manifest,
        base_dir=base_dir,
        run_id=run_id,
    )
    resolved_repo_root = (repo_root or base_dir).expanduser().resolve()
    deltas = [int(delta) for delta in manifest.get("deltas", DEFAULT_DELTAS)]
    default_diagnostic_top_k = (
        diagnostic_top_k
        if diagnostic_top_k is not None
        else _optional_int(manifest.get("diagnostic_top_k"))
    )

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
                diagnostic_top_k=_suite_diagnostic_top_k(
                    suite,
                    default=default_diagnostic_top_k,
                ),
            )
        elif suite_type == "local_project":
            metrics, results = run_local_project_suite(
                client=client,
                suite=suite,
                base_dir=base_dir,
                repo_root=resolved_repo_root,
                deltas=deltas,
            )
        elif suite_type == "retrieval_ablation":
            metrics, results = run_retrieval_ablation_suite(
                client=client,
                suite=suite,
                base_dir=base_dir,
                repo_root=resolved_repo_root,
                run_id=run_id,
                deltas=deltas,
            )
        elif suite_type == "retrieval_answer_matrix":
            metrics, results = run_retrieval_answer_matrix_suite(
                client=client,
                suite=suite,
                base_dir=base_dir,
                repo_root=resolved_repo_root,
                run_id=run_id,
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
    metrics_csv_path = resolved_output_dir / "metrics_summary.csv"
    query_results_path = resolved_output_dir / "query_results.jsonl"
    summary_path = resolved_output_dir / "summary.md"
    write_json(metrics_path, metrics_payload)
    _write_metrics_summary_csv(metrics_csv_path, metrics_payload)
    write_jsonl(query_results_path, all_results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_markdown(metrics_payload), encoding="utf-8")

    return BenchmarkRun(
        run_id=run_id,
        output_dir=resolved_output_dir,
        metrics_path=metrics_path,
        metrics_csv_path=metrics_csv_path,
        query_results_path=query_results_path,
        summary_path=summary_path,
        metrics=metrics_payload,
    )


def run_retrieval_ablation_suite(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
    run_id: str,
    deltas: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_id = str(suite.get("suite_id") or suite.get("project_id") or "retrieval_ablation")
    domain = str(suite.get("domain") or "public_synthetic")
    project = _load_ablation_project(suite=suite, base_dir=base_dir, repo_root=repo_root)
    queries = _read_ablation_queries(base_dir=base_dir, suite=suite)
    index_uid = str(suite["index"])
    visual_index_uid = _optional_str(suite.get("visual_index"))
    limit = int(suite.get("limit", 5))
    modes = _ablation_modes(suite)

    rows: list[dict[str, Any]] = []
    for query_index, query_row in enumerate(queries, start=1):
        query_text = _query_text(query_row, query_index=query_index)
        query_id = _query_id(query_row, query_index=query_index)
        expected_ranges = _expected_ranges(query_row)
        for mode in modes:
            if mode == "visual-only" and visual_index_uid is None:
                raise ValueError("retrieval_ablation visual-only mode requires visual_index")
            row = _run_ablation_query(
                client=client,
                project=project,
                run_id=run_id,
                suite_id=suite_id,
                domain=domain,
                index_uid=index_uid,
                visual_index_uid=visual_index_uid,
                query_id=query_id,
                query_row=query_row,
                query_text=query_text,
                expected_ranges=expected_ranges,
                mode=mode,
                limit=limit,
                deltas=deltas,
            )
            rows.append(row)

    mode_metrics = [
        _ablation_mode_metrics(
            mode=mode,
            rows=[row for row in rows if row.get("mode") == mode],
            deltas=deltas,
        )
        for mode in modes
    ]
    metric: dict[str, Any] = {
        "schema_version": PUBLIC_ABLATION_SCHEMA_VERSION,
        "suite_id": suite_id,
        "suite_type": "retrieval_ablation",
        "domain": domain,
        "query_count": len(queries),
        "result_count": len(rows),
        "mode_count": len(mode_metrics),
        "modes": mode_metrics,
        "mode_metrics": {str(item["mode"]): item for item in mode_metrics},
        "privacy": _public_ablation_privacy_payload(),
    }
    for delta in deltas:
        metric[f"hit_at_{delta}s"] = _mean_or_none(
            item.get(f"hit_at_{delta}s") for item in mode_metrics
        )
    metric["evidence_coverage_ratio"] = _mean_or_none(
        item.get("evidence_coverage_ratio") for item in mode_metrics
    )
    metric["frame_backed_ratio"] = _mean_or_none(
        item.get("frame_backed_ratio") for item in mode_metrics
    )
    metric["linked_entity_ratio"] = _mean_or_none(
        item.get("linked_entity_ratio") for item in mode_metrics
    )
    metric["linked_entity_backed_ratio"] = metric["linked_entity_ratio"]
    metric["mean_processing_time_ms"] = _mean_or_none(
        item.get("mean_processing_time_ms") for item in mode_metrics
    )
    return metric, rows


def run_retrieval_answer_matrix_suite(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
    run_id: str,
    deltas: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_id = str(suite.get("suite_id") or suite.get("project_id") or "retrieval_answer_matrix")
    domain = str(suite.get("domain") or "public_synthetic")
    project_dir = _project_dir_from_suite(suite=suite, base_dir=base_dir, repo_root=repo_root)
    queries = _read_ablation_queries(base_dir=base_dir, suite=suite)
    variants = _matrix_variants(suite)
    include_answer = _bool_config(suite, "include_answer", default=True)

    rows: list[dict[str, Any]] = []
    for query_index, query_row in enumerate(queries, start=1):
        query_text = _query_text(query_row, query_index=query_index)
        query_id = _query_id(query_row, query_index=query_index)
        expected_ranges = _expected_ranges(query_row)
        expected_segment_ids = _expected_segment_ids(query_row)
        expected_window_ids = _expected_window_ids(query_row)
        for variant in variants:
            rows.append(
                _run_matrix_query(
                    client=client,
                    suite=suite,
                    variant=variant,
                    project_dir=project_dir,
                    run_id=run_id,
                    suite_id=suite_id,
                    domain=domain,
                    query_id=query_id,
                    query_row=query_row,
                    query_text=query_text,
                    expected_ranges=expected_ranges,
                    expected_segment_ids=expected_segment_ids,
                    expected_window_ids=expected_window_ids,
                    include_answer=include_answer,
                    deltas=deltas,
                )
            )

    variant_metrics = [
        _matrix_variant_metrics(
            variant=variant,
            rows=[row for row in rows if row.get("variant_id") == variant["variant_id"]],
            deltas=deltas,
        )
        for variant in variants
    ]
    metric: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "suite_id": suite_id,
        "suite_type": "retrieval_answer_matrix",
        "domain": domain,
        "query_count": len(queries),
        "result_count": len(rows),
        "variant_count": len(variant_metrics),
        "supported_variant_ids": [variant["variant_id"] for variant in _matrix_variants({})],
        "variants": variant_metrics,
        "variant_metrics": {str(item["variant_id"]): item for item in variant_metrics},
        "dataset_descriptor": _dataset_descriptor(
            suite=suite,
            suite_id=suite_id,
            suite_type="retrieval_answer_matrix",
            domain=domain,
            query_count=len(queries),
        ),
        "privacy": _matrix_privacy_payload(),
    }
    for delta in deltas:
        metric[f"hit_at_{delta}s"] = _mean_or_none(
            item.get(f"hit_at_{delta}s") for item in variant_metrics
        )
    for key in (
        "mrr_at_max_delta",
        "evidence_coverage_ratio",
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
    ):
        metric[key] = _mean_or_none(item.get(key) for item in variant_metrics)
    return metric, rows


def run_eduvidqa_suite(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    base_dir: Path,
    deltas: list[int],
    diagnostic_top_k: int = 0,
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
        candidate_errors = eval_candidate_abs_errors(record, candidates)
        row = {
            "suite_id": suite_id,
            "suite_type": "eduvidqa",
            "domain": domain,
            "query_id": record.sample_id,
            "index": index_uid,
            "query_text": record.question,
            "best_abs_error": query_eval.best_abs_error,
            "hit_by_delta": {str(key): value for key, value in query_eval.hit_by_delta.items()},
            "topk_recall": {str(key): value for key, value in query_eval.recall_by_k.items()},
            "mrr": _reciprocal_rank(candidate_errors, deltas=max(deltas)),
            "top_candidate": _candidate_result(candidates[0], candidate_errors[0])
            if candidates
            else None,
            "processing_time_ms": response.get("processingTimeMs"),
        }
        if diagnostic_top_k > 0:
            row["diagnostic_candidates"] = candidate_diagnostics(
                record,
                candidates,
                top_k=diagnostic_top_k,
                include_private_fields=bool(suite.get("include_private_fields", False)),
            )
        rows.append(row)

    metric = summarize(evals, deltas=deltas)
    metric.update(
        {
            "suite_id": suite_id,
            "suite_type": "eduvidqa",
            "domain": domain,
            "index": index_uid,
            "limit": limit,
            "diagnostic_top_k": diagnostic_top_k,
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
    visual_index_uid = _optional_str(suite.get("visual_index"))
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
    hybrid_retrieval_enabled = bool(suite.get("hybrid_retrieval", False))
    hybrid_embedder = str(suite.get("hybrid_embedder") or DEFAULT_HYBRID_EMBEDDER)
    hybrid_semantic_ratio = (
        _optional_float(suite.get("hybrid_semantic_ratio")) or DEFAULT_HYBRID_SEMANTIC_RATIO
    )

    rows: list[dict[str, Any]] = []
    latencies = []
    domain_lexicon_metadata: dict[str, Any] | None = None
    rerank_metadata: dict[str, Any] | None = None
    hybrid_metadata: dict[str, Any] | None = None
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
            visual_index_uid=visual_index_uid,
            project_dir=project_dir,
            query=str(query_row["query_text"]),
            limit=limit,
            neighbor_count=neighbor_count,
            domain_lexicon_path=domain_lexicon_path,
            rerank=rerank_enabled,
            rerank_time_hint=row_rerank_time_hint,
            hybrid_retrieval=hybrid_retrieval_enabled,
            hybrid_embedder=hybrid_embedder,
            hybrid_semantic_ratio=hybrid_semantic_ratio,
        )
        if domain_lexicon_metadata is None:
            loaded_metadata = response.get("domain_lexicon")
            domain_lexicon_metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
        if rerank_metadata is None:
            loaded_rerank = (response.get("retrieval_context") or {}).get("rerank")
            rerank_metadata = loaded_rerank if isinstance(loaded_rerank, dict) else None
        if hybrid_metadata is None:
            loaded_hybrid = (response.get("retrieval_context") or {}).get("hybrid_retrieval")
            hybrid_metadata = loaded_hybrid if isinstance(loaded_hybrid, dict) else None
        elapsed_ms = (time.perf_counter() - started) * 1000
        processing_time_ms = response.get("processing_time_ms")
        latencies.append(_optional_float(processing_time_ms) or elapsed_ms)
        bundles = response.get("bundles", [])
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
                "hybrid_retrieval": (response.get("retrieval_context") or {}).get(
                    "hybrid_retrieval"
                ),
                "top_rerank": bundles[0].get("rerank") if bundles else None,
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
        "hybrid_retrieval": _suite_hybrid_metadata(
            enabled=hybrid_retrieval_enabled,
            observed=hybrid_metadata,
            embedder=hybrid_embedder,
            semantic_ratio=hybrid_semantic_ratio,
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


def _run_ablation_query(
    *,
    client: SearchClient,
    project: AblationProject,
    run_id: str,
    suite_id: str,
    domain: str,
    index_uid: str,
    visual_index_uid: str | None,
    query_id: str,
    query_row: dict[str, Any],
    query_text: str,
    expected_ranges: list[tuple[float, float]],
    mode: str,
    limit: int,
    deltas: list[int],
) -> dict[str, Any]:
    started = time.perf_counter()
    search_responses: list[dict[str, Any]] = []
    raw_hit_count = 0
    candidates: list[dict[str, Any]] = []

    if mode in {"transcript-only", "time-aligned", "object-aligned"}:
        response = client.search(index_uid, query_text, limit=limit)
        search_responses.append(response)
        hits = _list_of_dicts(response.get("hits"))
        raw_hit_count += len(hits)
        candidates.extend(
            _segment_candidates_from_hits(
                hits=hits,
                project=project,
                mode=mode,
                index_uid=index_uid,
            )
        )

    if mode in {"visual-only", "object-aligned"} and visual_index_uid is not None:
        response = client.search(visual_index_uid, query_text, limit=limit)
        search_responses.append(response)
        hits = _list_of_dicts(response.get("hits"))
        raw_hit_count += len(hits)
        candidates.extend(
            _visual_candidates_from_hits(
                hits=hits,
                project=project,
                mode=mode,
                index_uid=visual_index_uid,
            )
        )

    if mode == "object-aligned":
        candidates = _merge_ablation_candidates(candidates, limit=limit)
    else:
        candidates = sorted(candidates, key=lambda item: _candidate_mode_sort_key(item, mode))[
            :limit
        ]

    elapsed_ms = round((time.perf_counter() - started) * 1000, 4)
    candidate_errors = [_ablation_candidate_error(candidate, expected_ranges) for candidate in candidates]
    best_error = min((error for error in candidate_errors if error is not None), default=None)
    top_error = candidate_errors[0] if candidate_errors else None
    hit_by_delta = {
        str(delta): best_error is not None and best_error <= delta for delta in deltas
    }
    frame_backed = any(candidate.get("frame_backed") for candidate in candidates)
    linked_entity_backed = any(candidate.get("linked_entity_backed") for candidate in candidates)
    evidence_covered = _mode_evidence_covered(mode=mode, candidates=candidates)
    top_candidate = candidates[0] if candidates else None

    return {
        "schema_version": PUBLIC_ABLATION_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "suite_type": "retrieval_ablation",
        "domain": domain,
        "mode": mode,
        "query_id": query_id,
        "query_label": _optional_public_label(query_row),
        "privacy": _public_ablation_privacy_payload(),
        "index_ref": f"index:{_short_hash(index_uid)}",
        "visual_index_ref": f"index:{_short_hash(visual_index_uid)}"
        if visual_index_uid is not None
        else None,
        "expected_time_available": bool(expected_ranges),
        "search_hit_count": raw_hit_count,
        "candidate_count": len(candidates),
        "best_abs_error": best_error,
        "top1_abs_error": top_error,
        "hit_by_delta": hit_by_delta,
        "mrr": _reciprocal_rank(candidate_errors, deltas=max(deltas)),
        "evidence_covered": evidence_covered,
        "frame_backed": frame_backed,
        "linked_entity_backed": linked_entity_backed,
        "top_candidate": _public_ablation_candidate(top_candidate),
        "processing_time_ms": _combined_processing_time_ms(search_responses),
        "elapsed_time_ms": elapsed_ms,
    }


def _segment_candidates_from_hits(
    *,
    hits: list[dict[str, Any]],
    project: AblationProject,
    mode: str,
    index_uid: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        segment_id = str(hit.get("segment_id") or hit.get("sample_id") or "").strip()
        segment = project.segment_lookup.get(segment_id, hit)
        target_segment_id = str(segment.get("segment_id") or segment_id).strip()
        frame_ids = _segment_frame_ids(segment) if mode in {"time-aligned", "object-aligned"} else []
        linked_entity_ids = (
            _linked_entity_ids_for_segment(project, target_segment_id)
            if mode == "object-aligned"
            else []
        )
        candidates.append(
            {
                "source": "segment",
                "index": index_uid,
                "rank": rank,
                "candidate_id": target_segment_id or str(hit.get("sample_id") or rank),
                "target_segment_id": target_segment_id,
                "score": _optional_float(hit.get("_rankingScore")),
                "start_time": _first_float(hit.get("start_time"), segment.get("start_time")),
                "end_time": _first_float(hit.get("end_time"), segment.get("end_time")),
                "timestamp_center": _first_float(
                    hit.get("timestamp_center"),
                    segment.get("timestamp_center"),
                ),
                "has_transcript_evidence": bool(
                    str(hit.get("transcript_text") or segment.get("transcript_text") or "").strip()
                ),
                "frame_ids": frame_ids,
                "frame_backed": bool(frame_ids),
                "linked_entity_ids": linked_entity_ids,
                "linked_entity_backed": bool(linked_entity_ids),
            }
        )
    return candidates


def _visual_candidates_from_hits(
    *,
    hits: list[dict[str, Any]],
    project: AblationProject,
    mode: str,
    index_uid: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        entity = _visual_entity_for_hit(hit=hit, project=project)
        target_segment_id, resolution_method = _resolve_visual_target_segment(
            hit=hit,
            entity=entity,
            project=project,
            use_entity_links=mode == "object-aligned",
        )
        target = project.segment_lookup.get(target_segment_id or "", {})
        candidate_id = str(
            entity.get("entity_id") or hit.get("entity_id") or hit.get("frame_id") or rank
        )
        frame_ids = _visual_candidate_frame_ids(hit=hit, entity=entity, target=target)
        linked_entity_ids: list[str] = []
        if mode == "object-aligned":
            if resolution_method == "entity_link":
                linked_entity_ids.extend(_entity_id_values(entity, hit))
            if target_segment_id:
                linked_entity_ids.extend(_linked_entity_ids_for_segment(project, target_segment_id))
        linked_entity_ids = _unique_strings(linked_entity_ids)
        candidates.append(
            {
                "source": "visual_entity",
                "index": index_uid,
                "rank": rank,
                "candidate_id": candidate_id,
                "target_segment_id": target_segment_id,
                "target_resolution": resolution_method,
                "score": _optional_float(hit.get("_rankingScore")),
                "start_time": _first_float(target.get("start_time"), entity.get("timestamp")),
                "end_time": _first_float(target.get("end_time"), entity.get("timestamp")),
                "timestamp_center": _first_float(
                    target.get("timestamp_center"),
                    entity.get("timestamp"),
                    hit.get("timestamp"),
                ),
                "has_transcript_evidence": bool(str(target.get("transcript_text") or "").strip()),
                "frame_ids": frame_ids,
                "frame_backed": bool(frame_ids),
                "linked_entity_ids": linked_entity_ids,
                "linked_entity_backed": bool(linked_entity_ids),
            }
        )
    return candidates


def _merge_ablation_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        key = str(candidate.get("target_segment_id") or candidate.get("candidate_id") or "")
        grouped[key].append(candidate)

    merged: list[dict[str, Any]] = []
    for key, items in grouped.items():
        primary = sorted(items, key=lambda item: _candidate_mode_sort_key(item, "object-aligned"))[0]
        frame_ids = _unique_strings(
            frame_id_value
            for item in items
            for frame_id_value in _string_list(item.get("frame_ids"))
        )
        linked_entity_ids = _unique_strings(
            entity_id
            for item in items
            for entity_id in _string_list(item.get("linked_entity_ids"))
        )
        sources = _unique_strings(item.get("source") for item in items)
        merged_item = dict(primary)
        merged_item["candidate_id"] = str(primary.get("candidate_id") or key)
        merged_item["frame_ids"] = frame_ids
        merged_item["frame_backed"] = bool(frame_ids)
        merged_item["linked_entity_ids"] = linked_entity_ids
        merged_item["linked_entity_backed"] = bool(linked_entity_ids)
        merged_item["source_count"] = len(sources)
        merged_item["sources"] = sources
        merged.append(merged_item)
    return sorted(merged, key=lambda item: _candidate_mode_sort_key(item, "object-aligned"))[:limit]


def _ablation_mode_metrics(
    *,
    mode: str,
    rows: list[dict[str, Any]],
    deltas: list[int],
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "mode": mode,
        "query_count": len(rows),
        "mean_abs_error": _mean_or_none(
            row.get("best_abs_error") for row in rows if row.get("best_abs_error") is not None
        ),
        "top1_mean_abs_error": _mean_or_none(
            row.get("top1_abs_error") for row in rows if row.get("top1_abs_error") is not None
        ),
        "mrr_at_max_delta": _mean_or_none(row.get("mrr") for row in rows),
        "evidence_coverage_ratio": _ratio(rows, "evidence_covered"),
        "evidence_coverage": _ratio(rows, "evidence_covered"),
        "frame_backed_ratio": _ratio(rows, "frame_backed"),
        "linked_entity_ratio": _ratio(rows, "linked_entity_backed"),
        "linked_entity_backed_ratio": _ratio(rows, "linked_entity_backed"),
        "mean_processing_time_ms": _mean_or_none(row.get("processing_time_ms") for row in rows),
        "mean_elapsed_time_ms": _mean_or_none(row.get("elapsed_time_ms") for row in rows),
        "source_counts": _source_counts(rows),
    }
    for delta in deltas:
        metric[f"hit_at_{delta}s"] = _ratio_hit(rows, str(delta))
    return metric


def _load_ablation_project(
    *,
    suite: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
) -> AblationProject:
    project_dir = _project_dir_from_suite(suite=suite, base_dir=base_dir, repo_root=repo_root)
    segments_path = segment_artifact_path(project_dir, segments=_optional_path(suite.get("segments")))
    frames_manifest_path = resolve_project_path(
        project_dir,
        _optional_path(suite.get("frames_manifest")),
        default=project_dir / "manifests" / "frames_manifest.jsonl",
    )
    visual_entities_path = resolve_project_path(
        project_dir,
        _optional_path(suite.get("visual_entities")),
        default=project_dir / "manifests" / "visual_entities.jsonl",
    )
    entity_links_path = resolve_project_path(
        project_dir,
        _optional_path(suite.get("entity_links")),
        default=project_dir / "manifests" / "entity_links.jsonl",
    )
    segments = read_jsonl(segments_path)
    frames = read_jsonl(frames_manifest_path) if frames_manifest_path.exists() else []
    visual_entities = read_jsonl(visual_entities_path) if visual_entities_path.exists() else []
    entity_links = read_jsonl(entity_links_path) if entity_links_path.exists() else []
    links_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    links_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in entity_links:
        segment_id = str(link.get("segment_id") or "").strip()
        entity_id = str(link.get("entity_id") or "").strip()
        if segment_id:
            links_by_segment[segment_id].append(link)
        if entity_id:
            links_by_entity[entity_id].append(link)
    return AblationProject(
        project_dir=project_dir,
        segments=segments,
        segment_lookup={
            str(segment.get("segment_id") or segment.get("sample_id") or ""): segment
            for segment in segments
        },
        frame_lookup={frame_id(frame): frame for frame in frames},
        visual_entities=visual_entities,
        entity_lookup={
            str(entity.get("entity_id") or ""): entity
            for entity in visual_entities
            if entity.get("entity_id") is not None
        },
        links_by_segment=dict(links_by_segment),
        links_by_entity=dict(links_by_entity),
    )


def _read_ablation_queries(*, base_dir: Path, suite: dict[str, Any]) -> list[dict[str, Any]]:
    source = suite.get("queries", suite.get("query_file"))
    if isinstance(source, list):
        return [dict(item) for item in source if isinstance(item, dict)]
    if source is None:
        raise ValueError("retrieval_ablation suite requires queries or query_file")
    path = _resolve_path(base_dir, source)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_query_csv(path)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL query rows must be objects: {path}:{line_number}")
                rows.append(payload)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            return [dict(item) for item in payload["queries"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported retrieval_ablation query file format: {path}")


def _ablation_modes(suite: dict[str, Any]) -> list[str]:
    raw_modes = suite.get("modes") or DEFAULT_ABLATION_MODES
    if isinstance(raw_modes, str):
        raw_items: list[Any] = [item.strip() for item in raw_modes.split(",")]
    else:
        raw_items = list(raw_modes)
    modes: list[str] = []
    for raw_item in raw_items:
        mode_value = raw_item.get("mode") if isinstance(raw_item, dict) else raw_item
        key = str(mode_value or "").strip().lower().replace(" ", "-")
        normalized = ABLATION_MODE_ALIASES.get(key.replace("-", "_"), ABLATION_MODE_ALIASES.get(key))
        if normalized is None:
            raise ValueError(f"Unsupported retrieval_ablation mode: {mode_value}")
        if normalized not in modes:
            modes.append(normalized)
    if not modes:
        raise ValueError("retrieval_ablation requires at least one mode")
    return modes


def _matrix_variants(suite: dict[str, Any]) -> list[dict[str, Any]]:
    default_by_id = {
        str(variant["variant_id"]): dict(variant) for variant in DEFAULT_MATRIX_VARIANTS
    }
    raw_variants = suite.get("variants") or DEFAULT_MATRIX_VARIANTS
    if isinstance(raw_variants, str):
        raw_items: list[Any] = [item.strip() for item in raw_variants.split(",")]
    else:
        raw_items = list(raw_variants)

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if isinstance(raw_item, str):
            variant_id = raw_item.strip()
            if variant_id not in default_by_id:
                raise ValueError(f"Unsupported retrieval_answer_matrix variant: {variant_id}")
            variant = dict(default_by_id[variant_id])
        elif isinstance(raw_item, dict):
            raw_variant_id = raw_item.get("variant_id", raw_item.get("id", raw_item.get("name")))
            variant_id = str(raw_variant_id or "").strip()
            if not variant_id:
                raise ValueError("retrieval_answer_matrix variant requires variant_id")
            variant = dict(default_by_id.get(variant_id, {}))
            variant.update(raw_item)
            variant["variant_id"] = variant_id
        else:
            raise ValueError(f"Invalid retrieval_answer_matrix variant: {raw_item!r}")

        variant["variant_id"] = str(variant["variant_id"]).strip()
        variant["label"] = str(variant.get("label") or variant["variant_id"]).strip()
        variant["index_kind"] = _normalize_index_kind(variant.get("index_kind", SEGMENT_HIT_SOURCE))
        if variant["variant_id"] in seen:
            raise ValueError(f"Duplicate retrieval_answer_matrix variant_id: {variant['variant_id']}")
        seen.add(variant["variant_id"])
        variants.append(variant)
    if not variants:
        raise ValueError("retrieval_answer_matrix requires at least one variant")
    return variants


def _run_matrix_query(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    variant: dict[str, Any],
    project_dir: Path,
    run_id: str,
    suite_id: str,
    domain: str,
    query_id: str,
    query_row: dict[str, Any],
    query_text: str,
    expected_ranges: list[tuple[float, float]],
    expected_segment_ids: list[str],
    expected_window_ids: list[str],
    include_answer: bool,
    deltas: list[int],
) -> dict[str, Any]:
    started = time.perf_counter()
    index_kind = _normalize_index_kind(variant.get("index_kind", SEGMENT_HIT_SOURCE))
    index_uid = _matrix_index_uid(suite=suite, variant=variant, index_kind=index_kind)
    visual_index_uid = _optional_str(_variant_value(suite, variant, "visual_index"))
    use_domain_lexicon = _bool_config(variant, "use_domain_lexicon", default=False)
    domain_lexicon_path = (
        _optional_path(_variant_value(suite, variant, "domain_lexicon"))
        if use_domain_lexicon
        else None
    )
    rerank_time_hint = _matrix_rerank_time_hint(suite=suite, variant=variant, query_row=query_row)
    response = query_project(
        client=client,
        index_uid=index_uid,
        retrieval_index_kind=index_kind,
        visual_index_uid=visual_index_uid,
        project_dir=project_dir,
        query=query_text,
        limit=int(_variant_value(suite, variant, "limit") or 5),
        segments_path=_optional_path(_variant_value(suite, variant, "segments")),
        frames_manifest_path=_optional_path(_variant_value(suite, variant, "frames_manifest")),
        visual_entities_path=_optional_path(_variant_value(suite, variant, "visual_entities")),
        entity_links_path=_optional_path(_variant_value(suite, variant, "entity_links")),
        domain_lexicon_path=domain_lexicon_path,
        disable_domain_lexicon=not use_domain_lexicon,
        window_seconds=_optional_float(_variant_value(suite, variant, "window_seconds")),
        neighbor_count=int(_variant_value(suite, variant, "neighbor_count") or 1),
        previous_neighbor_count=_optional_int(
            _variant_value(suite, variant, "previous_neighbor_count")
        ),
        next_neighbor_count=_optional_int(_variant_value(suite, variant, "next_neighbor_count")),
        window_before_seconds=_optional_float(
            _variant_value(suite, variant, "window_before_seconds")
        ),
        window_after_seconds=_optional_float(_variant_value(suite, variant, "window_after_seconds")),
        rerank=_bool_config(variant, "rerank", default=False),
        rerank_time_hint=rerank_time_hint,
        rerank_backend=str(_variant_value(suite, variant, "rerank_backend") or "deterministic"),
        hybrid_retrieval=_bool_config(variant, "hybrid_retrieval", default=False),
        hybrid_embedder=str(
            _variant_value(suite, variant, "hybrid_embedder") or DEFAULT_HYBRID_EMBEDDER
        ),
        hybrid_semantic_ratio=(
            _optional_float(_variant_value(suite, variant, "hybrid_semantic_ratio"))
            or DEFAULT_HYBRID_SEMANTIC_RATIO
        ),
    )
    answer_enabled = _bool_config(variant, "include_answer", default=include_answer)
    answer = compose_answer(response) if answer_enabled else None
    answer_grounding = _answer_grounding_metrics(
        answer=answer,
        expected_segment_ids=expected_segment_ids,
        expected_window_ids=expected_window_ids,
        expected_ranges=expected_ranges,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 4)
    bundles = _list_of_dicts(response.get("bundles"))
    candidate_errors = [
        _local_candidate_error(
            bundle.get("candidate") if isinstance(bundle.get("candidate"), dict) else None,
            expected_ranges,
        )
        for bundle in bundles
    ]
    best_error = min((error for error in candidate_errors if error is not None), default=None)
    top_error = candidate_errors[0] if candidate_errors else None
    hit_by_delta = {
        str(delta): best_error is not None and best_error <= delta for delta in deltas
    }
    top_bundle = bundles[0] if bundles else None
    top_candidate = (
        top_bundle.get("candidate")
        if isinstance(top_bundle, dict) and isinstance(top_bundle.get("candidate"), dict)
        else None
    )

    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "suite_type": "retrieval_answer_matrix",
        "domain": domain,
        "variant_id": variant["variant_id"],
        "variant_label": variant["label"],
        "query_id": query_id,
        "query_label": _optional_public_label(query_row),
        "privacy": _matrix_privacy_payload(),
        "config": _public_matrix_config(
            suite=suite,
            variant=variant,
            index_uid=index_uid,
            visual_index_uid=visual_index_uid,
            index_kind=index_kind,
            use_domain_lexicon=use_domain_lexicon,
        ),
        "expected_time_available": bool(expected_ranges),
        "expected_segment_available": bool(expected_segment_ids),
        "expected_window_available": bool(expected_window_ids),
        "search_hit_count": (response.get("counts") or {}).get("search_hits"),
        "bundle_count": len(bundles),
        "best_abs_error": best_error,
        "top1_abs_error": top_error,
        "top1_expected_segment_match": _top_candidate_matches(
            top_candidate=top_candidate,
            expected_segment_ids=expected_segment_ids,
        ),
        "hit_by_delta": hit_by_delta,
        "mrr": _reciprocal_rank(candidate_errors, deltas=max(deltas)),
        "evidence_covered": bool(bundles),
        "frame_backed": _bundles_have_frames(bundles),
        "linked_entity_backed": _bundles_have_linked_entities(bundles),
        "query_expansion": _public_query_expansion(response.get("query_expansion")),
        "top_candidate": _public_matrix_candidate(top_bundle),
        "answer": _public_answer_summary(answer),
        "answer_grounding": answer_grounding,
        "processing_time_ms": response.get("processing_time_ms"),
        "elapsed_time_ms": elapsed_ms,
        "warning_count": len(response.get("warnings") or []),
    }


def _matrix_variant_metrics(
    *,
    variant: dict[str, Any],
    rows: list[dict[str, Any]],
    deltas: list[int],
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "variant_id": variant["variant_id"],
        "label": variant["label"],
        "query_count": len(rows),
        "config": _public_matrix_variant_config(variant),
        "mean_abs_error": _mean_or_none(
            row.get("best_abs_error") for row in rows if row.get("best_abs_error") is not None
        ),
        "top1_mean_abs_error": _mean_or_none(
            row.get("top1_abs_error") for row in rows if row.get("top1_abs_error") is not None
        ),
        "mrr_at_max_delta": _mean_or_none(row.get("mrr") for row in rows),
        "evidence_coverage_ratio": _ratio(rows, "evidence_covered"),
        "frame_backed_ratio": _ratio(rows, "frame_backed"),
        "linked_entity_backed_ratio": _ratio(rows, "linked_entity_backed"),
        "top1_expected_segment_match_ratio": _ratio(rows, "top1_expected_segment_match"),
        "grounded_answer_ratio": _answer_ratio(rows, "grounded_answer"),
        "candidate_evidence_only_ratio": _answer_ratio(rows, "candidate_evidence_only"),
        "citation_coverage_ratio": _answer_count_ratio(rows, "citation_count"),
        "answer_citation_precision": _mean_grounding_metric(rows, "citation_precision"),
        "answer_citation_recall": _mean_grounding_metric(rows, "citation_recall"),
        "expected_citation_hit_ratio": _grounding_bool_ratio(rows, "expected_citation_hit"),
        "mean_unsupported_claim_count": _mean_grounding_metric(rows, "unsupported_claim_count"),
        "unsupported_claim_ratio": _grounding_positive_ratio(rows, "unsupported_claim_count"),
        "mean_answer_citation_count": _mean_or_none(
            (row.get("answer") or {}).get("citation_count")
            for row in rows
            if isinstance(row.get("answer"), dict)
        ),
        "mean_processing_time_ms": _mean_or_none(row.get("processing_time_ms") for row in rows),
        "mean_elapsed_time_ms": _mean_or_none(row.get("elapsed_time_ms") for row in rows),
        "warning_ratio": _ratio(rows, "warning_count"),
        "source_counts": _source_counts(rows),
    }
    for delta in deltas:
        metric[f"hit_at_{delta}s"] = _ratio_hit(rows, str(delta))
    return metric


def _expected_ranges(query_row: dict[str, Any]) -> list[tuple[float, float]]:
    time_hint = str(query_row.get("expected_time_hint") or query_row.get("time_hint") or "")
    ranges = parse_time_hint(time_hint)
    if ranges:
        return ranges
    start = _optional_float(query_row.get("expected_start_time"))
    end = _optional_float(query_row.get("expected_end_time"))
    if start is not None or end is not None:
        if start is None:
            start = end
        if end is None:
            end = start
        if start is not None and end is not None:
            return [(min(start, end), max(start, end))]
    points = query_row.get("timestamp_points", query_row.get("expected_timestamp"))
    if not isinstance(points, list):
        points = [] if points is None else [points]
    ranges = []
    for point in points:
        parsed = _optional_float(point)
        if parsed is not None:
            ranges.append((parsed, parsed))
    return ranges


def _expected_segment_ids(query_row: dict[str, Any]) -> list[str]:
    values = query_row.get(
        "expected_segment_ids",
        query_row.get("expected_segment_id", query_row.get("target_segment_id")),
    )
    return _string_list(values)


def _expected_window_ids(query_row: dict[str, Any]) -> list[str]:
    values = query_row.get(
        "expected_window_ids",
        query_row.get("expected_window_id", query_row.get("target_window_id")),
    )
    return _string_list(values)


def _normalize_index_kind(value: Any) -> str:
    normalized = str(value or SEGMENT_HIT_SOURCE).strip().lower().replace("_", "-")
    aliases = {
        "segment": SEGMENT_HIT_SOURCE,
        "segments": SEGMENT_HIT_SOURCE,
        "segment-lexical": SEGMENT_HIT_SOURCE,
        "window": WINDOW_HIT_SOURCE,
        "windows": WINDOW_HIT_SOURCE,
        "lecture-window": WINDOW_HIT_SOURCE,
    }
    result = aliases.get(normalized)
    if result is None:
        valid = ", ".join(sorted({SEGMENT_HIT_SOURCE, WINDOW_HIT_SOURCE}))
        raise ValueError(f"Unsupported retrieval_answer_matrix index_kind: {value}. Valid: {valid}")
    return result


def _variant_value(suite: dict[str, Any], variant: dict[str, Any], key: str) -> Any:
    return variant[key] if key in variant else suite.get(key)


def _bool_config(config: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _matrix_index_uid(*, suite: dict[str, Any], variant: dict[str, Any], index_kind: str) -> str:
    explicit = _optional_str(variant.get("index"))
    if explicit:
        return explicit
    if index_kind == WINDOW_HIT_SOURCE:
        window_index = _optional_str(variant.get("window_index", suite.get("window_index")))
        if window_index:
            return window_index
        raise ValueError("retrieval_answer_matrix window variant requires window_index")
    index = _optional_str(suite.get("index"))
    if not index:
        raise ValueError("retrieval_answer_matrix suite requires index")
    return index


def _matrix_rerank_time_hint(
    *,
    suite: dict[str, Any],
    variant: dict[str, Any],
    query_row: dict[str, Any],
) -> str | None:
    field = _optional_str(_variant_value(suite, variant, "rerank_time_hint_field"))
    if field and query_row.get(field):
        return str(query_row[field])
    value = _variant_value(suite, variant, "rerank_time_hint")
    if value is not None:
        return str(value)
    if _bool_config(variant, "rerank", default=False):
        return str(query_row.get("expected_time_hint") or "") or None
    return None


def _public_matrix_config(
    *,
    suite: dict[str, Any],
    variant: dict[str, Any],
    index_uid: str,
    visual_index_uid: str | None,
    index_kind: str,
    use_domain_lexicon: bool,
) -> dict[str, Any]:
    return {
        **_public_matrix_variant_config(variant),
        "index_kind": index_kind,
        "index_ref": f"index:{_short_hash(index_uid)}",
        "visual_index_ref": f"index:{_short_hash(visual_index_uid)}"
        if visual_index_uid is not None
        else None,
        "limit": int(_variant_value(suite, variant, "limit") or 5),
        "domain_lexicon_enabled": use_domain_lexicon,
    }


def _public_matrix_variant_config(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "index_kind": _normalize_index_kind(variant.get("index_kind", SEGMENT_HIT_SOURCE)),
        "domain_lexicon": _bool_config(variant, "use_domain_lexicon", default=False),
        "hybrid_retrieval": _bool_config(variant, "hybrid_retrieval", default=False),
        "rerank": _bool_config(variant, "rerank", default=False),
        "answer": _bool_config(variant, "include_answer", default=True),
    }


def _public_query_expansion(value: Any) -> dict[str, Any]:
    metadata = value if isinstance(value, dict) else {}
    return {
        "enabled": bool(metadata.get("enabled")),
        "applied": bool(metadata.get("applied")),
        "added_term_count": int(metadata.get("added_term_count") or 0),
        "search_query_count": len(metadata.get("search_queries") or []),
    }


def _public_matrix_candidate(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(bundle, dict):
        return None
    candidate = bundle.get("candidate")
    if not isinstance(candidate, dict):
        return None
    source = str(candidate.get("source") or "candidate")
    candidate_id = str(
        candidate.get("segment_id")
        or candidate.get("target_segment_id")
        or candidate.get("window_id")
        or candidate.get("sample_id")
        or "unknown"
    )
    evidence_window = bundle.get("evidence_window") if isinstance(bundle, dict) else {}
    frame_refs = []
    if isinstance(evidence_window, dict):
        frame_refs = _list_of_dicts(evidence_window.get("frame_refs"))
    return {
        "ref": f"{source}:{_short_hash(candidate_id)}",
        "source": source,
        "modality": candidate.get("modality"),
        "rank": candidate.get("rank"),
        "retrieval_mode": candidate.get("retrieval_mode"),
        "timestamp_available": _first_float(
            candidate.get("timestamp_center"),
            candidate.get("start_time"),
            candidate.get("end_time"),
        )
        is not None,
        "frame_backed": bool(frame_refs),
        "linked_entity_backed": bool(bundle.get("linked_entities")),
        "source_count": (bundle.get("merge") or {}).get("source_count")
        if isinstance(bundle.get("merge"), dict)
        else None,
    }


def _public_answer_summary(answer: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(answer, dict):
        return {"enabled": False}
    policy = answer.get("no_answer_policy") if isinstance(answer.get("no_answer_policy"), dict) else {}
    llm = answer.get("llm") if isinstance(answer.get("llm"), dict) else {}
    return {
        "enabled": True,
        "schema_version": answer.get("schema_version"),
        "answer_type": answer.get("answer_type"),
        "claim_count": len(_list_of_dicts(answer.get("claims"))),
        "citation_count": len(_list_of_dicts(answer.get("citations"))),
        "candidate_evidence_count": len(_list_of_dicts(answer.get("candidate_evidence"))),
        "policy_reason": policy.get("reason"),
        "llm_enabled": bool(llm.get("enabled")),
    }


def _answer_grounding_metrics(
    *,
    answer: dict[str, Any] | None,
    expected_segment_ids: list[str],
    expected_window_ids: list[str],
    expected_ranges: list[tuple[float, float]],
) -> dict[str, Any]:
    expected_units = _expected_citation_units(
        expected_segment_ids=expected_segment_ids,
        expected_window_ids=expected_window_ids,
        expected_ranges=expected_ranges,
    )
    expected_available = bool(expected_units)
    if not isinstance(answer, dict):
        return {
            "enabled": False,
            "expected_available": expected_available,
            "expected_unit_count": len(expected_units),
            "citation_precision": None,
            "citation_recall": None,
            "expected_citation_hit": None,
            "unsupported_claim_count": None,
        }

    citations = _list_of_dicts(answer.get("citations"))
    claims = _list_of_dicts(answer.get("claims"))
    matched_citation_count = (
        sum(
            1
            for citation in citations
            if _citation_matches_expected(
                citation,
                expected_segment_ids=expected_segment_ids,
                expected_window_ids=expected_window_ids,
                expected_ranges=expected_ranges,
            )
        )
        if expected_available
        else None
    )
    covered_unit_count = (
        sum(1 for unit in expected_units if _expected_unit_covered(unit, citations))
        if expected_available
        else None
    )
    citation_precision = (
        round(float(matched_citation_count) / len(citations), 4)
        if expected_available and citations
        else 0.0
        if expected_available
        else None
    )
    citation_recall = (
        round(float(covered_unit_count) / len(expected_units), 4)
        if expected_available and expected_units
        else None
    )
    unsupported_claim_count = _unsupported_claim_count(
        claims=claims,
        citations=citations,
        expected_segment_ids=expected_segment_ids,
        expected_window_ids=expected_window_ids,
        expected_ranges=expected_ranges,
        expected_available=expected_available,
    )
    return {
        "enabled": True,
        "expected_available": expected_available,
        "expected_unit_count": len(expected_units),
        "claim_count": len(claims),
        "citation_count": len(citations),
        "matched_citation_count": matched_citation_count,
        "covered_expected_unit_count": covered_unit_count,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "expected_citation_hit": bool(matched_citation_count)
        if matched_citation_count is not None
        else None,
        "unsupported_claim_count": unsupported_claim_count,
    }


def _expected_citation_units(
    *,
    expected_segment_ids: list[str],
    expected_window_ids: list[str],
    expected_ranges: list[tuple[float, float]],
) -> list[tuple[str, Any]]:
    units: list[tuple[str, Any]] = []
    for segment_id in sorted(set(expected_segment_ids)):
        units.append(("segment", segment_id))
    for window_id in sorted(set(expected_window_ids)):
        units.append(("window", window_id))
    for index, expected_range in enumerate(expected_ranges):
        units.append(("time", (index, expected_range)))
    return units


def _expected_unit_covered(unit: tuple[str, Any], citations: list[dict[str, Any]]) -> bool:
    unit_type, value = unit
    for citation in citations:
        if unit_type == "segment" and value in _citation_segment_ids(citation):
            return True
        if unit_type == "window" and value in _citation_window_ids(citation):
            return True
        if unit_type == "time" and _citation_matches_time(citation, [value[1]]):
            return True
    return False


def _citation_matches_expected(
    citation: dict[str, Any],
    *,
    expected_segment_ids: list[str],
    expected_window_ids: list[str],
    expected_ranges: list[tuple[float, float]],
) -> bool:
    if expected_segment_ids and set(expected_segment_ids).intersection(
        _citation_segment_ids(citation)
    ):
        return True
    if expected_window_ids and set(expected_window_ids).intersection(_citation_window_ids(citation)):
        return True
    return bool(expected_ranges and _citation_matches_time(citation, expected_ranges))


def _citation_segment_ids(citation: dict[str, Any]) -> set[str]:
    values: list[Any] = [
        citation.get("segment_id"),
        citation.get("target_segment_id"),
        citation.get("sample_id"),
    ]
    for source in _list_of_dicts(citation.get("retrieval_sources")):
        values.extend(
            [
                source.get("segment_id"),
                source.get("target_segment_id"),
                source.get("sample_id"),
            ]
        )
        values.extend(_string_list(source.get("source_segment_ids")))
    return set(_string_list(values))


def _citation_window_ids(citation: dict[str, Any]) -> set[str]:
    values: list[Any] = [citation.get("window_id")]
    for source in _list_of_dicts(citation.get("retrieval_sources")):
        values.append(source.get("window_id"))
    return set(_string_list(values))


def _citation_matches_time(
    citation: dict[str, Any],
    expected_ranges: list[tuple[float, float]],
) -> bool:
    start = _optional_float(citation.get("start_time"))
    end = _optional_float(citation.get("end_time"))
    timestamp = _first_float(citation.get("timestamp"), citation.get("timestamp_center"))
    for expected_range in expected_ranges:
        if start is not None and end is not None:
            citation_range = (min(start, end), max(start, end))
            if _ranges_overlap(citation_range, expected_range):
                return True
        if timestamp is not None and _point_to_range_distance(timestamp, expected_range) == 0:
            return True
    return False


def _ranges_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _unsupported_claim_count(
    *,
    claims: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    expected_segment_ids: list[str],
    expected_window_ids: list[str],
    expected_ranges: list[tuple[float, float]],
    expected_available: bool,
) -> int:
    citation_by_id = {
        str(citation.get("citation_id")): citation
        for citation in citations
        if citation.get("citation_id") is not None
    }
    unsupported = 0
    for claim in claims:
        citation_ids = _string_list(claim.get("citation_ids"))
        if not citation_ids:
            unsupported += 1
            continue
        cited = [citation_by_id[citation_id] for citation_id in citation_ids if citation_id in citation_by_id]
        if not cited:
            unsupported += 1
            continue
        if expected_available and not any(
            _citation_matches_expected(
                citation,
                expected_segment_ids=expected_segment_ids,
                expected_window_ids=expected_window_ids,
                expected_ranges=expected_ranges,
            )
            for citation in cited
        ):
            unsupported += 1
    return unsupported


def _top_candidate_matches(
    *,
    top_candidate: dict[str, Any] | None,
    expected_segment_ids: list[str],
) -> bool | None:
    if not expected_segment_ids:
        return None
    if not isinstance(top_candidate, dict):
        return False
    candidate_ids = _string_list(
        [
            top_candidate.get("segment_id"),
            top_candidate.get("target_segment_id"),
            top_candidate.get("sample_id"),
        ]
    )
    return bool(set(candidate_ids).intersection(expected_segment_ids))


def _bundles_have_frames(bundles: list[dict[str, Any]]) -> bool:
    for bundle in bundles:
        evidence_window = bundle.get("evidence_window")
        if isinstance(evidence_window, dict) and evidence_window.get("frame_refs"):
            return True
    return False


def _bundles_have_linked_entities(bundles: list[dict[str, Any]]) -> bool:
    return any(bool(bundle.get("linked_entities")) for bundle in bundles)


def _answer_ratio(rows: list[dict[str, Any]], answer_type: str) -> float | None:
    answer_rows = [row for row in rows if isinstance(row.get("answer"), dict)]
    if not answer_rows:
        return None
    return round(
        sum(
            1.0
            if (row.get("answer") or {}).get("answer_type") == answer_type
            else 0.0
            for row in answer_rows
        )
        / len(answer_rows),
        4,
    )


def _answer_count_ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    answer_rows = [row for row in rows if isinstance(row.get("answer"), dict)]
    if not answer_rows:
        return None
    return round(
        sum(1.0 if ((row.get("answer") or {}).get(key) or 0) > 0 else 0.0 for row in answer_rows)
        / len(answer_rows),
        4,
    )


def _mean_grounding_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        grounding = row.get("answer_grounding")
        if isinstance(grounding, dict) and grounding.get(key) is not None:
            values.append(grounding[key])
    return _mean_or_none(values)


def _grounding_bool_ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        grounding = row.get("answer_grounding")
        if isinstance(grounding, dict) and grounding.get(key) is not None:
            values.append(bool(grounding[key]))
    if not values:
        return None
    return round(sum(1.0 if value else 0.0 for value in values) / len(values), 4)


def _grounding_positive_ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        grounding = row.get("answer_grounding")
        if isinstance(grounding, dict) and grounding.get(key) is not None:
            values.append(float(grounding[key]))
    if not values:
        return None
    return round(sum(1.0 if value > 0 else 0.0 for value in values) / len(values), 4)


def _matrix_privacy_payload() -> dict[str, Any]:
    return {
        "public_outputs_are_sanitized": True,
        "raw_query_text": "redacted",
        "answer_text": "redacted",
        "claim_text": "redacted",
        "candidate_evidence_text": "redacted",
        "transcript_excerpt": "redacted",
        "local_paths": "redacted",
        "raw_candidate_ids": "hashed",
        "raw_response_in_public_output": False,
        "grounding_proxy_metrics": "deterministic_expected_hint_overlap_not_full_llm_quality",
    }


def _dataset_descriptor(
    *,
    suite: dict[str, Any],
    suite_id: str,
    suite_type: str,
    domain: str,
    query_count: int,
) -> dict[str, Any]:
    raw = suite.get("dataset_descriptor") if isinstance(suite.get("dataset_descriptor"), dict) else {}
    descriptor = {
        "suite_id": suite_id,
        "suite_type": suite_type,
        "domain": domain,
        "query_count": query_count,
        "descriptor_id": str(raw.get("descriptor_id") or raw.get("dataset_id") or suite_id),
        "split": raw.get("split"),
        "version": raw.get("version"),
        "privacy": raw.get("privacy", "aggregate_only"),
    }
    if suite.get("project_id") is not None:
        descriptor["project_id"] = str(suite["project_id"])
    elif suite.get("project_dir") is not None:
        descriptor["project_ref"] = f"project:{_short_hash(str(suite['project_dir']))}"
    return {key: value for key, value in descriptor.items() if value is not None}


def _visual_entity_for_hit(*, hit: dict[str, Any], project: AblationProject) -> dict[str, Any]:
    entity_id = str(hit.get("entity_id") or "").strip()
    if entity_id and entity_id in project.entity_lookup:
        return project.entity_lookup[entity_id]
    return hit


def _resolve_visual_target_segment(
    *,
    hit: dict[str, Any],
    entity: dict[str, Any],
    project: AblationProject,
    use_entity_links: bool,
) -> tuple[str | None, str]:
    hit_segment_id = str(hit.get("segment_id") or entity.get("segment_id") or "").strip()
    if hit_segment_id and hit_segment_id in project.segment_lookup:
        return hit_segment_id, "hit_segment_id"

    if use_entity_links:
        entity_ids = _entity_id_values(entity, hit)
        linked_segments = [
            str(link.get("segment_id") or "").strip()
            for entity_id in entity_ids
            for link in project.links_by_entity.get(entity_id, [])
            if str(link.get("segment_id") or "").strip() in project.segment_lookup
        ]
        if linked_segments:
            return sorted(set(linked_segments))[0], "entity_link"

    frame_ids = _visual_candidate_frame_ids(hit=hit, entity=entity, target={})
    if frame_ids:
        for segment in project.segments:
            segment_id = str(segment.get("segment_id") or "").strip()
            if set(frame_ids).intersection(_segment_frame_ids(segment)) and segment_id:
                return segment_id, "frame_ref"

    timestamp = _first_float(entity.get("timestamp"), hit.get("timestamp"))
    if timestamp is not None:
        nearest = _nearest_segment_to_timestamp(project.segments, timestamp)
        if nearest is not None:
            return str(nearest.get("segment_id") or "").strip() or None, "timestamp_nearest"
    return None, "unresolved"


def _nearest_segment_to_timestamp(
    segments: list[dict[str, Any]],
    timestamp: float,
) -> dict[str, Any] | None:
    if not segments:
        return None
    return sorted(
        segments,
        key=lambda segment: (
            _segment_timestamp_distance(segment, timestamp),
            str(segment.get("segment_id") or ""),
        ),
    )[0]


def _segment_timestamp_distance(segment: dict[str, Any], timestamp: float) -> float:
    window = segment_window(segment)
    if window is None:
        center = _optional_float(segment.get("timestamp_center"))
        return abs(center - timestamp) if center is not None else float("inf")
    if window[0] <= timestamp <= window[1]:
        return 0.0
    return min(abs(timestamp - window[0]), abs(timestamp - window[1]))


def _candidate_mode_sort_key(candidate: dict[str, Any], mode: str) -> tuple[Any, ...]:
    rank = _optional_int(candidate.get("rank")) or 10**9
    source = str(candidate.get("source") or "")
    source_priority = 0 if source == "segment" else 1
    if mode == "object-aligned":
        return (
            not bool(candidate.get("linked_entity_backed")),
            not bool(candidate.get("frame_backed")),
            rank,
            source_priority,
            str(candidate.get("candidate_id") or ""),
        )
    if mode == "time-aligned":
        return (
            not bool(candidate.get("frame_backed")),
            rank,
            source_priority,
            str(candidate.get("candidate_id") or ""),
        )
    return (rank, source_priority, str(candidate.get("candidate_id") or ""))


def _ablation_candidate_error(
    candidate: dict[str, Any],
    expected_ranges: list[tuple[float, float]],
) -> float | None:
    if not expected_ranges:
        return None
    center = _first_float(candidate.get("timestamp_center"))
    if center is None:
        start = _optional_float(candidate.get("start_time"))
        end = _optional_float(candidate.get("end_time"))
        if start is not None and end is not None:
            center = (start + end) / 2.0
        else:
            center = start if start is not None else end
    if center is None:
        return None
    return min(_point_to_range_distance(center, expected_range) for expected_range in expected_ranges)


def _mode_evidence_covered(*, mode: str, candidates: list[dict[str, Any]]) -> bool:
    if not candidates:
        return False
    if mode == "transcript-only":
        return any(candidate.get("has_transcript_evidence") for candidate in candidates)
    if mode == "visual-only":
        return any(candidate.get("source") == "visual_entity" for candidate in candidates)
    if mode == "time-aligned":
        return any(candidate.get("frame_backed") for candidate in candidates)
    if mode == "object-aligned":
        return any(
            candidate.get("linked_entity_backed")
            or candidate.get("frame_backed")
            or candidate.get("source") == "visual_entity"
            for candidate in candidates
        )
    return bool(candidates)


def _public_ablation_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    source = str(candidate.get("source") or "candidate")
    candidate_id = str(
        candidate.get("target_segment_id") or candidate.get("candidate_id") or "unknown"
    )
    return {
        "ref": f"{source}:{_short_hash(candidate_id)}",
        "source": source,
        "timestamp_available": _first_float(
            candidate.get("timestamp_center"),
            candidate.get("start_time"),
            candidate.get("end_time"),
        )
        is not None,
        "source_count": candidate.get("source_count"),
    }


def _public_ablation_privacy_payload() -> dict[str, Any]:
    return {
        "public_outputs_are_sanitized": True,
        "raw_query_text": "redacted",
        "transcript_excerpt": "redacted",
        "local_paths": "redacted",
        "raw_candidate_ids": "hashed",
        "raw_visual_entity_text": "redacted",
        "raw_response_in_public_output": False,
    }


def _query_text(query_row: dict[str, Any], *, query_index: int) -> str:
    value = query_row.get("query_text", query_row.get("query"))
    if value is None or not str(value).strip():
        raise ValueError(f"retrieval_ablation query #{query_index} is missing query_text")
    return str(value)


def _query_id(query_row: dict[str, Any], *, query_index: int) -> str:
    value = query_row.get("query_id", query_row.get("id"))
    if value is None or not str(value).strip():
        return f"q{query_index:04d}"
    return str(value).strip()


def _optional_public_label(query_row: dict[str, Any]) -> str | None:
    value = query_row.get("public_label", query_row.get("query_label"))
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _visual_candidate_frame_ids(
    *,
    hit: dict[str, Any],
    entity: dict[str, Any],
    target: dict[str, Any],
) -> list[str]:
    values = [
        hit.get("frame_id"),
        entity.get("frame_id"),
        *(_string_list(target.get("frame_refs")) if target else []),
    ]
    return _unique_strings(values)


def _segment_frame_ids(segment: dict[str, Any]) -> list[str]:
    return _unique_strings(segment.get("frame_refs") or [])


def _linked_entity_ids_for_segment(project: AblationProject, segment_id: str | None) -> list[str]:
    if not segment_id:
        return []
    return _unique_strings(
        link.get("entity_id") for link in project.links_by_segment.get(segment_id, [])
    )


def _entity_id_values(*items: dict[str, Any]) -> list[str]:
    return _unique_strings(item.get("entity_id") for item in items)


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        top = row.get("top_candidate")
        source = top.get("source") if isinstance(top, dict) else None
        counts[str(source or "none")] += 1
    return dict(sorted(counts.items()))


def _combined_processing_time_ms(responses: list[dict[str, Any]]) -> float | None:
    values = [
        _optional_float(response.get("processingTimeMs"))
        for response in responses
        if response.get("processingTimeMs") is not None
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    total = sum(values)
    return int(total) if total.is_integer() else round(total, 4)


def _ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows), 4)


def _ratio_hit(rows: list[dict[str, Any]], delta: str) -> float | None:
    if not rows:
        return None
    return round(
        sum(1.0 if (row.get("hit_by_delta") or {}).get(delta) else 0.0 for row in rows)
        / len(rows),
        4,
    )


def _mean_or_none(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else ([] if value is None else [value])
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _short_hash(value: str | None) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:12]


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
    return eval_candidate_abs_errors(record, candidates)


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
    ablation_modes = [
        (suite, mode)
        for suite in metrics["suites"]
        for mode in suite.get("modes", [])
        if isinstance(mode, dict)
    ]
    if ablation_modes:
        lines.extend(
            [
                "",
                "## Retrieval Ablation Modes",
                "",
                "Public ablation outputs are sanitized: raw queries, transcript excerpts, "
                "local paths, visual labels, and raw candidate IDs are omitted.",
                "",
                "| suite | mode | queries | Hit@10s | evidence | frame-backed | linked-entity | latency ms |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for suite, mode in ablation_modes:
            lines.append(
                "| {suite_id} | {mode} | {query_count} | {hit10} | {evidence} | "
                "{frame} | {linked} | {latency} |".format(
                    suite_id=suite.get("suite_id"),
                    mode=mode.get("mode"),
                    query_count=mode.get("query_count") or 0,
                    hit10=_format_metric(mode.get("hit_at_10s")),
                    evidence=_format_metric(mode.get("evidence_coverage_ratio")),
                    frame=_format_metric(mode.get("frame_backed_ratio")),
                    linked=_format_metric(mode.get("linked_entity_ratio")),
                    latency=_format_metric(
                        mode.get("mean_processing_time_ms") or mode.get("mean_elapsed_time_ms")
                    ),
                )
            )
    matrix_variants = [
        (suite, variant)
        for suite in metrics["suites"]
        for variant in suite.get("variants", [])
        if isinstance(variant, dict)
    ]
    if matrix_variants:
        lines.extend(
            [
                "",
                "## Retrieval/Answer Matrix",
                "",
                "Matrix outputs are aggregate and public-safe: raw queries, answer text, "
                "candidate evidence, transcript excerpts, local paths, and raw candidate IDs "
                "are omitted.",
                "Grounding proxy metrics use deterministic expected hint overlap for regression "
                "tracking; they do not replace full LLM answer quality review.",
                "",
                "| suite | variant | queries | Hit@10s | MRR | frame-backed | linked-backed | grounded | cite P | cite R | expected hit | unsupported | latency ms |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for suite, variant in matrix_variants:
            lines.append(
                "| {suite_id} | {variant_id} | {query_count} | {hit10} | {mrr} | "
                "{frame} | {linked} | {grounded} | {precision} | {recall} | {hit} | "
                "{unsupported} | {latency} |".format(
                    suite_id=suite.get("suite_id"),
                    variant_id=variant.get("variant_id"),
                    query_count=variant.get("query_count") or 0,
                    hit10=_format_metric(variant.get("hit_at_10s")),
                    mrr=_format_metric(variant.get("mrr_at_max_delta")),
                    frame=_format_metric(variant.get("frame_backed_ratio")),
                    linked=_format_metric(variant.get("linked_entity_backed_ratio")),
                    grounded=_format_metric(variant.get("grounded_answer_ratio")),
                    precision=_format_metric(variant.get("answer_citation_precision")),
                    recall=_format_metric(variant.get("answer_citation_recall")),
                    hit=_format_metric(variant.get("expected_citation_hit_ratio")),
                    unsupported=_format_metric(variant.get("mean_unsupported_claim_count")),
                    latency=_format_metric(
                        variant.get("mean_processing_time_ms")
                        or variant.get("mean_elapsed_time_ms")
                    ),
                )
            )
    diagnostic_suites = [
        suite
        for suite in metrics["suites"]
        if any(suite.get(f"top{k}_recall") is not None for k in [1, 5, 10, 50])
    ]
    if diagnostic_suites:
        lines.extend(
            [
                "",
                "## Retrieval Diagnostics",
                "",
                "| suite | top1 | top5 | top10 | top50 | median err | p75 err | p90 err |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for suite in diagnostic_suites:
            lines.append(
                "| {suite_id} | {top1} | {top5} | {top10} | {top50} | {median} | {p75} | {p90} |".format(
                    suite_id=suite.get("suite_id"),
                    top1=_format_metric(suite.get("top1_recall")),
                    top5=_format_metric(suite.get("top5_recall")),
                    top10=_format_metric(suite.get("top10_recall")),
                    top50=_format_metric(suite.get("top50_recall")),
                    median=_format_metric(suite.get("median_abs_error")),
                    p75=_format_metric(suite.get("p75_abs_error")),
                    p90=_format_metric(suite.get("p90_abs_error")),
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


def _write_metrics_summary_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = _metrics_summary_rows(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "suite_id",
        "suite_type",
        "domain",
        "row_type",
        "variant_or_mode",
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _metrics_summary_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = metrics.get("run_id")
    for suite in metrics.get("suites", []):
        if not isinstance(suite, dict):
            continue
        base = {
            "run_id": run_id,
            "suite_id": suite.get("suite_id"),
            "suite_type": suite.get("suite_type"),
            "domain": suite.get("domain"),
        }
        variants = [item for item in suite.get("variants", []) if isinstance(item, dict)]
        modes = [item for item in suite.get("modes", []) if isinstance(item, dict)]
        if variants:
            for variant in variants:
                rows.append(
                    _metrics_summary_row(
                        base=base,
                        item=variant,
                        row_type="variant",
                        variant_or_mode=variant.get("variant_id"),
                    )
                )
        elif modes:
            for mode in modes:
                rows.append(
                    _metrics_summary_row(
                        base=base,
                        item=mode,
                        row_type="mode",
                        variant_or_mode=mode.get("mode"),
                    )
                )
        else:
            rows.append(
                _metrics_summary_row(
                    base=base,
                    item=suite,
                    row_type="suite",
                    variant_or_mode=None,
                )
            )
    return rows


def _metrics_summary_row(
    *,
    base: dict[str, Any],
    item: dict[str, Any],
    row_type: str,
    variant_or_mode: Any,
) -> dict[str, Any]:
    return {
        **base,
        "row_type": row_type,
        "variant_or_mode": variant_or_mode,
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


def _suite_hybrid_metadata(
    *,
    enabled: bool,
    observed: dict[str, Any] | None,
    embedder: str,
    semantic_ratio: float,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "modes": ["lexical"],
            "embedder": None,
            "semantic_ratio": None,
            "fusion": None,
        }
    return {
        "enabled": True,
        "modes": observed.get("modes") if observed else ["lexical", "semantic"],
        "embedder": observed.get("embedder") if observed else embedder,
        "semantic_ratio": observed.get("semantic_ratio") if observed else semantic_ratio,
        "fusion": observed.get("fusion") if observed else None,
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


def _suite_diagnostic_top_k(suite: dict[str, Any], *, default: int | None) -> int:
    value = suite.get("diagnostic_top_k", default)
    parsed = _optional_int(value)
    if parsed is None:
        return 0
    return max(parsed, 0)


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
