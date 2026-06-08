from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oarag.core.config import Neo4jConfig, default_paths
from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.dual_candidates import (
    GraphCandidateSession,
    query_project_dual_candidates,
)


PUBLIC_SCHEMA_VERSION = "cross-lecture-retrieval-smoke-public-v1"
DEFAULT_OUTPUT_ROOT = Path("reports") / "paper" / "cross_lecture_retrieval_smoke"
DEFAULT_VARIANTS = (
    "meili_only",
    "graph_only",
    "meili_graph",
    "graph_aware_rerank",
)
MEILI_UNAVAILABLE_SKIP = "meili_unavailable"
GRAPH_UNAVAILABLE_SKIP = "graph_db_unavailable"


class CrossLectureSmokeClient(Protocol):
    def health(self) -> dict[str, Any]: ...

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CrossLectureProjectConfig:
    project_ref: str
    project_dir: Path
    index_uid: str
    evidence_units: Path | None = None


@dataclass(frozen=True)
class CrossLectureQueryConfig:
    query_id: str
    query_text: str
    project_ref: str
    target_evidence_unit_ids: tuple[str, ...]
    target_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossLectureSuiteConfig:
    suite_id: str
    projects: dict[str, CrossLectureProjectConfig]
    queries: tuple[CrossLectureQueryConfig, ...]
    limit: int
    candidate_pool_limit: int
    graph_limit: int
    variants: tuple[str, ...]


@dataclass(frozen=True)
class CrossLectureSmokeManifest:
    run_id: str
    output_dir: Path
    suites: tuple[CrossLectureSuiteConfig, ...]
    manifest_path: Path


@dataclass(frozen=True)
class CrossLectureRetrievalSmokeRun:
    run_id: str
    output_dir: Path
    metrics_path: Path
    query_results_path: Path
    summary_path: Path
    payload: dict[str, Any]


def run_cross_lecture_retrieval_smoke(
    *,
    client: CrossLectureSmokeClient,
    manifest_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    dry_run: bool = False,
    graph_session: GraphCandidateSession | None = None,
    neo4j_config: Neo4jConfig | None = None,
) -> CrossLectureRetrievalSmokeRun:
    manifest = load_cross_lecture_smoke_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    meili = (
        {"available": False, "status": "dry_run", "skip_reason": "dry_run_requested"}
        if dry_run
        else _meili_health(client)
    )
    suite_summaries: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for suite in manifest.suites:
        suite_rows: list[dict[str, Any]] = []
        for query in suite.queries:
            project = suite.projects[query.project_ref]
            for variant in suite.variants:
                row = _run_variant(
                    client=client,
                    run_id=manifest.run_id,
                    suite=suite,
                    project=project,
                    query=query,
                    variant=variant,
                    meili=meili,
                    dry_run=dry_run,
                    graph_session=graph_session,
                    neo4j_config=neo4j_config,
                )
                suite_rows.append(row)
                query_rows.append(row)
        suite_summaries.append(_suite_summary(suite=suite, rows=suite_rows))

    payload = _summary_payload(
        run_id=manifest.run_id,
        manifest_path=manifest.manifest_path,
        suites=suite_summaries,
        query_rows=query_rows,
        meili=meili,
        dry_run=dry_run,
    )
    metrics_path = manifest.output_dir / "metrics.json"
    query_results_path = manifest.output_dir / "query_results.jsonl"
    summary_path = manifest.output_dir / "summary.md"
    write_json(metrics_path, payload)
    write_jsonl(query_results_path, query_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_markdown(payload), encoding="utf-8")
    return CrossLectureRetrievalSmokeRun(
        run_id=manifest.run_id,
        output_dir=manifest.output_dir,
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        summary_path=summary_path,
        payload=payload,
    )


def load_cross_lecture_smoke_manifest(
    *,
    manifest_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> CrossLectureSmokeManifest:
    resolved_manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(resolved_manifest_path)
    base_dir = resolved_manifest_path.parent
    resolved_repo_root = (repo_root or default_paths().repo_root).expanduser().resolve()
    run_id = str(manifest.get("run_id") or f"cross_lecture_retrieval_smoke_{int(time.time())}")
    resolved_output_dir = _resolve_output_dir(
        manifest=manifest,
        output_dir=output_dir,
        base_dir=base_dir,
        repo_root=resolved_repo_root,
        run_id=run_id,
    )
    suites_payload = manifest.get("suites")
    if suites_payload is None:
        suites_payload = [
            {
                "suite_id": manifest.get("suite_id") or "cross_lecture_suite",
                "projects": manifest.get("projects"),
                "queries": manifest.get("queries"),
                "index": manifest.get("index"),
                "limit": manifest.get("limit") or manifest.get("top_k"),
                "top_k": manifest.get("top_k"),
                "candidate_pool_limit": manifest.get("candidate_pool_limit"),
                "graph_limit": manifest.get("graph_limit"),
                "variants": manifest.get("variants"),
            }
        ]
    if not isinstance(suites_payload, list):
        raise ValueError("cross-lecture retrieval smoke manifest requires a list under 'suites'")

    suites: list[CrossLectureSuiteConfig] = []
    for suite_index, suite_payload in enumerate(suites_payload, start=1):
        if not isinstance(suite_payload, dict):
            raise ValueError(f"cross-lecture suite #{suite_index} must be a JSON object")
        suites.append(
            _parse_suite(
                suite_payload,
                suite_index=suite_index,
                manifest=manifest,
                base_dir=base_dir,
                repo_root=resolved_repo_root,
            )
        )
    return CrossLectureSmokeManifest(
        run_id=run_id,
        output_dir=resolved_output_dir,
        suites=tuple(suites),
        manifest_path=resolved_manifest_path,
    )


def build_candidate_source_metrics(
    response: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    diagnostics = _mapping(response.get("diagnostics"))
    source_counts = {
        source_type: _public_source_count(counts)
        for source_type, counts in sorted(_mapping(diagnostics.get("source_counts")).items())
    }
    generated_recall = _source_recall_summary(_mapping(diagnostics.get("source_recall")))
    returned_recall = _source_recall_summary(
        _mapping(diagnostics.get("returned_source_recall"))
    )
    target_count = max(
        int(generated_recall.get("target_count") or 0),
        int(returned_recall.get("target_count") or 0),
    )
    target_rank = _minimum_target_rank(generated_recall)
    top_k_recalled_count = int(returned_recall.get("recalled_target_count") or 0)
    return {
        "public_safe": True,
        "top_k": top_k,
        "target_count": target_count,
        "target_rank": target_rank,
        "target_rank_bucket": _target_rank_bucket(target_rank),
        "top_k_recall": _top_k_recall(target_count, top_k_recalled_count),
        "top_k_recalled_count": top_k_recalled_count,
        "generated_recalled_count": int(generated_recall.get("recalled_target_count") or 0),
        "candidate_pool": _mapping(diagnostics.get("candidate_pool")),
        "source_counts": source_counts,
        "by_source": _merge_source_recall(
            source_counts=source_counts,
            generated=generated_recall,
            returned=returned_recall,
            target_count=target_count,
        ),
        "omits": ["raw_query", "raw_transcript", "evidence_text", "local_paths"],
    }


def _run_variant(
    *,
    client: CrossLectureSmokeClient,
    run_id: str,
    suite: CrossLectureSuiteConfig,
    project: CrossLectureProjectConfig,
    query: CrossLectureQueryConfig,
    variant: str,
    meili: dict[str, Any],
    dry_run: bool,
    graph_session: GraphCandidateSession | None,
    neo4j_config: Neo4jConfig | None,
) -> dict[str, Any]:
    if dry_run:
        return _skipped_query_row(
            run_id=run_id,
            suite=suite,
            project=project,
            query=query,
            variant=variant,
            skip_reason="dry_run_requested",
        )
    options = _variant_options(variant)
    if options["requires_meili"] and meili.get("available") is not True:
        return _skipped_query_row(
            run_id=run_id,
            suite=suite,
            project=project,
            query=query,
            variant=variant,
            skip_reason=MEILI_UNAVAILABLE_SKIP,
        )

    try:
        response = query_project_dual_candidates(
            client=client,  # type: ignore[arg-type]
            index_uid=project.index_uid,
            project_dir=project.project_dir,
            query=query.query_text,
            limit=suite.limit,
            candidate_pool_limit=suite.candidate_pool_limit,
            graph_limit=suite.graph_limit,
            evidence_units=project.evidence_units,
            target_evidence_unit_ids=list(query.target_evidence_unit_ids),
            target_segment_ids=list(query.target_segment_ids),
            enable_meili=options["enable_meili"],
            enable_graph=options["enable_graph"],
            graph_aware_rerank=options["graph_aware_rerank"],
            graph_session=graph_session,
            neo4j_config=neo4j_config,
        )
    except Exception as exc:  # noqa: BLE001 - smoke runner records public-safe buckets.
        return _skipped_query_row(
            run_id=run_id,
            suite=suite,
            project=project,
            query=query,
            variant=variant,
            skip_reason=_exception_skip_reason(exc, requires_meili=options["requires_meili"]),
            status="skipped",
        )

    graph_status = _public_graph_status(
        _mapping(_mapping(response.get("retrieval_context")).get("graph"))
    )
    if options["requires_graph"] and graph_status.get("available") is False:
        return _skipped_query_row(
            run_id=run_id,
            suite=suite,
            project=project,
            query=query,
            variant=variant,
            skip_reason=str(graph_status.get("skip_reason") or GRAPH_UNAVAILABLE_SKIP),
            graph_status=graph_status,
        )

    metrics = build_candidate_source_metrics(response, top_k=suite.limit)
    top_candidate = _first_dict(response.get("candidates"))
    row = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite.suite_id,
        "query_id": query.query_id,
        "project_ref": project.project_ref,
        "variant": variant,
        "status": "queried",
        "privacy": _privacy_policy(),
        "top_k": suite.limit,
        "index_ref": _id_ref(project.index_uid, prefix="idx"),
        "target_configured": bool(
            query.target_evidence_unit_ids or query.target_segment_ids
        ),
        "target_rank": metrics["target_rank"],
        "target_rank_bucket": metrics["target_rank_bucket"],
        "top_k_recall": metrics["top_k_recall"],
        "candidate_source_metrics": metrics,
        "candidate_source_contribution": metrics["source_counts"],
        "top_candidate": _public_candidate(top_candidate),
        "meilisearch": _public_meili_status(meili, attempted=options["enable_meili"]),
        "graph": graph_status,
        "graph_aware_rerank": _public_graph_aware_rerank(response),
    }
    row["diagnostic_bucket"] = _diagnostic_bucket(row)
    return row


def _skipped_query_row(
    *,
    run_id: str,
    suite: CrossLectureSuiteConfig,
    project: CrossLectureProjectConfig,
    query: CrossLectureQueryConfig,
    variant: str,
    skip_reason: str,
    status: str = "skipped",
    graph_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite.suite_id,
        "query_id": query.query_id,
        "project_ref": project.project_ref,
        "variant": variant,
        "status": status,
        "skip_reason": skip_reason,
        "privacy": _privacy_policy(),
        "top_k": suite.limit,
        "index_ref": _id_ref(project.index_uid, prefix="idx"),
        "target_configured": bool(
            query.target_evidence_unit_ids or query.target_segment_ids
        ),
        "target_rank": None,
        "target_rank_bucket": "not_queried",
        "top_k_recall": None,
        "candidate_source_metrics": _empty_candidate_source_metrics(suite.limit),
        "candidate_source_contribution": {},
        "top_candidate": None,
        "meilisearch": {"attempted": False, "available": False, "status": "skipped"},
        "graph": graph_status or {"attempted": False, "available": False, "status": "skipped"},
        "graph_aware_rerank": {"enabled": variant == "graph_aware_rerank", "status": "not_queried"},
        "diagnostic_bucket": skip_reason,
    }
    if skip_reason == MEILI_UNAVAILABLE_SKIP:
        row["meilisearch"] = {
            "attempted": True,
            "available": False,
            "status": "unavailable",
            "skip_reason": MEILI_UNAVAILABLE_SKIP,
        }
    return row


def _parse_suite(
    suite: dict[str, Any],
    *,
    suite_index: int,
    manifest: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
) -> CrossLectureSuiteConfig:
    suite_id = str(suite.get("suite_id") or f"cross_lecture_suite_{suite_index}")
    limit = _positive_int(
        suite.get("top_k") or suite.get("limit") or manifest.get("top_k") or manifest.get("limit") or 5,
        field_name="top_k",
    )
    candidate_pool_limit = _positive_int(
        suite.get("candidate_pool_limit")
        or manifest.get("candidate_pool_limit")
        or max(limit, 10),
        field_name="candidate_pool_limit",
    )
    graph_limit = _positive_int(
        suite.get("graph_limit") or manifest.get("graph_limit") or candidate_pool_limit,
        field_name="graph_limit",
    )
    projects = _parse_projects(
        suite=suite,
        manifest=manifest,
        base_dir=base_dir,
        repo_root=repo_root,
    )
    queries = _parse_queries(
        suite=suite,
        manifest=manifest,
        base_dir=base_dir,
        project_refs=set(projects),
    )
    variants = _parse_variants(suite.get("variants") or manifest.get("variants"))
    return CrossLectureSuiteConfig(
        suite_id=suite_id,
        projects=projects,
        queries=tuple(queries),
        limit=limit,
        candidate_pool_limit=max(candidate_pool_limit, limit),
        graph_limit=graph_limit,
        variants=variants,
    )


def _parse_projects(
    *,
    suite: dict[str, Any],
    manifest: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
) -> dict[str, CrossLectureProjectConfig]:
    raw_projects = suite.get("projects") or manifest.get("projects")
    if not isinstance(raw_projects, list) or not raw_projects:
        raise ValueError("cross-lecture suite requires a non-empty 'projects' list")
    projects: dict[str, CrossLectureProjectConfig] = {}
    default_index = _optional_str(suite.get("index") or manifest.get("index"))
    for index, project in enumerate(raw_projects, start=1):
        if not isinstance(project, dict):
            raise ValueError(f"cross-lecture project #{index} must be a JSON object")
        project_ref = _optional_str(
            project.get("project_ref") or project.get("lecture_id") or project.get("project_id")
        )
        if project_ref is None:
            project_ref = f"lecture_{index}"
        index_uid = _optional_str(project.get("index") or default_index)
        if index_uid is None:
            raise ValueError(f"cross-lecture project '{project_ref}' requires an index")
        project_dir = _project_dir_from_project(project, base_dir=base_dir, repo_root=repo_root)
        evidence_units = _optional_path(
            project.get("evidence_units"),
            base_dir=project_dir,
            repo_root=repo_root,
        )
        projects[project_ref] = CrossLectureProjectConfig(
            project_ref=project_ref,
            project_dir=project_dir,
            index_uid=index_uid,
            evidence_units=evidence_units,
        )
    return projects


def _parse_queries(
    *,
    suite: dict[str, Any],
    manifest: dict[str, Any],
    base_dir: Path,
    project_refs: set[str],
) -> list[CrossLectureQueryConfig]:
    raw_queries = _query_rows(suite.get("queries") or manifest.get("queries"), base_dir=base_dir)
    if not raw_queries:
        raise ValueError("cross-lecture suite requires at least one query")
    default_project_ref = next(iter(project_refs)) if len(project_refs) == 1 else None
    queries: list[CrossLectureQueryConfig] = []
    for query_index, query in enumerate(raw_queries, start=1):
        if not isinstance(query, dict):
            raise ValueError(f"cross-lecture query #{query_index} must be a JSON object")
        query_text = _optional_str(query.get("query") or query.get("question"))
        if query_text is None:
            raise ValueError(f"cross-lecture query #{query_index} requires 'query'")
        project_ref = _optional_str(query.get("project_ref") or query.get("lecture_id"))
        if project_ref is None:
            project_ref = default_project_ref
        if project_ref is None or project_ref not in project_refs:
            raise ValueError(
                f"cross-lecture query #{query_index} references unknown project_ref: {project_ref}"
            )
        query_id = _optional_str(query.get("query_id") or query.get("id")) or f"q{query_index:03d}"
        queries.append(
            CrossLectureQueryConfig(
                query_id=query_id,
                query_text=query_text,
                project_ref=project_ref,
                target_evidence_unit_ids=tuple(
                    _targets(
                        query,
                        singular="target_evidence_unit_id",
                        plural="target_evidence_unit_ids",
                        aliases=("expected_evidence_unit_id", "expected_evidence_unit_ids"),
                    )
                ),
                target_segment_ids=tuple(
                    _targets(
                        query,
                        singular="target_segment_id",
                        plural="target_segment_ids",
                        aliases=("expected_segment_id", "expected_segment_ids"),
                    )
                ),
            )
        )
    return queries


def _query_rows(value: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        path = value.get("path")
        if path is None:
            rows = value.get("rows")
            return [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []
        value = path
    path = _optional_path(value, base_dir=base_dir, repo_root=base_dir)
    if path is None:
        return []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [row for row in (_json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if isinstance(row, dict)]
    if suffix == ".json":
        loaded = _read_json(path)
        return [dict(item) for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported cross-lecture query file extension: {path.suffix}")


def _parse_variants(value: Any) -> tuple[str, ...]:
    raw_variants = value if isinstance(value, list) and value else list(DEFAULT_VARIANTS)
    variants = tuple(str(item) for item in raw_variants)
    unsupported = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unsupported:
        raise ValueError(f"Unsupported cross-lecture retrieval smoke variants: {unsupported}")
    return variants


def _project_dir_from_project(project: dict[str, Any], *, base_dir: Path, repo_root: Path) -> Path:
    project_dir = _optional_path(project.get("project_dir"), base_dir=base_dir, repo_root=repo_root)
    if project_dir is not None:
        return project_dir
    project_id = _optional_str(project.get("project_id"))
    if project_id is None:
        raise ValueError("cross-lecture project requires either project_dir or project_id")
    return (repo_root / "artifacts" / "projects" / project_id).expanduser().resolve()


def _variant_options(variant: str) -> dict[str, bool]:
    if variant == "meili_only":
        return {
            "enable_meili": True,
            "enable_graph": False,
            "graph_aware_rerank": False,
            "requires_meili": True,
            "requires_graph": False,
        }
    if variant == "graph_only":
        return {
            "enable_meili": False,
            "enable_graph": True,
            "graph_aware_rerank": False,
            "requires_meili": False,
            "requires_graph": True,
        }
    if variant == "meili_graph":
        return {
            "enable_meili": True,
            "enable_graph": True,
            "graph_aware_rerank": False,
            "requires_meili": True,
            "requires_graph": False,
        }
    if variant == "graph_aware_rerank":
        return {
            "enable_meili": True,
            "enable_graph": True,
            "graph_aware_rerank": True,
            "requires_meili": True,
            "requires_graph": False,
        }
    raise ValueError(f"Unsupported cross-lecture retrieval smoke variant: {variant}")


def _source_recall_summary(recall: dict[str, Any]) -> dict[str, Any]:
    by_source = {}
    recalled_targets: set[str] = set()
    for source_type, summary in _mapping(recall.get("by_source")).items():
        source_summary = _mapping(summary)
        targets = _string_list(source_summary.get("recalled_targets"))
        recalled_targets.update(targets)
        ranks = [
            _public_rank_record(record)
            for record in _list_of_dicts(source_summary.get("ranks"))
        ]
        by_source[str(source_type)] = {
            "recalled_count": int(source_summary.get("recalled_count") or len(targets)),
            "recalled_target_refs": [_target_ref(target) for target in sorted(targets)],
            "ranks": ranks,
            "rank_bucket_counts": dict(
                Counter(_target_rank_bucket(_optional_int(rank.get("candidate_rank"))) for rank in ranks)
            ),
        }
    return {
        "target_count": int(recall.get("target_count") or 0),
        "recalled_target_count": len(recalled_targets),
        "recalled_target_refs": [_target_ref(target) for target in sorted(recalled_targets)],
        "by_source": by_source,
    }


def _merge_source_recall(
    *,
    source_counts: dict[str, dict[str, int]],
    generated: dict[str, Any],
    returned: dict[str, Any],
    target_count: int,
) -> dict[str, Any]:
    source_types = sorted(
        set(source_counts)
        | set(_mapping(generated.get("by_source")))
        | set(_mapping(returned.get("by_source")))
    )
    merged: dict[str, Any] = {}
    for source_type in source_types:
        generated_summary = _mapping(_mapping(generated.get("by_source")).get(source_type))
        returned_summary = _mapping(_mapping(returned.get("by_source")).get(source_type))
        top_k_count = int(returned_summary.get("recalled_count") or 0)
        generated_count = int(generated_summary.get("recalled_count") or 0)
        merged[source_type] = {
            "generated_recalled_count": generated_count,
            "top_k_recalled_count": top_k_count,
            "top_k_recall": _top_k_recall(target_count, top_k_count),
            "generated_recalled_target_refs": _string_list(
                generated_summary.get("recalled_target_refs")
            ),
            "top_k_recalled_target_refs": _string_list(
                returned_summary.get("recalled_target_refs")
            ),
            "rank_bucket_counts": _mapping(generated_summary.get("rank_bucket_counts")),
            "candidate_source_contribution": source_counts.get(
                source_type,
                {"source_hits": 0, "unique_candidates": 0, "returned_candidates": 0},
            ),
        }
    return merged


def _suite_summary(
    *,
    suite: CrossLectureSuiteConfig,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "project_count": len(suite.projects),
        "query_count": len(suite.queries),
        "variant_count": len(suite.variants),
        "top_k": suite.limit,
        "variants": list(suite.variants),
        "query_status_counts": dict(Counter(str(row.get("status") or "unknown") for row in rows)),
        "diagnostic_bucket_counts": _diagnostic_bucket_counts(rows),
        "variant_metrics": _variant_metrics(rows),
        "source_recall_by_variant": _source_recall_by_variant(rows),
        "candidate_source_contribution": _candidate_source_contribution(rows),
        "graph_skip_reasons": _skip_reason_counts(rows, component="graph"),
        "meili_skip_reasons": _skip_reason_counts(rows, component="meilisearch"),
        "rerank_diagnostics": _graph_aware_rerank_inspection(rows),
    }


def _summary_payload(
    *,
    run_id: str,
    manifest_path: Path,
    suites: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    meili: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest_ref": _short_hash(str(manifest_path.expanduser().resolve())),
        "privacy": _privacy_policy(),
        "dry_run": dry_run,
        "meilisearch": {
            "available": bool(meili.get("available")),
            "status": str(meili.get("status") or "unknown"),
            "skip_reason": meili.get("skip_reason"),
        },
        "suite_count": len(suites),
        "query_variant_count": len(query_rows),
        "query_status_counts": dict(
            Counter(str(row.get("status") or "unknown") for row in query_rows)
        ),
        "diagnostic_bucket_counts": _diagnostic_bucket_counts(query_rows),
        "variant_metrics": _variant_metrics(query_rows),
        "source_recall_by_variant": _source_recall_by_variant(query_rows),
        "candidate_source_contribution": _candidate_source_contribution(query_rows),
        "rerank_diagnostics": _graph_aware_rerank_inspection(query_rows),
        "graph_skip_reasons": _skip_reason_counts(query_rows, component="graph"),
        "meili_skip_reasons": _skip_reason_counts(query_rows, component="meilisearch"),
        "suites": suites,
        "public_note": (
            "This report stores query IDs, hashed refs, source types, ranks, counts, "
            "recall buckets, rerank deltas, and skip reasons only."
        ),
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Lecture Retrieval Smoke",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "This public-safe report redacts raw query text, evidence text, transcripts, local paths, and raw target IDs.",
        "",
        "## Runtime",
        "",
        f"- Meilisearch available: `{payload['meilisearch']['available']}`",
        f"- Meilisearch status: `{payload['meilisearch']['status']}`",
    ]
    if payload["meilisearch"].get("skip_reason"):
        lines.append(f"- Meilisearch skip reason: `{payload['meilisearch']['skip_reason']}`")
    lines.extend(
        [
            f"- query variants: `{payload['query_variant_count']}`",
            f"- statuses: `{json.dumps(payload['query_status_counts'], sort_keys=True)}`",
            f"- diagnostic buckets: `{json.dumps(payload['diagnostic_bucket_counts'], sort_keys=True)}`",
            "",
            "## Variant Summary",
            "",
        ]
    )
    for variant, summary in _mapping(payload.get("variant_metrics")).items():
        lines.extend(
            [
                f"### {variant}",
                "",
                f"- statuses: `{json.dumps(summary.get('status_counts', {}), sort_keys=True)}`",
                f"- target rank buckets: `{json.dumps(summary.get('target_rank_bucket_counts', {}), sort_keys=True)}`",
                f"- top-k recall: `{summary.get('top_k_recall_count', 0)}`/`{summary.get('target_configured_count', 0)}`",
                f"- source recall: `{json.dumps(_mapping(payload.get('source_recall_by_variant')).get(variant, {}), sort_keys=True)}`",
                f"- candidate source contribution: `{json.dumps(_mapping(payload.get('candidate_source_contribution')).get(variant, {}), sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Rerank",
            "",
            f"- graph-aware top changes: `{payload['rerank_diagnostics'].get('top_changed_count', 0)}`",
            f"- graph-aware target rank buckets: `{json.dumps(payload['rerank_diagnostics'].get('target_rank_bucket_counts', {}), sort_keys=True)}`",
            "",
            "## Skip Reasons",
            "",
            f"- Meili: `{json.dumps(payload.get('meili_skip_reasons', {}), sort_keys=True)}`",
            f"- Graph: `{json.dumps(payload.get('graph_skip_reasons', {}), sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in sorted({str(row.get("variant") or "unknown") for row in rows}):
        variant_rows = [row for row in rows if row.get("variant") == variant]
        configured = [row for row in variant_rows if row.get("target_configured") is True]
        recalled = [row for row in configured if row.get("top_k_recall") is True]
        metrics[variant] = {
            "query_variant_count": len(variant_rows),
            "status_counts": dict(Counter(str(row.get("status") or "unknown") for row in variant_rows)),
            "target_configured_count": len(configured),
            "top_k_recall_count": len(recalled),
            "target_rank_bucket_counts": dict(
                Counter(str(row.get("target_rank_bucket") or "unknown") for row in configured)
            ),
            "diagnostic_bucket_counts": _diagnostic_bucket_counts(variant_rows),
        }
    return metrics


def _source_recall_by_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = str(row.get("variant") or "unknown")
        metrics = _mapping(row.get("candidate_source_metrics"))
        for source_type, summary in _mapping(metrics.get("by_source")).items():
            target = by_variant.setdefault(variant, {}).setdefault(
                source_type,
                {
                    "generated_recalled_count": 0,
                    "top_k_recalled_count": 0,
                    "target_configured_count": 0,
                    "rank_bucket_counts": {},
                },
            )
            target["generated_recalled_count"] += int(
                _mapping(summary).get("generated_recalled_count") or 0
            )
            target["top_k_recalled_count"] += int(
                _mapping(summary).get("top_k_recalled_count") or 0
            )
            if int(metrics.get("target_count") or 0) > 0:
                target["target_configured_count"] += 1
            for bucket, count in _mapping(_mapping(summary).get("rank_bucket_counts")).items():
                target["rank_bucket_counts"][bucket] = (
                    int(target["rank_bucket_counts"].get(bucket) or 0) + int(count or 0)
                )
    return by_variant


def _candidate_source_contribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        variant = str(row.get("variant") or "unknown")
        for source_type, counts in _mapping(row.get("candidate_source_contribution")).items():
            target = by_variant.setdefault(variant, {}).setdefault(
                source_type,
                {"source_hits": 0, "unique_candidates": 0, "returned_candidates": 0},
            )
            for key in ("source_hits", "unique_candidates", "returned_candidates"):
                target[key] += int(_mapping(counts).get(key) or 0)
    return by_variant


def _graph_aware_rerank_inspection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        _mapping(row.get("graph_aware_rerank"))
        for row in rows
        if row.get("variant") == "graph_aware_rerank"
    ]
    computed = [diag for diag in diagnostics if diag.get("enabled") is True and diag.get("status") == "computed"]
    return {
        "enabled_query_count": len([diag for diag in diagnostics if diag.get("enabled") is True]),
        "computed_query_count": len(computed),
        "top_changed_count": sum(1 for diag in computed if diag.get("top_changed") is True),
        "target_rank_bucket_counts": dict(
            Counter(
                str(diag.get("target_rank_bucket") or "unknown")
                for diag in computed
            )
        ),
        "component_presence_counts": dict(
            Counter(
                component
                for diag in computed
                for component in _string_list(diag.get("component_names"))
            )
        ),
    }


def _public_graph_aware_rerank(response: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping(_mapping(response.get("diagnostics")).get("graph_aware_rerank"))
    if not diagnostics:
        return {"enabled": False, "status": "disabled"}
    metrics = build_candidate_source_metrics(response, top_k=int(response.get("limit") or 0))
    return {
        "enabled": True,
        "status": "computed",
        "strategy": diagnostics.get("strategy"),
        "query_type": diagnostics.get("query_type"),
        "candidate_count": int(diagnostics.get("candidate_count") or 0),
        "top_changed": bool(diagnostics.get("top_changed")),
        "base_top_ref": diagnostics.get("base_top_ref"),
        "reranked_top_ref": diagnostics.get("reranked_top_ref"),
        "reranked_top_original_rank": diagnostics.get("reranked_top_original_rank"),
        "reranked_top_rank": diagnostics.get("reranked_top_rank"),
        "component_names": _string_list(diagnostics.get("component_names")),
        "target_rank": metrics.get("target_rank"),
        "target_rank_bucket": metrics.get("target_rank_bucket"),
        "top_k_recall": metrics.get("top_k_recall"),
        "public_safe": True,
    }


def _public_graph_status(status: dict[str, Any]) -> dict[str, Any]:
    if not status:
        return {"attempted": False, "available": False, "status": "not_configured"}
    return {
        "attempted": bool(status.get("attempted")),
        "available": bool(status.get("available")),
        "status": str(status.get("status") or "unknown"),
        "skip_reason": status.get("skip_reason"),
        "hit_count": int(status.get("hit_count") or 0),
    }


def _public_meili_status(meili: dict[str, Any], *, attempted: bool) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "available": bool(meili.get("available")) if attempted else False,
        "status": str(meili.get("status") or "skipped") if attempted else "skipped",
        "skip_reason": meili.get("skip_reason") if attempted else "meili_disabled",
    }


def _public_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        "rank": _optional_int(candidate.get("rank")),
        "ref": _id_ref(_candidate_identity(candidate), prefix="evu"),
        "target_ref": _id_ref(_optional_str(candidate.get("target_segment_id")), prefix="seg"),
        "source_types": _string_list(candidate.get("candidate_source_types")),
        "candidate_source_count": int(candidate.get("candidate_source_count") or 0),
        "selected_source": _optional_str(candidate.get("selected_source")),
        "score_bucket": _score_bucket(
            _optional_float(candidate.get("score") or candidate.get("_rankingScore"))
        ),
        "timestamp_available": _optional_float(candidate.get("start_time")) is not None,
    }


def _public_rank_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_ref": _target_ref(str(record.get("target") or "")),
        "source_rank": _optional_int(record.get("source_rank")),
        "candidate_rank": _optional_int(record.get("candidate_rank")),
    }


def _public_source_count(counts: Any) -> dict[str, int]:
    mapping = _mapping(counts)
    return {
        "source_hits": int(mapping.get("source_hits") or 0),
        "unique_candidates": int(mapping.get("unique_candidates") or 0),
        "returned_candidates": int(mapping.get("returned_candidates") or 0),
    }


def _empty_candidate_source_metrics(top_k: int) -> dict[str, Any]:
    return {
        "public_safe": True,
        "top_k": top_k,
        "target_count": 0,
        "target_rank": None,
        "target_rank_bucket": "not_queried",
        "top_k_recall": None,
        "top_k_recalled_count": 0,
        "generated_recalled_count": 0,
        "candidate_pool": {"generated": 0, "returned": 0},
        "source_counts": {},
        "by_source": {},
        "omits": ["raw_query", "raw_transcript", "evidence_text", "local_paths"],
    }


def _diagnostic_bucket(row: dict[str, Any]) -> str:
    if row.get("status") != "queried":
        return str(row.get("skip_reason") or row.get("status") or "not_queried")
    metrics = _mapping(row.get("candidate_source_metrics"))
    if row.get("target_configured") is not True:
        return "target_not_configured"
    if int(metrics.get("generated_recalled_count") or 0) <= 0:
        return "candidate_recall_failure"
    if row.get("top_k_recall") is not True:
        return "top_k_recall_miss"
    if _optional_int(row.get("target_rank")) == 1:
        return "target_top1"
    if row.get("variant") == "graph_aware_rerank" and _mapping(row.get("graph_aware_rerank")).get("top_changed") is True:
        return "rerank_changed_top"
    return "target_found_not_top1"


def _diagnostic_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("diagnostic_bucket") or "unknown") for row in rows))


def _skip_reason_counts(rows: list[dict[str, Any]], *, component: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        status = _mapping(row.get(component))
        reason = _optional_str(status.get("skip_reason"))
        if reason:
            counts[reason] += 1
    return dict(counts)


def _minimum_target_rank(recall: dict[str, Any]) -> int | None:
    ranks = [
        _optional_int(record.get("candidate_rank"))
        for summary in _mapping(recall.get("by_source")).values()
        for record in _list_of_dicts(_mapping(summary).get("ranks"))
    ]
    ranks = [rank for rank in ranks if rank is not None]
    return min(ranks) if ranks else None


def _target_rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found"
    if rank <= 1:
        return "top1"
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "top10"
    if rank <= 50:
        return "top50"
    return "beyond_top50"


def _top_k_recall(target_count: int, recalled_count: int) -> bool | None:
    if target_count <= 0:
        return None
    return recalled_count >= target_count


def _target_ref(target: str) -> str:
    if ":" in target:
        prefix, value = target.split(":", 1)
        return _id_ref(value, prefix=prefix.replace("_", "-")) or f"{prefix}:missing"
    return _id_ref(target, prefix="target") or "target:missing"


def _candidate_identity(candidate: dict[str, Any]) -> str | None:
    return _optional_str(
        candidate.get("evidence_unit_id")
        or candidate.get("target_segment_id")
        or candidate.get("candidate_key")
    )


def _meili_health(client: CrossLectureSmokeClient) -> dict[str, Any]:
    try:
        response = client.health()
    except Exception:  # noqa: BLE001 - public report records only the bucket.
        return {
            "available": False,
            "status": "unavailable",
            "skip_reason": MEILI_UNAVAILABLE_SKIP,
        }
    status = str(response.get("status") or "").lower()
    available = status == "available" or response.get("available") is True
    return {
        "available": available,
        "status": status or "unknown",
        "skip_reason": None if available else MEILI_UNAVAILABLE_SKIP,
    }


def _exception_skip_reason(exc: Exception, *, requires_meili: bool) -> str:
    if requires_meili and _looks_like_meili_unavailable(exc):
        return MEILI_UNAVAILABLE_SKIP
    return "query_runtime_exception"


def _looks_like_meili_unavailable(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        token in message
        for token in (
            "connection",
            "refused",
            "meili",
            "timeout",
            "temporarily unavailable",
            "failed to establish",
        )
    )


def _privacy_policy() -> dict[str, Any]:
    return {
        "public_safe": True,
        "omits": [
            "raw_query",
            "raw_transcript",
            "evidence_text",
            "semantic_text",
            "local_paths",
            "frame_paths",
            "raw_target_ids",
        ],
    }


def _resolve_output_dir(
    *,
    manifest: dict[str, Any],
    output_dir: Path | None,
    base_dir: Path,
    repo_root: Path,
    run_id: str,
) -> Path:
    configured = output_dir or _optional_path(
        manifest.get("output_dir"),
        base_dir=base_dir,
        repo_root=repo_root,
    )
    if configured is not None:
        return configured.expanduser().resolve()
    return (repo_root / DEFAULT_OUTPUT_ROOT / run_id).expanduser().resolve()


def _optional_path(value: Any, *, base_dir: Path, repo_root: Path) -> Path | None:
    raw = _optional_str(value)
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    base_candidate = (base_dir / path).resolve()
    repo_candidate = (repo_root / path).resolve()
    if base_candidate.exists():
        return base_candidate
    if repo_candidate.exists():
        return repo_candidate
    return base_candidate


def _targets(
    row: dict[str, Any],
    *,
    singular: str,
    plural: str,
    aliases: tuple[str, ...],
) -> list[str]:
    values: list[str] = []
    for key in (singular, plural, *aliases):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value not in (None, ""):
            raw = str(value)
            values.extend(item.strip() for item in raw.split("|") if item.strip())
    return sorted(set(values))


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "missing"
    if score >= 0.9:
        return "0.90-1.00"
    if score >= 0.75:
        return "0.75-0.90"
    if score >= 0.5:
        return "0.50-0.75"
    if score > 0:
        return "0.00-0.50"
    return "zero"


def _id_ref(value: str | None, *, prefix: str) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return f"{prefix}:{_short_hash(raw)}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return dict(item)
    return {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


_json = json
