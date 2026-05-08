from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oarag.graph.graph_query import GraphTraversalConfig, graph_query
from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.project_query import SEGMENT_HIT_SOURCE, VISUAL_ENTITY_HIT_SOURCE, query_project


PUBLIC_SCHEMA_VERSION = "lecture-smoke-public-v1"
PRIVATE_SCHEMA_VERSION = "lecture-smoke-private-v1"
DEFAULT_OUTPUT_ROOT = Path("reports") / "lecture_smoke"


class SearchClient(Protocol):
    def search(self, index_uid: str, query: str, limit: int = 10) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LectureSmokeRun:
    run_id: str
    output_dir: Path
    metrics_path: Path
    query_results_path: Path
    summary_path: Path
    private_output_dir: Path | None
    private_query_outputs_path: Path | None
    summary: dict[str, Any]


def run_lecture_smoke(
    *,
    client: SearchClient,
    manifest_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    private_output_dir: Path | None = None,
    allow_private_output: bool = False,
) -> LectureSmokeRun:
    manifest = _read_json(manifest_path)
    base_dir = manifest_path.expanduser().resolve().parent
    run_id = str(manifest.get("run_id") or f"lecture_smoke_{int(time.time())}")
    resolved_output_dir = _resolve_output_dir(
        output_dir=output_dir,
        manifest=manifest,
        base_dir=base_dir,
        run_id=run_id,
    )
    resolved_repo_root = (repo_root or base_dir).expanduser().resolve()
    resolved_private_output_dir = _resolve_private_output_dir(
        private_output_dir=private_output_dir,
        manifest=manifest,
        base_dir=base_dir,
        allow_private_output=allow_private_output,
    )

    suite_metrics: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    suites = manifest.get("suites") or []
    if not isinstance(suites, list):
        raise ValueError("lecture-smoke manifest requires a list under 'suites'")

    for suite_index, suite in enumerate(suites, start=1):
        if not isinstance(suite, dict):
            raise ValueError(f"lecture-smoke suite #{suite_index} must be a JSON object")
        metrics, rows, raw_rows = run_lecture_smoke_suite(
            client=client,
            suite=suite,
            base_dir=base_dir,
            repo_root=resolved_repo_root,
            run_id=run_id,
            write_private=resolved_private_output_dir is not None,
        )
        suite_metrics.append(metrics)
        public_rows.extend(rows)
        private_rows.extend(raw_rows)

    summary = _summary_payload(
        run_id=run_id,
        suites=suite_metrics,
        query_count=len(public_rows),
        private_output_written=resolved_private_output_dir is not None,
    )
    metrics_path = resolved_output_dir / "metrics.json"
    query_results_path = resolved_output_dir / "query_results.jsonl"
    summary_path = resolved_output_dir / "summary.md"
    write_json(metrics_path, summary)
    write_jsonl(query_results_path, public_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_markdown(summary, public_rows), encoding="utf-8")

    private_query_outputs_path = None
    if resolved_private_output_dir is not None:
        private_query_outputs_path = resolved_private_output_dir / "private_query_outputs.jsonl"
        write_jsonl(private_query_outputs_path, private_rows)
        warning_path = resolved_private_output_dir / "PRIVATE_OUTPUT_WARNING.txt"
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(
            "Private lecture-smoke outputs may contain raw query text, transcript excerpts, "
            "local absolute paths, frame labels, and full retrieval responses. Keep this "
            "directory outside public PRs, docs, and tracked fixtures.\n",
            encoding="utf-8",
        )

    return LectureSmokeRun(
        run_id=run_id,
        output_dir=resolved_output_dir,
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        summary_path=summary_path,
        private_output_dir=resolved_private_output_dir,
        private_query_outputs_path=private_query_outputs_path,
        summary=summary,
    )


def run_lecture_smoke_suite(
    *,
    client: SearchClient,
    suite: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
    run_id: str,
    write_private: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    suite_type = str(suite.get("type") or "lecture_project").strip()
    if suite_type not in {"lecture_project", "local_project"}:
        raise ValueError(f"Unsupported lecture-smoke suite type: {suite_type}")
    suite_id = str(suite.get("suite_id") or suite.get("project_id") or "lecture_suite")
    domain = str(suite.get("domain") or "private_lecture")
    project_dir = _project_dir_from_suite(suite=suite, base_dir=base_dir, repo_root=repo_root)
    index_uid = str(suite["index"])
    visual_index_uid = _optional_str(suite.get("visual_index"))
    limit = int(suite.get("limit", 5))
    neighbor_count = int(suite.get("neighbor_count", 1))
    graph_config = _graph_config_from_suite(suite)
    queries = _read_queries(base_dir=base_dir, suite=suite)

    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for query_index, query_row in enumerate(queries, start=1):
        query_text = _query_text(query_row, query_index=query_index)
        query_id = _query_id(query_row, query_index=query_index)
        started = time.perf_counter()
        if graph_config is not None:
            response = graph_query(
                client=client,
                index_uid=index_uid,
                project_dir=project_dir,
                project_id=_optional_str(suite.get("project_id")),
                query=query_text,
                limit=limit,
                segments_path=_optional_path(suite.get("segments")),
                frames_manifest_path=_optional_path(suite.get("frames_manifest")),
                visual_entities_path=_optional_path(suite.get("visual_entities")),
                entity_links_path=_optional_path(suite.get("entity_links")),
                domain_lexicon_path=_optional_path(suite.get("domain_lexicon")),
                traversal_config=graph_config,
            )
        else:
            response = query_project(
                client=client,
                index_uid=index_uid,
                visual_index_uid=visual_index_uid,
                project_dir=project_dir,
                query=query_text,
                limit=limit,
                segments_path=_optional_path(suite.get("segments")),
                frames_manifest_path=_optional_path(suite.get("frames_manifest")),
                visual_entities_path=_optional_path(suite.get("visual_entities")),
                entity_links_path=_optional_path(suite.get("entity_links")),
                domain_lexicon_path=_optional_path(suite.get("domain_lexicon")),
                window_seconds=_optional_float(suite.get("window_seconds")),
                neighbor_count=neighbor_count,
                previous_neighbor_count=_optional_int(suite.get("previous_neighbor_count")),
                next_neighbor_count=_optional_int(suite.get("next_neighbor_count")),
                window_before_seconds=_optional_float(suite.get("window_before_seconds")),
                window_after_seconds=_optional_float(suite.get("window_after_seconds")),
                rerank=bool(suite.get("rerank", False)),
                rerank_time_hint=_rerank_time_hint(suite=suite, query_row=query_row),
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 4)
        rows.append(
            _public_query_row(
                run_id=run_id,
                suite_id=suite_id,
                suite_type=suite_type,
                domain=domain,
                index_uid=index_uid,
                visual_index_uid=visual_index_uid,
                query_id=query_id,
                query_row=query_row,
                response=response,
                elapsed_ms=elapsed_ms,
                graph_enabled=graph_config is not None,
            )
        )
        if write_private:
            private_rows.append(
                {
                    "schema_version": PRIVATE_SCHEMA_VERSION,
                    "run_id": run_id,
                    "suite_id": suite_id,
                    "query_id": query_id,
                    "query_text": query_text,
                    "elapsed_time_ms": elapsed_ms,
                    "response": response,
                }
            )

    return _suite_metrics(suite_id=suite_id, suite_type=suite_type, domain=domain, rows=rows), rows, private_rows


def _public_query_row(
    *,
    run_id: str,
    suite_id: str,
    suite_type: str,
    domain: str,
    index_uid: str,
    visual_index_uid: str | None,
    query_id: str,
    query_row: dict[str, Any],
    response: dict[str, Any],
    elapsed_ms: float,
    graph_enabled: bool,
) -> dict[str, Any]:
    top_bundle = _first_dict(response.get("bundles"))
    candidate = _top_candidate(response=response, top_bundle=top_bundle)
    retrieval_sources = _list_of_dicts(top_bundle.get("retrieval_sources") if top_bundle else None)
    top_source = _top_candidate_source(candidate=candidate, top_bundle=top_bundle, retrieval_sources=retrieval_sources)
    frame_ref_count = _top_frame_ref_count(top_bundle)
    linked_entity_count = len(_list_of_dicts(top_bundle.get("linked_entities") if top_bundle else None))
    visual_entity_count = len(_list_of_dicts(top_bundle.get("visual_entities") if top_bundle else None))
    semantic_fields = _semantic_source_fields(candidate=candidate, top_bundle=top_bundle)
    graph_evidence_count = _graph_evidence_count(response)
    processing_time_ms = _optional_float(response.get("processing_time_ms"))
    counts = response.get("counts") if isinstance(response.get("counts"), dict) else {}
    search_hit_count = _optional_int(counts.get("search_hits"))
    if search_hit_count is None:
        search_hit_count = len(_list_of_dicts(response.get("candidates")))
    timestamp_available = _timestamp_available(candidate)
    frame_backed = frame_ref_count > 0

    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "suite_type": suite_type,
        "domain": domain,
        "query_id": query_id,
        "query_label": _optional_public_label(query_row),
        "privacy": {
            "raw_query_text": "redacted",
            "transcript_excerpt": "redacted",
            "local_paths": "redacted",
            "raw_response_in_public_output": False,
        },
        "index_ref": f"index:{_short_hash(index_uid)}",
        "visual_index_configured": visual_index_uid is not None,
        "graph_configured": graph_enabled,
        "search_hit_count": search_hit_count,
        "top_candidate": {
            "ref": _candidate_ref(candidate=candidate, source=top_source),
            "source": top_source,
            "timestamp_available": timestamp_available,
        },
        "top_candidate_source": top_source,
        "top_candidate_timestamp_available": timestamp_available,
        "frame_backed_evidence": frame_backed,
        "frame_ref_count": frame_ref_count,
        "linked_entity_count": linked_entity_count,
        "visual_entity_count": visual_entity_count,
        "graph_evidence_count": graph_evidence_count,
        "transcript_only_fallback": bool(candidate)
        and top_source == SEGMENT_HIT_SOURCE
        and not frame_backed
        and linked_entity_count == 0,
        "semantic_source_fields_available": bool(semantic_fields),
        "semantic_source_fields_count": len(semantic_fields),
        "retrieval_source_count": len(retrieval_sources),
        "processing_time_ms": processing_time_ms,
        "elapsed_time_ms": elapsed_ms,
        "warning_count": len(response.get("warnings") or []),
    }


def _suite_metrics(
    *,
    suite_id: str,
    suite_type: str,
    domain: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    top_source_counts = Counter(row.get("top_candidate_source") or "none" for row in rows)
    graph_counts = [
        int(row["graph_evidence_count"])
        for row in rows
        if row.get("graph_evidence_count") is not None
    ]
    return {
        "suite_id": suite_id,
        "suite_type": suite_type,
        "domain": domain,
        "query_count": len(rows),
        "mean_search_hit_count": _mean(row.get("search_hit_count") for row in rows),
        "top_candidate_source_counts": dict(sorted(top_source_counts.items())),
        "timestamp_available_ratio": _ratio(rows, "top_candidate_timestamp_available"),
        "frame_backed_ratio": _ratio(rows, "frame_backed_evidence"),
        "transcript_only_fallback_ratio": _ratio(rows, "transcript_only_fallback"),
        "mean_linked_entity_count": _mean(row.get("linked_entity_count") for row in rows),
        "graph_evidence_count": sum(graph_counts) if graph_counts else None,
        "semantic_source_fields_available_ratio": _ratio(
            rows,
            "semantic_source_fields_available",
        ),
        "mean_semantic_source_fields_count": _mean(
            row.get("semantic_source_fields_count") for row in rows
        ),
        "mean_processing_time_ms": _mean(row.get("processing_time_ms") for row in rows),
        "mean_elapsed_time_ms": _mean(row.get("elapsed_time_ms") for row in rows),
    }


def _summary_payload(
    *,
    run_id: str,
    suites: list[dict[str, Any]],
    query_count: int,
    private_output_written: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_count": len(suites),
        "query_count": query_count,
        "privacy": {
            "public_outputs": ["metrics.json", "query_results.jsonl", "summary.md"],
            "public_outputs_are_sanitized": True,
            "public_omits": [
                "raw_query_text",
                "transcript_excerpt",
                "local_absolute_paths",
                "frame_paths",
                "visual_entity_text",
                "full_retrieval_response",
            ],
            "private_output_written": private_output_written,
            "private_output_requires_explicit_opt_in": True,
        },
        "suites": suites,
    }


def _summary_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Lecture Smoke: {summary['run_id']}",
        "",
        "## Privacy Boundary",
        "",
        "- Public files in this directory are sanitized.",
        "- Raw query text, transcript excerpts, frame paths, local absolute paths, entity labels, and full retrieval responses are omitted.",
        "- Private raw output is written only when `--allow-private-output` and `--private-output-dir` are both supplied.",
        "",
        "## Suites",
        "",
        "| suite | type | domain | queries | hits | top sources | frame-backed | transcript-only | linked | graph | semantic fields | latency ms |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite in summary["suites"]:
        lines.append(
            "| {suite_id} | {suite_type} | {domain} | {query_count} | {hits} | {sources} | "
            "{frame} | {fallback} | {linked} | {graph} | {semantic} | {latency} |".format(
                suite_id=suite.get("suite_id"),
                suite_type=suite.get("suite_type"),
                domain=suite.get("domain"),
                query_count=suite.get("query_count") or 0,
                hits=_format_metric(suite.get("mean_search_hit_count")),
                sources=_format_source_counts(suite.get("top_candidate_source_counts")),
                frame=_format_metric(suite.get("frame_backed_ratio")),
                fallback=_format_metric(suite.get("transcript_only_fallback_ratio")),
                linked=_format_metric(suite.get("mean_linked_entity_count")),
                graph=_format_metric(suite.get("graph_evidence_count")),
                semantic=_format_metric(suite.get("semantic_source_fields_available_ratio")),
                latency=_format_metric(suite.get("mean_processing_time_ms")),
            )
        )
    lines.extend(
        [
            "",
            "## Query Results",
            "",
            "| suite | query_id | hits | source | timestamp | frame-backed | linked | graph | transcript-only | semantic fields | latency ms |",
            "| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {suite} | {query} | {hits} | {source} | {timestamp} | {frame} | {linked} | "
            "{graph} | {fallback} | {semantic} | {latency} |".format(
                suite=row.get("suite_id"),
                query=row.get("query_id"),
                hits=row.get("search_hit_count"),
                source=row.get("top_candidate_source") or "-",
                timestamp=_format_bool(row.get("top_candidate_timestamp_available")),
                frame=_format_bool(row.get("frame_backed_evidence")),
                linked=row.get("linked_entity_count"),
                graph=_format_metric(row.get("graph_evidence_count")),
                fallback=_format_bool(row.get("transcript_only_fallback")),
                semantic=row.get("semantic_source_fields_count"),
                latency=_format_metric(row.get("processing_time_ms")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _top_candidate(response: dict[str, Any], top_bundle: dict[str, Any] | None) -> dict[str, Any]:
    if top_bundle and isinstance(top_bundle.get("candidate"), dict):
        return dict(top_bundle["candidate"])
    candidate = _first_dict(response.get("candidates"))
    return candidate or {}


def _top_candidate_source(
    *,
    candidate: dict[str, Any],
    top_bundle: dict[str, Any] | None,
    retrieval_sources: list[dict[str, Any]],
) -> str | None:
    merge = top_bundle.get("merge") if top_bundle else None
    if isinstance(merge, dict) and merge.get("selected_source"):
        return str(merge["selected_source"])
    if candidate.get("source"):
        return str(candidate["source"])
    if retrieval_sources and retrieval_sources[0].get("source"):
        return str(retrieval_sources[0]["source"])
    if candidate:
        return SEGMENT_HIT_SOURCE
    return None


def _top_frame_ref_count(top_bundle: dict[str, Any] | None) -> int:
    if not top_bundle:
        return 0
    evidence_window = top_bundle.get("evidence_window")
    if not isinstance(evidence_window, dict):
        return 0
    return len(_list_of_dicts(evidence_window.get("frame_refs")))


def _semantic_source_fields(
    *,
    candidate: dict[str, Any],
    top_bundle: dict[str, Any] | None,
) -> list[str]:
    fields = _string_list(candidate.get("semantic_source_fields"))
    if fields:
        return fields
    if not top_bundle:
        return []
    evidence_window = top_bundle.get("evidence_window")
    if not isinstance(evidence_window, dict):
        return []
    target_segment = evidence_window.get("target_segment")
    if not isinstance(target_segment, dict):
        return []
    return _string_list(target_segment.get("semantic_source_fields"))


def _timestamp_available(candidate: dict[str, Any]) -> bool:
    return any(
        _optional_float(candidate.get(field)) is not None
        for field in (
            "timestamp_center",
            "start_time",
            "end_time",
            "target_segment_timestamp_center",
            "target_segment_start_time",
            "target_segment_end_time",
        )
    )


def _graph_evidence_count(response: dict[str, Any]) -> int | None:
    counts = response.get("counts")
    if isinstance(counts, dict) and "graph_evidence" in counts:
        return _optional_int(counts.get("graph_evidence")) or 0
    evidence = response.get("graph_evidence")
    if isinstance(evidence, list):
        return len(evidence)
    return None


def _candidate_ref(candidate: dict[str, Any], source: str | None) -> str | None:
    raw_id = (
        candidate.get("entity_id")
        or candidate.get("segment_id")
        or candidate.get("frame_id")
        or candidate.get("sample_id")
    )
    if raw_id is None:
        return None
    prefix = source if source in {SEGMENT_HIT_SOURCE, VISUAL_ENTITY_HIT_SOURCE} else "candidate"
    return f"{prefix}:{_short_hash(str(raw_id))}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _read_queries(*, base_dir: Path, suite: dict[str, Any]) -> list[dict[str, Any]]:
    source = suite.get("queries", suite.get("query_file"))
    if isinstance(source, list):
        return [dict(item) for item in source if isinstance(item, dict)]
    if source is None:
        raise ValueError("lecture-smoke suite requires 'queries' or 'query_file'")
    path = _resolve_path(base_dir, source)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSONL query rows must be objects: {path}")
                    rows.append(payload)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            return [dict(item) for item in payload["queries"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported lecture-smoke query file format: {path}")


def _query_text(query_row: dict[str, Any], *, query_index: int) -> str:
    value = query_row.get("query_text", query_row.get("query"))
    if value is None or not str(value).strip():
        raise ValueError(f"lecture-smoke query #{query_index} is missing query_text")
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


def _rerank_time_hint(*, suite: dict[str, Any], query_row: dict[str, Any]) -> str | None:
    field_name = _optional_str(suite.get("rerank_time_hint_field"))
    if field_name and query_row.get(field_name):
        return str(query_row[field_name])
    return _optional_str(suite.get("rerank_time_hint"))


def _graph_config_from_suite(suite: dict[str, Any]) -> GraphTraversalConfig | None:
    graph = suite.get("graph")
    if graph is None:
        graph = suite.get("graph_query")
    if graph in (None, False):
        return None
    if graph is True:
        graph = {}
    if not isinstance(graph, dict):
        raise ValueError("lecture-smoke graph config must be true/false or an object")
    if graph.get("enabled") is False:
        return None
    return GraphTraversalConfig(
        lookback_segments=int(graph.get("lookback_segments", 3)),
        per_candidate_limit=int(graph.get("limit", graph.get("per_candidate_limit", 12))),
    )


def _project_dir_from_suite(*, suite: dict[str, Any], base_dir: Path, repo_root: Path) -> Path:
    if suite.get("project_dir") is not None:
        return _resolve_path(base_dir, suite["project_dir"])
    if suite.get("project_id") is not None:
        return (repo_root / "artifacts" / "projects" / str(suite["project_id"])).resolve()
    raise ValueError("lecture-smoke suite requires project_dir or project_id")


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
        path = DEFAULT_OUTPUT_ROOT / run_id
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_private_output_dir(
    *,
    private_output_dir: Path | None,
    manifest: dict[str, Any],
    base_dir: Path,
    allow_private_output: bool,
) -> Path | None:
    manifest_private_output = manifest.get("private_output_dir")
    if private_output_dir is None and manifest_private_output is not None:
        private_output_dir = Path(str(manifest_private_output))
    if private_output_dir is None:
        return None
    if not allow_private_output:
        raise ValueError(
            "Private lecture-smoke output requires allow_private_output=True. "
            "Private output can contain raw queries, transcripts, and local paths."
        )
    path = private_output_dir.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in lecture-smoke manifest: {path}")
    return payload


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


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
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, dict):
            return item
    return None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else ([] if value is None else [value])
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows), 4)


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _format_source_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{source}:{count}" for source, count in sorted(value.items()))


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_metric(value: Any) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "-"
    if parsed == int(parsed):
        return str(int(parsed))
    return f"{parsed:.4f}".rstrip("0").rstrip(".")
