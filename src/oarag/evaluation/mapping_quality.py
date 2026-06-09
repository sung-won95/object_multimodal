from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from oarag.core.io import write_json


MAPPING_QUALITY_SCHEMA_VERSION = "oarag-mapping-quality-report-v1"

METRICS_FILENAME = "mapping_quality_metrics.json"
MARKDOWN_FILENAME = "mapping_quality_summary.md"
CSV_FILENAME = "mapping_quality_artifacts.csv"

ARTIFACT_DEFAULTS = {
    "lecture_segments_aligned": Path("segments") / "lecture_segments_aligned.jsonl",
    "lecture_windows": Path("segments") / "lecture_windows.jsonl",
    "visual_states": Path("manifests") / "visual_states.jsonl",
    "visual_entities": Path("manifests") / "visual_entities.jsonl",
    "entity_links": Path("manifests") / "entity_links.jsonl",
    "evidence_units": Path("segments") / "evidence_units.jsonl",
    "concept_graph": Path("manifests") / "concept_graph.jsonl",
    "global_concept_graph": Path("manifests") / "global_concept_graph.json",
}

ARTIFACT_ALIASES = {
    "lecture_segments_aligned": ("lecture_segments_aligned", "segments_aligned", "segments"),
    "lecture_windows": ("lecture_windows", "windows"),
    "visual_states": ("visual_states",),
    "visual_entities": ("visual_entities",),
    "entity_links": ("entity_links",),
    "evidence_units": ("evidence_units",),
    "concept_graph": ("concept_graph", "lecture_concept_graph"),
    "global_concept_graph": ("global_concept_graph", "global_concepts"),
}

PROJECT_ENTRY_KEYS = ("projects", "project_dirs", "lectures", "lecture_projects", "suites")
PROJECT_PATH_KEYS = ("project_dir", "project_path", "path", "root")

TEXT_EXCLUSION_NOTICE = (
    "Raw transcript, local filesystem paths, raw queries, raw evidence text, visual labels, "
    "and raw candidate identifiers are excluded from this public-safe report."
)

TIMESTAMP_ONLY_STATUSES = {"timestamp-only", "timestamp-fallback"}
VERIFIED_STATUSES = {"verified", "strict-verified", "human-gold", "vlm-verified"}
RELATION_PREFIX = "GLOBAL_"


@dataclass(frozen=True)
class MappingQualityReportRun:
    output_dir: Path
    metrics_path: Path
    markdown_path: Path
    csv_path: Path | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class _ProjectInput:
    project_dir: Path
    manifest: Mapping[str, Any]
    source_index: int


def generate_mapping_quality_report(
    *,
    project_dirs: Sequence[Path] | None = None,
    manifest_path: Path | None = None,
    output_dir: Path,
    write_csv: bool = False,
) -> MappingQualityReportRun:
    projects = _load_project_inputs(project_dirs=project_dirs, manifest_path=manifest_path)
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    project_reports = [_project_report(project) for project in projects]
    payload = _aggregate_payload(project_reports)

    metrics_path = resolved_output_dir / METRICS_FILENAME
    markdown_path = resolved_output_dir / MARKDOWN_FILENAME
    csv_path = resolved_output_dir / CSV_FILENAME if write_csv else None
    write_json(metrics_path, payload)
    markdown_path.write_text(mapping_quality_markdown(payload), encoding="utf-8")
    if csv_path is not None:
        _write_artifact_csv(csv_path, payload)

    return MappingQualityReportRun(
        output_dir=resolved_output_dir,
        metrics_path=metrics_path,
        markdown_path=markdown_path,
        csv_path=csv_path,
        payload=payload,
    )


def mapping_quality_markdown(payload: Mapping[str, Any]) -> str:
    counts = _mapping(payload.get("counts"))
    artifact_counts = _mapping(payload.get("artifact_counts"))
    evidence = _mapping(payload.get("evidence_quality"))
    concepts = _mapping(payload.get("concept_quality"))
    regression = _mapping(payload.get("regression_checks"))
    lines = [
        "# Mapping Quality Report",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Projects: {counts.get('project_count', 0)}",
        f"- Lectures: {counts.get('lecture_count', 0)}",
        f"- Segments: {artifact_counts.get('segment_count', 0)}",
        f"- Windows: {artifact_counts.get('window_count', 0)}",
        f"- Evidence units: {artifact_counts.get('evidence_unit_count', 0)}",
        f"- Evidence units per minute: {_format_float(evidence.get('evidence_units_per_minute'))}",
        "",
        "## Alignment Quality",
        "",
        f"- Empty transcript segments: {evidence.get('empty_transcript_segment_count', 0)}",
        f"- Visual-state interval coverage: {_format_float(evidence.get('visual_state_interval_coverage_ratio'))}",
        f"- Evidence units with visual state: {_format_float(evidence.get('visual_state_evidence_ratio'))}",
        f"- Evidence units with visual entity: {_format_float(evidence.get('visual_entity_evidence_ratio'))}",
        f"- OCR-only evidence ratio: {_format_float(evidence.get('ocr_only_evidence_ratio'))}",
        f"- VLM object evidence ratio: {_format_float(evidence.get('vlm_object_evidence_ratio'))}",
        "",
        "## Entity Link Status",
        "",
        _markdown_counter_table(_mapping(evidence.get("entity_link_status_distribution"))),
        "",
        "## Concept Clustering",
        "",
        f"- Lecture-local concepts: {concepts.get('lecture_local_concept_count', 0)}",
        f"- Canonical concepts: {concepts.get('canonical_concept_count', 0)}",
        f"- Alias merge ratio: {_format_float(concepts.get('alias_merge_ratio'))}",
        f"- Singleton concept ratio: {_format_float(concepts.get('singleton_concept_ratio'))}",
        f"- Cross-lecture hubs: {concepts.get('cross_lecture_hub_count', 0)}",
        f"- Hub evidence unit diversity mean: {_format_float(concepts.get('hub_evidence_unit_diversity_mean'))}",
        "",
        "## Relation Types",
        "",
        _markdown_counter_table(_mapping(concepts.get("relation_type_counts"))),
        "",
        "## Regression Checks",
        "",
        (
            "- Timestamp-only counted as verified: "
            f"{regression.get('timestamp_only_counted_as_verified_count', 0)}"
        ),
        "",
        "## Artifact Coverage",
        "",
        "| project ref | artifact | status | reason | records |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for project in _list_of_dicts(payload.get("projects")):
        project_ref = str(project.get("project_ref") or "")
        for artifact in _list_of_dicts(project.get("artifacts")):
            lines.append(
                "| {project_ref} | {artifact} | {status} | {reason} | {records} |".format(
                    project_ref=project_ref,
                    artifact=artifact.get("artifact"),
                    status=artifact.get("status"),
                    reason=artifact.get("reason") or "",
                    records=artifact.get("record_count", 0),
                )
            )
    lines.extend(
        [
            "",
            "## Public-Safe Policy",
            "",
            TEXT_EXCLUSION_NOTICE,
            "Only aggregate counts, buckets, reason codes, artifact names, and hashed refs are emitted.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_project_inputs(
    *,
    project_dirs: Sequence[Path] | None,
    manifest_path: Path | None,
) -> list[_ProjectInput]:
    projects: list[_ProjectInput] = []
    manifest_base: Path | None = None
    if manifest_path is not None:
        resolved_manifest = manifest_path.expanduser().resolve()
        manifest_base = resolved_manifest.parent
        manifest = _read_json(resolved_manifest)
        for index, item in enumerate(_manifest_project_items(manifest, manifest_base)):
            project_dir_value = _project_dir_value(item)
            if not isinstance(project_dir_value, str) or not project_dir_value.strip():
                continue
            project_dir = _resolve_path(Path(project_dir_value), manifest_base)
            project_manifest = _project_manifest(project_dir, item)
            projects.append(
                _ProjectInput(project_dir=project_dir, manifest=project_manifest, source_index=index)
            )
    for index, project_dir in enumerate(project_dirs or (), start=len(projects)):
        resolved_dir = project_dir.expanduser().resolve()
        projects.append(
            _ProjectInput(
                project_dir=resolved_dir,
                manifest=_project_manifest(resolved_dir, {}),
                source_index=index,
            )
        )
    if not projects:
        raise ValueError("At least one --project-dir or manifest project entry is required")
    return projects


def _manifest_project_items(
    manifest: Mapping[str, Any],
    manifest_base: Path,
    *,
    seen: frozenset[Path] = frozenset(),
) -> list[Mapping[str, Any]]:
    for key in PROJECT_ENTRY_KEYS:
        value = manifest.get(key)
        if isinstance(value, list):
            items: list[Mapping[str, Any]] = []
            for item in value:
                if isinstance(item, str):
                    items.append({"project_dir": str(_resolve_path(Path(item), manifest_base))})
                elif isinstance(item, Mapping):
                    items.append(_manifest_project_item(item, manifest_base))
            if items:
                return items

    matrix_manifest = _paper_bundle_matrix_manifest(manifest, manifest_base)
    if matrix_manifest is None or matrix_manifest in seen or not matrix_manifest.exists():
        return []
    nested = _read_json(matrix_manifest)
    if not isinstance(nested, Mapping):
        return []
    return _manifest_project_items(nested, matrix_manifest.parent, seen=seen | {matrix_manifest})


def _project_dir_value(item: Mapping[str, Any]) -> Any:
    for key in PROJECT_PATH_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _manifest_project_item(item: Mapping[str, Any], manifest_base: Path) -> dict[str, Any]:
    normalized = dict(item)
    for key in PROJECT_PATH_KEYS:
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = str(_resolve_path(Path(value), manifest_base))
    return normalized


def _paper_bundle_matrix_manifest(manifest: Mapping[str, Any], manifest_base: Path) -> Path | None:
    paper_bundle = manifest.get("paper_bundle")
    if not isinstance(paper_bundle, Mapping):
        return None
    matrix_manifest = paper_bundle.get("matrix_manifest")
    if not isinstance(matrix_manifest, str) or not matrix_manifest.strip():
        return None
    return _resolve_path(Path(matrix_manifest), manifest_base)


def _project_manifest(project_dir: Path, inline: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest_path_value = inline.get("manifest")
    if isinstance(manifest_path_value, str) and manifest_path_value.strip():
        manifest_path = _resolve_path(Path(manifest_path_value), project_dir)
        if manifest_path.exists():
            return _merge_manifest(_read_json(manifest_path), inline)
    default_manifest = project_dir / "manifests" / "project_manifest.json"
    if default_manifest.exists():
        return _merge_manifest(_read_json(default_manifest), inline)
    return dict(inline)


def _merge_manifest(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "artifacts" and isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            artifacts = dict(base[key])
            artifacts.update(value)
            merged[key] = artifacts
        elif key not in {*PROJECT_PATH_KEYS, "manifest"}:
            merged[key] = value
    return merged


def _project_report(project: _ProjectInput) -> dict[str, Any]:
    artifact_rows: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    for artifact_name in ARTIFACT_DEFAULTS:
        status = _load_artifact(project, artifact_name)
        artifact_rows.append(
            {
                "artifact": artifact_name,
                "status": status["status"],
                "reason": status.get("reason"),
                "record_count": status.get("record_count", 0),
            }
        )
        loaded[artifact_name] = status.get("payload")

    segments = _records(loaded.get("lecture_segments_aligned"))
    windows = _records(loaded.get("lecture_windows"))
    visual_states = _records(loaded.get("visual_states"))
    visual_entities = _records(loaded.get("visual_entities"))
    entity_links = _records(loaded.get("entity_links"))
    evidence_units = _records(loaded.get("evidence_units"))
    concept_graph = _records(loaded.get("concept_graph"))
    global_graph = loaded.get("global_concept_graph")
    if not isinstance(global_graph, Mapping):
        global_graph = {}

    lecture_ids = _lecture_ids(project.manifest, segments, windows, evidence_units, concept_graph, global_graph)
    duration_seconds = _duration_seconds(segments, windows, evidence_units, visual_states)

    return {
        "project_ref": _hash_ref("project", _project_ref_seed(project)),
        "lecture_refs": [_hash_ref("lecture", lecture_id) for lecture_id in sorted(lecture_ids)],
        "artifacts": artifact_rows,
        "metrics": {
            "lecture_count": len(lecture_ids) or 1,
            "duration_seconds": round(duration_seconds, 3),
            "segments": _segment_metrics(segments),
            "windows": _timed_record_metrics(windows),
            "visual_states": _visual_state_metrics(visual_states, duration_seconds),
            "visual_entities": _visual_entity_metrics(visual_entities),
            "entity_links": _entity_link_metrics(entity_links),
            "evidence_units": _evidence_unit_metrics(evidence_units),
            "concept_graph": _concept_graph_metrics(concept_graph),
            "global_concept_graph": _global_concept_graph_metrics(global_graph),
        },
    }


def _load_artifact(project: _ProjectInput, artifact_name: str) -> dict[str, Any]:
    path, configured = _artifact_path(project, artifact_name)
    if path is None:
        return {"status": "missing", "reason": "not_configured", "record_count": 0}
    if not path.exists():
        return {
            "status": "missing",
            "reason": "missing_artifact" if configured else "not_configured",
            "record_count": 0,
        }
    try:
        payload = _read_jsonl(path) if path.suffix == ".jsonl" else _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {"status": "missing", "reason": "unreadable_artifact", "record_count": 0}
    return {
        "status": "ok",
        "reason": None,
        "record_count": len(payload) if isinstance(payload, list) else 1,
        "payload": payload,
    }


def _artifact_path(project: _ProjectInput, artifact_name: str) -> tuple[Path | None, bool]:
    artifacts = project.manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    for alias in ARTIFACT_ALIASES[artifact_name]:
        value = artifacts.get(alias) or project.manifest.get(alias)
        if isinstance(value, str) and value.strip():
            return _resolve_path(Path(value), project.project_dir), True
    default_path = project.project_dir / ARTIFACT_DEFAULTS[artifact_name]
    if default_path.exists():
        return default_path, False
    return default_path, False


def _aggregate_payload(project_reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "project_count": len(project_reports),
        "lecture_count": sum(_metric(report, "lecture_count") for report in project_reports),
    }
    artifact_counts = Counter()
    artifact_statuses = Counter()
    missing_reasons = Counter()
    segment_duration_buckets = Counter()
    timestamp_coverage_buckets = Counter()
    visual_state_coverage_buckets = Counter()
    entity_link_status_distribution = Counter()
    candidate_signal_counts = Counter()
    verified_source_counts = Counter()
    relation_type_counts = Counter()
    relation_supporting_evidence_count = Counter()
    over_merge_reason_codes = Counter()
    hub_lecture_coverage_buckets = Counter()

    duration_seconds = 0.0
    empty_transcripts = 0
    visual_state_covered_seconds = 0.0
    evu_with_state = 0
    evu_with_entity = 0
    evu_ocr_only = 0
    evu_vlm = 0
    timestamp_only_as_verified = 0
    verified_alignment_count = 0
    local_concepts = 0
    canonical_concepts = 0
    singleton_concepts = 0
    cross_lecture_hubs = 0
    hub_evidence_diversity_values: list[int] = []

    for report in project_reports:
        metrics = _mapping(report.get("metrics"))
        for artifact in _list_of_dicts(report.get("artifacts")):
            name = str(artifact.get("artifact") or "")
            status = str(artifact.get("status") or "")
            artifact_statuses[f"{name}:{status}"] += 1
            if status == "ok":
                artifact_counts[f"{name}_records"] += int(artifact.get("record_count") or 0)
            else:
                missing_reasons[str(artifact.get("reason") or "unknown")] += 1

        segment_metrics = _mapping(metrics.get("segments"))
        window_metrics = _mapping(metrics.get("windows"))
        visual_metrics = _mapping(metrics.get("visual_states"))
        evidence_metrics = _mapping(metrics.get("evidence_units"))
        concept_metrics = _mapping(metrics.get("concept_graph"))
        global_metrics = _mapping(metrics.get("global_concept_graph"))

        artifact_counts["segment_count"] += int(segment_metrics.get("count") or 0)
        artifact_counts["window_count"] += int(window_metrics.get("count") or 0)
        artifact_counts["evidence_unit_count"] += int(evidence_metrics.get("count") or 0)
        duration_seconds += float(metrics.get("duration_seconds") or 0.0)
        empty_transcripts += int(segment_metrics.get("empty_transcript_count") or 0)
        visual_state_covered_seconds += float(visual_metrics.get("covered_seconds") or 0.0)

        segment_duration_buckets.update(_mapping(segment_metrics.get("duration_buckets")))
        timestamp_coverage_buckets.update(_mapping(segment_metrics.get("timestamp_duration_buckets")))
        timestamp_coverage_buckets.update(_mapping(window_metrics.get("timestamp_duration_buckets")))
        timestamp_coverage_buckets.update(_mapping(evidence_metrics.get("timestamp_duration_buckets")))
        visual_state_coverage_buckets.update(_mapping(visual_metrics.get("coverage_buckets")))

        evu_with_state += int(evidence_metrics.get("with_visual_state_count") or 0)
        evu_with_entity += int(evidence_metrics.get("with_visual_entity_count") or 0)
        evu_ocr_only += int(evidence_metrics.get("ocr_only_count") or 0)
        evu_vlm += int(evidence_metrics.get("vlm_object_count") or 0)
        entity_link_status_distribution.update(
            _mapping(evidence_metrics.get("entity_link_status_distribution"))
        )
        entity_link_status_distribution.update(
            _mapping(_mapping(metrics.get("entity_links")).get("status_distribution"))
        )
        candidate_signal_counts.update(_mapping(evidence_metrics.get("candidate_signal_counts")))
        verified_source_counts.update(_mapping(evidence_metrics.get("verified_source_counts")))
        timestamp_only_as_verified += int(
            evidence_metrics.get("timestamp_only_counted_as_verified_count") or 0
        )
        verified_alignment_count += int(evidence_metrics.get("verified_alignment_count") or 0)

        local_concepts += int(concept_metrics.get("local_concept_count") or 0)
        canonical_concepts += int(global_metrics.get("canonical_concept_count") or 0)
        singleton_concepts += int(global_metrics.get("singleton_concept_count") or 0)
        cross_lecture_hubs += int(global_metrics.get("cross_lecture_hub_count") or 0)
        hub_lecture_coverage_buckets.update(_mapping(global_metrics.get("hub_lecture_coverage_buckets")))
        hub_evidence_diversity_values.extend(
            int(value) for value in _list(global_metrics.get("hub_evidence_unit_diversity_values"))
        )
        relation_type_counts.update(_mapping(concept_metrics.get("relation_type_counts")))
        relation_type_counts.update(_mapping(global_metrics.get("relation_type_counts")))
        relation_supporting_evidence_count.update(
            _mapping(concept_metrics.get("relation_supporting_evidence_count"))
        )
        relation_supporting_evidence_count.update(
            _mapping(global_metrics.get("relation_supporting_evidence_count"))
        )
        over_merge_reason_codes.update(_mapping(global_metrics.get("over_merge_reason_codes")))

    evidence_unit_count = int(artifact_counts.get("evidence_unit_count") or 0)
    duration_minutes = duration_seconds / 60.0 if duration_seconds > 0 else 0.0
    resolved_canonical = canonical_concepts or local_concepts
    return {
        "schema_version": MAPPING_QUALITY_SCHEMA_VERSION,
        "counts": totals,
        "artifact_counts": dict(sorted(artifact_counts.items())),
        "artifact_statuses": dict(sorted(artifact_statuses.items())),
        "missing_artifact_reasons": dict(sorted(missing_reasons.items())),
        "duration_coverage": {
            "total_duration_seconds": round(duration_seconds, 3),
            "segment_duration_buckets": dict(sorted(segment_duration_buckets.items())),
            "timestamp_duration_coverage_buckets": dict(sorted(timestamp_coverage_buckets.items())),
        },
        "evidence_quality": {
            "evidence_units_per_minute": _safe_ratio(evidence_unit_count, duration_minutes),
            "empty_transcript_segment_count": empty_transcripts,
            "visual_state_interval_coverage_ratio": _safe_ratio(
                visual_state_covered_seconds,
                duration_seconds,
            ),
            "visual_state_interval_coverage_buckets": dict(sorted(visual_state_coverage_buckets.items())),
            "visual_state_evidence_ratio": _safe_ratio(evu_with_state, evidence_unit_count),
            "visual_entity_evidence_ratio": _safe_ratio(evu_with_entity, evidence_unit_count),
            "ocr_only_evidence_ratio": _safe_ratio(evu_ocr_only, evidence_unit_count),
            "vlm_object_evidence_ratio": _safe_ratio(evu_vlm, evidence_unit_count),
            "entity_link_status_distribution": dict(sorted(entity_link_status_distribution.items())),
            "candidate_signal_counts": dict(sorted(candidate_signal_counts.items())),
            "verified_source_counts": dict(sorted(verified_source_counts.items())),
            "verified_alignment_count": verified_alignment_count,
        },
        "concept_quality": {
            "lecture_local_concept_count": local_concepts,
            "canonical_concept_count": resolved_canonical,
            "alias_merge_ratio": (
                round((local_concepts - resolved_canonical) / local_concepts, 6)
                if local_concepts > 0
                else 0.0
            ),
            "singleton_concept_ratio": _safe_ratio(singleton_concepts, resolved_canonical),
            "cross_lecture_hub_count": cross_lecture_hubs,
            "hub_lecture_coverage_buckets": dict(sorted(hub_lecture_coverage_buckets.items())),
            "hub_evidence_unit_diversity": _number_summary(hub_evidence_diversity_values),
            "hub_evidence_unit_diversity_mean": _mean(hub_evidence_diversity_values),
            "relation_type_counts": dict(sorted(relation_type_counts.items())),
            "relation_supporting_evidence_count": dict(sorted(relation_supporting_evidence_count.items())),
            "over_merge_reason_codes": dict(sorted(over_merge_reason_codes.items())),
        },
        "regression_checks": {
            "timestamp_only_counted_as_verified_count": timestamp_only_as_verified,
            "timestamp_only_counted_as_verified_passed": timestamp_only_as_verified == 0,
        },
        "projects": [
            {
                "project_ref": report["project_ref"],
                "lecture_refs": report["lecture_refs"],
                "artifacts": report["artifacts"],
            }
            for report in project_reports
        ],
        "public_safety": {
            "level": "aggregate_public_safe",
            "omits": [
                "raw_transcript",
                "local_filesystem_paths",
                "raw_queries",
                "raw_evidence_text",
                "visual_labels",
                "raw_candidate_ids",
            ],
            "hash_algorithm": "sha256_12",
        },
    }


def _segment_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _timed_record_metrics(rows)
    metrics["empty_transcript_count"] = sum(
        1 for row in rows if not str(row.get("transcript_text") or row.get("text") or "").strip()
    )
    metrics["duration_buckets"] = dict(sorted(_duration_buckets(rows).items()))
    return metrics


def _timed_record_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = Counter(_timestamp_duration_bucket(row) for row in rows)
    return {
        "count": len(rows),
        "timestamp_duration_buckets": dict(sorted(buckets.items())),
    }


def _visual_state_metrics(rows: Sequence[Mapping[str, Any]], duration_seconds: float) -> dict[str, Any]:
    intervals = []
    for row in rows:
        start, end = _record_interval(row)
        if start is not None and end is not None and end >= start:
            intervals.append((start, end))
    covered_seconds = _merged_interval_duration(intervals)
    coverage_ratio = _safe_ratio(covered_seconds, duration_seconds)
    return {
        "count": len(rows),
        "covered_seconds": round(covered_seconds, 3),
        "coverage_ratio": coverage_ratio,
        "coverage_buckets": {_ratio_bucket(coverage_ratio): 1} if rows else {"missing": 1},
    }


def _visual_entity_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "source_distribution": dict(sorted(Counter(_safe_status(row.get("source")) for row in rows).items())),
    }


def _entity_link_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter()
    for row in rows:
        statuses[_link_status(row)] += 1
    return {
        "count": len(rows),
        "status_distribution": dict(sorted(statuses.items())),
    }


def _evidence_unit_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timestamp_buckets = Counter()
    link_statuses = Counter()
    candidate_signal_counts = Counter()
    verified_source_counts = Counter()
    with_state = 0
    with_entity = 0
    ocr_only = 0
    vlm_object = 0
    timestamp_verified_regression = 0
    verified_alignment_count = 0
    for row in rows:
        timestamp_buckets[_timestamp_duration_bucket(row)] += 1
        source_quality = _mapping(row.get("source_quality"))
        if _list(row.get("visual_state_ids")) or source_quality.get("has_visual_state"):
            with_state += 1
        if _list(row.get("visual_entity_ids")) or source_quality.get("has_visual_entity"):
            with_entity += 1
        if source_quality.get("uses_ocr_only"):
            ocr_only += 1
        if source_quality.get("has_vlm_entity") or source_quality.get("has_visual_description"):
            vlm_object += 1
        for status in _candidate_statuses(row):
            link_statuses[status] += 1
        candidate_signal_counts.update(_int_counter(source_quality.get("candidate_link_signal_counts")))
        verified_source_counts.update(_int_counter(source_quality.get("verified_link_source_counts")))
        verified_alignment = _mapping(source_quality.get("verified_object_alignment"))
        candidate_support = _mapping(source_quality.get("candidate_visual_support"))
        if verified_alignment.get("timestamp_fallback_counted_as_verified"):
            timestamp_verified_regression += 1
        if bool(verified_alignment.get("has_verified_object_alignment")):
            verified_alignment_count += 1
        timestamp_fallback_count = int(
            source_quality.get("timestamp_fallback_link_count")
            or candidate_support.get("timestamp_fallback_link_count")
            or 0
        )
        if timestamp_fallback_count:
            link_statuses["timestamp-only"] += timestamp_fallback_count
    return {
        "count": len(rows),
        "timestamp_duration_buckets": dict(sorted(timestamp_buckets.items())),
        "with_visual_state_count": with_state,
        "with_visual_entity_count": with_entity,
        "ocr_only_count": ocr_only,
        "vlm_object_count": vlm_object,
        "entity_link_status_distribution": dict(sorted(link_statuses.items())),
        "candidate_signal_counts": dict(sorted(candidate_signal_counts.items())),
        "verified_source_counts": dict(sorted(verified_source_counts.items())),
        "timestamp_only_counted_as_verified_count": timestamp_verified_regression,
        "verified_alignment_count": verified_alignment_count,
    }


def _concept_graph_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    local_concepts = 0
    aliases = 0
    relation_types = Counter()
    relation_evidence_counts = Counter()
    lecture_ids: set[str] = set()
    for row in rows:
        record_type = str(row.get("record_type") or "")
        if row.get("lecture_id"):
            lecture_ids.add(str(row["lecture_id"]))
        if record_type == "concept_node":
            local_concepts += 1
            aliases += len(_list(row.get("aliases")))
        elif record_type == "relation_edge":
            relation_type = _safe_status(row.get("relation_type"))
            relation_types[relation_type] += 1
            relation_evidence_counts[relation_type] += len(
                _list(row.get("evidence_unit_ids"))
                or _list(row.get("source_evidence_unit_ids"))
                or _list(row.get("evidence_sources"))
            )
    return {
        "local_concept_count": local_concepts,
        "alias_count": aliases,
        "lecture_count": len(lecture_ids),
        "relation_type_counts": dict(sorted(relation_types.items())),
        "relation_supporting_evidence_count": dict(sorted(relation_evidence_counts.items())),
    }


def _global_concept_graph_metrics(document: Mapping[str, Any]) -> dict[str, Any]:
    counts = _mapping(document.get("counts"))
    nodes = _list_of_dicts(document.get("nodes"))
    relationships = _list_of_dicts(document.get("relationships"))
    conflicts = _list_of_dicts(document.get("conflicts"))
    canonical_count = int(counts.get("global_concepts") or 0)
    singleton_count = int(counts.get("singleton_global_concepts") or 0)
    hubs = []
    evidence_diversities: list[int] = []
    coverage_buckets = Counter()
    for node in nodes:
        labels = set(str(label) for label in _list(node.get("labels")))
        if "GlobalConcept" not in labels:
            continue
        properties = _mapping(node.get("properties"))
        lecture_ids = {str(value) for value in _list(properties.get("source_lecture_ids"))}
        if len(lecture_ids) < 2:
            continue
        hubs.append(node)
        coverage_buckets[_lecture_coverage_bucket(len(lecture_ids))] += 1
        evidence_refs = _list_of_dicts(properties.get("source_evidence_refs"))
        evidence_keys = {
            _hash_ref(
                "evidence",
                f"{ref.get('lecture_id')}:{ref.get('evidence_unit_id')}:{ref.get('source_id')}",
            )
            for ref in evidence_refs
        }
        evidence_diversities.append(len(evidence_keys))

    relation_types = Counter()
    relation_evidence_counts = Counter()
    for relationship in relationships:
        relation_type = str(relationship.get("type") or "")
        if relation_type.startswith(RELATION_PREFIX):
            properties = _mapping(relationship.get("properties"))
            public_type = _safe_status(properties.get("relation_type") or relation_type)
            relation_types[public_type] += 1
            relation_evidence_counts[public_type] += len(
                _list_of_dicts(properties.get("source_evidence_refs"))
            )

    reason_codes = Counter()
    for conflict in conflicts:
        reason_codes.update(str(reason) for reason in _list(conflict.get("reasons")))

    return {
        "canonical_concept_count": canonical_count,
        "singleton_concept_count": singleton_count,
        "cross_lecture_hub_count": len(hubs),
        "hub_lecture_coverage_buckets": dict(sorted(coverage_buckets.items())),
        "hub_evidence_unit_diversity_values": evidence_diversities,
        "relation_type_counts": dict(sorted(relation_types.items())),
        "relation_supporting_evidence_count": dict(sorted(relation_evidence_counts.items())),
        "over_merge_reason_codes": dict(sorted(reason_codes.items())),
    }


def _candidate_statuses(row: Mapping[str, Any]) -> list[str]:
    statuses_by_link_id: dict[str, str] = {}
    explicit = row.get("candidate_entity_link_statuses")
    if isinstance(explicit, Mapping):
        for link_id, status in explicit.items():
            if link_id:
                statuses_by_link_id[str(link_id)] = _merge_link_status(
                    statuses_by_link_id.get(str(link_id)),
                    _normalized_link_status(status),
                )
    for link_id in _list(row.get("verified_entity_link_ids")):
        if link_id:
            statuses_by_link_id[str(link_id)] = _merge_link_status(
                statuses_by_link_id.get(str(link_id)),
                "verified",
            )
    for link_id in _list(row.get("candidate_entity_link_ids")):
        if link_id:
            statuses_by_link_id[str(link_id)] = _merge_link_status(
                statuses_by_link_id.get(str(link_id)),
                "candidate",
            )
    if statuses_by_link_id:
        return [statuses_by_link_id[link_id] for link_id in sorted(statuses_by_link_id)]

    alignment_status = _normalized_link_status(row.get("alignment_status"))
    if alignment_status in {"verified", "candidate", "timestamp-only"}:
        return [alignment_status]
    return []


def _merge_link_status(current: str | None, incoming: str) -> str:
    priority = {"candidate": 0, "transcript-only": 0, "timestamp-only": 1, "verified": 2}
    if current is None:
        return incoming
    return incoming if priority.get(incoming, 0) > priority.get(current, 0) else current


def _link_status(row: Mapping[str, Any]) -> str:
    for key in ("status", "alignment_status", "link_status"):
        if row.get(key):
            return _normalized_link_status(row.get(key))
    if row.get("verified") is True:
        return "verified"
    evidence = [str(value).casefold() for value in _list(row.get("evidence"))]
    if any("timestamp" in value or value == "time_overlap" for value in evidence):
        return "timestamp-only" if len(evidence) == 1 else "candidate"
    return "candidate"


def _normalized_link_status(value: Any) -> str:
    status = str(value or "candidate").strip().casefold().replace("_", "-")
    if status in TIMESTAMP_ONLY_STATUSES:
        return "timestamp-only"
    if status in VERIFIED_STATUSES:
        return "verified"
    if status in {"candidate", "transcript-only"}:
        return status
    return "candidate"


def _duration_seconds(*record_groups: Sequence[Mapping[str, Any]]) -> float:
    max_end = 0.0
    for rows in record_groups:
        for row in rows:
            _, end = _record_interval(row)
            if end is not None:
                max_end = max(max_end, end)
    return max_end


def _duration_buckets(rows: Sequence[Mapping[str, Any]]) -> Counter:
    buckets = Counter()
    for row in rows:
        start, end = _record_interval(row)
        if start is None or end is None or end < start:
            buckets["unknown"] += 1
            continue
        duration = end - start
        if duration <= 0:
            buckets["zero"] += 1
        elif duration < 5:
            buckets["0-5s"] += 1
        elif duration < 15:
            buckets["5-15s"] += 1
        elif duration < 60:
            buckets["15-60s"] += 1
        else:
            buckets["60s+"] += 1
    return buckets


def _timestamp_duration_bucket(row: Mapping[str, Any]) -> str:
    start, end = _record_interval(row)
    center = _number(row.get("timestamp_center") or row.get("timestamp") or row.get("time"))
    if start is not None and end is not None and end >= start:
        return "complete_interval"
    if center is not None:
        return "timestamp_only"
    return "missing_or_invalid"


def _record_interval(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = _number(
        row.get("start_time")
        if row.get("start_time") is not None
        else row.get("valid_start_time", row.get("target_start_time"))
    )
    end = _number(
        row.get("end_time")
        if row.get("end_time") is not None
        else row.get("valid_end_time", row.get("target_end_time"))
    )
    return start, end


def _merged_interval_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return round(sum(end - start for start, end in merged), 6)


def _lecture_ids(
    manifest: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    evidence_units: Sequence[Mapping[str, Any]],
    concept_graph: Sequence[Mapping[str, Any]],
    global_graph: Mapping[str, Any],
) -> set[str]:
    ids = set()
    for key in ("lecture_id", "video_id", "project_id"):
        if manifest.get(key):
            ids.add(str(manifest[key]))
            break
    for rows in (segments, windows, evidence_units, concept_graph):
        for row in rows:
            for key in ("lecture_id", "video_id", "project_id"):
                if row.get(key):
                    ids.add(str(row[key]))
                    break
    for node in _list_of_dicts(global_graph.get("nodes")):
        properties = _mapping(node.get("properties"))
        ids.update(str(value) for value in _list(properties.get("source_lecture_ids")))
    return ids


def _project_ref_seed(project: _ProjectInput) -> str:
    for key in ("project_id", "lecture_id", "video_id", "id"):
        value = project.manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"input:{project.source_index}:{project.project_dir.name}"


def _write_artifact_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["project_ref", "artifact", "status", "reason", "record_count"],
        )
        writer.writeheader()
        for project in _list_of_dicts(payload.get("projects")):
            for artifact in _list_of_dicts(project.get("artifacts")):
                writer.writerow(
                    {
                        "project_ref": project.get("project_ref"),
                        "artifact": artifact.get("artifact"),
                        "status": artifact.get("status"),
                        "reason": artifact.get("reason") or "",
                        "record_count": artifact.get("record_count", 0),
                    }
                )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                raise ValueError(f"Expected JSON object row in {path.name}")
    return rows


def _resolve_path(path: Path, base_dir: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base_dir / expanded).resolve()


def _records(value: Any) -> list[Mapping[str, Any]]:
    return _list_of_dicts(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]


def _int_counter(value: Any) -> Counter:
    return Counter({str(key): int(count or 0) for key, count in _mapping(value).items()})


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(report: Mapping[str, Any], key: str) -> int:
    return int(_mapping(report.get("metrics")).get(key) or 0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _number_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": _mean(values)}


def _ratio_bucket(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 0.25:
        return "0-25%"
    if value < 0.5:
        return "25-50%"
    if value < 0.75:
        return "50-75%"
    if value < 1.0:
        return "75-100%"
    return "100%+"


def _lecture_coverage_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return "5+"


def _safe_status(value: Any) -> str:
    text = str(value or "unknown").strip().casefold().replace("_", "-")
    return text or "unknown"


def _hash_ref(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}_{digest}"


def _format_float(value: Any) -> str:
    number = _number(value)
    return "0.000" if number is None else f"{number:.3f}"


def _markdown_counter_table(counter: Mapping[str, Any]) -> str:
    if not counter:
        return "| value | count |\n| --- | ---: |\n| none | 0 |"
    lines = ["| value | count |", "| --- | ---: |"]
    for key, value in sorted(counter.items()):
        lines.append(f"| {key} | {int(value or 0)} |")
    return "\n".join(lines)
