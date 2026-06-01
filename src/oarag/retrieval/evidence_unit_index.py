from __future__ import annotations

from pathlib import Path
from typing import Any

from oarag.integrations.meili import MeiliClient
from oarag.retrieval.project_index import (
    evidence_unit_artifact_path,
    iter_jsonl_documents,
)


EVIDENCE_UNIT_RESULT_FIELDS = [
    "evidence_unit_id",
    "project_id",
    "video_id",
    "target_segment_id",
    "source_segment_ids",
    "start_time",
    "end_time",
    "visual_state_ids",
    "visual_entity_ids",
    "verified_entity_link_ids",
    "candidate_entity_link_ids",
    "candidate_entity_link_statuses",
    "alignment_status",
    "source_quality",
]


def query_project_evidence_units(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    limit: int = 5,
    evidence_units: Path | None = None,
) -> dict[str, Any]:
    result_limit = _positive_int(limit, field_name="limit")
    resolved_project_dir = project_dir.expanduser().resolve()
    project_id = _project_id_from_evidence_units(
        project_dir=resolved_project_dir,
        evidence_units=evidence_units,
    )
    response = client.search(
        index_uid,
        query,
        limit=result_limit,
        filter=f'project_id = "{_escape_meili_filter_string(project_id)}"',
    )
    hits = response.get("hits", [])
    candidates = [
        _evidence_unit_candidate(rank=rank, hit=hit)
        for rank, hit in enumerate(hits, start=1)
    ]
    return {
        "query": query,
        "index": index_uid,
        "project_id": project_id,
        "processing_time_ms": response.get("processingTimeMs"),
        "limit": result_limit,
        "candidates": candidates,
    }


def _project_id_from_evidence_units(
    *,
    project_dir: Path,
    evidence_units: Path | None,
) -> str:
    try:
        artifact_path = evidence_unit_artifact_path(project_dir, evidence_units=evidence_units)
    except FileNotFoundError:
        if evidence_units is not None:
            raise
        return project_dir.name
    for document in iter_jsonl_documents(artifact_path):
        project_id = str(document.get("project_id") or "").strip()
        if project_id:
            return project_id
    return project_dir.name


def _evidence_unit_candidate(*, rank: int, hit: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "evidence_unit_id": str(hit.get("evidence_unit_id") or ""),
        "project_id": str(hit.get("project_id") or ""),
        "video_id": str(hit.get("video_id") or ""),
        "target_segment_id": str(hit.get("target_segment_id") or ""),
        "source_segment_ids": _string_list(hit.get("source_segment_ids")),
        "start_time": _optional_float(hit.get("start_time")),
        "end_time": _optional_float(hit.get("end_time")),
        "visual_state_ids": _string_list(hit.get("visual_state_ids")),
        "visual_entity_ids": _string_list(hit.get("visual_entity_ids")),
        "verified_entity_link_ids": _string_list(hit.get("verified_entity_link_ids")),
        "candidate_entity_link_ids": _string_list(hit.get("candidate_entity_link_ids")),
        "candidate_entity_link_statuses": _mapping(hit.get("candidate_entity_link_statuses")),
        "alignment_status": str(hit.get("alignment_status") or ""),
        "source_quality": _mapping(hit.get("source_quality")),
    }
    candidate["rank"] = rank
    candidate["score"] = _optional_float(hit.get("_rankingScore"))
    candidate["evidence_text"] = hit.get("evidence_text")
    candidate["semantic_text"] = hit.get("semantic_text")
    candidate["transcript_window_text"] = hit.get("transcript_window_text")
    return candidate


def _positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _escape_meili_filter_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
