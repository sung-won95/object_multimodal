from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from oarag.core.io import write_json, write_jsonl
from oarag.core.schemas import slugify
from oarag.retrieval.evidence import (
    resolve_window_config,
    segment_sort_key,
    select_window_segments,
)
from oarag.retrieval.project_index import iter_jsonl_documents, segment_artifact_path
from oarag.vision.vlm_evidence_validator import is_paper_quality_vlm_entity


EVIDENCE_UNITS_ARTIFACT_RELATIVE_PATH = Path("segments") / "evidence_units.jsonl"
DEFAULT_STATE_PADDING_SECONDS = 15.0


def build_project_evidence_units(
    *,
    project_dir: Path,
    output_path: Path | None = None,
    segments: Path | None = None,
    frames_manifest: Path | None = None,
    visual_entities: Path | None = None,
    entity_links: Path | None = None,
    manifest_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    segments_path = segment_artifact_path(resolved_project_dir, segments=segments)
    frames_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=frames_manifest,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
        artifact_name="frames manifest",
    )
    visual_entities_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=visual_entities,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
        artifact_name="visual entities",
    )
    entity_links_path = _optional_existing_project_path(
        project_dir=resolved_project_dir,
        path=entity_links,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
        artifact_name="entity links",
    )
    resolved_output_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / EVIDENCE_UNITS_ARTIFACT_RELATIVE_PATH,
    )
    resolved_manifest_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )

    segment_rows = list(iter_jsonl_documents(segments_path))
    frame_rows = list(iter_jsonl_documents(frames_path)) if frames_path else []
    visual_entity_rows = list(iter_jsonl_documents(visual_entities_path)) if visual_entities_path else []
    entity_link_rows = list(iter_jsonl_documents(entity_links_path)) if entity_links_path else []
    documents = build_evidence_unit_documents(
        segment_rows,
        frames=frame_rows,
        visual_entities=visual_entity_rows,
        entity_links=entity_link_rows,
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
        state_padding_seconds=state_padding_seconds,
    )
    write_jsonl(resolved_output_path, documents)

    counts = {
        "segments_total": len(segment_rows),
        "frames_total": len(frame_rows),
        "visual_entities_total": len(visual_entity_rows),
        "entity_links_total": len(entity_link_rows),
        "evidence_units_total": len(documents),
        "units_with_visual_state": sum(1 for document in documents if document["source_quality"]["has_visual_state"]),
        "units_with_visual_entity": sum(1 for document in documents if document["source_quality"]["has_visual_entity"]),
        "units_with_vlm_entity": sum(1 for document in documents if document["source_quality"]["has_vlm_entity"]),
        "units_with_verified_link": sum(1 for document in documents if document["source_quality"]["has_verified_link"]),
        "units_with_detected_text": sum(1 for document in documents if document["source_quality"]["has_detected_text"]),
        "units_with_visual_description": sum(
            1 for document in documents if document["source_quality"]["has_visual_description"]
        ),
        "timestamp_fallback_links": sum(
            document["source_quality"]["timestamp_fallback_link_count"] for document in documents
        ),
        "candidate_links": sum(document["source_quality"]["candidate_link_count"] for document in documents),
        "verified_links": sum(document["source_quality"]["verified_link_count"] for document in documents),
    }
    summary = {
        "project_dir": str(resolved_project_dir),
        "paths": {
            "segments": str(segments_path),
            "frames_manifest": str(frames_path) if frames_path else None,
            "visual_entities": str(visual_entities_path) if visual_entities_path else None,
            "entity_links": str(entity_links_path) if entity_links_path else None,
            "evidence_units": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "window_config": _window_config(
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        ),
        "visual_state_config": {
            "mode": "sampled_frame_midpoints",
            "state_padding_seconds": _nonnegative_float(
                state_padding_seconds,
                default=DEFAULT_STATE_PADDING_SECONDS,
            ),
        },
        "counts": counts,
        "alignment_status_counts": _status_counts(documents),
    }
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        evidence_units_path=resolved_output_path,
        summary=summary,
    )
    return summary


def build_evidence_unit_documents(
    segments: list[dict[str, Any]],
    *,
    frames: list[dict[str, Any]] | None = None,
    visual_entities: list[dict[str, Any]] | None = None,
    entity_links: list[dict[str, Any]] | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
) -> list[dict[str, Any]]:
    sorted_segments = [
        segment for _, segment in sorted(enumerate(segments), key=lambda item: segment_sort_key(item[1], item[0]))
    ]
    visual_states = build_visual_states(
        frames or [],
        default_project_id=_first_text(sorted_segments, "project_id"),
        default_video_id=_first_text(sorted_segments, "video_id"),
        state_padding_seconds=state_padding_seconds,
    )
    states_by_id = {str(state["visual_state_id"]): state for state in visual_states}
    entities_by_frame_id = _group_by_text(visual_entities or [], "frame_id")
    entities_by_id = {str(entity.get("entity_id")): entity for entity in visual_entities or []}
    links_by_segment_id = _group_by_text(entity_links or [], "segment_id")
    window_config = _window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )

    documents: list[dict[str, Any]] = []
    for target in sorted_segments:
        target_segment_id = str(target.get("segment_id") or "")
        if not target_segment_id:
            continue
        window_segments = select_window_segments(
            sorted_segments,
            target_segment_id=target_segment_id,
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        )
        documents.append(
            _evidence_unit_document(
                target=target,
                window_segments=window_segments,
                states_by_id=states_by_id,
                visual_states=visual_states,
                entities_by_frame_id=entities_by_frame_id,
                entities_by_id=entities_by_id,
                links_by_segment_id=links_by_segment_id,
                window_config=window_config,
            )
        )
    return documents


def build_visual_states(
    frames: list[dict[str, Any]],
    *,
    default_project_id: str | None = None,
    default_video_id: str | None = None,
    state_padding_seconds: float = DEFAULT_STATE_PADDING_SECONDS,
) -> list[dict[str, Any]]:
    padding = _nonnegative_float(state_padding_seconds, default=DEFAULT_STATE_PADDING_SECONDS)
    grouped_frames: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        frame_id = _text(frame.get("frame_id"))
        timestamp = _optional_float(frame.get("timestamp"))
        if not frame_id or timestamp is None:
            continue
        video_id = _text(frame.get("video_id")) or default_video_id or "__default__"
        grouped_frames.setdefault(video_id, []).append(frame)

    states: list[dict[str, Any]] = []
    for video_id, video_frames in sorted(grouped_frames.items()):
        sorted_frames = sorted(
            video_frames,
            key=lambda frame: (
                _optional_float(frame.get("timestamp")) or 0.0,
                _text(frame.get("frame_id")),
            ),
        )
        timestamps = [_optional_float(frame.get("timestamp")) or 0.0 for frame in sorted_frames]
        for index, frame in enumerate(sorted_frames):
            frame_id = _text(frame.get("frame_id"))
            timestamp = timestamps[index]
            start_time = (
                (timestamps[index - 1] + timestamp) / 2.0
                if index > 0
                else max(0.0, timestamp - padding)
            )
            end_time = (
                (timestamp + timestamps[index + 1]) / 2.0
                if index < len(timestamps) - 1
                else timestamp + padding
            )
            state_summary = _compact_text(
                " ".join(
                    _text_values(
                        [
                            frame.get("state_summary"),
                            frame.get("visual_description"),
                            frame.get("detected_text"),
                        ]
                    )
                )
            )
            states.append(
                {
                    "visual_state_id": f"vstate_{slugify(video_id)}_{slugify(frame_id)}",
                    "project_id": _text(frame.get("project_id")) or default_project_id,
                    "video_id": video_id if video_id != "__default__" else default_video_id,
                    "representative_frame_id": frame_id,
                    "frame_ids": [frame_id],
                    "valid_start_time": round(start_time, 3),
                    "valid_end_time": round(end_time, 3),
                    "state_summary": state_summary,
                    "detected_text": _text_values([frame.get("detected_text")]),
                    "source": "sampled_frame_interval",
                    "confidence": _optional_float(frame.get("confidence")),
                }
            )
    return states


def _evidence_unit_document(
    *,
    target: dict[str, Any],
    window_segments: list[dict[str, Any]],
    states_by_id: dict[str, dict[str, Any]],
    visual_states: list[dict[str, Any]],
    entities_by_frame_id: dict[str, list[dict[str, Any]]],
    entities_by_id: dict[str, dict[str, Any]],
    links_by_segment_id: dict[str, list[dict[str, Any]]],
    window_config: dict[str, Any],
) -> dict[str, Any]:
    source_segment_ids = [_text(segment.get("segment_id")) for segment in window_segments]
    source_segment_ids = [segment_id for segment_id in source_segment_ids if segment_id]
    start_time, end_time = _window_bounds(window_segments)
    target_segment_id = _text(target.get("segment_id"))
    video_id = _text(target.get("video_id"))
    window_state_ids = _state_ids_for_window(
        visual_states,
        video_id=video_id,
        start_time=start_time,
        end_time=end_time,
    )
    window_states = [states_by_id[state_id] for state_id in window_state_ids if state_id in states_by_id]
    visual_entity_rows = _entities_for_states(window_states, entities_by_frame_id)
    window_links = [
        link
        for segment_id in source_segment_ids
        for link in links_by_segment_id.get(segment_id, [])
        if _text(link.get("entity_id")) in entities_by_id
    ]
    visual_entity_rows = _merge_entities(visual_entity_rows, window_links, entities_by_id)
    visual_entity_ids = [_text(entity.get("entity_id")) for entity in visual_entity_rows if _text(entity.get("entity_id"))]

    link_statuses = {
        _text(link.get("link_id")): _link_alignment_status(link)
        for link in window_links
        if _text(link.get("link_id"))
    }
    verified_link_ids = [
        link_id for link_id, status in link_statuses.items() if status == "verified"
    ]
    candidate_link_ids = [
        link_id for link_id, status in link_statuses.items() if status != "verified"
    ]
    transcript_window_text = _transcript_window_text(window_segments)
    visual_text = _visual_summary_text(window_states=window_states, visual_entities=visual_entity_rows)
    evidence_text = _compact_text(
        " ".join(
            part
            for part in (
                f"Transcript: {transcript_window_text}" if transcript_window_text else "",
                f"Visual: {visual_text}" if visual_text else "",
            )
            if part
        )
    )
    source_quality = _source_quality(
        visual_states=window_states,
        visual_entities=visual_entity_rows,
        link_statuses=link_statuses,
    )
    return {
        "evidence_unit_id": f"evu_{slugify(target_segment_id)}",
        "project_id": _text(target.get("project_id")),
        "video_id": video_id,
        "target_segment_id": target_segment_id,
        "source_segment_ids": source_segment_ids,
        "start_time": start_time,
        "end_time": end_time,
        "transcript_window_text": transcript_window_text,
        "visual_state_ids": window_state_ids,
        "visual_states": [_visual_state_context(state) for state in window_states],
        "visual_entity_ids": visual_entity_ids,
        "visual_entities": [_visual_entity_context(entity) for entity in visual_entity_rows],
        "verified_entity_link_ids": verified_link_ids,
        "candidate_entity_link_ids": candidate_link_ids,
        "candidate_entity_link_statuses": {
            link_id: status for link_id, status in link_statuses.items() if status != "verified"
        },
        "modality": ["speech", "visual"] if source_quality["has_visual_state"] or source_quality["has_visual_entity"] else ["speech"],
        "evidence_text": evidence_text,
        "semantic_text": _compact_text(f"{transcript_window_text} {visual_text}"),
        "alignment_score": _alignment_score(window_links, link_statuses),
        "alignment_status": _alignment_status(source_quality),
        "source_quality": source_quality,
        "window_config": window_config,
    }


def _state_ids_for_window(
    visual_states: list[dict[str, Any]],
    *,
    video_id: str,
    start_time: float | None,
    end_time: float | None,
) -> list[str]:
    if start_time is None or end_time is None:
        return []
    state_ids: list[str] = []
    for state in visual_states:
        state_video_id = _text(state.get("video_id"))
        if state_video_id and video_id and state_video_id != video_id:
            continue
        if _overlaps(
            _optional_float(state.get("valid_start_time")),
            _optional_float(state.get("valid_end_time")),
            start_time,
            end_time,
        ):
            state_ids.append(_text(state.get("visual_state_id")))
    return [state_id for state_id in state_ids if state_id]


def _entities_for_states(
    states: list[dict[str, Any]],
    entities_by_frame_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        for frame_id in _string_list(state.get("frame_ids")):
            for entity in entities_by_frame_id.get(frame_id, []):
                entity_id = _text(entity.get("entity_id"))
                if not entity_id or entity_id in seen:
                    continue
                seen.add(entity_id)
                entities.append(entity)
    return entities


def _merge_entities(
    entities: list[dict[str, Any]],
    links: list[dict[str, Any]],
    entities_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(entities)
    seen = {_text(entity.get("entity_id")) for entity in merged}
    for link in links:
        entity_id = _text(link.get("entity_id"))
        if not entity_id or entity_id in seen:
            continue
        entity = entities_by_id.get(entity_id)
        if entity is None:
            continue
        seen.add(entity_id)
        merged.append(entity)
    return merged


def _link_alignment_status(link: dict[str, Any]) -> str:
    explicit_status = _text(
        link.get("alignment_status")
        or link.get("verification_status")
        or link.get("status")
    ).casefold()
    if explicit_status == "verified" or link.get("verified") is True:
        return "verified"
    if _is_timestamp_fallback_link(link):
        return "timestamp_fallback"
    return "candidate"


def _is_timestamp_fallback_link(link: dict[str, Any]) -> bool:
    evidence = {str(item) for item in link.get("evidence", [])}
    link_type = _text(link.get("link_type")).casefold()
    reason_summary = _text(
        _mapping(link.get("reason_metadata")).get("summary")
    ).casefold()
    has_strong_signal = bool(
        evidence
        - {
            "time_overlap",
            "timestamp_fallback",
        }
    )
    return (
        not has_strong_signal
        or link_type == "time_overlap"
        or reason_summary == "timestamp_fallback_only"
    )


def _source_quality(
    *,
    visual_states: list[dict[str, Any]],
    visual_entities: list[dict[str, Any]],
    link_statuses: dict[str, str],
) -> dict[str, Any]:
    has_vlm_entity = any(_is_vlm_entity(entity) for entity in visual_entities)
    visual_state_detected_text_count = sum(
        1 for state in visual_states if _text_values([state.get("detected_text")])
    )
    visual_entity_detected_text_count = sum(
        1 for entity in visual_entities if _text_values([entity.get("detected_text")])
    )
    visual_description_count = sum(
        1 for entity in visual_entities if _text(entity.get("visual_description"))
    )
    return {
        "has_visual_state": bool(visual_states),
        "has_visual_entity": bool(visual_entities),
        "has_vlm_entity": has_vlm_entity,
        "has_verified_link": any(status == "verified" for status in link_statuses.values()),
        "uses_ocr_only": bool(visual_entities) and not has_vlm_entity,
        "has_detected_text": bool(visual_state_detected_text_count or visual_entity_detected_text_count),
        "has_visual_description": bool(visual_description_count),
        "visual_state_detected_text_count": visual_state_detected_text_count,
        "visual_entity_detected_text_count": visual_entity_detected_text_count,
        "visual_description_count": visual_description_count,
        "has_timestamp_fallback_link": any(
            status == "timestamp_fallback" for status in link_statuses.values()
        ),
        "candidate_link_count": sum(1 for status in link_statuses.values() if status == "candidate"),
        "timestamp_fallback_link_count": sum(
            1 for status in link_statuses.values() if status == "timestamp_fallback"
        ),
        "verified_link_count": sum(1 for status in link_statuses.values() if status == "verified"),
    }


def _alignment_status(source_quality: dict[str, Any]) -> str:
    if source_quality["has_verified_link"]:
        return "verified"
    if (
        source_quality["candidate_link_count"]
        or source_quality["timestamp_fallback_link_count"]
        or source_quality["has_visual_entity"]
        or source_quality["has_visual_state"]
    ):
        return "candidate"
    return "transcript_only"


def _alignment_score(links: list[dict[str, Any]], statuses: dict[str, str]) -> float:
    if not links:
        return 0.0
    weighted_scores: list[float] = []
    for link in links:
        link_id = _text(link.get("link_id"))
        score = _optional_float(link.get("score")) or 0.0
        status = statuses.get(link_id, "candidate")
        if status == "verified":
            weighted_scores.append(max(score, 1.0))
        elif status == "timestamp_fallback":
            weighted_scores.append(min(score, 0.2))
        else:
            weighted_scores.append(min(score, 0.8))
    return round(max(weighted_scores), 3)


def _is_vlm_entity(entity: dict[str, Any]) -> bool:
    return is_paper_quality_vlm_entity(entity)


def _visual_state_context(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "visual_state_id": state.get("visual_state_id"),
            "representative_frame_id": state.get("representative_frame_id"),
            "frame_ids": state.get("frame_ids", []),
            "valid_start_time": state.get("valid_start_time"),
            "valid_end_time": state.get("valid_end_time"),
            "state_summary": state.get("state_summary", ""),
            "detected_text": state.get("detected_text", []),
        }.items()
        if value not in (None, "", [])
    }


def _visual_entity_context(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "entity_id": entity.get("entity_id"),
            "frame_id": entity.get("frame_id"),
            "timestamp": entity.get("timestamp"),
            "entity_type": entity.get("entity_type"),
            "text": entity.get("text"),
            "visual_description": entity.get("visual_description"),
            "position": entity.get("position"),
            "relations": entity.get("relations"),
            "detected_text": entity.get("detected_text"),
            "confidence": entity.get("confidence"),
            "source": entity.get("source"),
            "source_model": entity.get("source_model"),
        }.items()
        if value not in (None, "", [])
    }


def _visual_summary_text(
    *,
    window_states: list[dict[str, Any]],
    visual_entities: list[dict[str, Any]],
) -> str:
    values: list[Any] = []
    for state in window_states:
        values.extend([state.get("state_summary"), state.get("detected_text")])
    for entity in visual_entities:
        values.extend(
            [
                entity.get("text"),
                entity.get("visual_description"),
                entity.get("entity_type"),
                entity.get("detected_text"),
                entity.get("position"),
                entity.get("relations"),
            ]
        )
    return _compact_text(" ".join(_unique_text_values(values)))


def _transcript_window_text(segments: list[dict[str, Any]]) -> str:
    return _compact_text(" ".join(_unique_text_values(segment.get("transcript_text") for segment in segments)))


def _window_bounds(segments: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    starts = [_optional_float(segment.get("start_time")) for segment in segments]
    ends = [_optional_float(segment.get("end_time")) for segment in segments]
    bounds = [value for value in [*starts, *ends] if value is not None]
    if not bounds:
        centers = [_optional_float(segment.get("timestamp_center")) for segment in segments]
        bounds = [value for value in centers if value is not None]
    if not bounds:
        return None, None
    return min(bounds), max(bounds)


def _overlaps(
    left_start: float | None,
    left_end: float | None,
    right_start: float | None,
    right_end: float | None,
) -> bool:
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return False
    return min(left_start, left_end) <= max(right_start, right_end) and min(right_start, right_end) <= max(left_start, left_end)


def _window_config(
    *,
    window_seconds: float | None,
    neighbor_count: int,
    previous_neighbor_count: int | None,
    next_neighbor_count: int | None,
    window_before_seconds: float | None,
    window_after_seconds: float | None,
) -> dict[str, Any]:
    return resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )


def _status_counts(documents: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        status = _text(document.get("alignment_status")) or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _optional_existing_project_path(
    *,
    project_dir: Path,
    path: Path | None,
    default: Path,
    artifact_name: str,
) -> Path | None:
    candidate = default if path is None else _resolve_project_output_path(project_dir=project_dir, path=path, default=default)
    if candidate.exists():
        return candidate
    if path is not None:
        raise FileNotFoundError(f"{artifact_name} artifact not found: {candidate}")
    return None


def _resolve_project_output_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _update_project_manifest(
    *,
    manifest_path: Path,
    evidence_units_path: Path,
    summary: dict[str, Any],
) -> None:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["evidence_units"] = str(evidence_units_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["evidence_units"] = int(summary["counts"]["evidence_units_total"])

    payload["evidence_unit_storage"] = {
        "window_config": summary["window_config"],
        "visual_state_config": summary["visual_state_config"],
        "counts": summary["counts"],
        "alignment_status_counts": summary["alignment_status_counts"],
    }
    write_json(manifest_path, payload)


def _group_by_text(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = _text(row.get(key))
        if not value:
            continue
        grouped.setdefault(value, []).append(row)
    return grouped


def _first_text(rows: Iterable[dict[str, Any]], key: str) -> str | None:
    for row in rows:
        value = _text(row.get(key))
        if value:
            return value
    return None


def _unique_text_values(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        for text in _text_values([value]):
            compacted = _compact_text(text)
            if compacted and compacted not in seen:
                seen.add(compacted)
                unique.append(compacted)
    return unique


def _text_values(values: Iterable[Any]) -> list[str]:
    text_values: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            text_values.extend(_text_values(value.values()))
        elif isinstance(value, list):
            text_values.extend(_text_values(value))
        else:
            text = _text(value)
            if text:
                text_values.append(text)
    return text_values


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_float(value: Any, *, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        return default
    return max(0.0, parsed)
