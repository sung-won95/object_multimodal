from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .project_index import segment_artifact_path
from .schemas import EvidenceWindow


def build_evidence_response(
    *,
    project_dir: Path,
    segment_ids: list[str],
    query: str | None = None,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    resolved_segments_path = segment_artifact_path(resolved_project_dir, segments=segments_path)
    resolved_frames_manifest_path = resolve_project_path(
        resolved_project_dir,
        frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )

    segments = read_jsonl(resolved_segments_path)
    frames = read_jsonl(resolved_frames_manifest_path) if resolved_frames_manifest_path.exists() else []
    frame_lookup = {frame_id(frame): frame for frame in frames}
    segment_lookup = {str(segment.get("segment_id")): segment for segment in segments}
    window_config = resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )

    top_segments = []
    evidence_windows = []
    for segment_id in segment_ids:
        if segment_id not in segment_lookup:
            raise ValueError(f"Segment not found in {resolved_segments_path}: {segment_id}")
        target = segment_lookup[segment_id]
        top_segments.append(segment_summary(target))
        window_segments = select_window_segments(
            segments,
            target_segment_id=segment_id,
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        )
        evidence_windows.append(
            make_evidence_window(
                target=target,
                window_segments=window_segments,
                frame_lookup=frame_lookup,
                window_config=window_config,
            ).to_dict()
        )

    return {
        "query": query,
        "project_dir": str(resolved_project_dir),
        "paths": {
            "segments": str(resolved_segments_path),
            "frames_manifest": str(resolved_frames_manifest_path),
        },
        "window_config": window_config,
        "top_segments": top_segments,
        "evidence_windows": evidence_windows,
    }


def select_window_segments(
    segments: list[dict[str, Any]],
    *,
    target_segment_id: str,
    window_seconds: float | None,
    neighbor_count: int,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
) -> list[dict[str, Any]]:
    window_config = resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )
    sorted_segments = sorted(enumerate(segments), key=lambda item: segment_sort_key(item[1], item[0]))
    sorted_rows = [segment for _, segment in sorted_segments]
    target_index = next(
        (
            index
            for index, segment in enumerate(sorted_rows)
            if str(segment.get("segment_id")) == target_segment_id
        ),
        None,
    )
    if target_index is None:
        raise ValueError(f"Segment not found: {target_segment_id}")

    if window_config["mode"] == "neighbors":
        previous_count = int(window_config["previous_neighbor_count"])
        next_count = int(window_config["next_neighbor_count"])
        start = max(0, target_index - previous_count)
        end = min(len(sorted_rows), target_index + next_count + 1)
        return sorted_rows[start:end]

    target_window = segment_window(sorted_rows[target_index])
    if target_window is None:
        return sorted_rows[target_index : target_index + 1]
    start_time = target_window[0] - float(window_config["window_before_seconds"])
    end_time = target_window[1] + float(window_config["window_after_seconds"])
    return [
        segment
        for segment in sorted_rows
        if overlaps_window(segment_window(segment), start_time=start_time, end_time=end_time)
    ]


def make_evidence_window(
    *,
    target: dict[str, Any],
    window_segments: list[dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
    window_config: dict[str, Any] | None = None,
) -> EvidenceWindow:
    frame_refs = collect_frame_metadata(window_segments, frame_lookup)
    window_bounds = [
        bound
        for segment in window_segments
        for bound in segment_window(segment) or ()
        if bound is not None
    ]
    annotated_segments = annotate_window_segments(target=target, window_segments=window_segments)
    target_segment = next(
        (segment for segment in annotated_segments if segment.get("window_role") == "target"),
        segment_context(target, window_role="target", target_distance=0),
    )
    neighbor_segments = [
        segment for segment in annotated_segments if segment.get("window_role") != "target"
    ]
    return EvidenceWindow(
        target_segment_id=str(target.get("segment_id", "")),
        project_id=optional_str(target.get("project_id")),
        video_id=optional_str(target.get("video_id")),
        start_time=min(window_bounds) if window_bounds else optional_float(target.get("start_time")),
        end_time=max(window_bounds) if window_bounds else optional_float(target.get("end_time")),
        transcript_segments=annotated_segments,
        frame_refs=frame_refs,
        target_segment=target_segment,
        neighbor_segments=neighbor_segments,
        window_config=window_config or {},
    )


def resolve_window_config(
    *,
    window_seconds: float | None,
    neighbor_count: int,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
) -> dict[str, Any]:
    if window_seconds is not None or window_before_seconds is not None or window_after_seconds is not None:
        symmetric_seconds = nonnegative_float(window_seconds, default=0.0)
        return {
            "mode": "time",
            "window_seconds": symmetric_seconds if window_seconds is not None else None,
            "window_before_seconds": nonnegative_float(
                window_before_seconds,
                default=symmetric_seconds,
            ),
            "window_after_seconds": nonnegative_float(
                window_after_seconds,
                default=symmetric_seconds,
            ),
        }

    symmetric_count = max(0, neighbor_count)
    return {
        "mode": "neighbors",
        "neighbor_count": symmetric_count,
        "previous_neighbor_count": nonnegative_int(
            previous_neighbor_count,
            default=symmetric_count,
        ),
        "next_neighbor_count": nonnegative_int(
            next_neighbor_count,
            default=symmetric_count,
        ),
    }


def annotate_window_segments(
    *,
    target: dict[str, Any],
    window_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_segment_id = str(target.get("segment_id", ""))
    target_index = next(
        (
            index
            for index, segment in enumerate(window_segments)
            if str(segment.get("segment_id")) == target_segment_id
        ),
        None,
    )
    if target_index is None:
        target_index = 0

    annotated = []
    for index, segment in enumerate(window_segments):
        target_distance = index - target_index
        if str(segment.get("segment_id")) == target_segment_id:
            target_distance = 0
            window_role = "target"
        elif target_distance < 0:
            window_role = "previous"
        else:
            window_role = "next"
        annotated.append(
            segment_context(
                segment,
                window_role=window_role,
                target_distance=target_distance,
            )
        )
    return annotated


def collect_frame_metadata(
    segments: Iterable[dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment in segments:
        for frame_ref in segment.get("frame_refs") or []:
            frame_id_value = str(frame_ref)
            if not frame_id_value or frame_id_value in seen:
                continue
            seen.add(frame_id_value)
            frame = dict(frame_lookup.get(frame_id_value, {}))
            frame.setdefault("frame_id", frame_id_value)
            collected.append(frame)
    return sorted(collected, key=frame_sort_key)


def frame_sort_key(frame: dict[str, Any]) -> tuple[bool, float, str]:
    timestamp = optional_float(frame.get("timestamp"))
    return (timestamp is None, timestamp or 0.0, str(frame.get("frame_id", "")))


def segment_context(
    segment: dict[str, Any],
    *,
    window_role: str | None = None,
    target_distance: int | None = None,
) -> dict[str, Any]:
    keys = (
        "segment_id",
        "sample_index",
        "video_id",
        "video_name",
        "start_time",
        "end_time",
        "timestamp_center",
        "transcript_text",
        "frame_refs",
    )
    context = {key: segment.get(key) for key in keys if key in segment}
    if window_role is not None:
        context["window_role"] = window_role
    if target_distance is not None:
        context["target_distance"] = target_distance
    duration = segment_duration(segment)
    if duration is not None:
        context["duration_seconds"] = duration
    transcript = str(segment.get("transcript_text", ""))
    context["text_char_count"] = len(transcript)
    metadata = segment_metadata(segment)
    if metadata:
        context["metadata"] = metadata
    return context


def segment_summary(segment: dict[str, Any]) -> dict[str, Any]:
    transcript = str(segment.get("transcript_text", ""))
    summary = {
        "segment_id": str(segment.get("segment_id", "")),
        "video_id": segment.get("video_id"),
        "video_name": segment.get("video_name"),
        "start_time": segment.get("start_time"),
        "end_time": segment.get("end_time"),
        "timestamp_center": segment.get("timestamp_center"),
        "duration_seconds": segment_duration(segment),
        "text_char_count": len(transcript),
        "transcript_excerpt": transcript[:240] + ("..." if len(transcript) > 240 else ""),
    }
    metadata = segment_metadata(segment)
    if metadata:
        summary["metadata"] = metadata
    return summary


def segment_metadata(segment: dict[str, Any]) -> dict[str, Any]:
    metadata_keys = (
        "project_id",
        "dataset_name",
        "subset_name",
        "split_name",
        "sample_id",
        "source",
        "title",
        "video_title",
        "video_name",
    )
    return {
        key: segment[key]
        for key in metadata_keys
        if segment.get(key) not in (None, "")
    }


def segment_duration(segment: dict[str, Any]) -> float | None:
    window = segment_window(segment)
    if window is None:
        return None
    return round(window[1] - window[0], 4)


def segment_sort_key(segment: dict[str, Any], fallback_index: int) -> tuple[float, int]:
    center = optional_float(segment.get("timestamp_center"))
    start = optional_float(segment.get("start_time"))
    sample_index = optional_int(segment.get("sample_index"), default=fallback_index)
    sort_time = center if center is not None else start
    return (sort_time if sort_time is not None else float("inf"), sample_index)


def segment_window(segment: dict[str, Any]) -> tuple[float, float] | None:
    start = optional_float(segment.get("start_time"))
    end = optional_float(segment.get("end_time"))
    if start is None and end is None:
        center = optional_float(segment.get("timestamp_center"))
        if center is None:
            return None
        return (center, center)
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def overlaps_window(
    window: tuple[float, float] | None,
    *,
    start_time: float,
    end_time: float,
) -> bool:
    if window is None:
        return False
    return window[0] <= end_time and window[1] >= start_time


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL not found: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {resolved}:{line_number}")
            rows.append(payload)
    return rows


def resolve_project_path(project_dir: Path, path: Path | None, *, default: Path) -> Path:
    if path is None:
        return default
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def frame_id(frame: dict[str, Any]) -> str:
    raw_frame_id = frame.get("frame_id")
    if raw_frame_id:
        return str(raw_frame_id)
    return Path(str(frame.get("frame_path", ""))).stem


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nonnegative_float(value: Any, *, default: float) -> float:
    parsed = optional_float(value)
    if parsed is None:
        return default
    return max(0.0, parsed)


def optional_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def nonnegative_int(value: Any, *, default: int) -> int:
    return max(0, optional_int(value, default=default))


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
