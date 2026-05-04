from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame_selection import DEFAULT_MIN_CONTRAST, DEFAULT_MIN_DETAIL_SCORE
from .io import write_json, write_jsonl
from .schemas import (
    VLM_ARTIFACT_PATHS,
    VLM_FRAME_CANDIDATES_ARTIFACT,
    VLMFrameCandidate,
    mention_candidates,
)


SELECTOR_BACKEND = "vlm-frame-candidate-selector"
SELECTION_STRATEGY = "segment_temporal_coverage"


@dataclass(frozen=True)
class VLMFrameCandidateConfig:
    max_candidates: int = 120
    max_per_segment: int = 2
    window_seconds: float = 60.0
    max_per_window: int = 6
    min_time_gap_seconds: float = 2.0
    segment_margin_seconds: float = 0.0
    duplicate_timestamp_epsilon_seconds: float = 0.05
    duplicate_distance_threshold: float = 0.025
    low_information_min_contrast: float = DEFAULT_MIN_CONTRAST
    low_information_min_detail_score: float = DEFAULT_MIN_DETAIL_SCORE


@dataclass(frozen=True)
class _FrameRow:
    frame_id: str
    frame_path: str | None
    timestamp: float | None
    video_id: str
    index: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class _SegmentRow:
    segment_id: str
    index: int
    start_time: float | None
    end_time: float | None
    center_time: float | None
    frame_ids: tuple[str, ...]
    visual_hints: tuple[str, ...]


def select_vlm_frame_candidates(
    *,
    frames: list[dict[str, Any]],
    project_id: str,
    video_id: str | None = None,
    segments: list[dict[str, Any]] | None = None,
    config: VLMFrameCandidateConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or VLMFrameCandidateConfig()
    _validate_config(resolved_config)

    frame_rows = _normalize_frames(frames=frames, default_video_id=video_id or project_id)
    frames_by_id = {frame.frame_id: frame for frame in frame_rows}
    segment_rows = _normalize_segments(
        segments=segments or [],
        frames=frame_rows,
        config=resolved_config,
    )
    frame_to_segments = _frame_to_segments(segment_rows)
    pre_excluded = _pre_excluded_frames(frame_rows, config=resolved_config)

    selected: list[VLMFrameCandidate] = []
    selected_ids: set[str] = set()
    selected_timestamps: list[float] = []
    selected_by_segment: dict[str, int] = {}
    selected_by_window: dict[int, int] = {}
    selection_context: dict[str, dict[str, Any]] = {}
    blocked_reasons: dict[str, str] = {}

    if resolved_config.max_candidates > 0:
        _select_segment_anchors(
            segment_rows=segment_rows,
            frames_by_id=frames_by_id,
            pre_excluded=pre_excluded,
            selected=selected,
            selected_ids=selected_ids,
            selected_timestamps=selected_timestamps,
            selected_by_segment=selected_by_segment,
            selected_by_window=selected_by_window,
            selection_context=selection_context,
            blocked_reasons=blocked_reasons,
            project_id=project_id,
            frame_to_segments=frame_to_segments,
            config=resolved_config,
        )

        _select_temporal_fill(
            frame_rows=frame_rows,
            pre_excluded=pre_excluded,
            selected=selected,
            selected_ids=selected_ids,
            selected_timestamps=selected_timestamps,
            selected_by_segment=selected_by_segment,
            selected_by_window=selected_by_window,
            selection_context=selection_context,
            blocked_reasons=blocked_reasons,
            project_id=project_id,
            frame_to_segments=frame_to_segments,
            config=resolved_config,
        )

    excluded_frames = _excluded_frames(
        frame_rows=frame_rows,
        selected_ids=selected_ids,
        pre_excluded=pre_excluded,
        blocked_reasons=blocked_reasons,
        frame_to_segments=frame_to_segments,
        selected_timestamps=selected_timestamps,
        selected_by_segment=selected_by_segment,
        selected_by_window=selected_by_window,
        config=resolved_config,
    )
    summary = _summary(
        frame_rows=frame_rows,
        candidates=selected,
        excluded_frames=excluded_frames,
        segment_rows=segment_rows,
        frame_to_segments=frame_to_segments,
        config=resolved_config,
    )

    return {
        "candidates": [candidate.to_dict() for candidate in selected],
        "excluded_frames": excluded_frames,
        "summary": summary,
    }


def generate_vlm_frame_candidates(
    *,
    project_dir: Path,
    frames_manifest_path: Path | None = None,
    segments_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    config: VLMFrameCandidateConfig | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    resolved_frames_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )
    resolved_segments_path = _resolve_segments_path(
        project_dir=resolved_project_dir,
        candidate=segments_path,
    )
    resolved_output_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=output_path,
        default=resolved_project_dir / VLM_ARTIFACT_PATHS[VLM_FRAME_CANDIDATES_ARTIFACT],
    )

    frames = _read_jsonl(resolved_frames_manifest_path, label="frames_manifest")
    segments = (
        _read_jsonl(resolved_segments_path, label="segments")
        if resolved_segments_path is not None
        else []
    )
    manifest = _read_manifest(resolved_manifest_path)
    project_id = str(manifest.get("project_id") or resolved_project_dir.name)
    video_id = _optional_str(manifest.get("video_id")) or project_id

    result = select_vlm_frame_candidates(
        frames=frames,
        segments=segments,
        project_id=project_id,
        video_id=video_id,
        config=config,
    )
    write_jsonl(resolved_output_path, result["candidates"])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        output_path=resolved_output_path,
        summary=result["summary"],
    )

    return {
        "project_id": project_id,
        "paths": {
            "project_dir": str(resolved_project_dir),
            "frames_manifest": str(resolved_frames_manifest_path),
            "segments": str(resolved_segments_path) if resolved_segments_path is not None else None,
            "vlm_frame_candidates": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "counts": result["summary"]["counts"],
        "summary": result["summary"],
    }


def _select_segment_anchors(
    *,
    segment_rows: list[_SegmentRow],
    frames_by_id: dict[str, _FrameRow],
    pre_excluded: dict[str, str],
    selected: list[VLMFrameCandidate],
    selected_ids: set[str],
    selected_timestamps: list[float],
    selected_by_segment: dict[str, int],
    selected_by_window: dict[int, int],
    selection_context: dict[str, dict[str, Any]],
    blocked_reasons: dict[str, str],
    project_id: str,
    frame_to_segments: dict[str, list[str]],
    config: VLMFrameCandidateConfig,
) -> None:
    queues = {
        segment.segment_id: _rank_segment_frames(segment=segment, frames_by_id=frames_by_id)
        for segment in segment_rows
        if segment.frame_ids
    }
    if not queues:
        return

    progress = True
    while progress and len(selected) < config.max_candidates:
        progress = False
        for segment in segment_rows:
            if len(selected) >= config.max_candidates:
                return
            if selected_by_segment.get(segment.segment_id, 0) >= config.max_per_segment:
                continue
            queue = queues.get(segment.segment_id, [])
            for frame_id in queue:
                if frame_id in selected_ids or frame_id in pre_excluded:
                    continue
                frame = frames_by_id[frame_id]
                reason = _blocking_reason(
                    frame=frame,
                    segment_id=segment.segment_id,
                    selected_timestamps=selected_timestamps,
                    selected_by_segment=selected_by_segment,
                    selected_by_window=selected_by_window,
                    selected_count=len(selected),
                    config=config,
                )
                if reason is not None:
                    blocked_reasons.setdefault(frame_id, reason)
                    continue

                candidate = _make_candidate(
                    frame=frame,
                    project_id=project_id,
                    rank=len(selected) + 1,
                    primary_segment_id=segment.segment_id,
                    segment_ids=frame_to_segments.get(frame_id, []),
                    selection_reason=(
                        "segment_visual_hint" if segment.visual_hints else "segment_coverage"
                    ),
                    selection_pass="segment_round_robin",
                    config=config,
                )
                selected.append(candidate)
                _record_selection(
                    frame=frame,
                    segment_id=segment.segment_id,
                    selected_ids=selected_ids,
                    selected_timestamps=selected_timestamps,
                    selected_by_segment=selected_by_segment,
                    selected_by_window=selected_by_window,
                    selection_context=selection_context,
                    selection_pass="segment_round_robin",
                    config=config,
                )
                progress = True
                break


def _select_temporal_fill(
    *,
    frame_rows: list[_FrameRow],
    pre_excluded: dict[str, str],
    selected: list[VLMFrameCandidate],
    selected_ids: set[str],
    selected_timestamps: list[float],
    selected_by_segment: dict[str, int],
    selected_by_window: dict[int, int],
    selection_context: dict[str, dict[str, Any]],
    blocked_reasons: dict[str, str],
    project_id: str,
    frame_to_segments: dict[str, list[str]],
    config: VLMFrameCandidateConfig,
) -> None:
    for frame in _temporal_spread_order(frame_rows, max_count=config.max_candidates):
        if len(selected) >= config.max_candidates:
            return
        if frame.frame_id in selected_ids or frame.frame_id in pre_excluded:
            continue
        segment_ids = frame_to_segments.get(frame.frame_id, [])
        primary_segment_id = segment_ids[0] if segment_ids else None
        reason = _blocking_reason(
            frame=frame,
            segment_id=primary_segment_id,
            selected_timestamps=selected_timestamps,
            selected_by_segment=selected_by_segment,
            selected_by_window=selected_by_window,
            selected_count=len(selected),
            config=config,
        )
        if reason is not None:
            blocked_reasons.setdefault(frame.frame_id, reason)
            continue

        selected.append(
            _make_candidate(
                frame=frame,
                project_id=project_id,
                rank=len(selected) + 1,
                primary_segment_id=primary_segment_id,
                segment_ids=segment_ids,
                selection_reason="temporal_spread",
                selection_pass="temporal_fill",
                config=config,
            )
        )
        _record_selection(
            frame=frame,
            segment_id=primary_segment_id,
            selected_ids=selected_ids,
            selected_timestamps=selected_timestamps,
            selected_by_segment=selected_by_segment,
            selected_by_window=selected_by_window,
            selection_context=selection_context,
            selection_pass="temporal_fill",
            config=config,
        )


def _make_candidate(
    *,
    frame: _FrameRow,
    project_id: str,
    rank: int,
    primary_segment_id: str | None,
    segment_ids: list[str],
    selection_reason: str,
    selection_pass: str,
    config: VLMFrameCandidateConfig,
) -> VLMFrameCandidate:
    metadata = {
        "selector": SELECTION_STRATEGY,
        "selection_pass": selection_pass,
        "segment_ids": segment_ids,
        "source_frame_index": frame.index,
        "window_index": _window_index(frame.timestamp, config.window_seconds),
        "min_time_gap_seconds": config.min_time_gap_seconds,
    }
    return VLMFrameCandidate(
        project_id=project_id,
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        segment_id=primary_segment_id,
        backend=SELECTOR_BACKEND,
        source_model=None,
        model_version=None,
        confidence=None,
        status="selected",
        frame_path=frame.frame_path,
        selection_reason=selection_reason,
        rank=rank,
        metadata=metadata,
    )


def _record_selection(
    *,
    frame: _FrameRow,
    segment_id: str | None,
    selected_ids: set[str],
    selected_timestamps: list[float],
    selected_by_segment: dict[str, int],
    selected_by_window: dict[int, int],
    selection_context: dict[str, dict[str, Any]],
    selection_pass: str,
    config: VLMFrameCandidateConfig,
) -> None:
    selected_ids.add(frame.frame_id)
    if frame.timestamp is not None:
        selected_timestamps.append(frame.timestamp)
    if segment_id is not None:
        selected_by_segment[segment_id] = selected_by_segment.get(segment_id, 0) + 1
    window_index = _window_index(frame.timestamp, config.window_seconds)
    if window_index is not None:
        selected_by_window[window_index] = selected_by_window.get(window_index, 0) + 1
    selection_context[frame.frame_id] = {"selection_pass": selection_pass}


def _blocking_reason(
    *,
    frame: _FrameRow,
    segment_id: str | None,
    selected_timestamps: list[float],
    selected_by_segment: dict[str, int],
    selected_by_window: dict[int, int],
    selected_count: int,
    config: VLMFrameCandidateConfig,
) -> str | None:
    if selected_count >= config.max_candidates:
        return "max_candidates"
    if (
        segment_id is not None
        and selected_by_segment.get(segment_id, 0) >= config.max_per_segment
    ):
        return "segment_cap"

    window_index = _window_index(frame.timestamp, config.window_seconds)
    if (
        window_index is not None
        and selected_by_window.get(window_index, 0) >= config.max_per_window
    ):
        return "window_cap"

    if frame.timestamp is not None and selected_timestamps:
        if min(abs(frame.timestamp - timestamp) for timestamp in selected_timestamps) < (
            config.min_time_gap_seconds
        ):
            return "min_time_gap"
    return None


def _excluded_frames(
    *,
    frame_rows: list[_FrameRow],
    selected_ids: set[str],
    pre_excluded: dict[str, str],
    blocked_reasons: dict[str, str],
    frame_to_segments: dict[str, list[str]],
    selected_timestamps: list[float],
    selected_by_segment: dict[str, int],
    selected_by_window: dict[int, int],
    config: VLMFrameCandidateConfig,
) -> list[dict[str, Any]]:
    excluded = []
    for frame in frame_rows:
        if frame.frame_id in selected_ids:
            continue
        segment_ids = frame_to_segments.get(frame.frame_id, [])
        reason = pre_excluded.get(frame.frame_id)
        if reason is None:
            reason = blocked_reasons.get(frame.frame_id)
        if reason is None and config.max_candidates == 0:
            reason = "max_candidates"
        if reason is None and segment_ids:
            capped_segments = [
                segment_id
                for segment_id in segment_ids
                if selected_by_segment.get(segment_id, 0) >= config.max_per_segment
            ]
            if capped_segments:
                reason = "segment_cap"
        if reason is None:
            window_index = _window_index(frame.timestamp, config.window_seconds)
            if (
                window_index is not None
                and selected_by_window.get(window_index, 0) >= config.max_per_window
            ):
                reason = "window_cap"
        if reason is None and frame.timestamp is not None and selected_timestamps:
            if min(abs(frame.timestamp - timestamp) for timestamp in selected_timestamps) < (
                config.min_time_gap_seconds
            ):
                reason = "min_time_gap"
        if reason is None and segment_ids:
            reason = "not_selected"
        if reason is None:
            reason = "outside_segment_coverage"

        excluded.append(
            {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "segment_ids": segment_ids,
                "reason": reason,
            }
        )
    return excluded


def _summary(
    *,
    frame_rows: list[_FrameRow],
    candidates: list[VLMFrameCandidate],
    excluded_frames: list[dict[str, Any]],
    segment_rows: list[_SegmentRow],
    frame_to_segments: dict[str, list[str]],
    config: VLMFrameCandidateConfig,
) -> dict[str, Any]:
    input_count = len(frame_rows)
    selected_count = len(candidates)
    excluded_count = len(excluded_frames)
    reason_counts = _reason_counts(excluded_frames)
    segments_with_frames = [segment for segment in segment_rows if segment.frame_ids]
    selected_segment_ids = {
        segment_id
        for candidate in candidates
        for segment_id in candidate.metadata.get("segment_ids", [])
    }
    candidate_timestamps = [
        candidate.timestamp for candidate in candidates if candidate.timestamp is not None
    ]
    input_timestamps = [frame.timestamp for frame in frame_rows if frame.timestamp is not None]
    input_windows = _window_set(input_timestamps, config.window_seconds)
    selected_windows = _window_set(candidate_timestamps, config.window_seconds)
    reduction_ratio = round((input_count - selected_count) / input_count, 4) if input_count else 0.0

    counts = {
        "input_frame_count": input_count,
        "selected_candidate_count": selected_count,
        "excluded_frame_count": excluded_count,
        "segments_total": len(segment_rows),
        "segments_with_frame_coverage": len(segments_with_frames),
        "segments_with_candidates": len(selected_segment_ids),
        "windows_with_input_frames": len(input_windows),
        "windows_with_candidates": len(selected_windows),
        "exclusion_reasons": reason_counts,
    }
    coverage = {
        "segment_candidate_coverage_ratio": _ratio(
            len(selected_segment_ids),
            len(segments_with_frames),
        ),
        "window_candidate_coverage_ratio": _ratio(len(selected_windows), len(input_windows)),
        "input_time_span_sec": _time_span(input_timestamps),
        "candidate_time_span_sec": _time_span(candidate_timestamps),
        "first_candidate_timestamp": min(candidate_timestamps) if candidate_timestamps else None,
        "last_candidate_timestamp": max(candidate_timestamps) if candidate_timestamps else None,
    }
    tradeoff = {
        "estimated_vlm_cost_reduction_ratio": reduction_ratio,
        "recall_risk": _recall_risk(
            reduction_ratio=reduction_ratio,
            selected_count=selected_count,
            input_count=input_count,
            segment_coverage_ratio=coverage["segment_candidate_coverage_ratio"],
            window_coverage_ratio=coverage["window_candidate_coverage_ratio"],
        ),
    }
    return {
        "strategy": SELECTION_STRATEGY,
        "status": "selected",
        "backend": SELECTOR_BACKEND,
        "counts": counts,
        "coverage": coverage,
        "tradeoff": tradeoff,
        "thresholds": {
            "max_candidates": config.max_candidates,
            "max_per_segment": config.max_per_segment,
            "window_seconds": config.window_seconds,
            "max_per_window": config.max_per_window,
            "min_time_gap_seconds": config.min_time_gap_seconds,
            "segment_margin_seconds": config.segment_margin_seconds,
            "duplicate_timestamp_epsilon_seconds": config.duplicate_timestamp_epsilon_seconds,
            "duplicate_distance_threshold": config.duplicate_distance_threshold,
            "low_information_min_contrast": config.low_information_min_contrast,
            "low_information_min_detail_score": config.low_information_min_detail_score,
        },
        "segments_without_candidates": [
            segment.segment_id
            for segment in segments_with_frames
            if segment.segment_id not in selected_segment_ids
        ],
        "selected_frame_ids": [candidate.frame_id for candidate in candidates],
        "frames_without_segment_coverage": [
            frame.frame_id for frame in frame_rows if frame.frame_id not in frame_to_segments
        ],
    }


def _normalize_frames(*, frames: list[dict[str, Any]], default_video_id: str) -> list[_FrameRow]:
    rows = []
    for index, frame in enumerate(frames):
        frame_path = _optional_str(frame.get("frame_path"))
        frame_id = _optional_str(frame.get("frame_id"))
        if frame_id is None and frame_path is not None:
            frame_id = Path(frame_path).stem
        if frame_id is None:
            frame_id = f"frame_{index:06d}"
        rows.append(
            _FrameRow(
                frame_id=frame_id,
                frame_path=frame_path,
                timestamp=_optional_float(frame.get("timestamp")),
                video_id=_optional_str(frame.get("video_id")) or default_video_id,
                index=index,
                payload=frame,
            )
        )
    return rows


def _normalize_segments(
    *,
    segments: list[dict[str, Any]],
    frames: list[_FrameRow],
    config: VLMFrameCandidateConfig,
) -> list[_SegmentRow]:
    frames_by_id = {frame.frame_id: frame for frame in frames}
    rows = []
    for index, segment in enumerate(segments):
        segment_id = _optional_str(segment.get("segment_id")) or f"seg_{index:06d}"
        start_time, end_time = _segment_window(segment, margin_seconds=config.segment_margin_seconds)
        center_time = _segment_center(segment, start_time=start_time, end_time=end_time)
        frame_ids = _segment_frame_ids(
            segment=segment,
            frames=frames,
            frames_by_id=frames_by_id,
            start_time=start_time,
            end_time=end_time,
        )
        rows.append(
            _SegmentRow(
                segment_id=segment_id,
                index=index,
                start_time=start_time,
                end_time=end_time,
                center_time=center_time,
                frame_ids=tuple(frame_ids),
                visual_hints=tuple(_segment_visual_hints(segment)),
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            row.center_time is None,
            row.center_time if row.center_time is not None else row.index,
            row.index,
        ),
    )


def _segment_frame_ids(
    *,
    segment: dict[str, Any],
    frames: list[_FrameRow],
    frames_by_id: dict[str, _FrameRow],
    start_time: float | None,
    end_time: float | None,
) -> list[str]:
    frame_refs = segment.get("frame_refs")
    if isinstance(frame_refs, list):
        return [
            frame_id
            for frame_id in dict.fromkeys(str(ref) for ref in frame_refs)
            if frame_id in frames_by_id
        ]
    if start_time is None and end_time is None:
        return []
    if start_time is None:
        start_time = end_time
    if end_time is None:
        end_time = start_time
    if start_time is None or end_time is None:
        return []
    if end_time < start_time:
        start_time, end_time = end_time, start_time
    return [
        frame.frame_id
        for frame in frames
        if frame.timestamp is not None and start_time <= frame.timestamp <= end_time
    ]


def _segment_window(
    segment: dict[str, Any],
    *,
    margin_seconds: float,
) -> tuple[float | None, float | None]:
    start = _optional_float(segment.get("start_time"))
    end = _optional_float(segment.get("end_time"))
    points = [
        value
        for value in (_optional_float(point) for point in segment.get("timestamp_points", []))
        if value is not None
    ]
    if start is None and points:
        start = min(points)
    if end is None and points:
        end = max(points)
    if start is None and end is None:
        center = _optional_float(segment.get("timestamp_center"))
        if center is not None:
            start = center
            end = center
    if start is None or end is None:
        return start, end
    if end < start:
        start, end = end, start
    margin = max(0.0, margin_seconds)
    return max(0.0, start - margin), end + margin


def _segment_center(
    segment: dict[str, Any],
    *,
    start_time: float | None,
    end_time: float | None,
) -> float | None:
    center = _optional_float(segment.get("timestamp_center"))
    if center is not None:
        return center
    if start_time is not None and end_time is not None:
        return (start_time + end_time) / 2.0
    return start_time if start_time is not None else end_time


def _segment_visual_hints(segment: dict[str, Any]) -> list[str]:
    raw_hints = segment.get("mention_candidates")
    if isinstance(raw_hints, list):
        return [str(hint) for hint in raw_hints if str(hint).strip()]
    text = " ".join(
        str(value)
        for value in (
            segment.get("transcript_text"),
            segment.get("normalized_text"),
            segment.get("question"),
        )
        if value is not None
    )
    return mention_candidates(text) if text else []


def _frame_to_segments(segment_rows: list[_SegmentRow]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for segment in segment_rows:
        for frame_id in segment.frame_ids:
            mapping.setdefault(frame_id, []).append(segment.segment_id)
    return mapping


def _rank_segment_frames(
    *,
    segment: _SegmentRow,
    frames_by_id: dict[str, _FrameRow],
) -> list[str]:
    def sort_key(frame_id: str) -> tuple[bool, float, int]:
        frame = frames_by_id[frame_id]
        if frame.timestamp is None or segment.center_time is None:
            distance = float(frame.index)
        else:
            distance = abs(frame.timestamp - segment.center_time)
        return frame.timestamp is None, distance, frame.index

    return sorted(segment.frame_ids, key=sort_key)


def _temporal_spread_order(
    frame_rows: list[_FrameRow],
    *,
    max_count: int,
) -> list[_FrameRow]:
    ordered = sorted(
        frame_rows,
        key=lambda frame: (
            frame.timestamp is None,
            frame.timestamp if frame.timestamp is not None else frame.index,
            frame.index,
        ),
    )
    if max_count <= 0 or len(ordered) <= 2:
        return ordered
    anchor_count = min(max_count, len(ordered))
    if anchor_count == 1:
        anchor_indices = [0]
    else:
        anchor_indices = [
            round(position * (len(ordered) - 1) / (anchor_count - 1))
            for position in range(anchor_count)
        ]
    seen_indices = set()
    spread = []
    for index in anchor_indices:
        if index not in seen_indices:
            spread.append(ordered[index])
            seen_indices.add(index)
    spread.extend(frame for index, frame in enumerate(ordered) if index not in seen_indices)
    return spread


def _pre_excluded_frames(
    frame_rows: list[_FrameRow],
    *,
    config: VLMFrameCandidateConfig,
) -> dict[str, str]:
    excluded: dict[str, str] = {}
    seen_paths: set[str] = set()
    kept_timestamps: list[float] = []
    for frame in sorted(
        frame_rows,
        key=lambda row: (
            row.timestamp is None,
            row.timestamp if row.timestamp is not None else row.index,
            row.index,
        ),
    ):
        if _is_low_information(frame, config=config):
            excluded[frame.frame_id] = "low_information"
            continue
        if _has_duplicate_signal(frame, config=config):
            excluded[frame.frame_id] = "near_duplicate"
            continue
        if frame.frame_path is not None:
            if frame.frame_path in seen_paths:
                excluded[frame.frame_id] = "near_duplicate"
                continue
            seen_paths.add(frame.frame_path)
        if frame.timestamp is not None:
            if any(
                abs(frame.timestamp - timestamp) <= config.duplicate_timestamp_epsilon_seconds
                for timestamp in kept_timestamps
            ):
                excluded[frame.frame_id] = "near_duplicate"
                continue
            kept_timestamps.append(frame.timestamp)
    return excluded


def _is_low_information(frame: _FrameRow, *, config: VLMFrameCandidateConfig) -> bool:
    selection = frame.payload.get("selection")
    if not isinstance(selection, dict):
        return False
    contrast = _optional_float(selection.get("contrast"))
    detail_score = _optional_float(selection.get("detail_score"))
    if contrast is None or detail_score is None:
        return False
    return (
        contrast < config.low_information_min_contrast
        and detail_score < config.low_information_min_detail_score
    )


def _has_duplicate_signal(frame: _FrameRow, *, config: VLMFrameCandidateConfig) -> bool:
    selection = frame.payload.get("selection")
    if not isinstance(selection, dict):
        return False
    distance = _optional_float(selection.get("nearest_selected_distance"))
    return distance is not None and distance < config.duplicate_distance_threshold


def _reason_counts(excluded_frames: list[dict[str, Any]]) -> dict[str, int]:
    reasons = {
        "near_duplicate": 0,
        "low_information": 0,
        "segment_cap": 0,
        "window_cap": 0,
        "min_time_gap": 0,
        "max_candidates": 0,
        "outside_segment_coverage": 0,
        "not_selected": 0,
    }
    for excluded in excluded_frames:
        reason = str(excluded.get("reason", "not_selected"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


def _window_set(timestamps: list[float], window_seconds: float) -> set[int]:
    return {
        window_index
        for window_index in (_window_index(timestamp, window_seconds) for timestamp in timestamps)
        if window_index is not None
    }


def _window_index(timestamp: float | None, window_seconds: float) -> int | None:
    if timestamp is None or not math.isfinite(window_seconds) or window_seconds <= 0:
        return None
    return int(timestamp // window_seconds)


def _time_span(timestamps: list[float]) -> float | None:
    if not timestamps:
        return None
    return round(max(timestamps) - min(timestamps), 3)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _recall_risk(
    *,
    reduction_ratio: float,
    selected_count: int,
    input_count: int,
    segment_coverage_ratio: float | None,
    window_coverage_ratio: float | None,
) -> str:
    if input_count > 0 and selected_count == 0:
        return "high"
    if segment_coverage_ratio is not None and segment_coverage_ratio < 0.5:
        return "high"
    if segment_coverage_ratio is not None and segment_coverage_ratio < 0.8:
        return "medium"
    if window_coverage_ratio is not None and window_coverage_ratio < 0.5:
        return "medium"
    if reduction_ratio >= 0.8:
        return "medium"
    if reduction_ratio > 0:
        return "low"
    return "none"


def _validate_config(config: VLMFrameCandidateConfig) -> None:
    if config.max_candidates < 0:
        raise ValueError("max_candidates must be >= 0")
    if config.max_per_segment < 0:
        raise ValueError("max_per_segment must be >= 0")
    if config.max_per_window < 0:
        raise ValueError("max_per_window must be >= 0")
    if config.window_seconds < 0:
        raise ValueError("window_seconds must be >= 0")
    if config.min_time_gap_seconds < 0:
        raise ValueError("min_time_gap_seconds must be >= 0")
    if config.duplicate_timestamp_epsilon_seconds < 0:
        raise ValueError("duplicate_timestamp_epsilon_seconds must be >= 0")


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} JSONL not found: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            rows.append(payload)
    return rows


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _update_project_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    payload = _read_manifest(manifest_path)
    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts[VLM_FRAME_CANDIDATES_ARTIFACT] = str(output_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts[VLM_FRAME_CANDIDATES_ARTIFACT] = summary["counts"]["selected_candidate_count"]

    payload["vlm_frame_candidate_selection"] = summary
    write_json(manifest_path, payload)


def _resolve_segments_path(*, project_dir: Path, candidate: Path | None) -> Path | None:
    if candidate is not None:
        return _resolve_path(
            project_dir=project_dir,
            candidate=candidate,
            default=project_dir / "segments" / "lecture_segments_aligned.jsonl",
        )
    aligned = project_dir / "segments" / "lecture_segments_aligned.jsonl"
    if aligned.exists():
        return aligned
    raw = project_dir / "segments" / "lecture_segments.jsonl"
    return raw if raw.exists() else None


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
