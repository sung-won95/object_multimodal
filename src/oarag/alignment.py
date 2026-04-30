from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl


def align_segments_to_frames(
    *,
    project_id: str,
    output_root: Path,
    margin_seconds: float = 0.0,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    project_dir = output_root / project_id
    resolved_segments_path = segments_path or (project_dir / "segments" / "lecture_segments.jsonl")
    resolved_frames_manifest_path = frames_manifest_path or (
        project_dir / "manifests" / "frames_manifest.jsonl"
    )
    resolved_output_path = output_path or (
        project_dir / "segments" / "lecture_segments_aligned.jsonl"
    )
    resolved_manifest_path = manifest_path or (project_dir / "manifests" / "project_manifest.json")
    margin = max(0.0, margin_seconds)

    segments = _read_jsonl(resolved_segments_path)
    frames = _read_jsonl(resolved_frames_manifest_path)

    frame_points = [_frame_point(frame) for frame in frames]
    aligned_rows: list[dict[str, Any]] = []
    segments_with_frames = 0
    referenced_frames: set[str] = set()
    total_frame_refs = 0

    for segment in segments:
        row = dict(segment)
        window = _segment_window(row, margin_seconds=margin)
        frame_refs = []
        if window is not None:
            for frame_id, timestamp in frame_points:
                if timestamp is None:
                    continue
                if window[0] <= timestamp <= window[1]:
                    frame_refs.append(frame_id)
                    referenced_frames.add(frame_id)
        row["frame_refs"] = frame_refs
        aligned_rows.append(row)
        if frame_refs:
            segments_with_frames += 1
            total_frame_refs += len(frame_refs)

    write_jsonl(resolved_output_path, aligned_rows)

    summary = {
        "margin_seconds": margin,
        "segments_total": len(segments),
        "segments_with_frames": segments_with_frames,
        "segments_without_frames": len(segments) - segments_with_frames,
        "segment_frame_coverage_ratio": (
            round(segments_with_frames / len(segments), 4) if segments else None
        ),
        "frame_refs_total": total_frame_refs,
        "unique_frames_referenced": len(referenced_frames),
        "available_frames": len(frames),
        "available_frame_time_span_sec": _frame_time_span(frames),
    }
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        aligned_segments_path=resolved_output_path,
        summary=summary,
    )

    return {
        "project_id": project_id,
        "paths": {
            "segments": str(resolved_segments_path),
            "frames_manifest": str(resolved_frames_manifest_path),
            "aligned_segments": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "counts": summary,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            rows.append(payload)
    return rows


def _segment_window(segment: dict[str, Any], margin_seconds: float) -> tuple[float, float] | None:
    start = _optional_float(segment.get("start_time"))
    end = _optional_float(segment.get("end_time"))
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return (start - margin_seconds, end + margin_seconds)


def _frame_point(frame: dict[str, Any]) -> tuple[str, float | None]:
    frame_id = str(frame.get("frame_id") or _frame_id_from_path(frame.get("frame_path")))
    return frame_id, _optional_float(frame.get("timestamp"))


def _frame_id_from_path(value: Any) -> str:
    if value is None:
        return ""
    return Path(str(value)).stem


def _frame_time_span(frames: list[dict[str, Any]]) -> float | None:
    timestamps = [
        timestamp
        for timestamp in (_optional_float(frame.get("timestamp")) for frame in frames)
        if timestamp is not None
    ]
    if not timestamps:
        return None
    return round(max(timestamps) - min(timestamps), 3)


def _update_project_manifest(
    *,
    manifest_path: Path,
    aligned_segments_path: Path,
    summary: dict[str, Any],
) -> None:
    payload: dict[str, Any]
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["lecture_segments_aligned"] = str(aligned_segments_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["lecture_segments_aligned"] = summary["segments_total"]

    payload["alignment"] = summary
    write_json(manifest_path, payload)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
