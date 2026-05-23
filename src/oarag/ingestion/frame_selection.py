from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


FRAME_SELECTION_STRATEGIES = {"none", "representative"}
DEFAULT_DUPLICATE_DISTANCE_THRESHOLD = 0.025
DEFAULT_MIN_CONTRAST = 0.015
DEFAULT_MIN_DETAIL_SCORE = 0.003
DEFAULT_MIN_TEMPORAL_COVERAGE_RATIO = 0.8
DEFAULT_MAX_FRAME_FREE_SEGMENT_RATIO = 0.2
ANALYSIS_SIZE = 32


@dataclass(frozen=True)
class FrameSelectionConfig:
    strategy: str = "none"
    duplicate_distance_threshold: float = DEFAULT_DUPLICATE_DISTANCE_THRESHOLD
    min_contrast: float = DEFAULT_MIN_CONTRAST
    min_detail_score: float = DEFAULT_MIN_DETAIL_SCORE


@dataclass(frozen=True)
class FrameSignal:
    frame_id: str
    brightness: float
    contrast: float
    detail_score: float
    pixels: tuple[int, ...]
    width: int
    height: int

    def to_manifest(self) -> dict[str, Any]:
        return {
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "detail_score": round(self.detail_score, 4),
        }


FrameAnalyzer = Callable[[dict[str, Any]], FrameSignal]


def summarize_frame_temporal_coverage(
    *,
    frames: list[dict[str, Any]],
    duration_sec: float | None,
    frame_rate: float,
    segments: list[dict[str, Any]] | None = None,
    min_temporal_coverage_ratio: float = DEFAULT_MIN_TEMPORAL_COVERAGE_RATIO,
    max_frame_free_segment_ratio: float = DEFAULT_MAX_FRAME_FREE_SEGMENT_RATIO,
) -> dict[str, Any]:
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than 0")

    timestamps = [
        timestamp
        for timestamp in (_optional_float(frame.get("timestamp")) for frame in frames)
        if timestamp is not None
    ]
    first_timestamp = min(timestamps) if timestamps else None
    last_timestamp = max(timestamps) if timestamps else None
    timestamp_span = (
        round(last_timestamp - first_timestamp, 3)
        if first_timestamp is not None and last_timestamp is not None
        else None
    )

    covered_until = None
    coverage_ratio = None
    if duration_sec is not None and duration_sec > 0 and last_timestamp is not None:
        covered_until = round(min(duration_sec, last_timestamp + (1 / frame_rate)), 3)
        coverage_ratio = round(min(1.0, covered_until / duration_sec), 4)

    segment_coverage = summarize_segment_frame_coverage(
        frames=frames,
        segments=segments or [],
    )
    frame_free_segment_ratio = segment_coverage["frame_free_segment_ratio"]
    warnings = _coverage_warnings(
        temporal_coverage_ratio=coverage_ratio,
        frame_free_segment_ratio=frame_free_segment_ratio,
        min_temporal_coverage_ratio=min_temporal_coverage_ratio,
        max_frame_free_segment_ratio=max_frame_free_segment_ratio,
    )
    return {
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "covered_until_sec": covered_until,
        "timestamp_span_sec": timestamp_span,
        "temporal_coverage_ratio": coverage_ratio,
        "segment_coverage": segment_coverage,
        "frame_free_segment_ratio": frame_free_segment_ratio,
        "warnings": warnings,
    }


def summarize_segment_frame_coverage(
    *,
    frames: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    segment_windows = [
        window for window in (_segment_window(segment) for segment in segments) if window is not None
    ]
    frame_timestamps = [
        timestamp
        for timestamp in (_optional_float(frame.get("timestamp")) for frame in frames)
        if timestamp is not None
    ]
    segments_with_frames = 0
    for start, end in segment_windows:
        if any(start <= timestamp <= end for timestamp in frame_timestamps):
            segments_with_frames += 1

    segment_count = len(segment_windows)
    segments_without_frames = max(0, segment_count - segments_with_frames)
    coverage_ratio = round(segments_with_frames / segment_count, 4) if segment_count else None
    frame_free_ratio = round(segments_without_frames / segment_count, 4) if segment_count else None
    return {
        "segment_count": segment_count,
        "segments_with_frames": segments_with_frames,
        "segments_without_frames": segments_without_frames,
        "segment_frame_coverage_ratio": coverage_ratio,
        "frame_free_segment_ratio": frame_free_ratio,
    }


def select_representative_frames(
    frames: list[dict[str, Any]],
    *,
    config: FrameSelectionConfig | None = None,
    analyzer: FrameAnalyzer | None = None,
) -> dict[str, Any]:
    resolved_config = config or FrameSelectionConfig()
    strategy = resolved_config.strategy.strip().lower()
    if strategy not in FRAME_SELECTION_STRATEGIES:
        raise ValueError(f"frame_selection must be one of {sorted(FRAME_SELECTION_STRATEGIES)}")

    if strategy == "none":
        return {
            "frames": frames,
            "dropped_frame_paths": [],
            "summary": _selection_summary(
                strategy=strategy,
                status="disabled",
                candidate_count=len(frames),
                selected_count=len(frames),
                drop_reasons=_empty_drop_reasons(),
                signals=[],
                config=resolved_config,
            ),
        }

    if not frames:
        return {
            "frames": [],
            "dropped_frame_paths": [],
            "summary": _selection_summary(
                strategy=strategy,
                status="selected",
                candidate_count=0,
                selected_count=0,
                drop_reasons=_empty_drop_reasons(),
                signals=[],
                config=resolved_config,
            ),
        }

    resolved_analyzer = analyzer or analyze_frame_file
    try:
        signals = [resolved_analyzer(frame) for frame in frames]
    except Exception as exc:
        return {
            "frames": frames,
            "dropped_frame_paths": [],
            "summary": _selection_summary(
                strategy=strategy,
                status="analysis_failed",
                candidate_count=len(frames),
                selected_count=len(frames),
                drop_reasons=_empty_drop_reasons(),
                signals=[],
                config=resolved_config,
                error=str(exc),
            ),
        }

    selected_frames: list[dict[str, Any]] = []
    selected_signals: list[FrameSignal] = []
    dropped_paths: list[str] = []
    dropped_reasons_by_index: dict[int, str] = {}
    drop_reasons = _empty_drop_reasons()

    for index, (frame, signal) in enumerate(zip(frames, signals, strict=True)):
        nearest_distance = _nearest_distance(signal, selected_signals)
        reason = _drop_reason(
            signal=signal,
            nearest_selected_distance=nearest_distance,
            config=resolved_config,
        )
        if reason is not None:
            drop_reasons[reason] += 1
            dropped_reasons_by_index[index] = reason
            frame_path = frame.get("frame_path")
            if frame_path is not None:
                dropped_paths.append(str(frame_path))
            continue

        selected_frames.append(
            _with_selection_metadata(
                frame=frame,
                signal=signal,
                rank=len(selected_frames) + 1,
                nearest_selected_distance=nearest_distance,
            )
        )
        selected_signals.append(signal)

    fallback_kept = False
    if not selected_frames:
        fallback_index = _best_signal_index(signals)
        fallback_kept = True
        fallback_reason = dropped_reasons_by_index.pop(fallback_index, None)
        if fallback_reason is not None:
            drop_reasons[fallback_reason] -= 1
        dropped_paths = [
            str(frame.get("frame_path"))
            for index, frame in enumerate(frames)
            if index in dropped_reasons_by_index and frame.get("frame_path") is not None
        ]
        selected_frames = [
            _with_selection_metadata(
                frame=frames[fallback_index],
                signal=signals[fallback_index],
                rank=1,
                nearest_selected_distance=None,
                fallback_kept=True,
            )
        ]

    summary = _selection_summary(
        strategy=strategy,
        status="selected",
        candidate_count=len(frames),
        selected_count=len(selected_frames),
        drop_reasons=drop_reasons,
        signals=signals,
        config=resolved_config,
        fallback_kept=fallback_kept,
    )
    return {
        "frames": selected_frames,
        "dropped_frame_paths": dropped_paths,
        "summary": summary,
    }


def analyze_frame_file(frame: dict[str, Any]) -> FrameSignal:
    frame_path = Path(str(frame.get("frame_path", ""))).expanduser()
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image not found: {frame_path}")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(frame_path),
        "-vf",
        f"scale={ANALYSIS_SIZE}:{ANALYSIS_SIZE},format=gray",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Required command not found: ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Frame analysis failed for {frame_path}: {stderr}") from exc

    return analyze_pixels(
        frame=frame,
        pixels=result.stdout,
        width=ANALYSIS_SIZE,
        height=ANALYSIS_SIZE,
    )


def analyze_pixels(
    *,
    frame: dict[str, Any],
    pixels: bytes | Sequence[int],
    width: int,
    height: int,
) -> FrameSignal:
    values = tuple(int(value) for value in pixels)
    expected_len = width * height
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} grayscale pixels, got {len(values)}")
    if not values:
        raise ValueError("Expected at least one grayscale pixel")

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    contrast = math.sqrt(variance) / 255.0

    diff_total = 0
    diff_count = 0
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            value = values[row_offset + x]
            if x + 1 < width:
                diff_total += abs(value - values[row_offset + x + 1])
                diff_count += 1
            if y + 1 < height:
                diff_total += abs(value - values[row_offset + width + x])
                diff_count += 1
    detail_score = (diff_total / diff_count / 255.0) if diff_count else 0.0

    frame_id = str(frame.get("frame_id") or Path(str(frame.get("frame_path", ""))).stem)
    return FrameSignal(
        frame_id=frame_id,
        brightness=mean_value / 255.0,
        contrast=contrast,
        detail_score=detail_score,
        pixels=values,
        width=width,
        height=height,
    )


def frame_distance(left: FrameSignal, right: FrameSignal) -> float:
    limit = min(len(left.pixels), len(right.pixels))
    if limit == 0:
        return 1.0
    return sum(abs(left.pixels[index] - right.pixels[index]) for index in range(limit)) / (
        limit * 255.0
    )


def _drop_reason(
    *,
    signal: FrameSignal,
    nearest_selected_distance: float | None,
    config: FrameSelectionConfig,
) -> str | None:
    if signal.contrast < config.min_contrast and signal.detail_score < config.min_detail_score:
        return "low_information"
    if (
        nearest_selected_distance is not None
        and nearest_selected_distance < config.duplicate_distance_threshold
    ):
        return "near_duplicate"
    return None


def _nearest_distance(signal: FrameSignal, selected_signals: list[FrameSignal]) -> float | None:
    if not selected_signals:
        return None
    return min(frame_distance(signal, selected) for selected in selected_signals)


def _best_signal_index(signals: list[FrameSignal]) -> int:
    return max(
        range(len(signals)),
        key=lambda index: (
            signals[index].detail_score,
            signals[index].contrast,
            -index,
        ),
    )


def _with_selection_metadata(
    *,
    frame: dict[str, Any],
    signal: FrameSignal,
    rank: int,
    nearest_selected_distance: float | None,
    fallback_kept: bool = False,
) -> dict[str, Any]:
    payload = dict(frame)
    payload["selection"] = {
        "strategy": "representative",
        "rank": rank,
        "fallback_kept": fallback_kept,
        "nearest_selected_distance": _round_optional(nearest_selected_distance),
        **signal.to_manifest(),
    }
    return payload


def _selection_summary(
    *,
    strategy: str,
    status: str,
    candidate_count: int,
    selected_count: int,
    drop_reasons: dict[str, int],
    signals: list[FrameSignal],
    config: FrameSelectionConfig,
    fallback_kept: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    dropped_count = max(0, candidate_count - selected_count)
    reduction_ratio = round(dropped_count / candidate_count, 4) if candidate_count else 0.0
    summary = {
        "strategy": strategy,
        "enabled": strategy != "none",
        "status": status,
        "candidate_frame_count": candidate_count,
        "selected_frame_count": selected_count,
        "dropped_frame_count": dropped_count,
        "drop_reasons": drop_reasons,
        "thresholds": {
            "duplicate_distance": config.duplicate_distance_threshold,
            "min_contrast": config.min_contrast,
            "min_detail_score": config.min_detail_score,
        },
        "quality_signals": _quality_summary(signals),
        "tradeoff": {
            "frame_reduction_ratio": reduction_ratio,
            "estimated_ocr_cost_reduction_ratio": reduction_ratio,
            "recall_risk": _recall_risk(reduction_ratio, fallback_kept=fallback_kept),
            "fallback_kept": fallback_kept,
        },
    }
    if error is not None:
        summary["error"] = error
    return summary


def _quality_summary(signals: list[FrameSignal]) -> dict[str, Any]:
    return {
        "brightness_mean": _mean_signal(signals, "brightness"),
        "contrast_mean": _mean_signal(signals, "contrast"),
        "detail_score_mean": _mean_signal(signals, "detail_score"),
    }


def _mean_signal(signals: list[FrameSignal], name: str) -> float | None:
    if not signals:
        return None
    return round(sum(float(getattr(signal, name)) for signal in signals) / len(signals), 4)


def _recall_risk(reduction_ratio: float, *, fallback_kept: bool) -> str:
    if fallback_kept:
        return "high"
    if reduction_ratio >= 0.5:
        return "medium"
    if reduction_ratio > 0:
        return "low"
    return "none"


def _empty_drop_reasons() -> dict[str, int]:
    return {
        "near_duplicate": 0,
        "low_information": 0,
    }


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _coverage_warnings(
    *,
    temporal_coverage_ratio: float | None,
    frame_free_segment_ratio: float | None,
    min_temporal_coverage_ratio: float,
    max_frame_free_segment_ratio: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if (
        temporal_coverage_ratio is not None
        and temporal_coverage_ratio < min_temporal_coverage_ratio
    ):
        warnings.append(
            {
                "code": "low_temporal_coverage",
                "severity": "warning",
                "metric": "temporal_coverage_ratio",
                "value": temporal_coverage_ratio,
                "threshold": min_temporal_coverage_ratio,
                "message": "Frame sampling does not cover enough of the video duration.",
            }
        )
    if (
        frame_free_segment_ratio is not None
        and frame_free_segment_ratio > max_frame_free_segment_ratio
    ):
        warnings.append(
            {
                "code": "high_frame_free_segment_ratio",
                "severity": "warning",
                "metric": "frame_free_segment_ratio",
                "value": frame_free_segment_ratio,
                "threshold": max_frame_free_segment_ratio,
                "message": "Frame sampling leaves too many transcript segments without nearby frames.",
            }
        )
    return warnings


def _segment_window(segment: dict[str, Any]) -> tuple[float, float] | None:
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
    return start, end


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
