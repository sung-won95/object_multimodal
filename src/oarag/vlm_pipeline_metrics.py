from __future__ import annotations

from typing import Any, Mapping

from .schemas import (
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT,
    VLM_FRAME_CANDIDATES_ARTIFACT,
)


PIPELINE_STAGE_CANDIDATES = "frame_candidates"
PIPELINE_STAGE_VISUAL_OBSERVATIONS = "visual_observations"
PIPELINE_STAGE_AUDIO_VISUAL_CONSISTENCY = "audio_visual_consistency"


def extract_vlm_smoke_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, CI-friendly metric view from a VLM pipeline summary."""
    stages = _mapping(summary.get("stages"))
    candidate_stage = _mapping(stages.get(PIPELINE_STAGE_CANDIDATES))
    observation_stage = _mapping(stages.get(PIPELINE_STAGE_VISUAL_OBSERVATIONS))
    consistency_stage = _mapping(stages.get(PIPELINE_STAGE_AUDIO_VISUAL_CONSISTENCY))

    candidate_counts = _candidate_counts(candidate_stage)
    observation_counts = _mapping(observation_stage.get("counts"))
    consistency_counts = _mapping(consistency_stage.get("counts"))

    input_frame_count = _int(candidate_counts.get("input_frame_count"), default=0)
    selected_candidate_count = _int(
        candidate_counts.get("selected_candidate_count"),
        candidate_counts.get(VLM_FRAME_CANDIDATES_ARTIFACT),
        default=0,
    )

    return {
        "pipeline_status": str(summary.get("status") or ""),
        "pipeline_elapsed_seconds": _float(summary.get("elapsed_seconds")),
        "candidate_frame_reduction_ratio": _candidate_reduction_ratio(
            candidate_stage=candidate_stage,
            input_frame_count=input_frame_count,
            selected_candidate_count=selected_candidate_count,
        ),
        "input_frame_count": input_frame_count,
        "selected_candidate_count": selected_candidate_count,
        "frames_total": _int(observation_counts.get("frames_total"), default=0),
        "frames_processed": _int(observation_counts.get("frames_processed"), default=0),
        "frames_failed": _int(observation_counts.get("frames_failed"), default=0),
        "frames_skipped": _int(observation_counts.get("frames_skipped_resumed"), default=0),
        "average_frame_latency_seconds": _float(
            observation_stage.get("average_frame_latency_seconds")
        ),
        "visual_observation_elapsed_seconds": _float(observation_stage.get("elapsed_seconds")),
        "frame_status_counts": _int_dict(observation_stage.get("frame_status_counts")),
        "audio_visual_consistency_records": _int(
            consistency_counts.get(AUDIO_VISUAL_CONSISTENCY_ARTIFACT),
            default=0,
        ),
        "audio_visual_status_counts": _int_dict(consistency_stage.get("status_counts")),
        "audio_visual_consistency_counts": _int_dict(
            consistency_stage.get("consistency_counts")
        ),
    }


def _candidate_counts(candidate_stage: Mapping[str, Any]) -> Mapping[str, Any]:
    nested_summary = _mapping(candidate_stage.get("summary"))
    summary_counts = _mapping(nested_summary.get("counts"))
    if summary_counts:
        return summary_counts
    return _mapping(candidate_stage.get("counts"))


def _candidate_reduction_ratio(
    *,
    candidate_stage: Mapping[str, Any],
    input_frame_count: int,
    selected_candidate_count: int,
) -> float | None:
    summary = _mapping(candidate_stage.get("summary"))
    tradeoff = _mapping(summary.get("tradeoff"))
    candidate_plan = _mapping(candidate_stage.get("candidate_plan"))
    plan_tradeoff = _mapping(candidate_plan.get("tradeoff"))
    configured_ratio = _float(
        tradeoff.get("estimated_vlm_cost_reduction_ratio"),
        plan_tradeoff.get("estimated_vlm_cost_reduction_ratio"),
    )
    if configured_ratio is not None:
        return configured_ratio
    if input_frame_count <= 0:
        return None
    return round((input_frame_count - selected_candidate_count) / input_frame_count, 4)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _int(raw_value, default=0) for key, raw_value in value.items()}
