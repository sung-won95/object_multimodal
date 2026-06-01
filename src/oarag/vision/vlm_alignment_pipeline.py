from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from oarag.vision.audio_visual_consistency import (
    AudioVisualConsistencyConfig,
    generate_audio_visual_consistency,
)
from oarag.retrieval.project_index import iter_jsonl_documents, segment_artifact_path
from oarag.core.schemas import (
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT,
    VLM_ARTIFACT_PATHS,
    VLM_FRAME_CANDIDATES_ARTIFACT,
    VLM_VISUAL_OBSERVATIONS_ARTIFACT,
)
from oarag.vision.vlm import (
    DEFAULT_VLM_BACKEND,
    make_vlm_backend,
    preflight_vlm_backend,
    run_vlm,
)
from oarag.vision.vlm_frame_candidates import (
    VLMFrameCandidateConfig,
    generate_vlm_frame_candidates,
    select_vlm_frame_candidates,
)
from oarag.vision.vlm_pipeline_metrics import extract_vlm_smoke_metrics


PIPELINE_STAGE_CANDIDATES = "frame_candidates"
PIPELINE_STAGE_VISUAL_OBSERVATIONS = "visual_observations"
PIPELINE_STAGE_AUDIO_VISUAL_CONSISTENCY = "audio_visual_consistency"


@dataclass(frozen=True)
class VLMAlignmentPipelineConfig:
    project_dir: Path
    vlm_backend: str = DEFAULT_VLM_BACKEND
    vlm_model: str | None = None
    vlm_device: str | None = None
    vlm_options: Mapping[str, Any] = field(default_factory=dict)
    frames_manifest_path: Path | None = None
    segments_path: Path | None = None
    frame_candidates_path: Path | None = None
    visual_observations_path: Path | None = None
    audio_visual_consistency_path: Path | None = None
    manifest_path: Path | None = None
    candidate_config: VLMFrameCandidateConfig = field(default_factory=VLMFrameCandidateConfig)
    consistency_config: AudioVisualConsistencyConfig = field(
        default_factory=AudioVisualConsistencyConfig
    )
    resume: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class _ResolvedPipelinePaths:
    project_dir: Path
    project_manifest: Path
    frames_manifest: Path
    segments: Path
    vlm_frame_candidates: Path
    vlm_visual_observations: Path
    audio_visual_consistency: Path


def run_vlm_alignment_pipeline(config: VLMAlignmentPipelineConfig) -> dict[str, Any]:
    started_at = time.perf_counter()
    project_dir = config.project_dir.expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    _validate_vlm_execution(backend=config.vlm_backend, model=config.vlm_model)
    if not config.dry_run:
        preflight_vlm_backend(
            project_dir=project_dir,
            backend=config.vlm_backend,
            model=config.vlm_model,
            device=config.vlm_device,
            options=config.vlm_options,
        )
    paths = _resolve_pipeline_paths(project_dir=project_dir, config=config)
    manifest = _read_json_object(paths.project_manifest) if paths.project_manifest.exists() else {}
    project_id = _optional_str(manifest.get("project_id")) or paths.project_dir.name
    video_id = _optional_str(manifest.get("video_id")) or project_id

    stage_summaries: dict[str, dict[str, Any]] = {}
    if config.dry_run:
        candidate_summary = _plan_frame_candidates(paths=paths, config=config)
        stage_summaries[PIPELINE_STAGE_CANDIDATES] = candidate_summary
        stage_summaries[PIPELINE_STAGE_VISUAL_OBSERVATIONS] = _plan_visual_observations(
            paths=paths,
            config=config,
            candidate_summary=candidate_summary,
        )
        stage_summaries[PIPELINE_STAGE_AUDIO_VISUAL_CONSISTENCY] = (
            _plan_audio_visual_consistency(paths=paths, config=config)
        )
    else:
        stage_summaries[PIPELINE_STAGE_CANDIDATES] = _run_frame_candidates(
            paths=paths,
            config=config,
        )
        stage_summaries[PIPELINE_STAGE_VISUAL_OBSERVATIONS] = _run_visual_observations(
            paths=paths,
            config=config,
        )
        stage_summaries[PIPELINE_STAGE_AUDIO_VISUAL_CONSISTENCY] = (
            _run_audio_visual_consistency(paths=paths, config=config)
        )

    elapsed_seconds = round(time.perf_counter() - started_at, 4)
    failure_by_stage = _failure_counts(stage_summaries)
    pipeline_summary = {
        "project_id": project_id,
        "video_id": video_id,
        "status": _pipeline_status(
            dry_run=config.dry_run,
            stage_summaries=stage_summaries,
        ),
        "dry_run": config.dry_run,
        "resume": config.resume,
        "elapsed_seconds": elapsed_seconds,
        "paths": _paths_dict(paths),
        "counts": _pipeline_counts(stage_summaries),
        "failures": {
            "count": sum(failure_by_stage.values()),
            "by_stage": failure_by_stage,
        },
        "stages": stage_summaries,
    }
    pipeline_summary["smoke_metrics"] = extract_vlm_smoke_metrics(pipeline_summary)
    return pipeline_summary


def _run_frame_candidates(
    *,
    paths: _ResolvedPipelinePaths,
    config: VLMAlignmentPipelineConfig,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if config.resume and paths.vlm_frame_candidates.exists():
        candidate_count = _count_jsonl(paths.vlm_frame_candidates)
        return {
            "status": "skipped_existing",
            "resume_reused": True,
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "artifact_path": str(paths.vlm_frame_candidates),
            "counts": {
                "selected_candidate_count": candidate_count,
                VLM_FRAME_CANDIDATES_ARTIFACT: candidate_count,
            },
            "failures": {"count": 0},
        }

    summary = generate_vlm_frame_candidates(
        project_dir=paths.project_dir,
        frames_manifest_path=paths.frames_manifest,
        segments_path=paths.segments,
        output_path=paths.vlm_frame_candidates,
        manifest_path=paths.project_manifest,
        config=config.candidate_config,
    )
    selected_count = int(summary["counts"]["selected_candidate_count"])
    return {
        "status": "completed",
        "resume_reused": False,
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "artifact_path": str(paths.vlm_frame_candidates),
        "counts": {
            **summary["counts"],
            VLM_FRAME_CANDIDATES_ARTIFACT: selected_count,
        },
        "failures": {"count": 0},
        "summary": summary["summary"],
    }


def _run_visual_observations(
    *,
    paths: _ResolvedPipelinePaths,
    config: VLMAlignmentPipelineConfig,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    summary = run_vlm(
        project_dir=paths.project_dir,
        backend=config.vlm_backend,
        model=config.vlm_model,
        device=config.vlm_device,
        options=config.vlm_options,
        frames_manifest_path=paths.frames_manifest,
        frame_candidates_path=paths.vlm_frame_candidates,
        output_path=paths.vlm_visual_observations,
        manifest_path=paths.project_manifest,
        resume=config.resume,
    )
    failures = int(summary["counts"]["frames_failed"])
    return {
        "status": summary["status"],
        "resume_enabled": config.resume,
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "artifact_path": str(paths.vlm_visual_observations),
        "counts": summary["counts"],
        "frame_status_counts": summary["frame_status_counts"],
        "average_frame_latency_seconds": summary["average_frame_latency_seconds"],
        "failures": {"count": failures},
    }


def _run_audio_visual_consistency(
    *,
    paths: _ResolvedPipelinePaths,
    config: VLMAlignmentPipelineConfig,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    summary = generate_audio_visual_consistency(
        project_dir=paths.project_dir,
        segments_path=paths.segments,
        observations_path=paths.vlm_visual_observations,
        output_path=paths.audio_visual_consistency,
        manifest_path=paths.project_manifest,
        config=config.consistency_config,
    )
    failures = int(summary["counts"]["source_failure_records"])
    return {
        "status": summary["status"],
        "rewrite_policy": "rewritten_each_run",
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "artifact_path": str(paths.audio_visual_consistency),
        "counts": summary["counts"],
        "status_counts": summary["status_counts"],
        "consistency_counts": summary["consistency_counts"],
        "failures": {"count": failures},
    }


def _plan_frame_candidates(
    *,
    paths: _ResolvedPipelinePaths,
    config: VLMAlignmentPipelineConfig,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if config.resume and paths.vlm_frame_candidates.exists():
        candidates = list(iter_jsonl_documents(paths.vlm_frame_candidates))
        return {
            "status": "would_skip_existing",
            "resume_reused": True,
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "artifact_path": str(paths.vlm_frame_candidates),
            "counts": {
                "selected_candidate_count": len(candidates),
                VLM_FRAME_CANDIDATES_ARTIFACT: len(candidates),
            },
            "candidate_plan": _candidate_plan_from_rows(candidates),
            "failures": {"count": 0},
        }

    frames = list(iter_jsonl_documents(paths.frames_manifest))
    segments = list(iter_jsonl_documents(paths.segments))
    manifest = _read_json_object(paths.project_manifest) if paths.project_manifest.exists() else {}
    project_id = _optional_str(manifest.get("project_id")) or paths.project_dir.name
    video_id = _optional_str(manifest.get("video_id")) or project_id
    result = select_vlm_frame_candidates(
        frames=frames,
        segments=segments,
        project_id=project_id,
        video_id=video_id,
        config=config.candidate_config,
    )
    selected_count = int(result["summary"]["counts"]["selected_candidate_count"])
    return {
        "status": "would_generate",
        "resume_reused": False,
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "artifact_path": str(paths.vlm_frame_candidates),
        "counts": {
            **result["summary"]["counts"],
            VLM_FRAME_CANDIDATES_ARTIFACT: selected_count,
        },
        "candidate_plan": {
            "selected_frame_ids": result["summary"]["selected_frame_ids"],
            "thresholds": result["summary"]["thresholds"],
            "coverage": result["summary"]["coverage"],
            "tradeoff": result["summary"]["tradeoff"],
            "exclusion_reasons": result["summary"]["counts"]["exclusion_reasons"],
        },
        "failures": {"count": 0},
    }


def _plan_visual_observations(
    *,
    paths: _ResolvedPipelinePaths,
    config: VLMAlignmentPipelineConfig,
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    candidate_count = int(candidate_summary.get("counts", {}).get(VLM_FRAME_CANDIDATES_ARTIFACT, 0))
    existing_count = (
        _count_jsonl(paths.vlm_visual_observations)
        if config.resume and paths.vlm_visual_observations.exists()
        else 0
    )
    frames_to_process = max(candidate_count - existing_count, 0) if config.resume else candidate_count
    return {
        "status": "would_run",
        "resume_enabled": config.resume,
        "artifact_path": str(paths.vlm_visual_observations),
        "counts": {
            "frames_total": candidate_count,
            "frames_processed": frames_to_process,
            "frames_skipped_resumed": existing_count if config.resume else 0,
            "vlm_visual_observations_existing": existing_count,
        },
        "failures": {"count": 0},
    }


def _plan_audio_visual_consistency(
    *,
    paths: _ResolvedPipelinePaths,
    config: VLMAlignmentPipelineConfig,
) -> dict[str, Any]:
    return {
        "status": "would_run",
        "rewrite_policy": "rewritten_each_run",
        "artifact_path": str(paths.audio_visual_consistency),
        "counts": {
            "segments_total": _count_jsonl(paths.segments),
            "vlm_visual_observations_existing": (
                _count_jsonl(paths.vlm_visual_observations)
                if paths.vlm_visual_observations.exists()
                else 0
            ),
        },
        "failures": {"count": 0},
    }


def _resolve_pipeline_paths(
    *,
    project_dir: Path,
    config: VLMAlignmentPipelineConfig,
) -> _ResolvedPipelinePaths:
    project_manifest = _resolve_path(
        project_dir=project_dir,
        candidate=config.manifest_path,
        default=project_dir / "manifests" / "project_manifest.json",
    )
    frames_manifest = _resolve_path(
        project_dir=project_dir,
        candidate=config.frames_manifest_path,
        default=project_dir / "manifests" / "frames_manifest.jsonl",
    )
    if not frames_manifest.exists():
        raise FileNotFoundError(f"frames_manifest JSONL not found: {frames_manifest}")

    segments = segment_artifact_path(project_dir, segments=config.segments_path)
    frame_candidates = _resolve_path(
        project_dir=project_dir,
        candidate=config.frame_candidates_path,
        default=project_dir / VLM_ARTIFACT_PATHS[VLM_FRAME_CANDIDATES_ARTIFACT],
    )
    visual_observations = _resolve_path(
        project_dir=project_dir,
        candidate=config.visual_observations_path,
        default=project_dir / VLM_ARTIFACT_PATHS[VLM_VISUAL_OBSERVATIONS_ARTIFACT],
    )
    consistency = _resolve_path(
        project_dir=project_dir,
        candidate=config.audio_visual_consistency_path,
        default=project_dir / VLM_ARTIFACT_PATHS[AUDIO_VISUAL_CONSISTENCY_ARTIFACT],
    )
    return _ResolvedPipelinePaths(
        project_dir=project_dir,
        project_manifest=project_manifest,
        frames_manifest=frames_manifest,
        segments=segments,
        vlm_frame_candidates=frame_candidates,
        vlm_visual_observations=visual_observations,
        audio_visual_consistency=consistency,
    )


def _validate_vlm_execution(*, backend: str, model: str | None) -> None:
    if _optional_str(model) is None:
        raise ValueError("--vlm-model is required for VLM alignment pipeline")
    make_vlm_backend(backend)


def _pipeline_status(
    *,
    dry_run: bool,
    stage_summaries: dict[str, dict[str, Any]],
) -> str:
    if dry_run:
        return "planned"
    if any(stage.get("failures", {}).get("count", 0) for stage in stage_summaries.values()):
        return "completed_with_errors"
    if any(stage.get("status") == "completed_with_skips" for stage in stage_summaries.values()):
        return "completed_with_skips"
    return "completed"


def _pipeline_counts(stage_summaries: dict[str, dict[str, Any]]) -> dict[str, int]:
    candidates = stage_summaries.get(PIPELINE_STAGE_CANDIDATES, {})
    observations = stage_summaries.get(PIPELINE_STAGE_VISUAL_OBSERVATIONS, {})
    consistency = stage_summaries.get(PIPELINE_STAGE_AUDIO_VISUAL_CONSISTENCY, {})
    candidate_counts = candidates.get("counts", {})
    observation_counts = observations.get("counts", {})
    consistency_counts = consistency.get("counts", {})
    return {
        VLM_FRAME_CANDIDATES_ARTIFACT: int(
            candidate_counts.get(VLM_FRAME_CANDIDATES_ARTIFACT, 0)
        ),
        "vlm_frames_total": int(observation_counts.get("frames_total", 0)),
        "vlm_frames_processed": int(observation_counts.get("frames_processed", 0)),
        "vlm_frames_failed": int(observation_counts.get("frames_failed", 0)),
        VLM_VISUAL_OBSERVATIONS_ARTIFACT: int(
            observation_counts.get(VLM_VISUAL_OBSERVATIONS_ARTIFACT, 0)
        ),
        AUDIO_VISUAL_CONSISTENCY_ARTIFACT: int(
            consistency_counts.get(AUDIO_VISUAL_CONSISTENCY_ARTIFACT, 0)
        ),
        "consistency_source_failures": int(
            consistency_counts.get("source_failure_records", 0)
        ),
    }


def _failure_counts(stage_summaries: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        stage_name: int(summary.get("failures", {}).get("count", 0))
        for stage_name, summary in stage_summaries.items()
    }


def _paths_dict(paths: _ResolvedPipelinePaths) -> dict[str, str]:
    return {
        "project_dir": str(paths.project_dir),
        "project_manifest": str(paths.project_manifest),
        "frames_manifest": str(paths.frames_manifest),
        "segments": str(paths.segments),
        VLM_FRAME_CANDIDATES_ARTIFACT: str(paths.vlm_frame_candidates),
        VLM_VISUAL_OBSERVATIONS_ARTIFACT: str(paths.vlm_visual_observations),
        AUDIO_VISUAL_CONSISTENCY_ARTIFACT: str(paths.audio_visual_consistency),
    }


def _candidate_plan_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_frame_ids": [
            str(row.get("frame_id"))
            for row in sorted(rows, key=lambda row: int(row.get("rank") or 0))
            if row.get("frame_id") is not None
        ]
    }


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl_documents(path))


def _read_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
