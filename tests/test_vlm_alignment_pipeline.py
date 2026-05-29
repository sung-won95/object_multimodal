import json
from pathlib import Path

import pytest

from oarag.vlm_alignment_pipeline import (
    PIPELINE_STAGE_CANDIDATES,
    PIPELINE_STAGE_VISUAL_OBSERVATIONS,
    VLMAlignmentPipelineConfig,
    run_vlm_alignment_pipeline,
)
from oarag.vlm_frame_candidates import VLMFrameCandidateConfig


def test_run_vlm_alignment_pipeline_dry_run_does_not_write_artifacts(tmp_path: Path) -> None:
    project_dir = _make_project(tmp_path)

    summary = run_vlm_alignment_pipeline(
        VLMAlignmentPipelineConfig(
            project_dir=project_dir,
            vlm_model="stub-vlm",
            candidate_config=VLMFrameCandidateConfig(
                max_candidates=1,
                max_per_segment=1,
                min_time_gap_seconds=0.0,
            ),
            dry_run=True,
        )
    )

    assert summary["status"] == "planned"
    assert summary["dry_run"] is True
    assert summary["counts"]["vlm_frame_candidates"] == 1
    assert summary["smoke_metrics"]["selected_candidate_count"] == 1
    assert summary["smoke_metrics"]["candidate_frame_reduction_ratio"] == 0.6667
    assert summary["stages"][PIPELINE_STAGE_CANDIDATES]["status"] == "would_generate"
    assert summary["stages"][PIPELINE_STAGE_VISUAL_OBSERVATIONS]["status"] == "would_run"
    assert not (project_dir / "manifests" / "vlm_frame_candidates.jsonl").exists()
    assert not (project_dir / "manifests" / "vlm_visual_observations.jsonl").exists()
    assert not (project_dir / "manifests" / "audio_visual_consistency.jsonl").exists()


def test_run_vlm_alignment_pipeline_generates_all_artifacts(tmp_path: Path) -> None:
    project_dir = _make_project(tmp_path)

    summary = run_vlm_alignment_pipeline(
        VLMAlignmentPipelineConfig(
            project_dir=project_dir,
            vlm_model="stub-vlm",
            vlm_options={
                "description_prefix": "Matrix observation",
                "detected_text": "matrix determinant",
            },
            candidate_config=VLMFrameCandidateConfig(
                max_candidates=3,
                max_per_segment=3,
                min_time_gap_seconds=0.0,
            ),
        )
    )

    candidates = _read_jsonl(project_dir / "manifests" / "vlm_frame_candidates.jsonl")
    observations = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")
    consistency = _read_jsonl(project_dir / "manifests" / "audio_visual_consistency.jsonl")
    manifest = json.loads(
        (project_dir / "manifests" / "project_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert summary["counts"]["vlm_frame_candidates"] == 3
    assert summary["counts"]["vlm_visual_observations"] == 3
    assert summary["counts"]["audio_visual_consistency"] == 3
    assert summary["smoke_metrics"]["candidate_frame_reduction_ratio"] == 0.0
    assert summary["smoke_metrics"]["selected_candidate_count"] == 3
    assert summary["smoke_metrics"]["frames_processed"] == 3
    assert summary["smoke_metrics"]["audio_visual_consistency_records"] == 3
    assert summary["smoke_metrics"]["average_frame_latency_seconds"] is not None
    assert candidates[0]["status"] == "selected"
    assert observations[0]["status"] == "success"
    assert consistency[0]["consistency"] == "aligned"
    assert manifest["artifacts"]["vlm_frame_candidates"].endswith("vlm_frame_candidates.jsonl")
    assert manifest["artifacts"]["vlm_visual_observations"].endswith(
        "vlm_visual_observations.jsonl"
    )
    assert manifest["artifacts"]["audio_visual_consistency"].endswith(
        "audio_visual_consistency.jsonl"
    )


def test_run_vlm_alignment_pipeline_accepts_jsonl_backend_fixture(tmp_path: Path) -> None:
    project_dir = _make_project(tmp_path)
    fixture = project_dir / "manifests" / "vlm_backend_fixture.jsonl"
    _write_jsonl(
        fixture,
        [
            {
                "frame_id": frame_id,
                "parser_version": "pipeline-jsonl-parser-v1",
                "source_model": "fixture-vlm",
                "observations": [
                    {
                        "observation_type": "diagram",
                        "visual_description": f"Fixture observation for {frame_id}",
                        "detected_text": "matrix determinant",
                        "confidence": 0.8,
                    }
                ],
            }
            for frame_id in ["frame_000000", "frame_000001", "frame_000002"]
        ],
    )

    summary = run_vlm_alignment_pipeline(
        VLMAlignmentPipelineConfig(
            project_dir=project_dir,
            vlm_backend="jsonl",
            vlm_model="fixture-fallback",
            vlm_options={"jsonl_path": "manifests/vlm_backend_fixture.jsonl"},
            candidate_config=VLMFrameCandidateConfig(
                max_candidates=3,
                max_per_segment=3,
                min_time_gap_seconds=0.0,
            ),
        )
    )

    observations = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")
    manifest = json.loads(
        (project_dir / "manifests" / "project_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert summary["counts"]["vlm_visual_observations"] == 3
    assert {row["backend"] for row in observations} == {"jsonl"}
    assert {row["source_model"] for row in observations} == {"fixture-vlm"}
    assert observations[0]["metadata"]["parser_version"] == "pipeline-jsonl-parser-v1"
    assert manifest["vlm_consistency"]["settings"]["options"]["jsonl_path"] == "<configured>"


def test_run_vlm_alignment_pipeline_resume_reuses_existing_candidates_and_observations(
    tmp_path: Path,
) -> None:
    project_dir = _make_project(tmp_path)
    first = run_vlm_alignment_pipeline(
        VLMAlignmentPipelineConfig(
            project_dir=project_dir,
            vlm_model="stub-vlm",
            candidate_config=VLMFrameCandidateConfig(
                max_candidates=1,
                max_per_segment=1,
                min_time_gap_seconds=0.0,
            ),
        )
    )
    second = run_vlm_alignment_pipeline(
        VLMAlignmentPipelineConfig(
            project_dir=project_dir,
            vlm_model="stub-vlm",
            candidate_config=VLMFrameCandidateConfig(
                max_candidates=1,
                max_per_segment=1,
                min_time_gap_seconds=0.0,
            ),
            resume=True,
        )
    )

    assert first["counts"]["vlm_frame_candidates"] == 1
    assert second["stages"][PIPELINE_STAGE_CANDIDATES]["status"] == "skipped_existing"
    assert second["stages"][PIPELINE_STAGE_VISUAL_OBSERVATIONS]["counts"][
        "frames_processed"
    ] == 0
    assert second["stages"][PIPELINE_STAGE_VISUAL_OBSERVATIONS]["counts"][
        "frames_skipped_resumed"
    ] == 1
    assert second["smoke_metrics"]["frames_processed"] == 0
    assert second["smoke_metrics"]["frames_skipped"] == 1
    assert _count_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl") == 1


def test_run_vlm_alignment_pipeline_rejects_unsupported_backend(tmp_path: Path) -> None:
    project_dir = _make_project(tmp_path)

    with pytest.raises(ValueError, match="Unsupported VLM backend"):
        run_vlm_alignment_pipeline(
            VLMAlignmentPipelineConfig(
                project_dir=project_dir,
                vlm_backend="missing-backend",
                vlm_model="stub-vlm",
            )
        )


def test_run_vlm_alignment_pipeline_reports_missing_segment_artifact(tmp_path: Path) -> None:
    project_dir = _make_project(tmp_path)
    (project_dir / "segments" / "lecture_segments_aligned.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="No segment artifact found"):
        run_vlm_alignment_pipeline(
            VLMAlignmentPipelineConfig(project_dir=project_dir, vlm_model="stub-vlm")
        )


def _make_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "artifacts" / "projects" / "pipeline_project"
    _write_json(
        project_dir / "manifests" / "project_manifest.json",
        {
            "project_id": "pipeline_project",
            "video_id": "lecture_01",
            "artifacts": {},
            "counts": {},
        },
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            _frame("frame_000000", 0.0),
            _frame("frame_000001", 5.0),
            _frame("frame_000002", 10.0),
        ],
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_a",
                "project_id": "pipeline_project",
                "video_id": "lecture_01",
                "start_time": 0.0,
                "end_time": 10.0,
                "timestamp_center": 5.0,
                "transcript_text": "The matrix determinant is shown.",
                "frame_refs": ["frame_000000", "frame_000001", "frame_000002"],
            }
        ],
    )
    return project_dir


def _frame(frame_id: str, timestamp: float) -> dict:
    return {
        "frame_id": frame_id,
        "frame_path": f"frames/{frame_id}.jpg",
        "timestamp": timestamp,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _count_jsonl(path: Path) -> int:
    return len(_read_jsonl(path))
