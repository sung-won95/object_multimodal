import json
from pathlib import Path

from oarag.audio_visual_consistency import (
    AVC_SKIPPED_STATUS,
    AVC_SOURCE_FAILURE_STATUS,
    AVC_SUCCESS_STATUS,
    CONSISTENCY_ALIGNED,
    CONSISTENCY_MISMATCH,
    CONSISTENCY_UNCERTAIN,
    MISSING_VISUAL_OBSERVATION,
    NO_FRAME_REFS,
    generate_audio_visual_consistency,
)


def test_generate_audio_visual_consistency_writes_stub_judgments_and_manifest(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    segments_path = project_dir / "segments" / "lecture_segments_aligned.jsonl"
    observations_path = project_dir / "manifests" / "vlm_visual_observations.jsonl"
    manifest_path = project_dir / "manifests" / "project_manifest.json"
    _write_jsonl(
        segments_path,
        [
            _segment("seg_aligned", "The matrix determinant appears here.", ["frame_a"]),
            _segment("seg_mismatch", "The river delta is discussed.", ["frame_b"]),
            _segment("seg_uncertain", "Look here.", ["frame_c"]),
        ],
    )
    _write_jsonl(
        observations_path,
        [
            _success_observation(
                "obs_a",
                "frame_a",
                1.0,
                "A matrix determinant calculation is visible.",
                detected_text="matrix determinant",
            ),
            _success_observation("obs_b", "frame_b", 2.0, "A poker stack size table."),
            _success_observation("obs_c", "frame_c", 3.0, "An arrow points to a blue line."),
        ],
    )
    _write_json(
        manifest_path,
        {
            "project_id": "sample_project",
            "video_id": "lecture_01",
            "artifacts": {},
            "counts": {},
        },
    )

    summary = generate_audio_visual_consistency(project_dir=project_dir)

    output_path = project_dir / "manifests" / "audio_visual_consistency.jsonl"
    rows = _read_jsonl(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["status"] == "completed"
    assert summary["counts"]["audio_visual_consistency"] == 3
    assert [row["status"] for row in rows] == [
        AVC_SUCCESS_STATUS,
        AVC_SUCCESS_STATUS,
        AVC_SUCCESS_STATUS,
    ]
    assert [row["consistency"] for row in rows] == [
        CONSISTENCY_ALIGNED,
        CONSISTENCY_MISMATCH,
        CONSISTENCY_UNCERTAIN,
    ]
    assert rows[0]["visual_observation_ids"] == ["obs_a"]
    assert rows[0]["evidence_refs"] == [
        "segment:seg_aligned",
        "frame:frame_a",
        "observation:obs_a",
    ]
    assert rows[0]["metadata"]["matched_terms"] == ["determinant", "matrix"]
    assert rows[1]["metadata"]["matched_terms"] == []
    assert rows[2]["confidence"] == 0.35
    assert manifest["artifacts"]["audio_visual_consistency"] == str(output_path)
    assert manifest["counts"]["audio_visual_consistency"] == 3
    assert manifest["vlm_consistency"]["audio_visual_consistency"]["status"] == "completed"
    assert manifest["vlm_consistency"]["audio_visual_consistency"]["consistency_counts"] == {
        CONSISTENCY_ALIGNED: 1,
        CONSISTENCY_MISMATCH: 1,
        CONSISTENCY_UNCERTAIN: 1,
    }


def test_audio_visual_consistency_records_missing_frames_and_failure_observations(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "edge_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_no_frame", "No frame for this segment.", []),
            _segment("seg_missing_observation", "Matrix appears.", ["frame_missing"]),
            _segment("seg_failure", "Stack size appears.", ["frame_failure"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "vlm_visual_observations.jsonl",
        [
            {
                "observation_id": "obs_failure",
                "project_id": "edge_project",
                "video_id": "lecture_01",
                "frame_id": "frame_failure",
                "timestamp": 10.0,
                "segment_id": "seg_failure",
                "backend": "deterministic",
                "source_model": "stub-vlm",
                "model_version": None,
                "confidence": None,
                "status": "backend_failure",
                "observation_type": "frame_error",
                "visual_description": "",
                "metadata": {"failure_reason": "fixture backend failure"},
            }
        ],
    )

    summary = generate_audio_visual_consistency(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "manifests" / "audio_visual_consistency.jsonl")
    rows_by_segment = {row["segment_id"]: row for row in rows}

    assert summary["status"] == "completed_with_errors"
    assert summary["counts"]["source_failure_records"] == 1
    assert summary["counts"]["skipped_records"] == 2
    assert rows_by_segment["seg_no_frame"]["status"] == AVC_SKIPPED_STATUS
    assert rows_by_segment["seg_no_frame"]["skip_reason"] == NO_FRAME_REFS
    assert rows_by_segment["seg_no_frame"]["frame_id"] == ""
    assert rows_by_segment["seg_missing_observation"]["status"] == AVC_SKIPPED_STATUS
    assert rows_by_segment["seg_missing_observation"]["skip_reason"] == MISSING_VISUAL_OBSERVATION
    assert rows_by_segment["seg_failure"]["status"] == AVC_SOURCE_FAILURE_STATUS
    assert rows_by_segment["seg_failure"]["consistency"] == CONSISTENCY_UNCERTAIN
    assert rows_by_segment["seg_failure"]["visual_observation_ids"] == ["obs_failure"]
    assert rows_by_segment["seg_failure"]["failure_reason"] == "fixture backend failure"
    assert rows_by_segment["seg_failure"]["metadata"]["source_observation_status_counts"] == {
        "backend_failure": 1
    }


def test_audio_visual_consistency_falls_back_to_raw_segments_with_time_window(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "raw_project"
    raw_segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    _write_jsonl(
        raw_segments_path,
        [
            {
                "segment_id": "seg_time",
                "project_id": "raw_project",
                "video_id": "lecture_01",
                "start_time": 5.0,
                "end_time": 7.0,
                "transcript_text": "The covariance matrix is on screen.",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "vlm_visual_observations.jsonl",
        [
            _success_observation("obs_in", "frame_in", 6.0, "Covariance matrix formula."),
            _success_observation("obs_out", "frame_out", 9.0, "River delta photo."),
        ],
    )

    summary = generate_audio_visual_consistency(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "manifests" / "audio_visual_consistency.jsonl")

    assert summary["paths"]["segments"] == str(raw_segments_path)
    assert len(rows) == 1
    assert rows[0]["segment_id"] == "seg_time"
    assert rows[0]["frame_id"] == "frame_in"
    assert rows[0]["consistency"] == CONSISTENCY_ALIGNED
    assert rows[0]["visual_observation_ids"] == ["obs_in"]


def _segment(segment_id: str, transcript_text: str, frame_refs: list[str]) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "start_time": 0.0,
        "end_time": 4.0,
        "timestamp_center": 2.0,
        "transcript_text": transcript_text,
        "frame_refs": frame_refs,
    }


def _success_observation(
    observation_id: str,
    frame_id: str,
    timestamp: float,
    visual_description: str,
    *,
    detected_text: str | None = None,
) -> dict:
    return {
        "observation_id": observation_id,
        "project_id": "sample_project",
        "video_id": "lecture_01",
        "frame_id": frame_id,
        "timestamp": timestamp,
        "segment_id": None,
        "backend": "deterministic",
        "source_model": "stub-vlm",
        "model_version": "test-v1",
        "confidence": 0.9,
        "status": "success",
        "observation_type": "frame_summary",
        "visual_description": visual_description,
        "detected_text": detected_text,
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
