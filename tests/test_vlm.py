import json
from pathlib import Path

import pytest

from oarag.vlm import make_vlm_backend, run_vlm


def test_run_vlm_deterministic_backend_writes_observations_and_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    output = project_dir / "manifests" / "vlm_visual_observations.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.25,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": "frames/frame_000002.jpg",
                "timestamp": 2.5,
            },
        ],
    )
    _write_json(
        project_manifest,
        {
            "project_id": "sample_project",
            "video_id": "lecture_01",
            "artifacts": {"frames_manifest": str(frames_manifest)},
            "counts": {"frames": 2},
        },
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="deterministic",
        model="stub-vlm",
        device="cpu",
        options={
            "description_prefix": "Fixture observation",
            "detected_text": "Matrix A",
            "confidence": "0.82",
            "model_version": "test-v1",
        },
    )

    rows = _read_jsonl(output)
    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))

    assert summary["backend"] == "deterministic"
    assert summary["source_model"] == "stub-vlm"
    assert summary["model_version"] == "test-v1"
    assert summary["counts"]["vlm_visual_observations"] == 2
    assert [row["frame_id"] for row in rows] == ["frame_000001", "frame_000002"]
    assert rows[0]["visual_description"] == "Fixture observation: frame_000001 at 1.250s"
    assert rows[0]["detected_text"] == "Matrix A"
    assert rows[0]["confidence"] == 0.82
    assert rows[0]["source_model"] == "stub-vlm"
    assert manifest["artifacts"]["frames_manifest"] == str(frames_manifest)
    assert manifest["artifacts"]["vlm_visual_observations"] == str(output)
    assert manifest["counts"]["frames"] == 2
    assert manifest["counts"]["vlm_visual_observations"] == 2
    assert manifest["vlm_consistency"]["backend"] == "deterministic"
    assert manifest["vlm_consistency"]["source_model"] == "stub-vlm"
    assert manifest["vlm_consistency"]["settings"]["device"] == "cpu"
    assert manifest["vlm_consistency"]["settings"]["run_id"].startswith("vlm_")


def test_run_vlm_uses_frame_candidates_when_provided(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "candidate_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    frame_candidates = project_dir / "manifests" / "vlm_frame_candidates.jsonl"
    _write_jsonl(
        frames_manifest,
        [
            {
                "frame_id": "frame_000001",
                "frame_path": "frames/frame_000001.jpg",
                "timestamp": 1.0,
            },
            {
                "frame_id": "frame_000002",
                "frame_path": "frames/frame_000002.jpg",
                "timestamp": 2.0,
            },
        ],
    )
    _write_jsonl(
        frame_candidates,
        [
            {
                "project_id": "candidate_project",
                "video_id": "lecture_01",
                "frame_id": "frame_000002",
                "timestamp": 2.0,
                "segment_id": "seg_0001",
                "backend": "vlm-frame-candidate-selector",
                "source_model": None,
                "model_version": None,
                "confidence": None,
                "status": "selected",
                "frame_path": "frames/frame_000002.jpg",
                "selection_reason": "segment_coverage",
                "rank": 1,
            }
        ],
    )

    summary = run_vlm(
        project_dir=project_dir,
        backend="deterministic",
        model="stub-vlm",
        frame_candidates_path=frame_candidates,
    )

    rows = _read_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl")

    assert summary["counts"]["frames_total"] == 1
    assert rows[0]["frame_id"] == "frame_000002"
    assert rows[0]["segment_id"] == "seg_0001"
    assert rows[0]["attributes"]["rank"] == 1


def test_vlm_backend_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="Unsupported VLM backend"):
        make_vlm_backend("missing-backend")


def test_run_vlm_requires_model(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    project_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="--vlm-model is required"):
        run_vlm(project_dir=project_dir, model=None)


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
