import json
from pathlib import Path

from oarag.vlm_frame_candidates import (
    VLMFrameCandidateConfig,
    generate_vlm_frame_candidates,
    select_vlm_frame_candidates,
)


def test_select_vlm_frame_candidates_balances_segment_coverage() -> None:
    frames = [
        _frame("frame_000000", 0.0),
        _frame("frame_000001", 5.0),
        _frame("frame_000002", 10.0),
        _frame("frame_000003", 15.0),
        _frame("frame_000004", 20.0),
        _frame("frame_000005", 25.0),
        _frame("frame_000006", 30.0),
        _frame("frame_000007", 35.0),
    ]
    segments = [
        {
            "segment_id": "seg_a",
            "start_time": 0.0,
            "end_time": 10.0,
            "frame_refs": ["frame_000000", "frame_000001", "frame_000002"],
            "mention_candidates": ["matrix"],
        },
        {
            "segment_id": "seg_b",
            "start_time": 20.0,
            "end_time": 30.0,
            "frame_refs": ["frame_000004", "frame_000005", "frame_000006"],
        },
    ]

    result = select_vlm_frame_candidates(
        frames=frames,
        segments=segments,
        project_id="sample_project",
        video_id="lecture_01",
        config=VLMFrameCandidateConfig(
            max_candidates=2,
            max_per_segment=1,
            max_per_window=2,
            min_time_gap_seconds=0.0,
        ),
    )

    candidates = result["candidates"]
    summary = result["summary"]

    assert [candidate["frame_id"] for candidate in candidates] == [
        "frame_000001",
        "frame_000005",
    ]
    assert [candidate["segment_id"] for candidate in candidates] == ["seg_a", "seg_b"]
    assert candidates[0]["selection_reason"] == "segment_visual_hint"
    assert candidates[1]["selection_reason"] == "segment_coverage"
    assert summary["counts"]["selected_candidate_count"] == 2
    assert summary["counts"]["segments_with_candidates"] == 2
    assert summary["coverage"]["segment_candidate_coverage_ratio"] == 1.0
    assert summary["tradeoff"]["estimated_vlm_cost_reduction_ratio"] == 0.75
    assert summary["tradeoff"]["recall_risk"] == "low"


def test_select_vlm_frame_candidates_records_exclusion_reasons() -> None:
    frames = [
        _frame("frame_000000", 0.0),
        _frame("frame_000001", 0.01),
        {
            **_frame("frame_000002", 5.0),
            "selection": {"contrast": 0.0, "detail_score": 0.0},
        },
        {
            **_frame("frame_000003", 10.0),
            "selection": {"nearest_selected_distance": 0.001},
        },
    ]

    result = select_vlm_frame_candidates(
        frames=frames,
        project_id="sample_project",
        config=VLMFrameCandidateConfig(max_candidates=4, min_time_gap_seconds=0.0),
    )

    summary = result["summary"]
    excluded_by_id = {
        excluded["frame_id"]: excluded["reason"]
        for excluded in result["excluded_frames"]
    }

    assert [candidate["frame_id"] for candidate in result["candidates"]] == ["frame_000000"]
    assert excluded_by_id == {
        "frame_000001": "near_duplicate",
        "frame_000002": "low_information",
        "frame_000003": "near_duplicate",
    }
    assert summary["counts"]["exclusion_reasons"]["near_duplicate"] == 2
    assert summary["counts"]["exclusion_reasons"]["low_information"] == 1


def test_generate_vlm_frame_candidates_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    frames_manifest = project_dir / "manifests" / "frames_manifest.jsonl"
    segments_path = project_dir / "segments" / "lecture_segments_aligned.jsonl"
    project_manifest = project_dir / "manifests" / "project_manifest.json"
    _write_jsonl(
        frames_manifest,
        [
            _frame("frame_000000", 0.0),
            _frame("frame_000001", 5.0),
            _frame("frame_000002", 10.0),
        ],
    )
    _write_jsonl(
        segments_path,
        [
            {
                "segment_id": "seg_a",
                "start_time": 0.0,
                "end_time": 10.0,
                "frame_refs": ["frame_000000", "frame_000001", "frame_000002"],
            }
        ],
    )
    _write_json(
        project_manifest,
        {
            "project_id": "sample_project",
            "video_id": "lecture_01",
            "artifacts": {"frames_manifest": str(frames_manifest)},
            "counts": {"frames": 3},
        },
    )

    summary = generate_vlm_frame_candidates(
        project_dir=project_dir,
        config=VLMFrameCandidateConfig(
            max_candidates=1,
            max_per_segment=1,
            min_time_gap_seconds=0.0,
        ),
    )

    output_path = project_dir / "manifests" / "vlm_frame_candidates.jsonl"
    rows = _read_jsonl(output_path)
    manifest = json.loads(project_manifest.read_text(encoding="utf-8"))

    assert summary["counts"]["selected_candidate_count"] == 1
    assert rows[0]["project_id"] == "sample_project"
    assert rows[0]["video_id"] == "lecture_01"
    assert rows[0]["status"] == "selected"
    assert manifest["artifacts"]["frames_manifest"] == str(frames_manifest)
    assert manifest["artifacts"]["vlm_frame_candidates"] == str(output_path)
    assert manifest["counts"]["frames"] == 3
    assert manifest["counts"]["vlm_frame_candidates"] == 1
    assert manifest["vlm_frame_candidate_selection"]["counts"]["input_frame_count"] == 3


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
