import json
from pathlib import Path

from oarag.evidence import build_evidence_response, select_window_segments


def test_build_evidence_response_with_neighbor_segments_and_frames(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    segments_path = project_dir / "segments" / "lecture_segments_aligned.jsonl"
    frames_path = project_dir / "manifests" / "frames_manifest.jsonl"
    _write_jsonl(
        segments_path,
        [
            _segment("seg_1", 1, 0.0, 4.0, ["frame_000001"]),
            _segment("seg_2", 2, 5.0, 9.0, ["frame_000006", "frame_000008"]),
            _segment("seg_3", 3, 10.0, 14.0, ["frame_000012"]),
        ],
    )
    _write_jsonl(
        frames_path,
        [
            {"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"},
            {"frame_id": "frame_000006", "timestamp": 5.0, "frame_path": "/tmp/f6.jpg"},
            {"frame_id": "frame_000008", "timestamp": 7.0, "frame_path": "/tmp/f8.jpg"},
            {"frame_id": "frame_000012", "timestamp": 11.0, "frame_path": "/tmp/f12.jpg"},
        ],
    )

    response = build_evidence_response(
        project_dir=project_dir,
        segment_ids=["seg_2"],
        query="bet size",
        neighbor_count=1,
    )

    assert response["query"] == "bet size"
    assert response["top_segments"][0]["segment_id"] == "seg_2"
    window = response["evidence_windows"][0]
    assert window["target_segment_id"] == "seg_2"
    assert [segment["segment_id"] for segment in window["transcript_segments"]] == [
        "seg_1",
        "seg_2",
        "seg_3",
    ]
    assert [frame["frame_id"] for frame in window["frame_refs"]] == [
        "frame_000001",
        "frame_000006",
        "frame_000008",
        "frame_000012",
    ]
    assert window["start_time"] == 0.0
    assert window["end_time"] == 14.0


def test_select_window_segments_by_time_overlap() -> None:
    segments = [
        _segment("seg_1", 1, 0.0, 2.0, []),
        _segment("seg_2", 2, 3.0, 5.0, []),
        _segment("seg_3", 3, 8.0, 10.0, []),
        _segment("seg_4", 4, 20.0, 22.0, []),
    ]

    selected = select_window_segments(
        segments,
        target_segment_id="seg_2",
        window_seconds=3.0,
        neighbor_count=0,
    )

    assert [segment["segment_id"] for segment in selected] == ["seg_1", "seg_2", "seg_3"]


def test_build_evidence_response_at_start_boundary(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 2.0, ["frame_000001"]),
            _segment("seg_2", 2, 3.0, 5.0, ["frame_000004"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 0.0}],
    )

    response = build_evidence_response(
        project_dir=project_dir,
        segment_ids=["seg_1"],
        neighbor_count=1,
    )

    window = response["evidence_windows"][0]
    assert [segment["segment_id"] for segment in window["transcript_segments"]] == ["seg_1", "seg_2"]
    assert window["frame_refs"] == [
        {"frame_id": "frame_000001", "timestamp": 0.0},
        {"frame_id": "frame_000004"},
    ]


def test_build_evidence_response_handles_missing_frame_metadata(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, ["missing_frame"])],
    )
    _write_jsonl(project_dir / "manifests" / "frames_manifest.jsonl", [])

    response = build_evidence_response(
        project_dir=project_dir,
        segment_ids=["seg_1"],
        neighbor_count=0,
    )

    assert response["evidence_windows"][0]["frame_refs"] == [{"frame_id": "missing_frame"}]


def _segment(
    segment_id: str,
    sample_index: int,
    start_time: float,
    end_time: float,
    frame_refs: list[str],
) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "project",
        "video_id": "video",
        "sample_index": sample_index,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2,
        "transcript_text": f"Transcript {segment_id}",
        "frame_refs": frame_refs,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
