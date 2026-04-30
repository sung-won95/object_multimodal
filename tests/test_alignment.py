import json
from pathlib import Path

from oarag.alignment import align_segments_to_frames


def test_align_segments_to_frames_with_overlap_and_margin(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts" / "projects"
    project_id = "sample"
    project_dir = output_root / project_id
    segments_path = project_dir / "segments" / "lecture_segments.jsonl"
    frames_manifest_path = project_dir / "manifests" / "frames_manifest.jsonl"
    manifest_path = project_dir / "manifests" / "project_manifest.json"

    _write_jsonl(
        segments_path,
        [
            {"segment_id": "seg_1", "start_time": 10.0, "end_time": 12.0},
            {"segment_id": "seg_2", "start_time": 30.0, "end_time": 31.0},
        ],
    )
    _write_jsonl(
        frames_manifest_path,
        [
            {"frame_id": "f1", "timestamp": 9.8},
            {"frame_id": "f2", "timestamp": 10.0},
            {"frame_id": "f3", "timestamp": 11.5},
            {"frame_id": "f4", "timestamp": 12.2},
            {"frame_id": "f5", "timestamp": 30.49},
        ],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"artifacts": {}, "counts": {}}), encoding="utf-8")

    summary = align_segments_to_frames(
        project_id=project_id,
        output_root=output_root,
        margin_seconds=0.5,
    )

    aligned_rows = _read_jsonl(project_dir / "segments" / "lecture_segments_aligned.jsonl")
    assert aligned_rows[0]["frame_refs"] == ["f1", "f2", "f3", "f4"]
    assert aligned_rows[1]["frame_refs"] == ["f5"]

    assert summary["counts"] == {
        "margin_seconds": 0.5,
        "segments_total": 2,
        "segments_with_frames": 2,
        "segments_without_frames": 0,
        "segment_frame_coverage_ratio": 1.0,
        "frame_refs_total": 5,
        "unique_frames_referenced": 5,
        "available_frames": 5,
        "available_frame_time_span_sec": 20.69,
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["lecture_segments_aligned"].endswith(
        "/sample/segments/lecture_segments_aligned.jsonl"
    )
    assert manifest["counts"]["lecture_segments_aligned"] == 2
    assert manifest["alignment"]["segments_with_frames"] == 2


def test_align_segments_to_frames_includes_start_and_end_boundaries(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts" / "projects"
    project_id = "boundary"
    project_dir = output_root / project_id
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [{"segment_id": "seg_1", "start_time": 5.0, "end_time": 7.0}],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "f_start", "timestamp": 5.0},
            {"frame_id": "f_end", "timestamp": 7.0},
            {"frame_id": "f_before", "timestamp": 4.999},
            {"frame_id": "f_after", "timestamp": 7.001},
        ],
    )

    align_segments_to_frames(project_id=project_id, output_root=output_root, margin_seconds=0.0)

    aligned_rows = _read_jsonl(project_dir / "segments" / "lecture_segments_aligned.jsonl")
    assert aligned_rows[0]["frame_refs"] == ["f_start", "f_end"]


def test_align_segments_to_frames_with_no_frames(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts" / "projects"
    project_id = "no_frames"
    project_dir = output_root / project_id
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [{"segment_id": "seg_1", "start_time": 1.0, "end_time": 2.0}],
    )
    _write_jsonl(project_dir / "manifests" / "frames_manifest.jsonl", [])

    summary = align_segments_to_frames(project_id=project_id, output_root=output_root)

    aligned_rows = _read_jsonl(project_dir / "segments" / "lecture_segments_aligned.jsonl")
    assert aligned_rows[0]["frame_refs"] == []
    assert summary["counts"]["segments_with_frames"] == 0
    assert summary["counts"]["segments_without_frames"] == 1
    assert summary["counts"]["frame_refs_total"] == 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
