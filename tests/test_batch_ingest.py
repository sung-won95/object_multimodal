import csv
import json
from pathlib import Path

from oarag.ingest import (
    BatchIngestConfig,
    batch_ingest_videos,
    discover_video_files,
    make_project_id_for_video,
    select_frame_timestamps,
    summarize_frame_sampling,
    write_batch_summary_csv,
    write_batch_summary_json,
    write_batch_summary_jsonl,
)


def test_discover_video_files_recurses_and_filters_extensions(tmp_path: Path) -> None:
    root = tmp_path / "lectures"
    (root / "week1").mkdir(parents=True)
    (root / "week2").mkdir(parents=True)
    (root / "week1" / "intro.mp4").write_bytes(b"x")
    (root / "week2" / "demo.MOV").write_bytes(b"x")
    (root / "week2" / "notes.txt").write_text("ignore", encoding="utf-8")

    videos = discover_video_files(root)

    assert [path.name for path in videos] == ["intro.mp4", "demo.MOV"]


def test_make_project_id_for_video_is_stable_and_distinct(tmp_path: Path) -> None:
    root = tmp_path / "lectures"
    video_a = root / "week1" / "lesson.mp4"
    video_b = root / "week2" / "lesson.mp4"
    video_a.parent.mkdir(parents=True)
    video_b.parent.mkdir(parents=True)
    video_a.write_bytes(b"x")
    video_b.write_bytes(b"x")

    project_id_a_1 = make_project_id_for_video(video_a, root_dir=root, prefix="pilot")
    project_id_a_2 = make_project_id_for_video(video_a, root_dir=root, prefix="pilot")
    project_id_b = make_project_id_for_video(video_b, root_dir=root, prefix="pilot")

    assert project_id_a_1 == project_id_a_2
    assert project_id_a_1 != project_id_b
    assert project_id_a_1.startswith("pilot__")


def test_uniform_frame_sampling_spreads_cap_across_duration() -> None:
    timestamps = select_frame_timestamps(
        duration_sec=300.0,
        frame_rate=1.0,
        max_frames=4,
        frame_sampling="uniform",
    )

    assert timestamps == [0.0, 100.0, 199.0, 299.0]


def test_prefix_frame_sampling_keeps_existing_front_loaded_behavior() -> None:
    timestamps = select_frame_timestamps(
        duration_sec=300.0,
        frame_rate=1.0,
        max_frames=4,
        frame_sampling="prefix",
    )

    assert timestamps == [0.0, 1.0, 2.0, 3.0]


def test_frame_sampling_summary_reports_temporal_coverage() -> None:
    summary = summarize_frame_sampling(
        frames=[
            {"frame_id": "frame_000001", "timestamp": 0.0},
            {"frame_id": "frame_000002", "timestamp": 299.0},
        ],
        duration_sec=300.0,
        frame_rate=1.0,
        max_frames=2,
        frame_sampling="uniform",
    )

    assert summary["selected_frame_count"] == 2
    assert summary["candidate_frame_count"] == 300
    assert summary["capped"] is True
    assert summary["covered_until_sec"] == 300.0
    assert summary["timestamp_span_sec"] == 299.0
    assert summary["temporal_coverage_ratio"] == 1.0


def test_batch_ingest_skips_existing_by_default_and_force_reingests(tmp_path: Path) -> None:
    root = tmp_path / "lectures"
    output_root = tmp_path / "artifacts" / "projects"
    first = root / "a" / "first.mp4"
    second = root / "b" / "second.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    first_project_id = make_project_id_for_video(first, root_dir=root)
    existing_manifest = output_root / first_project_id / "manifests" / "project_manifest.json"
    existing_manifest.parent.mkdir(parents=True, exist_ok=True)
    existing_manifest.write_text("{}", encoding="utf-8")

    calls: list[str] = []

    def fake_ingest(video_config) -> dict:
        calls.append(video_config.project_id)
        return {
            "transcript_source": "srt",
            "counts": {"lecture_segments": 2, "frames": 10},
        }

    summary = batch_ingest_videos(
        BatchIngestConfig(root_dir=root, output_root=output_root),
        ingest_fn=fake_ingest,
    )

    assert summary["counts"] == {
        "discovered": 2,
        "ingested": 1,
        "skipped_existing": 1,
        "failed": 0,
    }
    assert len(calls) == 1

    forced_summary = batch_ingest_videos(
        BatchIngestConfig(root_dir=root, output_root=output_root, force=True),
        ingest_fn=fake_ingest,
    )

    assert forced_summary["counts"] == {
        "discovered": 2,
        "ingested": 2,
        "skipped_existing": 0,
        "failed": 0,
    }
    assert len(calls) == 3


def test_batch_ingest_non_strict_continues_after_failure(tmp_path: Path) -> None:
    root = tmp_path / "lectures"
    output_root = tmp_path / "artifacts" / "projects"
    good = root / "z" / "good.mp4"
    bad = root / "a" / "bad.mp4"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_bytes(b"x")
    bad.write_bytes(b"x")

    def flaky_ingest(video_config) -> dict:
        if video_config.video_id == "bad":
            raise RuntimeError("boom")
        return {"transcript_source": "srt", "counts": {"lecture_segments": 1, "frames": 2}}

    summary = batch_ingest_videos(
        BatchIngestConfig(root_dir=root, output_root=output_root),
        ingest_fn=flaky_ingest,
    )

    assert summary["aborted"] is False
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["ingested"] == 1
    assert len(summary["results"]) == 2


def test_batch_ingest_passes_frame_selection_to_video_ingest(tmp_path: Path) -> None:
    root = tmp_path / "lectures"
    output_root = tmp_path / "artifacts" / "projects"
    video = root / "lesson.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    seen_frame_selection = None

    def fake_ingest(video_config) -> dict:
        nonlocal seen_frame_selection
        seen_frame_selection = video_config.frame_selection
        return {"transcript_source": "none", "counts": {"lecture_segments": 0, "frames": 1}}

    batch_ingest_videos(
        BatchIngestConfig(
            root_dir=root,
            output_root=output_root,
            frame_selection="representative",
        ),
        ingest_fn=fake_ingest,
    )

    assert seen_frame_selection == "representative"


def test_batch_ingest_strict_stops_after_first_failure(tmp_path: Path) -> None:
    root = tmp_path / "lectures"
    output_root = tmp_path / "artifacts" / "projects"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "first.mp4").write_bytes(b"x")
    (root / "b" / "second.mp4").write_bytes(b"x")

    calls = 0

    def failing_ingest(_video_config) -> dict:
        nonlocal calls
        calls += 1
        raise RuntimeError("first failure")

    summary = batch_ingest_videos(
        BatchIngestConfig(root_dir=root, output_root=output_root, strict=True),
        ingest_fn=failing_ingest,
    )

    assert summary["aborted"] is True
    assert summary["counts"]["failed"] == 1
    assert len(summary["results"]) == 1
    assert calls == 1


def test_write_batch_summary_outputs_json_jsonl_and_csv(tmp_path: Path) -> None:
    summary = {
        "counts": {"discovered": 1, "ingested": 1, "skipped_existing": 0, "failed": 0},
        "results": [
            {
                "status": "ingested",
                "video_path": "/tmp/video.mp4",
                "project_id": "proj",
                "video_id": "video",
                "project_dir": "/tmp/artifacts/projects/proj",
                "manifest_path": "/tmp/artifacts/projects/proj/manifests/project_manifest.json",
                "transcript_source": "srt",
                "segments": 3,
                "frames": 11,
            }
        ],
    }
    json_path = tmp_path / "summary.json"
    jsonl_path = tmp_path / "summary.jsonl"
    csv_path = tmp_path / "summary.csv"

    write_batch_summary_json(json_path, summary)
    write_batch_summary_jsonl(jsonl_path, summary)
    write_batch_summary_csv(csv_path, summary)

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    loaded_jsonl = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert loaded_json["counts"]["ingested"] == 1
    assert loaded_jsonl[0]["status"] == "ingested"
    assert csv_rows[0]["project_id"] == "proj"
