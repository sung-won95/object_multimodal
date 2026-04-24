import json
from pathlib import Path

from oarag.cli import build_parser
from oarag.project_query import query_project


class FakeClient:
    def __init__(self, hits: list[dict], processing_time_ms: int = 7) -> None:
        self.hits = hits
        self.processing_time_ms = processing_time_ms

    def search(self, index_uid: str, query: str, limit: int = 10) -> dict:
        return {
            "hits": self.hits[:limit],
            "processingTimeMs": self.processing_time_ms,
            "indexUid": index_uid,
            "query": query,
        }


def test_query_project_returns_multimodal_bundle(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 4.0, "Intro", ["frame_000001"]),
            _segment("seg_2", 2, 5.0, 9.0, "Bet size is on this board", ["frame_000006", "frame_000008"]),
            _segment("seg_3", 3, 10.0, 14.0, "Next street", ["frame_000012"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"},
            {"frame_id": "frame_000006", "timestamp": 5.0, "frame_path": "/tmp/f6.jpg"},
            {"frame_id": "frame_000008", "timestamp": 7.0, "frame_path": "/tmp/f8.jpg"},
            {"frame_id": "frame_000012", "timestamp": 11.0, "frame_path": "/tmp/f12.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_board",
                "project_id": "project",
                "frame_id": "frame_000006",
                "timestamp": 5.0,
                "frame_path": "/tmp/f6.jpg",
                "bbox": None,
                "text": "bet size board",
                "entity_type": "ocr_text",
                "confidence": 0.91,
                "source": "local-ocr",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_2_entity_board",
                "project_id": "project",
                "segment_id": "seg_2",
                "entity_id": "entity_board",
                "frame_id": "frame_000006",
                "link_type": "time_overlap+lexical_match+mention_candidate",
                "score": 1.35,
                "evidence": ["time_overlap", "lexical_match", "mention_candidate"],
                "time_overlap": True,
                "lexical_match": ["bet", "board", "size"],
                "mention_candidate": ["board"],
            }
        ],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 5.0,
                    "end_time": 9.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "Bet size is on this board",
                    "_rankingScore": 0.87,
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="bet size",
        neighbor_count=1,
    )

    assert response["artifact_availability"] == {
        "frames_manifest": True,
        "visual_entities": True,
        "entity_links": True,
    }
    assert response["counts"]["search_hits"] == 1
    assert response["counts"]["bundles"] == 1
    bundle = response["bundles"][0]
    assert bundle["candidate"]["segment_id"] == "seg_2"
    assert [segment["segment_id"] for segment in bundle["evidence_window"]["transcript_segments"]] == [
        "seg_1",
        "seg_2",
        "seg_3",
    ]
    assert [frame["frame_path"] for frame in bundle["evidence_window"]["frame_refs"]] == [
        "/tmp/f1.jpg",
        "/tmp/f6.jpg",
        "/tmp/f8.jpg",
        "/tmp/f12.jpg",
    ]
    assert bundle["visual_entities"][0]["entity_id"] == "entity_board"
    assert bundle["linked_entities"][0]["entity"]["text"] == "bet size board"
    assert "lexical match: bet, board, size" in bundle["linked_entities"][0]["explanation"]
    assert bundle["summary"]["frame_paths"] == ["/tmp/f1.jpg", "/tmp/f6.jpg", "/tmp/f8.jpg", "/tmp/f12.jpg"]
    assert "bet size board" in response["summary_lines"][0]


def test_query_project_without_visual_artifacts_falls_back_to_transcript_and_frames(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "Only transcript evidence", ["frame_000001"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"}],
    )

    response = query_project(
        client=FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Only transcript evidence",
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="transcript evidence",
        neighbor_count=0,
    )

    assert response["artifact_availability"] == {
        "frames_manifest": True,
        "visual_entities": False,
        "entity_links": False,
    }
    assert response["warnings"] == []
    bundle = response["bundles"][0]
    assert bundle["visual_entities"] == []
    assert bundle["linked_entities"] == []
    assert bundle["evidence_window"]["frame_refs"] == [
        {"frame_id": "frame_000001", "timestamp": 0.0, "frame_path": "/tmp/f1.jpg"}
    ]
    assert "linked_entities=none" in bundle["summary"]["text"]


def test_query_project_cli_prints_summary_and_writes_output(tmp_path: Path, monkeypatch, capsys) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 1.0, 3.0, "Shown on slide", ["frame_000001"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "/tmp/f1.jpg"}],
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "query-project",
            "--index",
            "local_segments",
            "--project-dir",
            str(project_dir),
            "--query",
            "shown",
            "--output",
            "query-result.json",
        ]
    )
    monkeypatch.setattr(
        "oarag.cli.client_from_args",
        lambda _: FakeClient(
            hits=[
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "timestamp_center": 2.0,
                    "transcript_text": "Shown on slide",
                }
            ]
        ),
    )

    args.func(args)
    captured = capsys.readouterr()
    assert "#1 video @1.0-3.0s: Shown on slide" in captured.err

    payload = json.loads(captured.out)
    assert payload["summary_lines"][0].startswith("#1 video @1.0-3.0s")
    output_path = project_dir / "query-result.json"
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["bundles"][0]["candidate"]["segment_id"] == "seg_1"


def _segment(
    segment_id: str,
    sample_index: int,
    start_time: float,
    end_time: float,
    transcript_text: str,
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
        "transcript_text": transcript_text,
        "frame_refs": frame_refs,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
