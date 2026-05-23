import json
from pathlib import Path

from oarag.answer import ask_project
from oarag.cli import build_parser


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


def test_ask_project_returns_extractive_answer_with_evidence(tmp_path: Path) -> None:
    project_dir = _project_fixture(tmp_path)

    response = ask_project(
        client=FakeClient([_hit()]),
        index_uid="local_segments",
        project_dir=project_dir,
        query="gradient loss curve",
        neighbor_count=0,
    )

    assert response["answer_text"] == (
        "The loss curve slopes downward after the gradient update. "
        "Linked visual evidence: loss curve annotation"
    )
    assert response["answer"] == {
        "mode": "extractive",
        "text": response["answer_text"],
        "source_bundle_count": 1,
    }
    bundle = response["bundles"][0]
    assert bundle["candidate"]["segment_id"] == "seg_1"
    assert bundle["evidence_window"]["target_segment"]["segment_id"] == "seg_1"
    assert bundle["evidence_window"]["frame_refs"][0]["frame_id"] == "frame_000001"
    assert bundle["linked_entities"][0]["entity"]["text"] == "loss curve annotation"


def test_ask_project_cli_prints_answer_and_writes_output(tmp_path: Path, monkeypatch, capsys) -> None:
    project_dir = _project_fixture(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "ask-project",
            "--index",
            "local_segments",
            "--project-dir",
            str(project_dir),
            "--query",
            "gradient loss curve",
            "--neighbor-count",
            "0",
            "--output",
            "answer-result.json",
        ]
    )
    monkeypatch.setattr("oarag.cli.client_from_args", lambda _: FakeClient([_hit()]))

    args.func(args)

    captured = capsys.readouterr()
    assert "The loss curve slopes downward" in captured.err
    payload = json.loads(captured.out)
    assert payload["answer_text"].startswith("The loss curve slopes downward")
    written = json.loads((project_dir / "answer-result.json").read_text(encoding="utf-8"))
    assert written["answer"]["mode"] == "extractive"


def _project_fixture(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_1",
                "project_id": "project",
                "video_id": "video",
                "sample_index": 1,
                "start_time": 0.0,
                "end_time": 8.0,
                "timestamp_center": 4.0,
                "transcript_text": "The loss curve slopes downward after the gradient update.",
                "frame_refs": ["frame_000001"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "timestamp": 4.0,
                "frame_path": "fixtures/public/frames/frame_000001.png",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_loss_curve",
                "project_id": "project",
                "frame_id": "frame_000001",
                "timestamp": 4.0,
                "frame_path": "fixtures/public/frames/frame_000001.png",
                "bbox": None,
                "text": "loss curve annotation",
                "entity_type": "synthetic_visual_note",
                "confidence": 1.0,
                "source": "test",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_1_entity_loss_curve",
                "project_id": "project",
                "segment_id": "seg_1",
                "entity_id": "entity_loss_curve",
                "frame_id": "frame_000001",
                "link_type": "synthetic_alignment",
                "score": 1.0,
                "evidence": ["public_synthetic_fixture"],
                "time_overlap": True,
                "lexical_match": ["loss"],
                "mention_candidate": ["curve"],
            }
        ],
    )
    return project_dir


def _hit() -> dict:
    return {
        "segment_id": "seg_1",
        "sample_id": "seg_1",
        "video_id": "video",
        "start_time": 0.0,
        "end_time": 8.0,
        "timestamp_center": 4.0,
        "transcript_text": "The loss curve slopes downward after the gradient update.",
        "_rankingScore": 0.91,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
