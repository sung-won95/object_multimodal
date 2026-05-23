import json
from pathlib import Path

from oarag.web.server import build_ask_project_response, build_ask_project_response_from_payload


class FakeClient:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits

    def search(self, index_uid: str, query: str, limit: int = 10) -> dict:
        return {
            "hits": self.hits[:limit],
            "processingTimeMs": 4,
            "indexUid": index_uid,
            "query": query,
        }


def test_web_ask_response_uses_shared_answer_schema(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "Gradient points toward lower loss.", [])],
    )

    response = build_ask_project_response(
        client=FakeClient(
            [
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Gradient points toward lower loss.",
                    "_rankingScore": 0.84,
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="gradient lower loss",
        neighbor_count=0,
    )

    assert response["answer_type"] == "grounded_answer"
    assert response["answer"]["answer_type"] == "grounded_answer"
    assert response["answer"]["claims"][0]["modalities"] == ["transcript"]
    assert response["answer"]["citations"][0]["segment_id"] == "seg_1"


def test_web_payload_builder_accepts_http_style_payload(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "Gradient points toward lower loss.", [])],
    )

    response = build_ask_project_response_from_payload(
        payload={
            "index": "local_segments",
            "project_dir": str(project_dir),
            "query": "unrelated river",
            "neighbor_count": 0,
        },
        client=FakeClient(
            [
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "timestamp_center": 1.0,
                    "transcript_text": "Gradient points toward lower loss.",
                    "_rankingScore": 0.84,
                }
            ]
        ),
    )

    assert response["answer_type"] == "candidate_evidence_only"
    assert response["answer"]["claims"] == []
    assert response["answer"]["candidate_evidence"][0]["support"]["matched_query_terms"] == []


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
