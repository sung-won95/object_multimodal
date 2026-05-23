import json
from pathlib import Path

from oarag.retrieval.answer import CANDIDATE_EVIDENCE_ONLY, GROUNDED_ANSWER, ask_project


class FakeClient:
    def __init__(self, hits: list[dict], processing_time_ms: int = 5) -> None:
        self.hits = hits
        self.processing_time_ms = processing_time_ms
        self.searches: list[tuple[str, str, int, dict | None]] = []

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
    ) -> dict:
        self.searches.append((index_uid, query, limit, hybrid))
        return {
            "hits": self.hits[:limit],
            "processingTimeMs": self.processing_time_ms,
            "indexUid": index_uid,
            "query": query,
        }


def test_ask_project_returns_grounded_answer_with_citations(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_grounded_project(project_dir)

    response = ask_project(
        client=FakeClient(
            [
                {
                    "segment_id": "seg_2",
                    "sample_id": "seg_2",
                    "video_id": "video",
                    "start_time": 5.0,
                    "end_time": 9.0,
                    "timestamp_center": 7.0,
                    "transcript_text": "Bet size appears on the board.",
                    "_rankingScore": 0.88,
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="bet size board",
        neighbor_count=0,
    )

    assert response["answer_type"] == GROUNDED_ANSWER
    answer = response["answer"]
    assert answer["answer_type"] == GROUNDED_ANSWER
    assert answer["llm"] == {"enabled": False, "backend": None}
    assert answer["claims"] == [
        {
            "claim_id": "claim_1",
            "text": "Bet size appears on the board.",
            "modalities": ["transcript"],
            "citation_ids": ["citation_1"],
        },
        {
            "claim_id": "claim_2",
            "text": "Visual evidence includes bet size board.",
            "modalities": ["visual"],
            "citation_ids": ["citation_1"],
        },
    ]
    citation = answer["citations"][0]
    assert citation["segment_id"] == "seg_2"
    assert citation["start_time"] == 5.0
    assert citation["end_time"] == 9.0
    assert citation["timestamp"] == 7.0
    assert citation["frame_refs"] == [
        {"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"}
    ]
    assert citation["visual_entity_ids"] == ["entity_board"]
    assert citation["entity_link_ids"] == ["link_seg_2_entity_board"]
    assert citation["modalities"] == ["transcript", "visual"]
    assert answer["no_answer_policy"]["reason"] == "query_terms_grounded_in_candidate"


def test_ask_project_preserves_window_hybrid_rerank_metadata_in_answer_evidence(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 2.0, "Intro context", []),
            _segment("seg_2", 2, 3.0, 5.0, "Target range grid appears", []),
            _segment("seg_3", 3, 6.0, 8.0, "Next context", []),
        ],
    )
    client = FakeClient(
        [
            {
                "window_id": "window_seg_2_abc123",
                "target_segment_id": "seg_2",
                "sample_id": "seg_2",
                "video_id": "video",
                "start_time": 0.0,
                "end_time": 8.0,
                "timestamp_center": 4.0,
                "target_start_time": 3.0,
                "target_end_time": 5.0,
                "target_timestamp_center": 4.0,
                "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
                "transcript_window_text": "Intro context Target range grid appears Next context",
                "_rankingScore": 0.91,
            }
        ]
    )

    response = ask_project(
        client=client,
        index_uid="local_windows",
        retrieval_index_kind="window",
        project_dir=project_dir,
        query="target range grid",
        neighbor_count=0,
        rerank=True,
        rerank_backend="stub",
        hybrid_retrieval=True,
        hybrid_embedder="lecture_embedder",
        hybrid_semantic_ratio=0.75,
    )

    assert client.searches == [
        ("local_windows", "target range grid", 5, None),
        (
            "local_windows",
            "target range grid",
            5,
            {"embedder": "lecture_embedder", "semanticRatio": 0.75},
        ),
    ]
    assert response["index_kind"] == "window"
    assert response["retrieval_context"]["hybrid_retrieval"]["enabled"] is True
    assert response["retrieval_context"]["rerank"]["enabled"] is True
    answer = response["answer"]
    citation = answer["citations"][0]
    evidence = answer["candidate_evidence"][0]
    for item in (citation, evidence):
        assert item["retrieval_index_kind"] == "window"
        assert item["hybrid_retrieval"]["embedder"] == "lecture_embedder"
        assert item["hybrid_retrieval"]["semantic_ratio"] == 0.75
        assert item["rerank"]["backend"] == "stub"
        assert item["retrieval_sources"][0]["source"] == "window"
        assert item["retrieval_sources"][0]["window_id"] == "window_seg_2_abc123"
        assert item["retrieval_sources"][0]["source_segment_ids"] == [
            "seg_1",
            "seg_2",
            "seg_3",
        ]
        assert [match["retrieval_mode"] for match in item["retrieval_sources"][0]["matches"]] == [
            "lexical",
            "semantic",
        ]


def test_ask_project_returns_candidate_evidence_only_when_query_is_weakly_grounded(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_1", 1, 0.0, 3.0, "Bet size appears on the board.", [])],
    )

    response = ask_project(
        client=FakeClient(
            [
                {
                    "segment_id": "seg_1",
                    "sample_id": "seg_1",
                    "video_id": "video",
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "timestamp_center": 1.5,
                    "transcript_text": "Bet size appears on the board.",
                    "_rankingScore": 0.91,
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="river geology",
        neighbor_count=0,
    )

    answer = response["answer"]
    assert response["answer_type"] == CANDIDATE_EVIDENCE_ONLY
    assert answer["answer_type"] == CANDIDATE_EVIDENCE_ONLY
    assert answer["claims"] == []
    assert answer["candidate_evidence"][0]["segment_id"] == "seg_1"
    assert answer["candidate_evidence"][0]["support"]["matched_query_terms"] == []
    assert answer["no_answer_policy"]["reason"] == "insufficient_query_overlap"
    assert "not strong enough" in answer["answer"]


def _write_grounded_project(project_dir: Path) -> None:
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [_segment("seg_2", 2, 5.0, 9.0, "Bet size appears on the board.", ["frame_000006"])],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_board",
                "project_id": "project",
                "frame_id": "frame_000006",
                "timestamp": 6.0,
                "frame_path": "/tmp/f6.jpg",
                "bbox": None,
                "text": "bet size board",
                "entity_type": "ocr_text",
                "confidence": 0.91,
                "source": "test",
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
                "link_type": "time_overlap+lexical_match",
                "score": 1.2,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["bet", "size", "board"],
                "mention_candidate": ["board"],
            }
        ],
    )


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
