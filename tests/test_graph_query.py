import json
import socket
from pathlib import Path

import pytest

from oarag.config import load_neo4j_config
from oarag.graph_query import (
    GraphTraversalConfig,
    detect_graph_query_hints,
    graph_query,
    graph_traversal_cypher,
    run_graph_traversal,
    serialize_traversal_record,
)
from oarag.graph_ingest import ingest_project_graph


class FakeClient:
    def __init__(self, hits: list[dict], processing_time_ms: int = 5) -> None:
        self.hits = hits
        self.processing_time_ms = processing_time_ms

    def search(self, index_uid: str, query: str, limit: int = 10) -> dict:
        return {
            "hits": self.hits[:limit],
            "processingTimeMs": self.processing_time_ms,
            "indexUid": index_uid,
            "query": query,
        }


class FakeSession:
    def __init__(self, rows_by_segment: dict[str, list[dict]]) -> None:
        self.rows_by_segment = rows_by_segment
        self.calls: list[dict] = []

    def run(self, query: str, parameters: dict | None = None) -> list[dict]:
        active_parameters = parameters or {}
        self.calls.append(active_parameters)
        return self.rows_by_segment.get(str(active_parameters.get("segment_id")), [])


def test_detect_graph_query_hints_finds_korean_temporal_reference_terms() -> None:
    hints = detect_graph_query_hints("아까 말했던 개념이 이 부분에서 뭐였지?")

    assert hints["has_graph_hint"] is True
    assert hints["has_temporal_hint"] is True
    assert hints["has_reference_hint"] is True
    assert "아까" in hints["matched_terms"]
    assert "말했던" in hints["matched_terms"]


def test_serialize_traversal_record_preserves_graph_evidence_fields() -> None:
    record = {
        "type": "reference_resolution",
        "candidate_segment_id": "seg_3",
        "candidate_rank": 1,
        "candidate_score": 0.91,
        "graph_path": ["segment:seg_3", "reference_mention:ref_1", "concept:pot_odds"],
        "resolved_concept": {"key": "concept:pot_odds", "canonical": "pot odds"},
        "source_segment": {"segment_id": "seg_1", "transcript_text": "Pot odds are the price."},
        "visual_entity": {"entity_id": "entity_board", "text": "pot odds board"},
        "frame": {"frame_id": "frame_000001", "frame_path": "/tmp/frame.jpg"},
        "reason": "reference resolved to previous concept",
        "evidence": ["REFERS_TO", "RESOLVES_TO"],
        "score": "0.82",
    }

    serialized = serialize_traversal_record(record)

    assert serialized["candidate"] == {"segment_id": "seg_3", "rank": 1, "score": 0.91}
    assert serialized["resolved_concept"]["canonical"] == "pot odds"
    assert serialized["source_segment"]["segment_id"] == "seg_1"
    assert serialized["visual_entity"]["entity_id"] == "entity_board"
    assert serialized["frame"]["frame_id"] == "frame_000001"
    assert serialized["reason"] == "reference resolved to previous concept"
    assert serialized["evidence"] == ["REFERS_TO", "RESOLVES_TO"]
    assert serialized["score"] == 0.82


def test_run_graph_traversal_returns_previous_concept_and_visual_evidence() -> None:
    session = FakeSession(
        {
            "seg_3": [
                {
                    "type": "previous_segment",
                    "candidate_segment_id": "seg_3",
                    "candidate_rank": 1,
                    "candidate_score": 0.7,
                    "graph_path": ["segment:seg_1", "segment:seg_2", "segment:seg_3"],
                    "resolved_concept": {"key": "concept:pot_odds", "canonical": "pot odds"},
                    "source_segment": {"segment_id": "seg_1", "transcript_text": "Pot odds first."},
                    "visual_entity": {"entity_id": "entity_board", "text": "pot odds"},
                    "frame": {"frame_id": "frame_000001"},
                    "reason": "Previous transcript segment reached through NEXT_SEGMENT lookback",
                    "evidence": ["NEXT_SEGMENT", "MENTIONS", "REPRESENTS", "ALIGNED_WITH"],
                    "score": 0.72,
                }
            ]
        }
    )

    evidence = run_graph_traversal(
        session=session,
        project_id="project",
        candidates=[{"segment_id": "seg_3", "rank": 1, "score": 0.7}],
        config=GraphTraversalConfig(lookback_segments=2, per_candidate_limit=3),
    )

    assert session.calls[0]["project_id"] == "project"
    assert session.calls[0]["limit"] == 3
    assert evidence[0]["source_segment"]["segment_id"] == "seg_1"
    assert evidence[0]["resolved_concept"]["canonical"] == "pot odds"
    assert "ALIGNED_WITH" in evidence[0]["evidence"]


def test_graph_traversal_cypher_scopes_project_evidence_nodes() -> None:
    cypher = graph_traversal_cypher(lookback_segments=3)

    assert "MATCH (start:GraphNode:Segment {project_id: $project_id, segment_id: $segment_id})" in cypher
    assert "WHERE all(segment IN nodes(path) WHERE segment.project_id = $project_id)" in cypher
    assert cypher.count("WHERE entity.project_id = $project_id") == 2
    assert "WHERE entity_frame.project_id = $project_id" in cypher
    assert "WHERE aligned_frame.project_id = $project_id" in cypher
    assert "WHERE target.project_id IS NULL OR target.project_id = $project_id" in cypher
    assert "WHERE segment_frame.project_id = $project_id" in cypher
    assert "WHERE contained_entity.project_id = $project_id" in cypher
    assert "WHERE frame.project_id = $project_id" in cypher
    assert "CASE WHEN target:Frame THEN target ELSE coalesce(segment_frame, entity_frame) END AS frame" in cypher


def test_graph_query_extends_query_project_response_with_graph_summary(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_project_fixture(project_dir)
    session = FakeSession(
        {
            "seg_3": [
                {
                    "type": "previous_segment",
                    "candidate_segment_id": "seg_3",
                    "candidate_rank": 1,
                    "candidate_score": 0.88,
                    "graph_path": ["segment:seg_1", "segment:seg_2", "segment:seg_3"],
                    "resolved_concept": {"key": "concept:pot_odds", "canonical": "pot odds"},
                    "source_segment": {"segment_id": "seg_1", "transcript_text": "Pot odds concept."},
                    "visual_entity": {"entity_id": "entity_board", "text": "pot odds board"},
                    "frame": {"frame_id": "frame_000001", "frame_path": "/tmp/f1.jpg"},
                    "reason": "Previous transcript segment reached through NEXT_SEGMENT lookback",
                    "evidence": ["NEXT_SEGMENT", "MENTIONS", "REPRESENTS", "ALIGNED_WITH"],
                    "score": 0.72,
                }
            ]
        }
    )

    response = graph_query(
        client=FakeClient(
            [
                {
                    "segment_id": "seg_3",
                    "sample_id": "seg_3",
                    "video_id": "video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "이 부분을 다시 보자",
                    "_rankingScore": 0.88,
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="아까 말했던 개념",
        session=session,
    )

    assert response["counts"]["search_hits"] == 1
    assert response["counts"]["graph_evidence"] == 1
    assert response["graph_availability"]["meilisearch"]["status"] == "hit"
    assert response["graph_availability"]["neo4j"]["status"] == "queried"
    assert response["graph_availability"]["graph_evidence"]["status"] == "hit"
    assert response["traversal_summary"]["evidence_by_type"] == {"previous_segment": 1}
    assert response["graph_evidence"][0]["source_segment"]["segment_id"] == "seg_1"


def test_graph_query_distinguishes_meili_miss_from_neo4j_miss(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_project_fixture(project_dir)

    meili_miss = graph_query(
        client=FakeClient([]),
        index_uid="local_segments",
        project_dir=project_dir,
        query="아까 말했던 개념",
        session=FakeSession({}),
    )
    assert meili_miss["graph_availability"]["meilisearch"]["status"] == "miss"
    assert meili_miss["graph_availability"]["graph_evidence"]["status"] == "skipped_no_meilisearch_hits"

    neo4j_miss = graph_query(
        client=FakeClient(
            [
                {
                    "segment_id": "seg_3",
                    "sample_id": "seg_3",
                    "video_id": "video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "이 부분을 다시 보자",
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="아까 말했던 개념",
        session=FakeSession({"seg_3": []}),
    )
    assert neo4j_miss["graph_availability"]["meilisearch"]["status"] == "hit"
    assert neo4j_miss["graph_availability"]["neo4j"]["status"] == "queried"
    assert neo4j_miss["graph_availability"]["graph_evidence"]["status"] == "miss"


def test_neo4j_smoke_traversal_when_runtime_is_available(tmp_path: Path) -> None:
    config = load_neo4j_config()
    if not _neo4j_socket_open(config.uri):
        pytest.skip("Neo4j is not running")

    project_dir = tmp_path / "project"
    _write_project_fixture(project_dir)
    ingest_project_graph(project_dir=project_dir, config=config)

    response = graph_query(
        client=FakeClient(
            [
                {
                    "segment_id": "seg_3",
                    "sample_id": "seg_3",
                    "video_id": "video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "이 부분을 다시 보자",
                }
            ]
        ),
        index_uid="local_segments",
        project_dir=project_dir,
        query="아까 말했던 개념",
        config=config,
    )

    assert response["graph_availability"]["neo4j"]["status"] == "queried"
    assert response["graph_availability"]["graph_evidence"]["status"] == "hit"


def _write_project_fixture(project_dir: Path) -> None:
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 4.0, "Pot odds concept appears first", ["frame_000001"]),
            _segment("seg_2", 2, 5.0, 9.0, "Bridge explanation", ["frame_000006"]),
            _segment("seg_3", 3, 10.0, 14.0, "이 부분을 다시 보자", ["frame_000012"]),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_000001", "timestamp": 1.0, "frame_path": "/tmp/f1.jpg"},
            {"frame_id": "frame_000006", "timestamp": 6.0, "frame_path": "/tmp/f6.jpg"},
            {"frame_id": "frame_000012", "timestamp": 12.0, "frame_path": "/tmp/f12.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_board",
                "project_id": "project",
                "frame_id": "frame_000001",
                "timestamp": 1.0,
                "frame_path": "/tmp/f1.jpg",
                "text": "pot odds board",
                "entity_type": "ocr_text",
                "confidence": 0.9,
                "source": "fixture",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_seg_1_entity_board",
                "project_id": "project",
                "segment_id": "seg_1",
                "entity_id": "entity_board",
                "frame_id": "frame_000001",
                "link_type": "time_overlap+lexical_match",
                "score": 1.0,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["pot", "odds"],
                "mention_candidate": [],
            }
        ],
    )
    (project_dir / "domain_lexicon.json").write_text(
        json.dumps({"aliases": {"pot odds": ["pot", "odds", "팟오즈"]}}),
        encoding="utf-8",
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
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _neo4j_socket_open(uri: str) -> bool:
    from oarag.neo4j import endpoint_from_uri

    endpoint = endpoint_from_uri(uri)
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=0.25):
            return True
    except OSError:
        return False
