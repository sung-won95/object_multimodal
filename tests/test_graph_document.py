from __future__ import annotations

import json
from pathlib import Path

from oarag.graph_document import GRAPH_DOCUMENT_SCHEMA_VERSION, build_graph_document


def test_build_graph_document_converts_project_artifacts_deterministically(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_json(
        project_dir / "manifests" / "project_manifest.json",
        {"project_id": "sample_project", "title": "Sample Lecture"},
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [_segment("seg_fallback", 99, 99.0, 100.0, "unused fallback", [])],
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_1", 1, 0.0, 4.0, "Intro board", ["frame_000001"]),
            _segment(
                "seg_2",
                2,
                5.0,
                9.0,
                "Bet size is on this board",
                [{"frame_id": "frame_000006", "frame_path": "frames/frame_000006.jpg"}],
            ),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_000001",
                "project_id": "sample_project",
                "video_id": "video_a",
                "timestamp": 1.0,
                "frame_path": "frames/frame_000001.jpg",
            },
            {
                "frame_id": "frame_000006",
                "project_id": "sample_project",
                "video_id": "video_a",
                "timestamp": 6.0,
                "frame_path": "frames/frame_000006.jpg",
            },
            {
                "frame_id": "frame_000007",
                "project_id": "sample_project",
                "video_id": "video_a",
                "timestamp": 7.0,
                "frame_path": "frames/frame_000007.jpg",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_board",
                "project_id": "sample_project",
                "frame_id": "frame_000006",
                "timestamp": 6.0,
                "frame_path": "frames/frame_000006.jpg",
                "bbox": None,
                "text": "BET SIZE BOARD",
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
                "project_id": "sample_project",
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
    _write_json(
        project_dir / "domain_lexicon.json",
        {"aliases": {"bet": ["wager"], "size": ["sizing"], "board": ["보드"]}},
    )

    first = build_graph_document(project_dir=project_dir).to_dict()
    second = build_graph_document(project_dir=project_dir).to_dict()

    assert first == second
    assert first["schema_version"] == GRAPH_DOCUMENT_SCHEMA_VERSION
    assert first["project_id"] == "sample_project"
    assert first["availability"]["artifacts"]["segments"] == {
        "available": True,
        "path": str(project_dir / "segments" / "lecture_segments_aligned.jsonl"),
        "selected": "lecture_segments_aligned.jsonl",
    }
    assert first["availability"]["counts"] == {
        "segments": 2,
        "frames": 3,
        "visual_entities": 1,
        "entity_links": 1,
    }
    assert first["counts"]["nodes_by_label"] == {
        "Concept": 4,
        "Frame": 3,
        "Project": 1,
        "Segment": 2,
        "Video": 1,
        "VisualEntity": 1,
    }
    assert first["counts"]["relationships_by_type"] == {
        "ALIGNED_WITH": 2,
        "CONTAINS": 1,
        "HAS_FRAME": 3,
        "HAS_SEGMENT": 2,
        "HAS_VIDEO": 1,
        "LINKED_TO": 1,
        "MENTIONS": 5,
        "NEXT_SEGMENT": 1,
        "REPRESENTS": 3,
        "TEMPORALLY_NEAR": 1,
    }

    node_keys = {node["key"] for node in first["nodes"]}
    assert {
        "project:sample_project",
        "video:video_a",
        "segment:seg_2",
        "frame:sample_project:frame_000006",
        "visual_entity:sample_project:entity_board",
        "concept:bet",
        "concept:board",
        "concept:size",
    } <= node_keys

    relationship_pairs = {
        (relationship["type"], relationship["start_node_key"], relationship["end_node_key"])
        for relationship in first["relationships"]
    }
    assert ("HAS_VIDEO", "project:sample_project", "video:video_a") in relationship_pairs
    assert ("HAS_SEGMENT", "video:video_a", "segment:seg_2") in relationship_pairs
    assert ("HAS_FRAME", "video:video_a", "frame:sample_project:frame_000006") in relationship_pairs
    assert ("ALIGNED_WITH", "segment:seg_2", "frame:sample_project:frame_000006") in relationship_pairs
    assert ("TEMPORALLY_NEAR", "segment:seg_2", "frame:sample_project:frame_000007") in relationship_pairs
    assert (
        "CONTAINS",
        "frame:sample_project:frame_000006",
        "visual_entity:sample_project:entity_board",
    ) in relationship_pairs
    assert ("MENTIONS", "segment:seg_2", "concept:bet") in relationship_pairs
    assert ("REPRESENTS", "visual_entity:sample_project:entity_board", "concept:bet") in relationship_pairs
    assert ("LINKED_TO", "segment:seg_2", "visual_entity:sample_project:entity_board") in relationship_pairs


def test_build_graph_document_keeps_scoped_frame_and_visual_keys_when_ids_collide(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_json(
        project_dir / "manifests" / "project_manifest.json",
        {"project_id": "sample_project", "title": "Sample Lecture"},
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment(
                "seg_1",
                1,
                0.0,
                2.0,
                "Shared id appears on the board",
                [{"frame_id": "shared_id"}],
            )
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "shared_id",
                "project_id": "sample_project",
                "video_id": "video_a",
                "timestamp": 1.0,
                "frame_path": "frames/shared_id.jpg",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "shared_id",
                "project_id": "sample_project",
                "frame_id": "shared_id",
                "timestamp": 1.0,
                "text": "BOARD",
                "entity_type": "ocr_text",
                "confidence": 0.91,
            }
        ],
    )

    document = build_graph_document(project_dir=project_dir).to_dict()

    nodes_by_key = {node["key"]: node for node in document["nodes"]}
    assert nodes_by_key["frame:sample_project:shared_id"]["labels"] == ["Frame"]
    assert nodes_by_key["visual_entity:sample_project:shared_id"]["labels"] == ["VisualEntity"]

    relationship_pairs = {
        (relationship["type"], relationship["start_node_key"], relationship["end_node_key"])
        for relationship in document["relationships"]
    }
    assert (
        "CONTAINS",
        "frame:sample_project:shared_id",
        "visual_entity:sample_project:shared_id",
    ) in relationship_pairs
    assert (
        "ALIGNED_WITH",
        "segment:seg_1",
        "frame:sample_project:shared_id",
    ) in relationship_pairs


def test_build_graph_document_falls_back_when_optional_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [_segment("seg_1", 1, 0.0, 2.0, "Transcript only", [])],
    )

    document = build_graph_document(project_dir=project_dir).to_dict()

    assert document["project_id"] == "sample_project"
    assert document["availability"]["counts"] == {
        "segments": 1,
        "frames": 0,
        "visual_entities": 0,
        "entity_links": 0,
    }
    assert document["availability"]["artifacts"]["segments"]["selected"] == "lecture_segments.jsonl"
    assert document["availability"]["artifacts"]["frames_manifest"]["available"] is False
    assert document["availability"]["artifacts"]["visual_entities"]["available"] is False
    assert document["availability"]["artifacts"]["entity_links"]["available"] is False
    assert document["counts"]["nodes_by_label"] == {
        "Concept": 2,
        "Project": 1,
        "Segment": 1,
        "Video": 1,
    }
    assert document["counts"]["relationships_by_type"] == {
        "HAS_SEGMENT": 1,
        "HAS_VIDEO": 1,
        "MENTIONS": 2,
    }
    assert any("frames_manifest artifact is missing" in warning for warning in document["warnings"])


def _segment(
    segment_id: str,
    sample_index: int,
    start_time: float,
    end_time: float,
    transcript_text: str,
    frame_refs: list[object],
) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "sample_project",
        "video_id": "video_a",
        "sample_id": segment_id,
        "sample_index": sample_index,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2.0,
        "transcript_text": transcript_text,
        "mention_candidates": ["board"] if "board" in transcript_text.casefold() else [],
        "frame_refs": frame_refs,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
