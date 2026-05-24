from __future__ import annotations

import json
from pathlib import Path

from oarag.graph_document import build_graph_document
from oarag.reference_resolution import (
    REFERENCE_LOOKBACK_SECONDS,
    REFERENCE_LOOKBACK_SEGMENT_COUNT,
    resolve_references,
)
from oarag.domain_lexicon import domain_lexicon_from_payload
from oarag.schemas import EntityLink, VisualEntity


def test_reference_resolution_uses_time_ordered_lookback_and_readable_evidence() -> None:
    lexicon = domain_lexicon_from_payload(
        {"aliases": {"pot odds": ["pot odds"], "board": ["보드"]}},
        source_path=Path("domain_lexicon.json"),
    )
    entity = VisualEntity.from_dict(
        {
            "entity_id": "entity_pot_odds_board",
            "project_id": "sample_project",
            "frame_id": "frame_000001",
            "timestamp": 1.0,
            "text": "POT ODDS",
            "entity_type": "ocr_text",
            "confidence": 0.92,
        }
    )
    link = EntityLink.from_dict(
        {
            "link_id": "link_seg_intro_entity_pot_odds_board",
            "project_id": "sample_project",
            "segment_id": "seg_intro",
            "entity_id": "entity_pot_odds_board",
            "frame_id": "frame_000001",
            "link_type": "time_overlap+lexical_match",
            "score": 1.2,
            "evidence": ["time_overlap", "lexical_match"],
        }
    )

    resolutions = resolve_references(
        segments=[
            _segment("seg_intro", 0.0, 4.0, "We define pot odds on the board.", ["frame_000001"], ["pot odds"]),
            _segment("seg_gap", 8.0, 10.0, "Now compare ranges.", [], ["range"]),
            _segment("seg_ref", 12.0, 14.0, "아까 말했던 개념을 다시 보겠습니다.", [], []),
        ],
        domain_lexicon=lexicon,
        entities=[entity],
        links=[link],
    )

    assert len(resolutions) == 1
    resolution = resolutions[0]
    assert resolution.segment_id == "seg_ref"
    assert resolution.hint_text == "아까 말했던 개념"
    assert resolution.lookback_segment_id == "seg_gap"
    assert resolution.lookback_segment_count == 1
    assert resolution.lookback_seconds == 4.0
    assert resolution.score == 1.0
    assert "Detected reference hint" in resolution.reason
    assert "lookback_transcript=Now compare ranges." in resolution.evidence
    assert [(target.target_type, target.local_id) for target in resolution.targets] == [
        ("segment", "seg_gap"),
        ("concept", "range"),
    ]


def test_reference_resolution_respects_fixed_lookback_count_and_seconds() -> None:
    lexicon = domain_lexicon_from_payload(
        {"aliases": {"alpha": ["alpha"], "beta": ["beta"]}},
        source_path=Path("domain_lexicon.json"),
    )

    too_far_by_count = resolve_references(
        segments=[
            _segment("seg_alpha", 0.0, 2.0, "Alpha concept.", [], ["alpha"]),
            _segment("seg_a", 4.0, 5.0, "x", [], []),
            _segment("seg_b", 6.0, 7.0, "y", [], []),
            _segment("seg_c", 8.0, 9.0, "z", [], []),
            _segment("seg_ref", 10.0, 11.0, "this concept comes back", [], []),
        ],
        domain_lexicon=lexicon,
    )
    too_far_by_seconds = resolve_references(
        segments=[
            _segment("seg_beta", 0.0, 2.0, "Beta concept.", [], ["beta"]),
            _segment(
                "seg_ref",
                REFERENCE_LOOKBACK_SECONDS + 10.0,
                REFERENCE_LOOKBACK_SECONDS + 11.0,
                "previous concept matters here",
                [],
                [],
            ),
        ],
        domain_lexicon=lexicon,
    )

    assert REFERENCE_LOOKBACK_SEGMENT_COUNT == 3
    assert REFERENCE_LOOKBACK_SECONDS == 180.0
    assert too_far_by_count == []
    assert too_far_by_seconds == []


def test_graph_document_adds_reference_mention_and_resolution_relationships(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_json(
        project_dir / "manifests" / "project_manifest.json",
        {"project_id": "sample_project", "title": "Reference Lecture"},
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment(
                "seg_intro",
                0.0,
                4.0,
                "Pot odds appears on this board.",
                [{"frame_id": "frame_000001", "frame_path": "frames/frame_000001.jpg"}],
                ["pot odds"],
            ),
            _segment("seg_ref", 8.0, 10.0, "아까 말했던 개념을 다시 보겠습니다.", [], []),
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
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_pot_odds_board",
                "project_id": "sample_project",
                "frame_id": "frame_000001",
                "timestamp": 1.0,
                "text": "POT ODDS",
                "entity_type": "ocr_text",
                "confidence": 0.92,
            }
        ],
    )
    _write_json(
        project_dir / "domain_lexicon.json",
        {"aliases": {"pot odds": ["pot odds"], "board": ["보드"]}},
    )

    document = build_graph_document(project_dir=project_dir).to_dict()

    reference_nodes = [
        node for node in document["nodes"] if node["labels"] == ["ReferenceMention"]
    ]
    assert len(reference_nodes) == 1
    assert reference_nodes[0]["properties"]["hint_text"] == "아까 말했던 개념"
    assert reference_nodes[0]["properties"]["lookback_segment_count"] == 1
    assert "Pot odds appears" in " ".join(reference_nodes[0]["properties"]["evidence"])

    relationship_pairs = {
        (relationship["type"], relationship["start_node_key"], relationship["end_node_key"])
        for relationship in document["relationships"]
    }
    reference_key = reference_nodes[0]["key"]
    assert ("REFERS_TO", "segment:seg_ref", reference_key) in relationship_pairs
    assert ("RESOLVES_TO", reference_key, "segment:seg_intro") in relationship_pairs
    assert ("RESOLVES_TO", reference_key, "concept:pot_odds") in relationship_pairs
    assert (
        "RESOLVES_TO",
        reference_key,
        "visual_entity:sample_project:entity_pot_odds_board",
    ) in relationship_pairs

    resolves_relationships = [
        relationship
        for relationship in document["relationships"]
        if relationship["type"] == "RESOLVES_TO"
    ]
    assert all("reason" in relationship["properties"] for relationship in resolves_relationships)
    assert all("evidence" in relationship["properties"] for relationship in resolves_relationships)


def _segment(
    segment_id: str,
    start_time: float,
    end_time: float,
    transcript_text: str,
    frame_refs: list[object],
    mention_candidates: list[str],
) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "sample_project",
        "video_id": "video_a",
        "sample_id": segment_id,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2.0,
        "transcript_text": transcript_text,
        "frame_refs": frame_refs,
        "mention_candidates": mention_candidates,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
