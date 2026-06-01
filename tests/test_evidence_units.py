import json
from pathlib import Path

from oarag.retrieval.evidence_units import build_project_evidence_units


def test_build_project_evidence_units_marks_timestamp_only_as_candidate_fallback(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_before", 0.0, 4.0, "We set up optimization."),
            _segment("seg_target", 10.0, 14.0, "This gradient arrow shows the descent direction."),
            _segment("seg_after", 20.0, 24.0, "Now the loss curve gets smaller."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_gradient", "timestamp": 12.0, "frame_path": "frames/gradient.jpg"},
            {"frame_id": "frame_loss", "timestamp": 22.0, "frame_path": "frames/loss.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_gradient_arrow",
                "project_id": "sample_project",
                "frame_id": "frame_gradient",
                "timestamp": 12.0,
                "frame_path": "frames/gradient.jpg",
                "bbox": None,
                "text": "gradient arrow",
                "entity_type": "diagram_component",
                "confidence": 0.91,
                "source": "vlm",
                "visual_description": "Arrow indicating the descent direction.",
            },
            {
                "entity_id": "ent_loss_ocr",
                "project_id": "sample_project",
                "frame_id": "frame_loss",
                "timestamp": 22.0,
                "frame_path": "frames/loss.jpg",
                "bbox": None,
                "text": "loss",
                "entity_type": "ocr_text",
                "confidence": 0.8,
                "source": "ocr",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_gradient",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_gradient_arrow",
                "frame_id": "frame_gradient",
                "link_type": "time_overlap+lexical_match",
                "score": 0.9,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["gradient"],
                "mention_candidate": [],
            },
            {
                "link_id": "link_loss_timestamp",
                "project_id": "sample_project",
                "segment_id": "seg_after",
                "entity_id": "ent_loss_ocr",
                "frame_id": "frame_loss",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
            },
        ],
    )

    summary = build_project_evidence_units(
        project_dir=project_dir,
        previous_neighbor_count=0,
        next_neighbor_count=1,
    )

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    target = next(row for row in rows if row["target_segment_id"] == "seg_target")
    assert summary["counts"]["evidence_units_total"] == 3
    assert target["source_segment_ids"] == ["seg_target", "seg_after"]
    assert "This gradient arrow" in target["transcript_window_text"]
    assert target["visual_state_ids"]
    assert target["visual_entity_ids"] == ["ent_gradient_arrow", "ent_loss_ocr"]
    assert target["candidate_entity_link_ids"] == ["link_gradient", "link_loss_timestamp"]
    assert target["verified_entity_link_ids"] == []
    assert target["candidate_entity_link_statuses"]["link_gradient"] == "candidate"
    assert target["candidate_entity_link_statuses"]["link_loss_timestamp"] == "timestamp_fallback"
    assert target["alignment_status"] == "candidate"
    assert target["source_quality"]["has_vlm_entity"] is True
    assert target["source_quality"]["has_verified_link"] is False
    assert target["source_quality"]["has_timestamp_fallback_link"] is True
    assert "gradient arrow" in target["semantic_text"]

    manifest = json.loads((project_dir / "manifests" / "project_manifest.json").read_text())
    assert manifest["artifacts"]["evidence_units"].endswith("segments/evidence_units.jsonl")
    assert manifest["counts"]["evidence_units"] == 3


def test_build_project_evidence_units_supports_transcript_only_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "transcript_only"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [
            _segment("seg_only", 0.0, 5.0, "A transcript-only explanation."),
        ],
    )

    build_project_evidence_units(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    assert rows[0]["alignment_status"] == "transcript_only"
    assert rows[0]["modality"] == ["speech"]
    assert rows[0]["visual_state_ids"] == []
    assert rows[0]["visual_entity_ids"] == []


def _segment(segment_id: str, start_time: float, end_time: float, text: str) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "sample_project",
        "video_id": "sample_video",
        "sample_id": segment_id,
        "sample_index": 0,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2.0,
        "transcript_text": text,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
