from __future__ import annotations

import json
from pathlib import Path

from oarag.vlm_evidence_validator import (
    is_paper_quality_vlm_entity,
    validate_vlm_object_evidence,
)


def test_validator_reports_ocr_only_as_issue_200_skip(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ocr_private",
                "frame_id": "frame_private",
                "timestamp": 1.0,
                "frame_path": str(project_dir / "frames" / "frame_private.jpg"),
                "text": "PRIVATE OCR TEXT",
                "entity_type": "ocr_text",
                "source": "ocr:tesseract",
            }
        ],
    )

    report = validate_vlm_object_evidence(project_dir=project_dir)

    entity = report["visual_entity_coverage"]
    assert report["status"] == "skipped"
    assert report["blocker"] is True
    assert report["skip_reason"] == "missing_real_vlm_artifacts_from_issue_200"
    assert entity["ocr_only_entity_count"] == 1
    assert entity["vlm_source_entity_count"] == 0
    assert entity["paper_quality_vlm_entity_count"] == 0


def test_validator_rejects_mock_deterministic_vlm_entities_and_units(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl", [])
    deterministic_entity = {
        "entity_id": "ent_mock_private",
        "frame_id": "frame_private",
        "timestamp": 2.0,
        "frame_path": str(project_dir / "frames" / "frame_private.jpg"),
        "text": "Deterministic VLM observation: frame_private at 2.000s",
        "entity_type": "frame_summary",
        "source": "vlm:stub-vlm",
        "source_model": "stub-vlm",
        "visual_description": "Deterministic VLM observation: frame_private at 2.000s",
        "detected_text": "PRIVATE DETECTED TEXT",
    }
    _write_jsonl(project_dir / "manifests" / "visual_entities.jsonl", [deterministic_entity])
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_private",
                "visual_entities": [deterministic_entity],
                "source_quality": {
                    "has_vlm_entity": True,
                    "has_visual_description": True,
                    "has_detected_text": True,
                },
            }
        ],
    )

    report = validate_vlm_object_evidence(project_dir=project_dir)

    entity = report["visual_entity_coverage"]
    units = report["evidence_unit_coverage"]
    assert report["status"] == "blocked"
    assert report["skip_reason"] == "no_paper_quality_vlm_object_evidence"
    assert entity["vlm_source_entity_count"] == 1
    assert entity["paper_quality_vlm_entity_count"] == 0
    assert entity["deterministic_or_mock_vlm_entity_count"] == 1
    assert entity["rejection_reason_counts"]["deterministic_or_mock_vlm_output"] == 1
    assert units["coverage_source"] == "visual_entity_context"
    assert units["units_with_vlm_entity"] == 0
    assert units["units_with_rejected_vlm_entity"] == 1


def test_validator_reports_empty_vlm_source_as_invalid(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _write_jsonl(project_dir / "manifests" / "vlm_parser_output.jsonl", [])
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_empty_private",
                "frame_id": "frame_private",
                "timestamp": 3.0,
                "frame_path": str(project_dir / "frames" / "frame_private.jpg"),
                "text": "",
                "entity_type": "diagram",
                "source": "vlm:gpt-4o-mini",
                "source_model": "gpt-4o-mini",
            }
        ],
    )

    report = validate_vlm_object_evidence(project_dir=project_dir)

    entity = report["visual_entity_coverage"]
    assert report["status"] == "blocked"
    assert entity["vlm_source_entity_count"] == 1
    assert entity["paper_quality_vlm_entity_count"] == 0
    assert entity["empty_invalid_entity_count"] == 1
    assert entity["rejection_reason_counts"]["empty_or_invalid_entity"] == 1


def test_validator_accepts_real_vlm_like_fixture_and_redacts_outputs(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    output_path = tmp_path / "public" / "metrics.json"
    summary_path = tmp_path / "public" / "summary.md"
    _write_jsonl(project_dir / "manifests" / "vlm_visual_observations.jsonl", [])
    real_like_entity = {
        "entity_id": "ent_real_private",
        "frame_id": "frame_private",
        "timestamp": 4.0,
        "frame_path": str(project_dir / "frames" / "frame_private.jpg"),
        "text": "PRIVATE object label",
        "entity_type": "diagram_component",
        "source": "vlm:gpt-4o-mini",
        "source_model": "gpt-4o-mini",
        "visual_description": "PRIVATE diagram object description",
        "detected_text": "PRIVATE detected label",
        "position": {"region": "center"},
        "relations": [{"type": "points_to", "target": "axis"}],
    }
    _write_jsonl(project_dir / "manifests" / "visual_entities.jsonl", [real_like_entity])
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_private",
                "visual_entities": [real_like_entity],
                "visual_states": [{"detected_text": ["PRIVATE state text"]}],
                "source_quality": {
                    "has_vlm_entity": True,
                    "has_visual_description": True,
                    "has_detected_text": True,
                },
            }
        ],
    )

    report = validate_vlm_object_evidence(
        project_dir=project_dir,
        output_path=output_path,
        summary_path=summary_path,
    )

    entity = report["visual_entity_coverage"]
    units = report["evidence_unit_coverage"]
    assert report["status"] == "passed"
    assert report["blocker"] is False
    assert entity["vlm_source_entity_count"] == 1
    assert entity["paper_quality_vlm_entity_count"] == 1
    assert entity["paper_quality_field_coverage"]["visual_description"]["count"] == 1
    assert entity["paper_quality_field_coverage"]["position"]["count"] == 1
    assert entity["paper_quality_field_coverage"]["relations"]["count"] == 1
    assert entity["paper_quality_field_coverage"]["detected_text"]["count"] == 1
    assert units["units_with_vlm_entity"] == 1
    assert units["units_with_visual_description"] == 1
    assert units["units_with_detected_text"] == 1
    assert is_paper_quality_vlm_entity(real_like_entity) is True

    public_text = output_path.read_text(encoding="utf-8") + summary_path.read_text(
        encoding="utf-8"
    )
    for sensitive in [
        "PRIVATE",
        "ent_real_private",
        "evu_private",
        "frame_private.jpg",
        str(project_dir),
    ]:
        assert sensitive not in public_text


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
