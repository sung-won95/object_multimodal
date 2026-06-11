import json
from pathlib import Path

from oarag.retrieval.evidence_units import build_project_evidence_units
from oarag.vision.strict_deterministic_verifier import (
    STRICT_DETERMINISTIC_LINK_SOURCE,
    verify_entity_links_strict_deterministic,
)


def test_strict_verifier_promotes_exact_visible_text_overlap(tmp_path: Path) -> None:
    project_dir = _write_project_fixture(tmp_path)

    summary = verify_entity_links_strict_deterministic(
        project_dir=project_dir,
        output_path=Path("manifests/entity_links.strict.jsonl"),
        report_path=Path("reports/strict_report.json"),
        lexical_overlap_threshold=0.5,
        sweep_thresholds=[0.25, 0.5, 0.75],
    )

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.strict.jsonl")
    promoted = rows[0]
    assert promoted["link_id"] == "link_visible_text"
    assert promoted["alignment_status"] == "verified"
    assert promoted["verification_status"] == "verified"
    assert promoted["verified_link_source"] == STRICT_DETERMINISTIC_LINK_SOURCE
    assert promoted["verification_source"] == STRICT_DETERMINISTIC_LINK_SOURCE
    assert (
        promoted["reason_metadata"]["strict_deterministic_verifier"]["source"]
        == STRICT_DETERMINISTIC_LINK_SOURCE
    )
    assert promoted["reason_metadata"]["strict_deterministic_verifier"]["matched_term_count"] == 4
    assert "matched_terms" not in promoted["reason_metadata"]["strict_deterministic_verifier"]

    report = json.loads((project_dir / "reports" / "strict_report.json").read_text())
    assert report["privacy"]["raw_transcript"] == "redacted"
    assert report["counts"]["promoted_links"] == 1
    assert report["counts"]["time_overlap_only_excluded"] == 1
    assert summary["counts"]["promoted_links"] == 1
    assert any(row["threshold"] == 0.5 and row["promoted_links"] == 1 for row in report["threshold_sweep"])


def test_strict_verifier_never_marks_timestamp_fallback_verified(tmp_path: Path) -> None:
    project_dir = _write_project_fixture(tmp_path)

    verify_entity_links_strict_deterministic(
        project_dir=project_dir,
        output_path=Path("manifests/entity_links.strict.jsonl"),
        lexical_overlap_threshold=0.1,
    )

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.strict.jsonl")
    fallback = next(row for row in rows if row["link_id"] == "link_timestamp_fallback")
    assert fallback.get("alignment_status") != "verified"
    assert fallback.get("verification_status") != "verified"
    assert fallback.get("verification_source") != STRICT_DETERMINISTIC_LINK_SOURCE


def test_strict_verifier_output_counts_as_verified_source_in_evidence_units(tmp_path: Path) -> None:
    project_dir = _write_project_fixture(tmp_path)

    verify_entity_links_strict_deterministic(
        project_dir=project_dir,
        output_path=Path("manifests/entity_links.strict.jsonl"),
        lexical_overlap_threshold=0.5,
    )
    summary = build_project_evidence_units(
        project_dir=project_dir,
        entity_links=Path("manifests/entity_links.strict.jsonl"),
        neighbor_count=0,
    )

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    verified_unit = next(row for row in rows if row["target_segment_id"] == "seg_visible")
    fallback_unit = next(row for row in rows if row["target_segment_id"] == "seg_fallback")
    source_counts = verified_unit["source_quality"]["verified_link_source_counts"]

    assert verified_unit["alignment_status"] == "verified"
    assert verified_unit["source_quality"]["has_verified_link"] is True
    assert source_counts[STRICT_DETERMINISTIC_LINK_SOURCE] == 1
    assert fallback_unit["alignment_status"] == "candidate"
    assert fallback_unit["source_quality"]["verified_link_count"] == 0
    assert summary["counts"]["units_with_verified_object_alignment"] == 1
    assert summary["counts"]["verified_links"] == 1
    assert (
        summary["link_diagnostics"]["verified_object_alignment"]["verified_link_source_counts"][
            STRICT_DETERMINISTIC_LINK_SOURCE
        ]
        == 1
    )


def _write_project_fixture(tmp_path: Path) -> Path:
    project_dir = tmp_path / "artifacts" / "projects" / "strict_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_visible", 10.0, 14.0, "Gradient descent follows this loss curve."),
            _segment("seg_below", 20.0, 24.0, "Gradient appears without enough slide terms."),
            _segment("seg_fallback", 30.0, 34.0, "Loss curve is visible here."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_visible", "timestamp": 12.0, "frame_path": "frames/visible.jpg"},
            {"frame_id": "frame_below", "timestamp": 22.0, "frame_path": "frames/below.jpg"},
            {"frame_id": "frame_fallback", "timestamp": 32.0, "frame_path": "frames/fallback.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            _entity(
                "ent_visible",
                "frame_visible",
                12.0,
                text="loss curve",
                detected_text=["gradient", "descent"],
            ),
            _entity(
                "ent_below",
                "frame_below",
                22.0,
                text="gradient descent objective update",
                detected_text=[],
            ),
            _entity(
                "ent_fallback",
                "frame_fallback",
                32.0,
                text="loss curve",
                detected_text=[],
            ),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            _link(
                "link_visible_text",
                "seg_visible",
                "ent_visible",
                "frame_visible",
                score=0.9,
                evidence=["time_overlap", "lexical_match", "visual_text_match"],
            ),
            _link(
                "link_below_threshold",
                "seg_below",
                "ent_below",
                "frame_below",
                score=0.7,
                evidence=["time_overlap", "lexical_match", "visual_text_match"],
            ),
            _link(
                "link_timestamp_fallback",
                "seg_fallback",
                "ent_fallback",
                "frame_fallback",
                score=0.2,
                evidence=["time_overlap", "timestamp_fallback"],
                link_type="time_overlap",
            ),
        ],
    )
    return project_dir


def _segment(segment_id: str, start_time: float, end_time: float, text: str) -> dict:
    return {
        "segment_id": segment_id,
        "project_id": "strict_project",
        "video_id": "strict_video",
        "sample_id": segment_id,
        "sample_index": 0,
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": (start_time + end_time) / 2.0,
        "transcript_text": text,
    }


def _entity(
    entity_id: str,
    frame_id: str,
    timestamp: float,
    *,
    text: str,
    detected_text: list[str],
) -> dict:
    return {
        "entity_id": entity_id,
        "project_id": "strict_project",
        "frame_id": frame_id,
        "timestamp": timestamp,
        "frame_path": f"frames/{frame_id}.jpg",
        "bbox": None,
        "text": text,
        "entity_type": "diagram_component",
        "confidence": 0.9,
        "source": "vlm",
        "visible_text": detected_text,
        "detected_text": detected_text,
    }


def _link(
    link_id: str,
    segment_id: str,
    entity_id: str,
    frame_id: str,
    *,
    score: float,
    evidence: list[str],
    link_type: str = "time_overlap+lexical_match",
) -> dict:
    return {
        "link_id": link_id,
        "project_id": "strict_project",
        "segment_id": segment_id,
        "entity_id": entity_id,
        "frame_id": frame_id,
        "link_type": link_type,
        "score": score,
        "evidence": evidence,
        "time_overlap": True,
        "lexical_match": ["gradient"],
        "mention_candidate": [],
        "score_breakdown": {"time_overlap": 0.2, "visual_text_match": 0.5},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
