import json
from pathlib import Path

from oarag.retrieval.evidence_units import build_project_evidence_units
from oarag.vision.entity_link_verifier import (
    VLM_ENTITY_LINK_VERIFIER_SOURCE,
    load_human_audit_agreement,
    verify_entity_links_vlm,
)


def test_vlm_link_verifier_promotes_only_verified_and_counts_source(
    tmp_path: Path,
) -> None:
    project_dir = _write_project_fixture(tmp_path)
    _write_jsonl(
        project_dir / "manifests" / "vlm_link_decisions.jsonl",
        [
            {
                "link_id": "link_verified",
                "decision": "verified",
                "reason_code": "speaker_refers_to_visible_chart",
                "public_reason": "Speaker refers to the visible chart.",
                "confidence": 0.91,
            },
            {
                "link_id": "link_rejected",
                "decision": "rejected",
                "reason_code": "visual_entity_not_referenced",
                "public_reason": "Visual entity is not referenced.",
                "confidence": 0.84,
            },
            {
                "link_id": "link_uncertain",
                "decision": "uncertain",
                "reason_code": "ambiguous_deictic_reference",
                "public_reason": "Reference is ambiguous.",
                "confidence": 0.51,
            },
        ],
    )

    summary = verify_entity_links_vlm(
        project_dir=project_dir,
        backend="jsonl",
        model="fixture-vlm",
        options={"jsonl_path": "manifests/vlm_link_decisions.jsonl"},
        output_path=Path("manifests/entity_links.vlm.jsonl"),
        cache_path=Path("manifests/entity_links.vlm.cache.json"),
        report_path=Path("reports/vlm_report.json"),
        audit_template_path=Path("reports/vlm_audit_template.jsonl"),
    )

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.vlm.jsonl")
    promoted = next(row for row in rows if row["link_id"] == "link_verified")
    rejected = next(row for row in rows if row["link_id"] == "link_rejected")
    uncertain = next(row for row in rows if row["link_id"] == "link_uncertain")

    assert promoted["alignment_status"] == "verified"
    assert promoted["verification_status"] == "verified"
    assert promoted["verified_link_source"] == VLM_ENTITY_LINK_VERIFIER_SOURCE
    assert promoted["verification_source"] == VLM_ENTITY_LINK_VERIFIER_SOURCE
    assert promoted["verifier"] == VLM_ENTITY_LINK_VERIFIER_SOURCE
    assert promoted["reason_metadata"]["vlm_verifier"]["decision"] == "verified"
    assert rejected.get("alignment_status") != "verified"
    assert rejected.get("verification_status") != "verified"
    assert rejected["reason_metadata"]["vlm_verifier"]["decision"] == "rejected"
    assert uncertain.get("alignment_status") != "verified"
    assert uncertain.get("verification_status") != "verified"
    assert uncertain["reason_metadata"]["vlm_verifier"]["decision"] == "uncertain"

    evidence_summary = build_project_evidence_units(
        project_dir=project_dir,
        entity_links=Path("manifests/entity_links.vlm.jsonl"),
        neighbor_count=0,
    )
    unit = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")[0]
    source_counts = unit["source_quality"]["verified_link_source_counts"]
    assert unit["source_quality"]["verified_link_count"] == 1
    assert source_counts["vlm_verifier"] == 1
    assert evidence_summary["counts"]["verified_links"] == 1
    assert (
        evidence_summary["link_diagnostics"]["verified_object_alignment"][
            "verified_link_source_counts"
        ]["vlm_verifier"]
        == 1
    )

    report = json.loads((project_dir / "reports" / "vlm_report.json").read_text())
    assert report["counts"]["verified"] == 1
    assert report["counts"]["rejected"] == 1
    assert report["counts"]["uncertain"] == 1
    assert report["manual_audit"]["human_audit_completed"] is False
    assert summary["counts"]["promoted_links"] == 1


def test_vlm_link_verifier_cache_is_public_safe_and_replays_without_backend(
    tmp_path: Path,
) -> None:
    project_dir = _write_project_fixture(tmp_path)
    fixture = project_dir / "manifests" / "vlm_link_decisions.jsonl"
    cache = project_dir / "manifests" / "entity_links.vlm.cache.json"
    _write_jsonl(
        fixture,
        [
            {
                "link_id": "link_verified",
                "decision": "verified",
                "reason_code": "speaker_refers_to_visible_chart",
                "public_reason": "Speaker refers to the visible chart.",
            },
            {"link_id": "link_rejected", "decision": "rejected", "reason_code": "not_referenced"},
            {"link_id": "link_uncertain", "decision": "uncertain", "reason_code": "ambiguous"},
        ],
    )

    first = verify_entity_links_vlm(
        project_dir=project_dir,
        backend="jsonl",
        model="fixture-vlm",
        options={"jsonl_path": fixture},
        output_path=Path("manifests/entity_links.vlm.jsonl"),
        cache_path=cache,
    )
    fixture.unlink()
    second = verify_entity_links_vlm(
        project_dir=project_dir,
        backend="jsonl",
        model="fixture-vlm",
        options={"jsonl_path": fixture},
        output_path=Path("manifests/entity_links.vlm.second.jsonl"),
        cache_path=cache,
    )

    cache_text = cache.read_text(encoding="utf-8")
    assert "Gradient descent follows this chart" not in cache_text
    assert "frames/frame_target.jpg" not in cache_text
    assert first["counts"]["links_fresh"] == 3
    assert second["counts"]["links_cached"] == 3
    assert second["status"] == "completed"


def test_vlm_link_verifier_unavailable_backend_writes_explicit_skip(
    tmp_path: Path,
) -> None:
    project_dir = _write_project_fixture(tmp_path)

    summary = verify_entity_links_vlm(
        project_dir=project_dir,
        backend="jsonl",
        model="fixture-vlm",
        options={"jsonl_path": "manifests/missing_decisions.jsonl"},
        output_path=Path("manifests/entity_links.vlm.jsonl"),
        report_path=Path("reports/vlm_report.json"),
        skip_on_unavailable=True,
    )

    rows = _read_jsonl(project_dir / "manifests" / "entity_links.vlm.jsonl")
    report = json.loads((project_dir / "reports" / "vlm_report.json").read_text())
    assert summary["status"] == "skipped"
    assert report["skip"]["skipped"] is True
    assert report["skip"]["no_candidate_auto_promotion"] is True
    assert report["counts"]["promoted_links"] == 0
    assert all(row.get("verification_status") != "verified" for row in rows)


def test_human_audit_agreement_loader_requires_real_completed_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    _write_jsonl(
        path,
        [
            {"vlm_decision": "verified", "human_decision": "verified"},
            {"vlm_decision": "rejected", "human_decision": "uncertain"},
            {"vlm_decision": "uncertain", "human_decision": ""},
        ],
    )

    summary = load_human_audit_agreement(path)

    assert summary["completed_rows"] == 2
    assert summary["matches"] == 1
    assert summary["agreement_rate"] == 0.5
    assert summary["meets_issue_281_sample_size"] is False


def _write_project_fixture(tmp_path: Path) -> Path:
    project_dir = tmp_path / "artifacts" / "projects" / "vlm_verifier_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_target",
                "project_id": "vlm_verifier_project",
                "video_id": "lecture_fixture",
                "sample_id": "sample_target",
                "sample_index": 0,
                "start_time": 10.0,
                "end_time": 14.0,
                "timestamp_center": 12.0,
                "transcript_text": "Gradient descent follows this chart.",
                "mention_candidates": ["this", "chart"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_target",
                "timestamp": 12.0,
                "frame_path": "frames/frame_target.jpg",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            _entity("ent_verified", "loss chart"),
            _entity("ent_rejected", "optimizer table"),
            _entity("ent_uncertain", "background formula"),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            _link("link_verified", "ent_verified"),
            _link("link_rejected", "ent_rejected"),
            _link("link_uncertain", "ent_uncertain"),
        ],
    )
    return project_dir


def _entity(entity_id: str, text: str) -> dict:
    return {
        "entity_id": entity_id,
        "project_id": "vlm_verifier_project",
        "frame_id": "frame_target",
        "timestamp": 12.0,
        "frame_path": "frames/frame_target.jpg",
        "bbox": None,
        "text": text,
        "entity_type": "diagram_component",
        "confidence": 0.9,
        "source": "vlm",
        "visual_description": f"Synthetic visual entity: {text}",
        "detected_text": text,
    }


def _link(link_id: str, entity_id: str) -> dict:
    return {
        "link_id": link_id,
        "project_id": "vlm_verifier_project",
        "segment_id": "seg_target",
        "entity_id": entity_id,
        "frame_id": "frame_target",
        "link_type": "time_overlap+visual_description_match",
        "score": 0.7,
        "evidence": ["time_overlap", "visual_description_match"],
        "time_overlap": True,
        "lexical_match": [],
        "mention_candidate": ["this"],
        "score_breakdown": {"time_overlap": 0.2, "visual_description_match": 0.5},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
