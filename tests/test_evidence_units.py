import json
from pathlib import Path

from oarag.retrieval.evidence_unit_index import query_project_evidence_units
from oarag.retrieval.evidence_units import build_project_evidence_units, build_project_visual_states
from oarag.retrieval.project_index import index_project_evidence_units


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
            {
                "frame_id": "frame_gradient",
                "timestamp": 12.0,
                "frame_path": "frames/gradient.jpg",
                "detected_text": ["Gradient Descent"],
            },
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
                "detected_text": ["descent"],
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
    assert target["source_quality"]["has_detected_text"] is True
    assert target["source_quality"]["has_visual_description"] is True
    assert target["source_quality"]["visual_state_detected_text_count"] == 1
    assert target["source_quality"]["visual_entity_detected_text_count"] == 1
    assert target["source_quality"]["visual_description_count"] == 1
    assert target["visual_states"][0]["detected_text"] == ["Gradient Descent"]
    assert target["visual_entities"][0]["detected_text"] == ["descent"]
    assert summary["counts"]["units_with_detected_text"] >= 1
    assert summary["counts"]["units_with_visual_description"] >= 1
    assert "gradient arrow" in target["semantic_text"]

    manifest = json.loads((project_dir / "manifests" / "project_manifest.json").read_text())
    assert manifest["artifacts"]["evidence_units"].endswith("segments/evidence_units.jsonl")
    assert manifest["counts"]["evidence_units"] == 3


def test_build_project_evidence_units_reports_link_signal_and_verified_source_aggregates(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "link_diagnostics_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_target", 10.0, 14.0, "This diagram shows the matrix rule here."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [{"frame_id": "frame_target", "timestamp": 12.0}],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_matrix",
                "project_id": "sample_project",
                "frame_id": "frame_target",
                "timestamp": 12.0,
                "text": "matrix diagram",
                "entity_type": "diagram_component",
                "confidence": 0.95,
                "source": "vlm",
                "source_model": "real-vlm",
                "visual_description": "Matrix rule diagram with highlighted position.",
                "position": "upper left",
                "relations": [{"type": "points_to", "target": "rule"}],
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_candidate_visual",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_matrix",
                "frame_id": "frame_target",
                "link_type": "time_overlap+lexical_match+mention_candidate",
                "score": 0.8,
                "evidence": [
                    "time_overlap",
                    "lexical_match",
                    "mention_candidate",
                    "visual_text_match",
                    "visual_description_match",
                    "position_match",
                    "semantic_hint",
                    "domain_lexicon_match",
                ],
                "time_overlap": True,
                "lexical_match": ["matrix"],
                "mention_candidate": ["this"],
            },
            {
                "link_id": "link_timestamp_only",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_matrix",
                "frame_id": "frame_target",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
            },
            {
                "link_id": "link_verified_human",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_matrix",
                "frame_id": "frame_target",
                "link_type": "time_overlap+lexical_match",
                "score": 0.95,
                "evidence": ["time_overlap", "lexical_match"],
                "time_overlap": True,
                "lexical_match": ["matrix"],
                "verified": True,
                "verified_by": "human_gold",
            },
            {
                "link_id": "link_verified_vlm",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_matrix",
                "frame_id": "frame_target",
                "link_type": "time_overlap+visual_description_match",
                "score": 0.94,
                "evidence": ["time_overlap", "visual_description_match"],
                "time_overlap": True,
                "verification_status": "verified",
                "verifier": "vlm_verifier",
            },
            {
                "link_id": "link_verified_strict",
                "project_id": "sample_project",
                "segment_id": "seg_target",
                "entity_id": "ent_matrix",
                "frame_id": "frame_target",
                "link_type": "time_overlap+domain_lexicon_match",
                "score": 0.93,
                "evidence": ["time_overlap", "domain_lexicon_match"],
                "time_overlap": True,
                "status": "verified",
                "verification_source": "strict_deterministic_rule",
            },
        ],
    )

    summary = build_project_evidence_units(
        project_dir=project_dir,
        previous_neighbor_count=0,
        next_neighbor_count=0,
    )

    unit = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")[0]
    candidate_counts = unit["source_quality"]["candidate_link_signal_counts"]
    verified_sources = unit["source_quality"]["verified_link_source_counts"]
    assert unit["source_quality"]["candidate_link_count"] == 1
    assert unit["source_quality"]["timestamp_fallback_link_count"] == 1
    assert unit["source_quality"]["verified_link_count"] == 3
    assert candidate_counts["temporal_overlap"] == 2
    assert candidate_counts["lexical_overlap"] == 1
    assert candidate_counts["mention_deictic_hook"] == 1
    assert candidate_counts["spatial_position"] == 1
    assert candidate_counts["visual_text_overlap"] == 1
    assert candidate_counts["vlm_object_visual_description_overlap"] == 1
    assert candidate_counts["semantic_domain_hint"] == 1
    assert candidate_counts["timestamp_fallback"] == 1
    assert verified_sources["explicit_verified_flag"] == 1
    assert verified_sources["explicit_verified_status"] == 2
    assert verified_sources["human_gold"] == 1
    assert verified_sources["vlm_verifier"] == 1
    assert verified_sources["strict_deterministic_rule"] == 1
    assert unit["source_quality"]["candidate_visual_support"]["paper_claim_eligible"] is False
    assert unit["source_quality"]["verified_object_alignment"]["paper_claim_eligible"] is True

    diagnostics = summary["link_diagnostics"]
    assert diagnostics["candidate_visual_support"]["units_with_candidate_link"] == 1
    assert diagnostics["candidate_visual_support"]["units_with_timestamp_fallback_link"] == 1
    assert diagnostics["verified_object_alignment"]["units_with_verified_object_alignment"] == 1
    assert diagnostics["verified_object_alignment"]["verified_links"] == 3
    assert diagnostics["verified_object_alignment"]["timestamp_fallback_counted_as_verified"] is False
    assert summary["counts"]["units_with_candidate_link"] == 1
    assert summary["counts"]["units_with_timestamp_fallback_link"] == 1
    assert summary["counts"]["units_with_verified_object_alignment"] == 1


def test_build_project_visual_states_writes_public_safe_artifact_and_manifest(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "visual_state_project"
    manifest_path = project_dir / "manifests" / "project_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"project_id": "visual_state_project", "video_id": "lecture_video"}),
        encoding="utf-8",
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {
                "frame_id": "frame_intro",
                "timestamp": 10.0,
                "frame_path": "/private/raw/frame_intro.jpg",
                "detected_text": ["Intro"],
            },
            {
                "frame_id": "frame_middle",
                "timestamp": 20.0,
                "frame_path": "/private/raw/frame_middle.jpg",
                "visual_description": "A public-safe board summary.",
            },
            {
                "frame_id": "frame_end",
                "timestamp": 35.0,
                "frame_path": "/private/raw/frame_end.jpg",
            },
        ],
    )

    summary = build_project_visual_states(
        project_dir=project_dir,
        state_padding_seconds=5.0,
        min_visual_states=3,
        fail_on_visual_state_gate=True,
    )

    visual_states_path = project_dir / "manifests" / "visual_states.jsonl"
    rows = _read_jsonl(visual_states_path)
    artifact_text = visual_states_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["counts"]["visual_states_total"] == 3
    assert summary["interval_duration_seconds"]["buckets"]["5-15s"] == 3
    assert summary["coverage_gate"]["status"] == "passed"
    assert rows[0]["schema_version"] == "oarag-visual-states-jsonl-v1"
    assert rows[0]["valid_start_time"] == 5.0
    assert rows[0]["valid_end_time"] == 15.0
    assert "frame_path" not in artifact_text
    assert "/private/raw" not in artifact_text
    assert manifest["artifacts"]["visual_states"].endswith("manifests/visual_states.jsonl")
    assert manifest["counts"]["visual_states"] == 3
    assert manifest["visual_state_storage"]["schema_version"] == "oarag-visual-states-jsonl-v1"


def test_build_project_evidence_units_loads_visual_states_artifact_and_reports_gate(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "external_visual_state_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_external", 10.0, 12.0, "A transcript-only mention."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_states.jsonl",
        [
            {
                "visual_state_id": "vstate_external",
                "project_id": "sample_project",
                "video_id": "sample_video",
                "representative_frame_id": "frame_external",
                "frame_ids": ["frame_external"],
                "valid_start_time": 9.0,
                "valid_end_time": 13.0,
                "state_summary": "External diagram state.",
                "source": "fixture_external",
                "frame_path": "/private/raw/frame_external.jpg",
            },
        ],
    )

    summary = build_project_evidence_units(
        project_dir=project_dir,
        visual_states=Path("manifests/visual_states.jsonl"),
        previous_neighbor_count=0,
        next_neighbor_count=0,
        visual_state_min_coverage_ratio=1.0,
        visual_state_min_total=1,
        fail_on_visual_state_gate=True,
    )

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    unit = rows[0]
    output_text = (project_dir / "segments" / "evidence_units.jsonl").read_text(encoding="utf-8")
    manifest = json.loads((project_dir / "manifests" / "project_manifest.json").read_text())
    assert unit["visual_state_ids"] == ["vstate_external"]
    assert unit["alignment_status"] == "candidate"
    assert unit["source_quality"]["has_visual_state"] is True
    assert unit["source_quality"]["has_verified_link"] is False
    assert summary["visual_state_config"]["mode"] == "external_visual_states_artifact"
    assert summary["visual_state_config"]["artifact_loaded"] is True
    assert summary["visual_state_coverage"]["coverage_gate"]["status"] == "passed"
    assert summary["visual_state_coverage"]["source"] == "external_visual_states_artifact"
    assert summary["visual_state_coverage"]["transcript_only_units"] == 0
    assert "frame_path" not in output_text
    assert "/private/raw" not in output_text
    assert manifest["artifacts"]["visual_states"].endswith("manifests/visual_states.jsonl")


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
    assert rows[0]["concept_ids"] == []
    assert rows[0]["source_quality"]["has_concept"] is False


def test_build_project_evidence_units_adds_concept_graph_search_fields(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "concept_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments.jsonl",
        [
            _segment(
                "public_001_0001",
                10.0,
                14.0,
                "We define the optimization objective.",
            ),
            _segment(
                "public_001_0002",
                20.0,
                24.0,
                "Step size controls the next update.",
            ),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "concept_graph.jsonl",
        [
            {
                "schema_version": "oarag-concept-graph-v1",
                "record_type": "concept_node",
                "concept_id": "concept_gradient_descent",
                "lecture_id": "public_lecture_graph_001",
                "label": "Gradient descent",
                "aliases": ["steepest descent"],
                "concept_type": "algorithm",
                "source_evidence_unit_ids": ["evu_public_001_0001"],
                "evidence_sources": [
                    {
                        "evidence_unit_id": "evu_public_001_0001",
                        "source_type": "transcript",
                        "source_signal": "transcript_statement",
                        "confidence": 0.86,
                    }
                ],
                "confidence": 0.86,
            },
            {
                "schema_version": "oarag-concept-graph-v1",
                "record_type": "concept_node",
                "concept_id": "concept_loss",
                "lecture_id": "public_lecture_graph_001",
                "label": "Loss function",
                "aliases": ["objective function"],
                "concept_type": "metric",
                "source_evidence_unit_ids": ["evu_public_001_0001"],
                "evidence_sources": [
                    {
                        "evidence_unit_id": "evu_public_001_0001",
                        "source_type": "slide_text",
                        "source_signal": "slide_text",
                        "confidence": 0.82,
                    }
                ],
                "confidence": 0.82,
            },
            {
                "schema_version": "oarag-concept-graph-v1",
                "record_type": "concept_node",
                "concept_id": "concept_learning_rate",
                "lecture_id": "public_lecture_graph_001",
                "label": "Learning rate",
                "aliases": ["step size"],
                "concept_type": "hyperparameter",
                "source_evidence_unit_ids": ["evu_public_001_0002"],
                "evidence_sources": [
                    {
                        "evidence_unit_id": "evu_public_001_0002",
                        "source_type": "transcript",
                        "source_signal": "transcript_statement",
                        "confidence": 0.8,
                    }
                ],
                "confidence": 0.8,
            },
            {
                "schema_version": "oarag-concept-graph-v1",
                "record_type": "relation_edge",
                "edge_id": "edge_gradient_descent_related_loss",
                "lecture_id": "public_lecture_graph_001",
                "source_concept_id": "concept_gradient_descent",
                "relation_type": "related_to",
                "target_concept_id": "concept_loss",
                "evidence_unit_ids": ["evu_public_001_0001"],
                "source_signals": ["timestamp_overlap"],
                "confidence": 0.35,
                "relation_status": "candidate",
            },
            {
                "schema_version": "oarag-concept-graph-v1",
                "record_type": "relation_edge",
                "edge_id": "edge_gradient_descent_uses_learning_rate",
                "lecture_id": "public_lecture_graph_001",
                "source_concept_id": "concept_gradient_descent",
                "relation_type": "uses",
                "target_concept_id": "concept_learning_rate",
                "evidence_unit_ids": ["evu_public_001_0002"],
                "source_signals": ["transcript_statement", "strict_deterministic_rule"],
                "confidence": 0.78,
                "relation_status": "verified",
            },
        ],
    )

    summary = build_project_evidence_units(
        project_dir=project_dir,
        previous_neighbor_count=0,
        next_neighbor_count=0,
    )

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    first = next(row for row in rows if row["target_segment_id"] == "public_001_0001")
    second = next(row for row in rows if row["target_segment_id"] == "public_001_0002")
    assert first["concept_ids"] == ["concept_gradient_descent", "concept_loss"]
    assert first["concept_labels"] == ["Gradient descent", "Loss function"]
    assert first["concept_aliases"] == ["steepest descent", "objective function"]
    assert "Gradient descent related to Loss function" in first["concept_relation_text"]
    assert "steepest descent" in first["semantic_text"]
    assert first["source_quality"]["has_concept"] is True
    assert first["source_quality"]["has_concept_relation"] is True
    assert first["source_quality"]["timestamp_only_concept_relation_count"] == 1
    assert first["source_quality"]["verified_object_alignment"][
        "timestamp_fallback_counted_as_verified"
    ] is False
    assert second["concept_labels"] == ["Learning rate", "Gradient descent"]
    assert "Gradient descent uses Learning rate" in second["concept_relation_text"]
    assert summary["counts"]["concept_graph_records_total"] == 5
    assert summary["counts"]["units_with_concept"] == 2
    assert summary["counts"]["units_with_concept_relation"] == 2
    assert summary["counts"]["units_with_concept_search_text"] == 2
    assert summary["concept_field_coverage"]["concept_graph_loaded"] is True
    assert summary["concept_field_coverage"]["timestamp_only_concept_relation_mentions"] == 1
    assert summary["concept_field_coverage"][
        "timestamp_only_counted_as_verified_object_alignment"
    ] is False

    manifest = json.loads((project_dir / "manifests" / "project_manifest.json").read_text())
    assert manifest["evidence_unit_storage"]["concept_field_coverage"][
        "evidence_units_with_concept_search_text"
    ] == 2


def test_index_project_evidence_units_indexes_artifact_and_preserves_fallback_status(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_seg_1",
                "project_id": "sample_project",
                "video_id": "sample_video",
                "target_segment_id": "seg_1",
                "source_segment_ids": ["seg_1"],
                "start_time": 10.0,
                "end_time": 14.0,
                "transcript_window_text": "This gradient arrow shows descent.",
                "visual_state_ids": ["vstate_1"],
                "visual_entity_ids": ["ent_gradient_arrow"],
                "verified_entity_link_ids": [],
                "candidate_entity_link_ids": ["link_timestamp"],
                "candidate_entity_link_statuses": {
                    "link_timestamp": "timestamp_fallback",
                },
                "alignment_status": "candidate",
                "source_quality": {
                    "has_visual_state": True,
                    "has_visual_entity": True,
                    "has_vlm_entity": False,
                    "has_verified_link": False,
                    "has_timestamp_fallback_link": True,
                },
                "evidence_text": "Transcript: This gradient arrow shows descent.",
                "semantic_text": "gradient arrow descent",
                "concept_ids": ["concept_gradient_descent"],
                "concepts": [
                    {
                        "concept_id": "concept_gradient_descent",
                        "label": "Gradient descent",
                        "aliases": ["steepest descent"],
                    }
                ],
                "concept_relations": [
                    {
                        "edge_id": "edge_gradient_descent_uses_learning_rate",
                        "source_label": "Gradient descent",
                        "target_label": "Learning rate",
                        "relation_text": "Gradient descent uses Learning rate",
                    }
                ],
            }
        ],
    )
    client = _FakeMeiliClient()

    summary = index_project_evidence_units(
        client,
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        reset=True,
    )

    assert client.created_indexes == [("sample_evidence_units", "evidence_unit_id")]
    assert client.deleted_indexes == ["sample_evidence_units"]
    assert client.settings["searchableAttributes"][:2] == ["semantic_text", "evidence_text"]
    assert "concept_labels" in client.settings["searchableAttributes"]
    assert "concept_aliases" in client.settings["searchableAttributes"]
    assert "concept_relation_text" in client.settings["searchableAttributes"]
    assert summary["indexed_documents"] == 1
    assert summary["alignment_status_counts"] == {"candidate": 1}
    assert summary["source_quality_counts"]["has_concept"] == 1
    assert summary["source_quality_counts"]["has_concept_relation"] == 1
    indexed = client.documents[0]
    assert indexed["evidence_unit_id"] == "evu_seg_1"
    assert indexed["concept_labels"] == ["Gradient descent", "Learning rate"]
    assert indexed["concept_aliases"] == ["steepest descent"]
    assert indexed["concept_relation_text"] == "Gradient descent uses Learning rate"
    assert indexed["candidate_entity_link_statuses"] == {
        "link_timestamp": "timestamp_fallback"
    }
    assert indexed["verified_entity_link_ids"] == []


def test_query_project_evidence_units_returns_required_fields(tmp_path: Path) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "sample_project"
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_seg_1",
                "project_id": "sample_project",
            }
        ],
    )
    hit = {
        "evidence_unit_id": "evu_seg_1",
        "project_id": "sample_project",
        "video_id": "sample_video",
        "target_segment_id": "seg_1",
        "source_segment_ids": ["seg_1"],
        "start_time": 10.0,
        "end_time": 14.0,
        "visual_state_ids": ["vstate_1"],
        "visual_entity_ids": ["ent_gradient_arrow"],
        "verified_entity_link_ids": [],
        "candidate_entity_link_ids": ["link_timestamp"],
        "candidate_entity_link_statuses": {
            "link_timestamp": "timestamp_fallback",
        },
        "alignment_status": "candidate",
        "source_quality": {
            "has_verified_link": False,
            "timestamp_fallback_link_count": 1,
        },
        "evidence_text": "Transcript: gradient arrow",
        "semantic_text": "gradient arrow",
        "concept_ids": ["concept_gradient_descent"],
        "concept_labels": ["Gradient descent"],
        "concept_aliases": ["steepest descent"],
        "concept_relation_text": "Gradient descent uses Learning rate",
        "concepts": [{"concept_id": "concept_gradient_descent", "label": "Gradient descent"}],
        "concept_relations": [
            {
                "edge_id": "edge_gradient_descent_uses_learning_rate",
                "relation_text": "Gradient descent uses Learning rate",
            }
        ],
        "_rankingScore": 0.9,
    }
    client = _FakeMeiliClient(search_hits=[hit])

    response = query_project_evidence_units(
        client=client,
        index_uid="sample_evidence_units",
        project_dir=project_dir,
        query="gradient arrow",
        limit=1,
    )

    assert client.search_calls == [
        {
            "index_uid": "sample_evidence_units",
            "query": "gradient arrow",
            "limit": 1,
            "filter": 'project_id = "sample_project"',
        }
    ]
    candidate = response["candidates"][0]
    required_fields = {
        "evidence_unit_id",
        "project_id",
        "video_id",
        "target_segment_id",
        "source_segment_ids",
        "start_time",
        "end_time",
        "visual_state_ids",
        "visual_entity_ids",
        "concept_ids",
        "concept_labels",
        "concept_aliases",
        "concept_relation_text",
        "concepts",
        "concept_relations",
        "candidate_entity_link_ids",
        "candidate_entity_link_statuses",
        "alignment_status",
        "source_quality",
    }
    assert required_fields <= set(candidate)
    assert candidate["verified_entity_link_ids"] == []
    assert candidate["candidate_entity_link_statuses"]["link_timestamp"] == "timestamp_fallback"
    assert candidate["concept_labels"] == ["Gradient descent"]
    assert candidate["concept_relation_text"] == "Gradient descent uses Learning rate"
    assert candidate["source_quality"]["has_verified_link"] is False


def test_explicit_verified_link_is_not_downgraded_to_timestamp_fallback(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "artifacts" / "projects" / "verified_project"
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            _segment("seg_verified", 10.0, 14.0, "The instructor refers to this object."),
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "frames_manifest.jsonl",
        [
            {"frame_id": "frame_verified", "timestamp": 12.0, "frame_path": "frames/verified.jpg"},
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "ent_verified",
                "project_id": "verified_project",
                "frame_id": "frame_verified",
                "timestamp": 12.0,
                "frame_path": "frames/verified.jpg",
                "bbox": None,
                "text": "object",
                "entity_type": "diagram_component",
                "confidence": 0.9,
                "source": "vlm",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "link_explicit_verified_bool",
                "project_id": "verified_project",
                "segment_id": "seg_verified",
                "entity_id": "ent_verified",
                "frame_id": "frame_verified",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap", "timestamp_fallback"],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
                "verified": True,
            },
            {
                "link_id": "link_explicit_verified_status",
                "project_id": "verified_project",
                "segment_id": "seg_verified",
                "entity_id": "ent_verified",
                "frame_id": "frame_verified",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": [],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
                "verification_status": "verified",
            },
            {
                "link_id": "link_plain_timestamp",
                "project_id": "verified_project",
                "segment_id": "seg_verified",
                "entity_id": "ent_verified",
                "frame_id": "frame_verified",
                "link_type": "time_overlap",
                "score": 0.2,
                "evidence": ["time_overlap"],
                "time_overlap": True,
                "lexical_match": [],
                "mention_candidate": [],
            },
        ],
    )

    summary = build_project_evidence_units(project_dir=project_dir)

    rows = _read_jsonl(project_dir / "segments" / "evidence_units.jsonl")
    unit = rows[0]
    assert unit["verified_entity_link_ids"] == [
        "link_explicit_verified_bool",
        "link_explicit_verified_status",
    ]
    assert unit["candidate_entity_link_ids"] == ["link_plain_timestamp"]
    assert unit["candidate_entity_link_statuses"] == {
        "link_plain_timestamp": "timestamp_fallback"
    }
    assert unit["source_quality"]["has_verified_link"] is True
    assert unit["source_quality"]["verified_link_count"] == 2
    assert unit["source_quality"]["timestamp_fallback_link_count"] == 1
    assert unit["alignment_status"] == "verified"
    assert summary["counts"]["verified_links"] == 2
    assert summary["counts"]["timestamp_fallback_links"] == 1
    assert summary["alignment_status_counts"] == {"verified": 1}


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


class _FakeMeiliClient:
    def __init__(self, search_hits: list[dict] | None = None) -> None:
        self.search_hits = search_hits or []
        self.created_indexes: list[tuple[str, str]] = []
        self.deleted_indexes: list[str] = []
        self.settings: dict = {}
        self.documents: list[dict] = []
        self.search_calls: list[dict] = []

    def delete_index(self, index_uid: str):
        self.deleted_indexes.append(index_uid)
        return {"taskUid": 1}

    def create_index(self, index_uid: str, primary_key: str):
        self.created_indexes.append((index_uid, primary_key))
        return {"taskUid": 2}

    def update_settings(self, index_uid: str, settings: dict):
        self.settings = settings
        return {"taskUid": 3}

    def add_documents(self, index_uid: str, documents: list[dict]):
        self.documents.extend(documents)
        return {"taskUid": 4}

    def search(self, index_uid: str, query: str, *, limit: int, filter: str):
        self.search_calls.append(
            {
                "index_uid": index_uid,
                "query": query,
                "limit": limit,
                "filter": filter,
            }
        )
        return {"hits": self.search_hits[:limit], "processingTimeMs": 1}

    def wait_task(self, task, ignored_error_codes=None):
        return task
