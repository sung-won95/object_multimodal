from __future__ import annotations

import json
from pathlib import Path

from oarag.cli import build_parser
from oarag.evaluation.mapping_quality import (
    MAPPING_QUALITY_SCHEMA_VERSION,
    generate_mapping_quality_report,
)


def test_mapping_quality_report_public_safe_aggregate_outputs(tmp_path: Path) -> None:
    project_a = tmp_path / "projects" / "lecture_a"
    project_b = tmp_path / "projects" / "lecture_b"
    _write_project_a(project_a)
    (project_b / "manifests").mkdir(parents=True)
    (project_b / "segments").mkdir(parents=True)
    (project_b / "manifests" / "project_manifest.json").write_text(
        json.dumps({"project_id": "public_missing_project"}),
        encoding="utf-8",
    )
    _write_jsonl(
        project_b / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_missing_raw",
                "lecture_id": "public_lecture_missing",
                "start_time": 0.0,
                "end_time": 10.0,
                "transcript_text": "",
            }
        ],
    )

    manifest_path = tmp_path / "mapping_manifest.json"
    manifest_path.write_text(
        json.dumps({"projects": [{"project_dir": str(project_a)}, {"project_dir": str(project_b)}]}),
        encoding="utf-8",
    )

    run = generate_mapping_quality_report(
        manifest_path=manifest_path,
        output_dir=tmp_path / "report",
        write_csv=True,
    )

    assert run.metrics_path.exists()
    assert run.markdown_path.exists()
    assert run.csv_path is not None
    assert run.csv_path.exists()
    assert run.payload["schema_version"] == MAPPING_QUALITY_SCHEMA_VERSION
    assert run.payload["counts"]["project_count"] == 2
    assert run.payload["artifact_counts"]["segment_count"] == 3
    assert run.payload["artifact_counts"]["evidence_unit_count"] == 3
    assert run.payload["missing_artifact_reasons"]["not_configured"] >= 1

    evidence = run.payload["evidence_quality"]
    assert evidence["entity_link_status_distribution"]["timestamp-only"] >= 1
    assert evidence["entity_link_status_distribution"]["verified"] >= 1
    assert evidence["verified_alignment_count"] == 1
    assert evidence["candidate_signal_counts"]["timestamp_fallback"] == 1
    assert evidence["verified_source_counts"]["vlm_verifier"] == 1

    concepts = run.payload["concept_quality"]
    assert concepts["lecture_local_concept_count"] == 4
    assert concepts["canonical_concept_count"] == 3
    assert concepts["alias_merge_ratio"] == 0.25
    assert concepts["cross_lecture_hub_count"] == 1
    assert concepts["hub_lecture_coverage_buckets"]["2"] == 1
    assert concepts["over_merge_reason_codes"]["same_label_or_alias_without_merge"] == 1
    assert concepts["over_merge_reason_codes"]["missing_relation_context_overlap"] == 1
    assert concepts["relation_type_counts"]["uses"] >= 1
    assert concepts["relation_supporting_evidence_count"]["uses"] >= 1

    regression = run.payload["regression_checks"]
    assert regression["timestamp_only_counted_as_verified_count"] == 0
    assert regression["timestamp_only_counted_as_verified_passed"] is True

    combined_output = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.markdown_path.read_text(encoding="utf-8"),
            run.csv_path.read_text(encoding="utf-8"),
        ]
    )
    for forbidden in (
        "SECRET_TRANSCRIPT",
        "SECRET_QUERY",
        "SECRET_EVIDENCE",
        "VISUAL_LABEL_SECRET",
        "candidate_raw_123",
        "seg_secret",
        "/private/raw",
        str(tmp_path),
    ):
        assert forbidden not in combined_output
    assert "project_" in combined_output
    assert "lecture_" in combined_output


def test_mapping_quality_report_accepts_suites_manifest_with_safe_overrides(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "suite_project"
    (project_dir / "segments").mkdir(parents=True)
    (project_dir / "manifests").mkdir(parents=True)
    (project_dir / "manifests" / "project_manifest.json").write_text(
        json.dumps(
            {
                "project_id": "base_project_should_be_overridden",
                "artifacts": {
                    "lecture_segments_aligned": "segments/base_segments.jsonl",
                    "visual_entities": "manifests/base_visual_entities.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        project_dir / "segments" / "base_segments.jsonl",
        [
            {
                "segment_id": "base_secret_segment",
                "lecture_id": "base_secret_lecture",
                "start_time": 0.0,
                "end_time": 5.0,
                "transcript_text": "BASE_SECRET_TRANSCRIPT",
            }
        ],
    )
    _write_jsonl(
        project_dir / "suite_artifacts" / "segments.jsonl",
        [
            {
                "segment_id": "suite_secret_segment_1",
                "start_time": 0.0,
                "end_time": 5.0,
                "transcript_text": "SUITE_SECRET_TRANSCRIPT",
            },
            {
                "segment_id": "suite_secret_segment_2",
                "start_time": 5.0,
                "end_time": 10.0,
                "transcript_text": "SUITE_SECRET_TRANSCRIPT",
            },
        ],
    )

    manifest_path = tmp_path / "suite_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite_id": "suite_raw_identifier_should_not_render",
                        "project_path": str(project_dir),
                        "project_id": "suite_project_public_id",
                        "video_id": "suite_video_public_id",
                        "lecture_id": "suite_lecture_public_id",
                        "artifacts": {
                            "lecture_segments_aligned": "suite_artifacts/segments.jsonl",
                            "visual_entities": str(tmp_path / "private_missing_visual_entities.jsonl"),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run = generate_mapping_quality_report(
        manifest_path=manifest_path,
        output_dir=tmp_path / "report",
        write_csv=True,
    )

    assert run.payload["counts"]["project_count"] == 1
    assert run.payload["counts"]["lecture_count"] == 1
    assert run.payload["artifact_counts"]["segment_count"] == 2
    assert run.payload["artifact_statuses"]["visual_entities:missing"] == 1
    assert run.payload["missing_artifact_reasons"]["missing_artifact"] >= 1

    combined_output = "\n".join(
        [
            run.metrics_path.read_text(encoding="utf-8"),
            run.markdown_path.read_text(encoding="utf-8"),
            run.csv_path.read_text(encoding="utf-8") if run.csv_path else "",
        ]
    )
    for forbidden in (
        "BASE_SECRET_TRANSCRIPT",
        "SUITE_SECRET_TRANSCRIPT",
        "base_secret_segment",
        "suite_secret_segment_1",
        "suite_raw_identifier_should_not_render",
        "suite_project_public_id",
        "suite_video_public_id",
        "suite_lecture_public_id",
        "private_missing_visual_entities",
        str(tmp_path),
    ):
        assert forbidden not in combined_output


def test_mapping_quality_report_resolves_paper_bundle_matrix_manifest(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "nested_suite_project"
    (project_dir / "segments").mkdir(parents=True)
    (project_dir / "manifests").mkdir(parents=True)
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "nested_secret_segment",
                "lecture_id": "nested_public_lecture",
                "start_time": 0.0,
                "end_time": 5.0,
                "transcript_text": "NESTED_SECRET_TRANSCRIPT",
            }
        ],
    )

    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    matrix_path = matrix_dir / "benchmark_matrix_manifest.json"
    matrix_path.write_text(
        json.dumps({"suites": [{"suite_id": "nested_suite", "root": "../projects/nested_suite_project"}]}),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "paper_bundle_manifest.json"
    bundle_path.write_text(
        json.dumps({"paper_bundle": {"matrix_manifest": "matrix/benchmark_matrix_manifest.json"}}),
        encoding="utf-8",
    )

    run = generate_mapping_quality_report(
        manifest_path=bundle_path,
        output_dir=tmp_path / "report",
    )

    assert run.payload["counts"]["project_count"] == 1
    assert run.payload["artifact_counts"]["segment_count"] == 1
    rendered = run.metrics_path.read_text(encoding="utf-8")
    assert "NESTED_SECRET_TRANSCRIPT" not in rendered
    assert str(tmp_path) not in rendered


def test_mapping_quality_report_cli_subcommand_parses() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "mapping-quality-report",
            "--project-dir",
            "public_project_a",
            "--project-dir",
            "public_project_b",
            "--output-dir",
            "reports/mapping_quality",
            "--csv",
        ]
    )

    assert args.project_dirs == [Path("public_project_a"), Path("public_project_b")]
    assert args.output_dir == Path("reports/mapping_quality")
    assert args.csv is True


def test_mapping_quality_link_status_normalization_and_dedupes_by_link_id(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "status_regression"
    (project_dir / "segments").mkdir(parents=True)
    (project_dir / "manifests").mkdir(parents=True)
    (project_dir / "manifests" / "project_manifest.json").write_text(
        json.dumps(
            {
                "project_id": "public_status_regression",
                "artifacts": {"evidence_units": "segments/evidence_units.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            _status_evidence_unit(
                "evu_status_timestamp",
                {"link_status_timestamp_raw": "timestamp_fallback"},
            ),
            _status_evidence_unit("evu_status_human", {"link_status_human_raw": "human_gold"}),
            _status_evidence_unit("evu_status_vlm", {"link_status_vlm_raw": "vlm_verified"}),
            _status_evidence_unit(
                "evu_status_strict",
                {"link_status_strict_raw": "strict_verified"},
            ),
            _status_evidence_unit(
                "evu_status_duplicate",
                {"link_status_duplicate_raw": "verified"},
                verified_ids=["link_status_duplicate_raw"],
                candidate_ids=["link_status_duplicate_raw", "link_status_extra_candidate_raw"],
            ),
        ],
    )

    run = generate_mapping_quality_report(
        project_dirs=[project_dir],
        output_dir=tmp_path / "report",
    )

    distribution = run.payload["evidence_quality"]["entity_link_status_distribution"]
    assert distribution["timestamp-only"] == 1
    assert distribution["verified"] == 4
    assert distribution["candidate"] == 1

    rendered = run.metrics_path.read_text(encoding="utf-8")
    assert "link_status_duplicate_raw" not in rendered
    assert "link_status_extra_candidate_raw" not in rendered


def _write_project_a(project_dir: Path) -> None:
    (project_dir / "segments").mkdir(parents=True)
    (project_dir / "manifests").mkdir(parents=True)
    (project_dir / "manifests" / "project_manifest.json").write_text(
        json.dumps(
            {
                "project_id": "public_project_a",
                "artifacts": {
                    "lecture_segments_aligned": "segments/lecture_segments_aligned.jsonl",
                    "lecture_windows": "segments/lecture_windows.jsonl",
                    "visual_states": "manifests/visual_states.jsonl",
                    "visual_entities": "manifests/visual_entities.jsonl",
                    "entity_links": "manifests/entity_links.jsonl",
                    "evidence_units": "segments/evidence_units.jsonl",
                    "concept_graph": "manifests/concept_graph.jsonl",
                    "global_concept_graph": "manifests/global_concept_graph.json",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_segments_aligned.jsonl",
        [
            {
                "segment_id": "seg_secret_loss",
                "lecture_id": "public_lecture_a",
                "start_time": 0.0,
                "end_time": 10.0,
                "transcript_text": "SECRET_TRANSCRIPT loss curve.",
            },
            {
                "segment_id": "seg_secret_empty",
                "lecture_id": "public_lecture_a",
                "start_time": 10.0,
                "end_time": 20.0,
                "transcript_text": "",
            },
        ],
    )
    _write_jsonl(
        project_dir / "segments" / "lecture_windows.jsonl",
        [
            {
                "window_id": "window_secret",
                "lecture_id": "public_lecture_a",
                "start_time": 0.0,
                "end_time": 20.0,
                "transcript_window_text": "SECRET_QUERY should never appear.",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_states.jsonl",
        [
            {
                "visual_state_id": "state_secret",
                "lecture_id": "public_lecture_a",
                "valid_start_time": 0.0,
                "valid_end_time": 15.0,
                "state_summary": "VISUAL_LABEL_SECRET",
                "frame_path": "/private/raw/frame.jpg",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "visual_entities.jsonl",
        [
            {
                "entity_id": "entity_secret",
                "lecture_id": "public_lecture_a",
                "timestamp": 5.0,
                "text": "VISUAL_LABEL_SECRET",
                "frame_path": "/private/raw/entity.jpg",
                "source": "fixture",
            }
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "entity_links.jsonl",
        [
            {
                "link_id": "candidate_raw_123",
                "lecture_id": "public_lecture_a",
                "status": "timestamp_only",
                "evidence": ["timestamp_fallback"],
            }
        ],
    )
    _write_jsonl(
        project_dir / "segments" / "evidence_units.jsonl",
        [
            {
                "evidence_unit_id": "evu_secret_timestamp",
                "lecture_id": "public_lecture_a",
                "start_time": 0.0,
                "end_time": 5.0,
                "visual_state_ids": ["state_secret"],
                "visual_entity_ids": [],
                "candidate_entity_link_statuses": {"candidate_raw_123": "timestamp_only"},
                "alignment_status": "candidate",
                "source_quality": {
                    "has_visual_state": True,
                    "has_visual_entity": False,
                    "uses_ocr_only": True,
                    "has_vlm_entity": False,
                    "timestamp_fallback_link_count": 1,
                    "candidate_link_signal_counts": {"timestamp_fallback": 1},
                    "verified_link_source_counts": {},
                    "verified_object_alignment": {
                        "has_verified_object_alignment": False,
                        "timestamp_fallback_counted_as_verified": False,
                    },
                },
                "evidence_text": "SECRET_EVIDENCE timestamp-only.",
            },
            {
                "evidence_unit_id": "evu_secret_verified",
                "lecture_id": "public_lecture_b",
                "start_time": 5.0,
                "end_time": 10.0,
                "visual_state_ids": ["state_secret"],
                "visual_entity_ids": ["entity_secret"],
                "candidate_entity_link_statuses": {"candidate_raw_verified": "verified"},
                "alignment_status": "verified",
                "source_quality": {
                    "has_visual_state": True,
                    "has_visual_entity": True,
                    "uses_ocr_only": False,
                    "has_vlm_entity": True,
                    "candidate_link_signal_counts": {"vlm_object_visual_description_overlap": 1},
                    "verified_link_source_counts": {"vlm_verifier": 1},
                    "verified_object_alignment": {
                        "has_verified_object_alignment": True,
                        "timestamp_fallback_counted_as_verified": False,
                    },
                },
                "evidence_text": "SECRET_EVIDENCE verified.",
            },
            {
                "evidence_unit_id": "evu_secret_candidate",
                "lecture_id": "public_lecture_b",
                "timestamp": 12.0,
                "visual_state_ids": [],
                "visual_entity_ids": [],
                "candidate_entity_link_ids": ["candidate_raw_candidate"],
                "alignment_status": "candidate",
                "source_quality": {
                    "has_visual_state": False,
                    "has_visual_entity": False,
                    "uses_ocr_only": False,
                    "has_vlm_entity": False,
                    "candidate_link_signal_counts": {"temporal_overlap": 1},
                    "verified_link_source_counts": {},
                    "verified_object_alignment": {
                        "has_verified_object_alignment": False,
                        "timestamp_fallback_counted_as_verified": False,
                    },
                },
                "evidence_text": "SECRET_EVIDENCE candidate.",
            },
        ],
    )
    _write_jsonl(
        project_dir / "manifests" / "concept_graph.jsonl",
        [
            _concept("concept_gd", "public_lecture_a", "Gradient descent", ["steepest descent"]),
            _concept("concept_step", "public_lecture_a", "Learning rate", ["step size"]),
            _concept("concept_sd", "public_lecture_b", "Steepest descent", ["gradient descent"]),
            _concept("concept_kernel", "public_lecture_b", "Kernel", ["core"]),
            _relation("edge_a", "public_lecture_a", "concept_gd", "uses", "concept_step"),
            _relation("edge_b", "public_lecture_b", "concept_sd", "uses", "concept_kernel"),
        ],
    )
    (project_dir / "manifests" / "global_concept_graph.json").write_text(
        json.dumps(
            {
                "schema_version": "oarag-global-concept-merge-v1",
                "counts": {
                    "local_concepts": 4,
                    "global_concepts": 3,
                    "merged_global_concepts": 1,
                    "singleton_global_concepts": 2,
                },
                "nodes": [
                    {
                        "labels": ["GraphNode", "GlobalConcept"],
                        "properties": {
                            "global_concept_id": "global_gradient_descent",
                            "canonical_label": "Gradient descent",
                            "source_lecture_ids": ["public_lecture_a", "public_lecture_b"],
                            "local_concept_count": 2,
                            "source_evidence_refs": [
                                {
                                    "lecture_id": "public_lecture_a",
                                    "evidence_unit_id": "evu_secret_timestamp",
                                    "source_id": "seg_secret_loss",
                                },
                                {
                                    "lecture_id": "public_lecture_b",
                                    "evidence_unit_id": "evu_secret_verified",
                                    "source_id": "seg_secret_verified",
                                },
                            ],
                        },
                    },
                    {
                        "labels": ["GraphNode", "GlobalConcept"],
                        "properties": {
                            "global_concept_id": "global_learning_rate",
                            "canonical_label": "Learning rate",
                            "source_lecture_ids": ["public_lecture_a"],
                            "local_concept_count": 1,
                        },
                    },
                    {
                        "labels": ["GraphNode", "GlobalConcept"],
                        "properties": {
                            "global_concept_id": "global_kernel",
                            "canonical_label": "Kernel",
                            "source_lecture_ids": ["public_lecture_b"],
                            "local_concept_count": 1,
                        },
                    },
                ],
                "relationships": [
                    {
                        "type": "GLOBAL_USES",
                        "properties": {
                            "relation_type": "uses",
                            "source_evidence_refs": [{"evidence_unit_id": "evu_secret_verified"}],
                        },
                    }
                ],
                "conflicts": [
                    {
                        "reasons": [
                            "same_label_or_alias_without_merge",
                            "missing_relation_context_overlap",
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _concept(concept_id: str, lecture_id: str, label: str, aliases: list[str]) -> dict[str, object]:
    return {
        "schema_version": "oarag-concept-graph-v1",
        "record_type": "concept_node",
        "concept_id": concept_id,
        "lecture_id": lecture_id,
        "label": label,
        "aliases": aliases,
        "concept_type": "algorithm",
        "source_evidence_unit_ids": ["evu_secret_timestamp"],
        "evidence_sources": [
            {
                "evidence_unit_id": "evu_secret_timestamp",
                "source_type": "transcript",
                "source_signal": "transcript_statement",
                "source_id": "seg_secret_loss",
            }
        ],
        "confidence": 0.8,
    }


def _relation(
    edge_id: str,
    lecture_id: str,
    source_concept_id: str,
    relation_type: str,
    target_concept_id: str,
) -> dict[str, object]:
    return {
        "schema_version": "oarag-concept-graph-v1",
        "record_type": "relation_edge",
        "edge_id": edge_id,
        "lecture_id": lecture_id,
        "source_concept_id": source_concept_id,
        "relation_type": relation_type,
        "target_concept_id": target_concept_id,
        "evidence_unit_ids": ["evu_secret_verified"],
        "source_signals": ["transcript_statement"],
        "confidence": 0.8,
    }


def _status_evidence_unit(
    evidence_unit_id: str,
    statuses: dict[str, str],
    *,
    verified_ids: list[str] | None = None,
    candidate_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "evidence_unit_id": evidence_unit_id,
        "lecture_id": "public_status_lecture",
        "start_time": 0.0,
        "end_time": 10.0,
        "candidate_entity_link_statuses": statuses,
        "verified_entity_link_ids": verified_ids or [],
        "candidate_entity_link_ids": candidate_ids or [],
        "source_quality": {
            "verified_object_alignment": {
                "has_verified_object_alignment": False,
                "timestamp_fallback_counted_as_verified": False,
            }
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
