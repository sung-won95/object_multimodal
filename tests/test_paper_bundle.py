from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.cli import build_parser
from oarag.evaluation.paper_bundle import PaperBundleError, run_paper_bundle


MIT_PAPER_MATRIX_VARIANTS = [
    "segment_lexical",
    "domain_lexicon",
    "hybrid",
    "window",
    "window_hybrid",
    "rerank",
    "evidence_unit_candidate",
    "evidence_unit_verified",
    "evidence_unit_quality_rerank",
]
GRAPH_CONCEPT_MATRIX_VARIANTS = [
    "meili_only",
    "graph_only",
    "meili_graph",
    "graph_aware_rerank",
]
GRAPH_CONCEPT_ALLOWED_PUBLIC_FIELDS = [
    "query_id",
    "hashed_refs",
    "source_types",
    "ranks",
    "counts",
    "recall_buckets",
    "rerank_deltas",
    "skip_reasons",
]


class FakePaperBundleClient:
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
        vector: list[float] | None = None,
        filter: str | list[str] | None = None,
    ) -> dict:
        del vector, filter
        mode = "semantic" if hybrid else "lexical"
        if index_uid == "public_evidence_units":
            hits = _public_evidence_unit_hits()
            return {"hits": hits[:limit], "processingTimeMs": 6, "indexUid": index_uid}
        if index_uid == "public_windows":
            return {
                "hits": [
                    {
                        "window_id": "window_loss_public",
                        "target_segment_id": "seg_loss_public",
                        "sample_id": "seg_loss_public",
                        "video_id": "public_demo_video",
                        "start_time": 0.0,
                        "end_time": 34.0,
                        "timestamp_center": 12.0,
                        "target_start_time": 10.0,
                        "target_end_time": 14.0,
                        "target_timestamp_center": 12.0,
                        "source_segment_ids": [
                            "seg_intro_public",
                            "seg_loss_public",
                            "seg_wrap_public",
                        ],
                        "transcript_window_text": "PUBLIC SYNTHETIC window must not leak",
                        "_rankingScore": 0.98 if mode == "semantic" else 0.89,
                    }
                ][:limit],
                "processingTimeMs": 4,
                "indexUid": index_uid,
            }
        if index_uid == "public_visual_entities":
            return {
                "hits": [
                    {
                        "entity_id": "ent_loss_public",
                        "frame_id": "frame_loss_public",
                        "timestamp": 12.0,
                        "text": "PUBLIC VISUAL LABEL loss curve",
                        "_rankingScore": 0.9,
                    }
                ][:limit],
                "processingTimeMs": 2,
                "indexUid": index_uid,
            }
        if index_uid == "public_segments":
            hits = [
                {
                    "segment_id": "seg_wrap_public",
                    "sample_id": "seg_wrap_public",
                    "video_id": "public_demo_video",
                    "start_time": 30.0,
                    "end_time": 34.0,
                    "timestamp_center": 32.0,
                    "transcript_text": "PUBLIC SYNTHETIC wrong transcript must not leak",
                    "_rankingScore": 0.95 if mode == "lexical" else 0.7,
                },
                {
                    "segment_id": "seg_loss_public",
                    "sample_id": "seg_loss_public",
                    "video_id": "public_demo_video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "PUBLIC SYNTHETIC correct transcript must not leak",
                    "_rankingScore": 0.55 if mode == "lexical" else 0.96,
                },
            ]
            if "validation loss" in query:
                hits = list(reversed(hits))
            return {"hits": hits[:limit], "processingTimeMs": 5, "indexUid": index_uid}
        return {"hits": [], "processingTimeMs": 1, "indexUid": index_uid}


def _public_evidence_unit_hits() -> list[dict]:
    candidate_signal_counts = {
        "temporal_overlap": 1,
        "lexical_overlap": 1,
        "mention_deictic_hook": 0,
        "spatial_position": 0,
        "visual_text_overlap": 1,
        "vlm_object_visual_description_overlap": 0,
        "semantic_domain_hint": 1,
        "timestamp_fallback": 0,
    }
    verified_signal_counts = {
        "temporal_overlap": 1,
        "lexical_overlap": 1,
        "mention_deictic_hook": 1,
        "spatial_position": 1,
        "visual_text_overlap": 1,
        "vlm_object_visual_description_overlap": 1,
        "semantic_domain_hint": 1,
        "timestamp_fallback": 0,
    }
    empty_verified_sources = {
        "explicit_verified_flag": 0,
        "explicit_verified_status": 0,
        "human_gold": 0,
        "vlm_verifier": 0,
        "strict_deterministic_rule": 0,
        "unspecified_verified": 0,
    }
    verified_sources = {
        "explicit_verified_flag": 1,
        "explicit_verified_status": 1,
        "human_gold": 0,
        "vlm_verifier": 1,
        "strict_deterministic_rule": 0,
        "unspecified_verified": 0,
    }
    return [
        {
            "evidence_unit_id": "evu_wrap_candidate_public",
            "project_id": "public_retrieval_ablation",
            "video_id": "public_demo_video",
            "target_segment_id": "seg_wrap_public",
            "source_segment_ids": ["seg_wrap_public"],
            "start_time": 30.0,
            "end_time": 34.0,
            "visual_state_ids": ["state_wrap_public"],
            "visual_entity_ids": ["ent_wrap_public"],
            "verified_entity_link_ids": [],
            "candidate_entity_link_ids": ["link_wrap_candidate_public"],
            "candidate_entity_link_statuses": {"link_wrap_candidate_public": "candidate"},
            "alignment_status": "candidate",
            "source_quality": {
                "has_visual_state": True,
                "has_visual_entity": True,
                "has_vlm_entity": False,
                "has_verified_link": False,
                "uses_ocr_only": True,
                "has_detected_text": True,
                "has_visual_description": False,
                "visual_state_detected_text_count": 1,
                "visual_entity_detected_text_count": 1,
                "visual_description_count": 0,
                "has_timestamp_fallback_link": False,
                "candidate_link_count": 1,
                "timestamp_fallback_link_count": 0,
                "verified_link_count": 0,
                "candidate_link_signal_counts": candidate_signal_counts,
                "verified_link_source_counts": empty_verified_sources,
                "candidate_visual_support": {
                    "has_candidate_visual_support": True,
                    "visual_state_count": 1,
                    "visual_entity_count": 1,
                    "candidate_link_count": 1,
                    "timestamp_fallback_link_count": 0,
                    "candidate_link_signal_counts": candidate_signal_counts,
                    "paper_claim_eligible": False,
                },
                "verified_object_alignment": {
                    "has_verified_object_alignment": False,
                    "verified_link_count": 0,
                    "verified_link_source_counts": empty_verified_sources,
                    "timestamp_fallback_counted_as_verified": False,
                    "paper_claim_eligible": False,
                },
            },
            "transcript_window_text": (
                "PUBLIC SYNTHETIC wrap evidence-unit transcript mentions loss curve slope "
                "but must not leak"
            ),
            "_rankingScore": 0.93,
        },
        {
            "evidence_unit_id": "evu_loss_verified_public",
            "project_id": "public_retrieval_ablation",
            "video_id": "public_demo_video",
            "target_segment_id": "seg_loss_public",
            "source_segment_ids": ["seg_intro_public", "seg_loss_public", "seg_wrap_public"],
            "start_time": 10.0,
            "end_time": 14.0,
            "visual_state_ids": ["state_loss_public"],
            "visual_entity_ids": ["ent_loss_public"],
            "verified_entity_link_ids": ["link_loss_verified_public"],
            "candidate_entity_link_ids": ["link_loss_verified_public"],
            "candidate_entity_link_statuses": {"link_loss_verified_public": "verified"},
            "alignment_status": "verified",
            "source_quality": {
                "has_visual_state": True,
                "has_visual_entity": True,
                "has_vlm_entity": True,
                "has_verified_link": True,
                "uses_ocr_only": False,
                "has_detected_text": True,
                "has_visual_description": True,
                "visual_state_detected_text_count": 1,
                "visual_entity_detected_text_count": 1,
                "visual_description_count": 1,
                "has_timestamp_fallback_link": False,
                "candidate_link_count": 0,
                "timestamp_fallback_link_count": 0,
                "verified_link_count": 1,
                "candidate_link_signal_counts": verified_signal_counts,
                "verified_link_source_counts": verified_sources,
                "candidate_visual_support": {
                    "has_candidate_visual_support": True,
                    "visual_state_count": 1,
                    "visual_entity_count": 1,
                    "candidate_link_count": 0,
                    "timestamp_fallback_link_count": 0,
                    "candidate_link_signal_counts": verified_signal_counts,
                    "paper_claim_eligible": False,
                },
                "verified_object_alignment": {
                    "has_verified_object_alignment": True,
                    "verified_link_count": 1,
                    "verified_link_source_counts": verified_sources,
                    "timestamp_fallback_counted_as_verified": False,
                    "paper_claim_eligible": True,
                },
            },
            "transcript_window_text": (
                "PUBLIC SYNTHETIC loss evidence-unit transcript mentions loss curve slope "
                "but must not leak"
            ),
            "_rankingScore": 0.91,
        },
    ]


def test_run_paper_bundle_public_fixture_writes_private_safe_bundle(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    output_dir = tmp_path / "paper_bundle"

    run = run_paper_bundle(
        client=FakePaperBundleClient(),
        manifest_path=fixture_dir / "benchmark_matrix_manifest.json",
        output_dir=output_dir,
        run_id="paper_bundle_fixture",
        gate_config_path=fixture_dir / "retrieval_quality_gate.json",
        baseline_variant_id="segment_lexical",
        sample_count=100,
        seed=7,
        repo_root=Path.cwd(),
        command=[
            "oarag",
            "run-paper-bundle",
            "--manifest",
            str(fixture_dir / "benchmark_matrix_manifest.json"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run.run_id == "paper_bundle_fixture"
    assert (output_dir / "experiment_manifest.json").exists()
    assert (output_dir / "paper_metric_intervals" / "paper_metric_intervals.json").exists()
    assert (output_dir / "paper_readiness" / "paper_readiness_audit.json").exists()
    assert (output_dir / "paper_claims" / "claim_evidence_matrix.json").exists()
    assert (output_dir / "paper_artifact_registry.json").exists()
    assert (output_dir / "paper_bundle_result.json").exists()

    assert run.metric_intervals.payload["status"] == "needs_evidence"
    assert run.claims.payload["summary"]["robustness_status"] == "needs_evidence"
    assert run.registry.payload["status"] == {
        "gate": "passed",
        "readiness": "ready",
        "claims": "needs_evidence",
        "robustness": "needs_evidence",
    }
    assert run.registry.payload["status_links"]["robustness"] == {
        "artifact": "paper_metric_intervals.json",
        "status": "needs_evidence",
    }
    assert run.payload["registry_status"] == run.registry.payload["status"]
    assert run.payload["status"] == "passed"

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            output_dir / "paper_metric_intervals" / "paper_metric_intervals.json",
            output_dir / "paper_metric_intervals" / "paper_metric_intervals.md",
            output_dir / "paper_readiness" / "paper_readiness_audit.json",
            output_dir / "paper_claims" / "claim_evidence_matrix.json",
            output_dir / "paper_artifact_registry.json",
            output_dir / "paper_bundle_result.json",
        ]
    )
    for sensitive in [
        "PUBLIC RAW QUERY",
        "loss curve slope should stay private-safe",
        "PUBLIC SYNTHETIC",
        "PUBLIC VISUAL LABEL",
        "seg_loss_public",
        "window_loss_public",
        "ent_loss_public",
        "frame_loss_public",
        str(fixture_dir),
        str(tmp_path),
    ]:
        assert sensitive not in public_text


def test_run_paper_bundle_gate_failure_records_stage_and_partial_artifacts(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    gate_config = tmp_path / "strict_gate.json"
    gate_config.write_text(
        json.dumps(
            {
                "gate_id": "strict_gate",
                "suites": [
                    {
                        "suite_id": "public_matrix",
                        "suite_type": "retrieval_answer_matrix",
                        "thresholds": {"expected_citation_hit_ratio": 1.1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "failed_bundle"

    with pytest.raises(PaperBundleError) as exc_info:
        run_paper_bundle(
            client=FakePaperBundleClient(),
            manifest_path=fixture_dir / "benchmark_matrix_manifest.json",
            output_dir=output_dir,
            gate_config_path=gate_config,
            repo_root=Path.cwd(),
        )

    assert exc_info.value.stage == "run_paper_experiment.gate"
    assert exc_info.value.result_path == output_dir.resolve() / "paper_bundle_result.json"
    assert (output_dir / "quality_gate_result.json").exists()
    assert (output_dir / "experiment_manifest.json").exists()
    assert not (output_dir / "paper_metric_intervals").exists()

    payload = json.loads((output_dir / "paper_bundle_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "run_paper_experiment.gate"
    assert payload["error"]["stage"] == "run_paper_experiment.gate"
    assert "quality_gate_result.json" in payload["artifacts"].values()
    assert str(tmp_path) not in json.dumps(payload)


def test_run_paper_bundle_cli_help_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["run-paper-bundle", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--manifest" in help_text
    assert "--baseline-variant-id" in help_text
    assert "--sample-count" in help_text


def test_mit_paper_bundle_manifest_and_skeleton_contract_are_private_safe() -> None:
    manifest_path = Path("eval/mit_deep_learning_stt/paper_bundle_manifest.json")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    suites = manifest["suites"]
    bundle = manifest["paper_bundle"]

    assert manifest["run_id"] == "mit_deep_learning_stt_paper_bundle_v1"
    assert manifest["output_dir"] == "../../reports/mit_deep_learning_eval/paper_matrix_v1"
    assert bundle["baseline_variant_id"] == "segment_lexical"
    assert bundle["domain_lexicon"] == "domain_lexicon.json"
    assert bundle["quality_gate_config"] == (
        "../../reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json"
    )
    assert bundle["required_variants"] == MIT_PAPER_MATRIX_VARIANTS
    assert len(suites) == 24
    assert {suite["suite_id"] for suite in suites} == {
        f"mitdl_lec{lecture:02d}" for lecture in range(1, 22)
    } | {"mitdl_lec23", "mitdl_lec24", "mitdl_review"}
    for suite in suites:
        assert suite["type"] == "retrieval_answer_matrix"
        assert suite["variants"] == MIT_PAPER_MATRIX_VARIANTS
        assert suite["domain_lexicon"] == "domain_lexicon.json"
        assert suite["evidence_unit_index"] == "mit_deep_learning_stt_evidence_units"
        assert suite["include_answer"] is True
        assert not Path(suite["project_dir"]).is_absolute()
        assert not Path(suite["queries"]).is_absolute()

    gate_path = Path("reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["gate_id"] == "mit_deep_learning_stt_paper_matrix_completeness_gate_v1"
    assert len(gate["suites"]) == len(suites)
    assert {suite["suite_id"] for suite in gate["suites"]} == {
        suite["suite_id"] for suite in suites
    }
    assert all(
        suite["required_variants"] == MIT_PAPER_MATRIX_VARIANTS for suite in gate["suites"]
    )
    assert all(
        set(suite["thresholds"]) == {
            "hit_at_10s",
            "mrr_at_max_delta",
            "grounded_answer_ratio",
            "expected_citation_hit_ratio",
        }
        for suite in gate["suites"]
    )

    expected_artifacts_path = Path(
        "reports/mit_deep_learning_eval/paper_matrix_v1/expected_artifacts.json"
    )
    expected_artifacts = json.loads(expected_artifacts_path.read_text(encoding="utf-8"))
    assert expected_artifacts["required_variants"] == MIT_PAPER_MATRIX_VARIANTS
    assert expected_artifacts["dynamic_concept_graph_variants"] == (
        GRAPH_CONCEPT_MATRIX_VARIANTS
    )
    assert expected_artifacts["allowed_public_result_fields"] == (
        GRAPH_CONCEPT_ALLOWED_PUBLIC_FIELDS
    )

    skeleton_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("reports/mit_deep_learning_eval/README.md"),
            Path("reports/mit_deep_learning_eval/paper_matrix_v1/README.md"),
            expected_artifacts_path,
            Path("reports/mit_deep_learning_eval/paper_matrix_v1/paper_report_skeleton.md"),
            gate_path,
            manifest_path,
        ]
    )
    for sensitive in [
        "/Users/",
        "/private/tmp",
        "transcript_text",
        "reference_answer",
        "query_text",
    ]:
        assert sensitive not in skeleton_text
    assert "candidate_evidence_text" in skeleton_text
    assert "local_paths" in skeleton_text
    for variant in GRAPH_CONCEPT_MATRIX_VARIANTS:
        assert variant in skeleton_text
    for issue_ref in ["#213", "#214", "#217", "#219", "#220", "#221", "#222"]:
        assert issue_ref in skeleton_text
    for field in GRAPH_CONCEPT_ALLOWED_PUBLIC_FIELDS:
        assert field in skeleton_text
    lowered_skeleton = skeleton_text.lower()
    assert "timestamp-only overlap" in lowered_skeleton
    assert "candidate/fallback evidence" in lowered_skeleton
