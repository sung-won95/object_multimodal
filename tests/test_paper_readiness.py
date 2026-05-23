from __future__ import annotations

import json
from pathlib import Path

from oarag.evaluation.experiment import run_paper_experiment
from oarag.evaluation.readiness import audit_paper_readiness


class FakeReadinessExperimentClient:
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
    ) -> dict:
        mode = "semantic" if hybrid else "lexical"
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


def test_paper_readiness_audit_public_fixture_writes_private_safe_reports(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    experiment_dir = tmp_path / "paper_experiment"
    audit_dir = tmp_path / "paper_readiness"
    run_paper_experiment(
        client=FakeReadinessExperimentClient(),
        manifest_path=fixture_dir / "benchmark_matrix_manifest.json",
        output_dir=experiment_dir,
        run_id="paper_readiness_fixture",
        gate_config_path=fixture_dir / "retrieval_quality_gate.json",
        repo_root=Path.cwd(),
        command=[
            "oarag",
            "run-paper-experiment",
            "--manifest",
            str(fixture_dir / "benchmark_matrix_manifest.json"),
            "--output-dir",
            str(experiment_dir),
        ],
    )

    audit = audit_paper_readiness(
        experiment_manifest_path=experiment_dir / "experiment_manifest.json",
        output_dir=audit_dir,
    )

    assert audit.payload["schema_version"] == "paper-readiness-audit-v1"
    assert audit.payload["ready"] is True
    assert audit.payload["gap_count"] == 0
    assert "not a paper acceptance guarantee" in audit.payload["disclaimer"]
    assert audit.json_path == audit_dir / "paper_readiness_audit.json"
    assert audit.markdown_path == audit_dir / "paper_readiness_audit.md"
    assert {check["status"] for check in audit.payload["checks"]} == {"pass"}

    semantic_check = next(
        check for check in audit.payload["checks"] if check["id"] == "semantic_live_smoke_evidence"
    )
    assert semantic_check["evidence"]["semantic_query_result_count"] > 0

    public_text = "\n".join(
        [
            audit.json_path.read_text(encoding="utf-8"),
            audit.markdown_path.read_text(encoding="utf-8"),
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


def test_paper_readiness_audit_reports_public_safe_gaps(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "bad_experiment"
    experiment_dir.mkdir()
    (experiment_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "bad_readiness_fixture",
                "suites": [
                    {
                        "suite_id": "public_matrix",
                        "suite_type": "retrieval_answer_matrix",
                        "schema_version": "retrieval-answer-ablation-matrix-v1",
                        "variants": [{"variant_id": "segment_lexical", "hit_at_10s": 1.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "quality_gate_result.json").write_text(
        json.dumps(
            {
                "schema_version": "retrieval-quality-gate-result-v1",
                "gate_id": "bad_public_fixture_gate",
                "passed": False,
                "failure_count": 1,
                "failures": [{"code": "metric_below_threshold"}],
                "privacy": {
                    "payload": "aggregate_metrics_only",
                    "query_content": "excluded",
                    "transcript_content": "excluded",
                    "evidence_content": "excluded",
                },
            }
        ),
        encoding="utf-8",
    )
    report_dir = experiment_dir / "paper_report"
    report_dir.mkdir()
    (report_dir / "reproducibility.json").write_text(
        json.dumps(
            {
                "schema_version": "evaluation-report-v1",
                "run_id": "bad_readiness_fixture",
                "privacy": {"raw_queries": "excluded"},
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper-experiment-manifest-v1",
                "run_id": "bad_readiness_fixture",
                "artifacts": {
                    "metrics": "metrics.json",
                    "query_results": "query_results.jsonl",
                    "quality_gate_result": "quality_gate_result.json",
                    "reproducibility_json": "paper_report/reproducibility.json",
                },
                "privacy": {
                    "payload": "artifact_schema_and_aggregate_metadata_only",
                    "raw_queries": "excluded",
                    "answer_text": "excluded",
                    "transcript_content": "excluded",
                    "candidate_evidence_text": "excluded",
                    "local_paths": "excluded",
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_paper_readiness(
        experiment_manifest_path=experiment_dir / "experiment_manifest.json",
        output_dir=tmp_path / "bad_readiness",
    )

    assert audit.payload["ready"] is False
    codes = {gap["code"] for gap in audit.payload["gaps"]}
    assert {
        "matrix_variant_missing",
        "semantic_live_smoke_missing",
        "answer_citation_metric_missing",
        "answer_citation_variant_metric_missing",
        "quality_gate_not_passed",
        "reproducibility_metadata_missing",
    }.issubset(codes)

    output_text = audit.json_path.read_text(encoding="utf-8")
    assert "metric_below_threshold" in output_text
    assert str(tmp_path) not in output_text
