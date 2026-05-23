from __future__ import annotations

import json
from pathlib import Path

from oarag.evaluation.paper_registry import build_paper_artifact_registry


def test_build_paper_artifact_registry_uses_filenames_and_coarse_statuses(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "paper_experiment"
    readiness_dir = tmp_path / "paper_readiness"
    claims_dir = tmp_path / "paper_claims"
    registry_dir = tmp_path / "registry"
    for path in (experiment_dir, readiness_dir, claims_dir, registry_dir):
        path.mkdir()

    experiment_manifest = experiment_dir / "experiment_manifest.json"
    quality_gate = experiment_dir / "quality_gate_result.json"
    readiness = readiness_dir / "paper_readiness_audit.json"
    claims = claims_dir / "claim_evidence_matrix.json"
    output = registry_dir / "paper_artifact_registry.json"

    _write_json(
        experiment_manifest,
        {
            "schema_version": "paper-experiment-manifest-v1",
            "run_id": "registry_fixture",
            "commit": {"sha": "abc123"},
            "artifacts": {
                "metrics": "metrics.json",
                "query_results": "query_results.jsonl",
                "paper_report_dir": "paper_report/",
                "reproducibility_json": "paper_report/reproducibility.json",
                "quality_gate_result": "quality_gate_result.json",
            },
            "private_path": str(tmp_path / "private" / "experiment_manifest.json"),
        },
    )
    _write_json(
        quality_gate,
        {
            "schema_version": "retrieval-quality-gate-result-v1",
            "passed": True,
            "privacy": {"query_content": "excluded"},
            "raw_query": "PUBLIC RAW QUERY should not be registered",
        },
    )
    _write_json(
        readiness,
        {
            "schema_version": "paper-readiness-audit-v1",
            "run_id": "registry_fixture",
            "ready": True,
            "privacy": {"raw_queries": "excluded"},
            "private_path": str(tmp_path / "private" / "readiness.json"),
        },
    )
    _write_json(
        claims,
        {
            "schema_version": "paper-claim-evidence-matrix-v1",
            "run_id": "registry_fixture",
            "summary": {"overall_status": "needs_evidence"},
            "privacy": {"private_eval_values": "excluded"},
            "claim": "SENSITIVE_EVAL_MARKER must not be registered",
        },
    )

    run = build_paper_artifact_registry(
        experiment_manifest_path=experiment_manifest,
        readiness_audit_path=readiness,
        claim_matrix_path=claims,
        output_path=output,
    )

    assert run.json_path == output
    assert run.payload["schema_version"] == "paper-artifact-registry-v1"
    assert run.payload["run_id"] == "registry_fixture"
    assert run.payload["commit"]["sha"] == "abc123"
    assert run.payload["artifacts"]["experiment_manifest"] == "experiment_manifest.json"
    assert run.payload["artifacts"]["reproducibility_json"] == "reproducibility.json"
    assert run.payload["status"] == {
        "gate": "passed",
        "readiness": "ready",
        "claims": "needs_evidence",
        "robustness": "missing",
    }

    public_text = output.read_text(encoding="utf-8")
    for sensitive in [
        str(tmp_path),
        "PUBLIC RAW QUERY should not be registered",
        "SENSITIVE_EVAL_MARKER",
    ]:
        assert sensitive not in public_text


def test_build_paper_artifact_registry_accepts_optional_robustness_status(
    tmp_path: Path,
) -> None:
    experiment_manifest = tmp_path / "experiment_manifest.json"
    quality_gate = tmp_path / "quality_gate_result.json"
    readiness = tmp_path / "paper_readiness_audit.json"
    claims = tmp_path / "claim_evidence_matrix.json"
    robustness = tmp_path / "robustness.json"

    _write_json(
        experiment_manifest,
        {
            "schema_version": "paper-experiment-manifest-v1",
            "run_id": "registry_fixture",
            "commit": {"sha": "abc123"},
            "artifacts": {"quality_gate_result": "quality_gate_result.json"},
        },
    )
    _write_json(quality_gate, {"passed": True})
    _write_json(readiness, {"ready": True})
    _write_json(claims, {"summary": {"overall_status": "supported"}})
    _write_json(robustness, {"schema_version": "robustness-summary-v1", "status": "passed"})

    run = build_paper_artifact_registry(
        experiment_manifest_path=experiment_manifest,
        readiness_audit_path=readiness,
        claim_matrix_path=claims,
        robustness_path=robustness,
    )

    assert run.json_path is None
    assert run.payload["status"]["robustness"] == "passed"
    assert run.payload["artifacts"]["robustness"] == "robustness.json"
    assert run.payload["status_links"]["robustness"] == {
        "artifact": "robustness.json",
        "status": "passed",
    }


def test_build_paper_artifact_registry_reads_metric_intervals_status(
    tmp_path: Path,
) -> None:
    experiment_manifest = tmp_path / "experiment_manifest.json"
    quality_gate = tmp_path / "quality_gate_result.json"
    readiness = tmp_path / "paper_readiness_audit.json"
    claims = tmp_path / "claim_evidence_matrix.json"
    robustness = tmp_path / "paper_metric_intervals.json"

    _write_json(
        experiment_manifest,
        {
            "schema_version": "paper-experiment-manifest-v1",
            "run_id": "registry_fixture",
            "commit": {"sha": "abc123"},
            "artifacts": {"quality_gate_result": "quality_gate_result.json"},
        },
    )
    _write_json(quality_gate, {"passed": True})
    _write_json(readiness, {"ready": True})
    _write_json(claims, {"summary": {"overall_status": "needs_evidence"}})
    _write_json(
        robustness,
        {
            "schema_version": "paper-metric-intervals-v1",
            "status": "needs_evidence",
            "summary": {"robustness_status": "needs_evidence"},
            "paired_deltas": [{"metric": "hit_at_10s"}],
            "caveats": ["bootstrap_interval_descriptive_only"],
        },
    )

    run = build_paper_artifact_registry(
        experiment_manifest_path=experiment_manifest,
        readiness_audit_path=readiness,
        claim_matrix_path=claims,
        robustness_path=robustness,
    )

    assert run.payload["status"]["robustness"] == "needs_evidence"
    assert run.payload["status_links"]["robustness"] == {
        "artifact": "paper_metric_intervals.json",
        "status": "needs_evidence",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
