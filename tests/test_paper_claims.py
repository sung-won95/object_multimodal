from __future__ import annotations

import json
from pathlib import Path

from oarag.evaluation.claims import build_paper_claims
from oarag.evaluation.quality_gate import DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS
from oarag.evaluation.readiness import ANSWER_CITATION_METRICS


def test_build_paper_claims_writes_private_safe_matrix(tmp_path: Path) -> None:
    paths = _write_claim_inputs(tmp_path)

    run = build_paper_claims(
        metrics_path=paths["metrics"],
        reproducibility_path=paths["reproducibility"],
        quality_gate_result_path=paths["gate"],
        readiness_audit_path=paths["readiness"],
        output_dir=tmp_path / "claims",
    )

    assert run.payload["schema_version"] == "paper-claim-evidence-matrix-v1"
    assert run.payload["summary"]["quality_gate_status"] == "passed"
    assert run.payload["summary"]["readiness_status"] == "ready"
    assert run.payload["summary"]["robustness_status"] == "missing"

    statuses = {claim["claim_id"]: claim["status"] for claim in run.payload["claims"]}
    assert statuses["readiness_audit_passed"] == "supported"
    assert statuses["quality_gate_passed"] == "supported"
    assert statuses["retrieval_answer_matrix_complete"] == "supported"
    assert statuses["answer_citation_metrics_available"] == "supported"
    assert statuses["private_safe_artifact_bundle"] == "supported"
    assert statuses["robustness_artifact_recorded"] == "needs_evidence"
    assert run.payload["summary"]["overall_status"] == "needs_evidence"

    public_text = "\n".join(
        [
            run.json_path.read_text(encoding="utf-8"),
            run.markdown_path.read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "PUBLIC RAW QUERY should never leave inputs",
        "PUBLIC SYNTHETIC transcript must not leak",
        "SENSITIVE ANSWER MARKER",
        "SENSITIVE_EVAL_MARKER",
        str(tmp_path),
    ]:
        assert sensitive not in public_text


def test_build_paper_claims_blocks_supported_claims_when_gate_or_readiness_fails(
    tmp_path: Path,
) -> None:
    paths = _write_claim_inputs(tmp_path, gate_passed=False, readiness_ready=False)

    run = build_paper_claims(
        metrics_path=paths["metrics"],
        reproducibility_path=paths["reproducibility"],
        quality_gate_result_path=paths["gate"],
        readiness_audit_path=paths["readiness"],
        output_dir=tmp_path / "claims",
    )

    statuses = {claim["claim_id"]: claim["status"] for claim in run.payload["claims"]}
    assert set(statuses.values()) <= {"blocked", "needs_evidence"}
    assert statuses["readiness_audit_passed"] == "blocked"
    assert statuses["quality_gate_passed"] == "blocked"
    assert statuses["retrieval_answer_matrix_complete"] == "blocked"
    assert statuses["answer_citation_metrics_available"] == "blocked"
    assert statuses["private_safe_artifact_bundle"] == "blocked"
    assert statuses["robustness_artifact_recorded"] == "needs_evidence"
    assert run.payload["summary"]["overall_status"] == "blocked"


def test_build_paper_claims_reads_metric_intervals_as_robustness_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_claim_inputs(tmp_path)
    robustness = tmp_path / "paper_metric_intervals.json"
    _write_json(
        robustness,
        {
            "schema_version": "paper-metric-intervals-v1",
            "status": "needs_evidence",
            "summary": {
                "robustness_status": "needs_evidence",
                "paired_delta_count": 1,
                "caveat_count": 1,
            },
            "paired_deltas": [{"metric": "hit_at_10s", "paired_query_count": 2}],
            "caveats": ["small_sample_intervals_should_not_be_used_as_confirmatory_evidence"],
            "privacy": {
                "raw_queries": "excluded",
                "answer_text": "excluded",
                "transcript_content": "excluded",
                "candidate_evidence_text": "excluded",
                "local_paths": "excluded",
            },
        },
    )

    run = build_paper_claims(
        metrics_path=paths["metrics"],
        reproducibility_path=paths["reproducibility"],
        quality_gate_result_path=paths["gate"],
        readiness_audit_path=paths["readiness"],
        robustness_path=robustness,
        output_dir=tmp_path / "claims",
    )

    claims = {claim["claim_id"]: claim for claim in run.payload["claims"]}
    robustness_claim = claims["robustness_artifact_recorded"]
    assert run.payload["summary"]["robustness_status"] == "needs_evidence"
    assert robustness_claim["status"] == "needs_evidence"
    assert robustness_claim["missing_evidence"] == ["uncaveated paired robustness evidence"]
    assert run.payload["summary"]["overall_status"] == "needs_evidence"


def _write_claim_inputs(
    tmp_path: Path,
    *,
    gate_passed: bool = True,
    readiness_ready: bool = True,
) -> dict[str, Path]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    paths = {
        "metrics": input_dir / "metrics.json",
        "reproducibility": input_dir / "reproducibility.json",
        "gate": input_dir / "quality_gate_result.json",
        "readiness": input_dir / "paper_readiness_audit.json",
    }
    _write_json(paths["metrics"], _metrics_payload())
    _write_json(paths["reproducibility"], _reproducibility_payload(tmp_path))
    _write_json(paths["gate"], _gate_payload(passed=gate_passed))
    _write_json(paths["readiness"], _readiness_payload(ready=readiness_ready))
    return paths


def _metrics_payload() -> dict:
    variant_metrics = {
        metric_name: "SENSITIVE_EVAL_MARKER" if metric_name == "grounded_answer_ratio" else 1.0
        for metric_name in ANSWER_CITATION_METRICS
    }
    variants = [
        {"variant_id": variant_id, **variant_metrics}
        for variant_id in DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS
    ]
    return {
        "run_id": "claims_fixture",
        "suites": [
            {
                "schema_version": "retrieval-answer-ablation-matrix-v1",
                "suite_id": "public_matrix",
                "suite_type": "retrieval_answer_matrix",
                "raw_query": "PUBLIC RAW QUERY should never leave inputs",
                "transcript_text": "PUBLIC SYNTHETIC transcript must not leak",
                "answer_text": "SENSITIVE ANSWER MARKER",
                "privacy": {
                    "raw_query_text": "redacted",
                    "answer_text": "redacted",
                    "candidate_evidence_text": "redacted",
                    "transcript_excerpt": "redacted",
                    "local_paths": "redacted",
                },
                **variant_metrics,
                "variants": variants,
            }
        ],
    }


def _reproducibility_payload(tmp_path: Path) -> dict:
    return {
        "schema_version": "evaluation-report-v1",
        "run_id": "claims_fixture",
        "commit": {"sha": "abc123"},
        "command": {"argv": ["oarag", "report-evaluation", "--metrics", "metrics.json"]},
        "artifacts": {"metrics": "metrics.json"},
        "private_path": str(tmp_path / "private" / "metrics.json"),
        "dataset_descriptors": [{"descriptor_id": "public_descriptor"}],
        "benchmark": {"suite_count": 1, "query_count": 1, "deltas": [5, 10, 15]},
        "privacy": {
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_excerpts": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
        },
    }


def _gate_payload(*, passed: bool) -> dict:
    return {
        "schema_version": "retrieval-quality-gate-result-v1",
        "gate_id": "public_gate",
        "passed": passed,
        "failure_count": 0 if passed else 1,
        "failures": [] if passed else [{"code": "metric_below_threshold"}],
        "privacy": {
            "query_content": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "evidence_content": "excluded",
            "local_paths": "excluded",
        },
    }


def _readiness_payload(*, ready: bool) -> dict:
    return {
        "schema_version": "paper-readiness-audit-v1",
        "run_id": "claims_fixture",
        "ready": ready,
        "gaps": [] if ready else [{"code": "quality_gate_not_passed"}],
        "privacy": {
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
