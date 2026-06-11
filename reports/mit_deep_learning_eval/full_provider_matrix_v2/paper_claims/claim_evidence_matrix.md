# Paper Claim/Evidence Matrix: mit_deep_learning_stt_full_provider_matrix_v2

This matrix is conservative and private-safe. It records claim status, artifact names, and aggregate schema evidence only. Private eval values, raw queries, answer text, transcripts, candidate evidence text, and local filesystem paths are excluded.

## Summary

- Overall status: `blocked`
- Quality gate: `failed`
- Readiness: `gaps_found`
- Robustness: `passed`

## Matrix

| claim id | status | evidence artifacts | note |
| --- | --- | --- | --- |
| readiness_audit_passed | blocked | paper_readiness_audit.json | Paper readiness audit reports gaps. Blocked by: paper_readiness_audit_not_ready |
| quality_gate_passed | blocked | quality_gate_result.json | Retrieval quality gate did not pass. Blocked by: quality_gate_not_passed |
| retrieval_answer_matrix_complete | blocked | metrics.json | Required retrieval-answer matrix variants are present. Blocked by: paper_readiness_audit_not_ready, quality_gate_not_passed |
| answer_citation_metrics_available | blocked | metrics.json | Answer/citation metric fields are present. Blocked by: paper_readiness_audit_not_ready, quality_gate_not_passed |
| private_safe_artifact_bundle | blocked | metrics.json, paper_readiness_audit.json, quality_gate_result.json, reproducibility.json | Private-safe exclusion statements are present. Blocked by: paper_readiness_audit_not_ready, quality_gate_not_passed |
| robustness_artifact_recorded | blocked | paper_metric_intervals.json | Robustness artifact reports a passing status. Blocked by: paper_readiness_audit_not_ready, quality_gate_not_passed |
