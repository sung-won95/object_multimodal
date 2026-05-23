# Private-safe Paper Experiment Runbook

This runbook records the public-safe paper artifact flow. Keep raw queries, transcripts,
answer text, candidate evidence text, private eval values, and local filesystem paths out of
committed reports, GitHub issues, and PR descriptions.

## Command Checklist

Run the steps in this order:

1. `run-paper-experiment`

   ```bash
   PYTHONPATH=src python3 -m oarag run-paper-experiment \
     --manifest <benchmark_manifest.json> \
     --output-dir <run_output_dir> \
     --gate-config <quality_gate.json>
   ```

2. `audit-paper-readiness`

   ```bash
   PYTHONPATH=src python3 -m oarag audit-paper-readiness \
     --experiment-manifest <run_output_dir>/experiment_manifest.json \
     --output-dir <readiness_output_dir>
   ```

3. `build-paper-claims`

   ```bash
   PYTHONPATH=src python3 -m oarag build-paper-claims \
     --metrics <run_output_dir>/metrics.json \
     --reproducibility <run_output_dir>/paper_report/reproducibility.json \
     --quality-gate-result <run_output_dir>/quality_gate_result.json \
     --readiness-audit <readiness_output_dir>/paper_readiness_audit.json \
     --output-dir <claims_output_dir>
   ```

4. `robustness`

   Run the project-specific robustness check after the claim matrix step and write a
   private-safe summary artifact such as `robustness.json`. The summary should expose only
   schema metadata and coarse pass/fail status.

5. Optional registry

   ```bash
   PYTHONPATH=src python3 -m oarag build-paper-registry \
     --experiment-manifest <run_output_dir>/experiment_manifest.json \
     --readiness-audit <readiness_output_dir>/paper_readiness_audit.json \
     --claim-matrix <claims_output_dir>/claim_evidence_matrix.json \
     --robustness <robustness.json> \
     --output <registry_output_dir>/paper_artifact_registry.json
   ```

## Artifact Contract

- `metrics.json`: aggregate schema and metric fields only.
- `paper_report/reproducibility.json`: commit, sanitized command tokens, dataset descriptors,
  and artifact filenames only.
- `quality_gate_result.json`: gate status, threshold failure codes, and aggregate schema metadata.
- `paper_readiness_audit.json`: checklist statuses and gap codes only.
- `claim_evidence_matrix.json`: claim status, evidence artifact filenames, and field names only.
- `paper_artifact_registry.json`: run id, commit sha, artifact filenames, and coarse statuses only.

## Review Gate

Before sharing a report publicly, inspect the generated JSON and Markdown for:

- raw query or answer text
- transcript or candidate evidence text
- private eval values
- local absolute paths
- raw private identifiers that are not already explicitly allowed by the artifact schema
