# Private-safe Paper Experiment Runbook

This runbook records the public-safe paper artifact flow. Keep raw queries, transcripts,
answer text, candidate evidence text, private eval values, and local filesystem paths out of
committed reports, GitHub issues, and PR descriptions.

## Command Checklist

Default path: run the full bundle with one command.

```bash
PYTHONPATH=src python3 -m oarag run-paper-bundle \
  --manifest <benchmark_manifest.json> \
  --output-dir <bundle_output_dir> \
  --gate-config <quality_gate.json> \
  --baseline-variant-id <baseline_variant_id>
```

Use `--seed`, `--sample-count`, and `--confidence-level` when the paper run needs
explicit bootstrap settings. Use `--no-fail-on-gate` only when you need the bundle
artifacts for inspection after a failing quality gate.

Before using any generated number in paper prose, compare the run output against
`reports/paper/current_claim_status.md`. That file records which claims are
currently supported, which are held, and which limitations must stay visible.

The bundle runs these stages in order and records the stage status in
`paper_bundle_result.json`:

1. `run-paper-experiment`
2. `paper-metric-intervals`
3. `audit-paper-readiness`
4. `build-paper-claims`
5. `build-paper-registry`

If a stage fails, the command names the failed stage and leaves any already generated
private-safe artifacts in the output directory.

## Hybrid Vector Smoke

For local paper-docker runs that exercise `hybrid` or `window_hybrid`, enable
Meilisearch vector store before indexing:

```bash
docker compose up -d meilisearch
curl -X PATCH 'http://127.0.0.1:7700/experimental-features/' \
  -H 'Authorization: Bearer dev-master-key' \
  -H 'Content-Type: application/json' \
  --data-binary '{"vectorStore": true}'
```

Use a `userProvided` embedder profile when indexing segment/window/visual documents.
OARAG will attach `_vectors.<embedder>` for the local smoke path when the
document does not already provide one. If a query vector manifest is unavailable,
`hybrid_query_vector_dimensions` can activate the deterministic `local_hash_v1`
fallback. This fallback is only for dependency-free local reproducibility and
smoke completion; it is not semantic quality evidence. Public artifacts must keep
only aggregate/sanitized metadata such as `purpose: local_reproducibility_smoke_fallback`
and `quality_claim: none`, never raw query vectors.

## Dynamic Concept Graph Smoke

Use the cross-lecture smoke when the paper question is source contribution or
dynamic concept graph candidate recall rather than final answer quality:

```bash
PYTHONPATH=src python3 -m oarag cross-lecture-retrieval-smoke \
  --manifest <cross_lecture_smoke_manifest.json> \
  --output-dir <cross_lecture_output_dir>
```

The public-safe graph matrix variants are:

- `meili_only`: Meilisearch candidates only.
- `graph_only`: graph traversal candidates only.
- `meili_graph`: combined Meili and graph candidate pool.
- `graph_aware_rerank`: combined pool plus deterministic graph-aware rerank.

This smoke supports claims about dynamic concept graph guided candidate recall,
source contribution diagnostics, graph-aware deterministic rerank diagnostics, and
public-safe smoke/report generation. It does not by itself support a claim that
Graph DB availability proves verified object alignment or final answer quality.

Public cross-lecture result tables and reports may expose only:

- `query_id`
- hashed refs
- source types
- ranks
- counts
- recall buckets
- rerank deltas
- skip reasons

They must not expose raw query text, answer text, transcript content, candidate
evidence text, private eval values, raw vectors, local absolute paths, or raw private
identifiers. Timestamp-only overlap is a candidate/fallback signal; it must stay
separate from verified object alignment in every table and narrative.

## Advanced / Fallback Manual Flow

Use the manual flow only for debugging, partial reruns, or comparing an individual
artifact builder.

1. `run-paper-experiment`

   ```bash
   PYTHONPATH=src python3 -m oarag run-paper-experiment \
     --manifest <benchmark_manifest.json> \
     --output-dir <run_output_dir> \
     --gate-config <quality_gate.json>
   ```

2. `paper-metric-intervals`

   ```bash
   PYTHONPATH=src python3 -m oarag paper-metric-intervals \
     --query-results <run_output_dir>/query_results.jsonl \
     --metrics <run_output_dir>/metrics.json \
     --report <run_output_dir>/paper_report/paper_table.md \
     --output-dir <intervals_output_dir> \
     --baseline-variant-id <baseline_variant_id>
   ```

3. `audit-paper-readiness`

   ```bash
   PYTHONPATH=src python3 -m oarag audit-paper-readiness \
     --experiment-manifest <run_output_dir>/experiment_manifest.json \
     --output-dir <readiness_output_dir>
   ```

4. `build-paper-claims`

   ```bash
   PYTHONPATH=src python3 -m oarag build-paper-claims \
     --metrics <run_output_dir>/metrics.json \
     --reproducibility <run_output_dir>/paper_report/reproducibility.json \
     --quality-gate-result <run_output_dir>/quality_gate_result.json \
     --readiness-audit <readiness_output_dir>/paper_readiness_audit.json \
     --robustness <intervals_output_dir>/paper_metric_intervals.json \
     --output-dir <claims_output_dir>
   ```

5. `build-paper-registry`

   ```bash
   PYTHONPATH=src python3 -m oarag build-paper-registry \
     --experiment-manifest <run_output_dir>/experiment_manifest.json \
     --readiness-audit <readiness_output_dir>/paper_readiness_audit.json \
     --claim-matrix <claims_output_dir>/claim_evidence_matrix.json \
     --robustness <intervals_output_dir>/paper_metric_intervals.json \
     --output <registry_output_dir>/paper_artifact_registry.json
   ```

## Artifact Contract

- `metrics.json`: aggregate schema and metric fields only.
- `paper_metric_intervals.json`: aggregate bootstrap intervals, paired delta status,
  caveats, and coarse robustness status only.
- `paper_report/reproducibility.json`: commit, sanitized command tokens, dataset descriptors,
  and artifact filenames only.
- `quality_gate_result.json`: gate status, threshold failure codes, and aggregate schema metadata.
- `paper_readiness_audit.json`: checklist statuses and gap codes only.
- `claim_evidence_matrix.json`: claim status, evidence artifact filenames, and field names only.
- `paper_artifact_registry.json`: run id, commit sha, artifact filenames, and coarse statuses only.
- `paper_bundle_result.json`: bundle stage statuses, artifact filenames, registry statuses,
  and failure stage metadata only.
- `cross-lecture-retrieval-smoke` artifacts: public-safe `metrics.json`,
  `query_results.jsonl`, and `summary.md` with query ids, hashed refs, source types,
  ranks, counts, recall buckets, rerank deltas, and skip reasons only.

The artifact registry links the quality gate, readiness audit, claim matrix, and robustness
interval status by filename and coarse status. Treat small-sample, missing-baseline, or
missing-paired-delta caveats as `needs_evidence`, not as a passing robustness claim.

## Claim Interpretation Rules

- `local_hash_v1` vectors are smoke fallback only and must carry `quality_claim: none`.
- Provider-backed embedding claims require provider/source model metadata in the vector
  or query-vector manifest and a passing paper bundle registry.
- OCR-only visual entity outputs are baseline/fallback artifacts. The target method is
  the VLM-first path, with mock/jsonl/command backends available for reproducible tests.
- Frame coverage, timestamp-only link ratio, and VLM/OCR comparisons are held claims
  until regenerated aggregate artifacts demonstrate them for the target suite.
- Dynamic concept graph claims are limited to candidate recall, source contribution
  diagnostics, deterministic graph-aware rerank diagnostics, and public-safe smoke/report
  readiness until a final privacy-checked run supports a stronger claim.
- Graph DB availability must not be interpreted as verified object alignment.
  Timestamp-only overlap remains candidate/fallback evidence and is never counted as
  verified object alignment.

## Review Gate

Before sharing a report publicly, inspect the generated JSON and Markdown for:

- raw query or answer text
- transcript or candidate evidence text
- private eval values
- local absolute paths
- raw private identifiers that are not already explicitly allowed by the artifact schema
