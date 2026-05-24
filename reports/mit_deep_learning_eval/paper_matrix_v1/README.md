# MIT Paper Matrix v1 Runbook

This folder is the public-safe landing zone for the MIT Deep Learning paper bundle.
It is intentionally a skeleton until a full local run produces fresh artifacts.

## Matrix Contract

The bundle manifest compares these six variants for every MIT lecture suite:

- `segment_lexical`
- `domain_lexicon`
- `hybrid`
- `window`
- `window_hybrid`
- `rerank`

Every suite references the shared `eval/mit_deep_learning_stt/domain_lexicon.json`
through the manifest-relative value `domain_lexicon.json`.

## Prerequisites

Build and index the shared window corpus before running the bundle:

```bash
PYTHONPATH=src python3 scripts/index_mit_deep_learning_windows.py --build-only
PYTHONPATH=src python3 scripts/index_mit_deep_learning_windows.py --index-only --reset
```

For `window_hybrid`, pass the same hybrid embedder profile and dimensions used by
the segment index when required by the local Meilisearch configuration.

## Full Bundle Command

```bash
PYTHONPATH=src python3 -m oarag run-paper-bundle \
  --manifest eval/mit_deep_learning_stt/paper_bundle_manifest.json \
  --output-dir reports/mit_deep_learning_eval/paper_matrix_v1 \
  --gate-config reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json \
  --baseline-variant-id segment_lexical \
  --sample-count 1000 \
  --seed 139
```

If the local Meilisearch state is still being prepared, run a 1 to 2 suite smoke by
copying the manifest locally and keeping only the target lecture suites. Do not commit
that local smoke manifest or generated outputs unless they pass the privacy checklist.

## Expected Generated Files

`run-paper-bundle` should generate:

- `metrics.json`
- `metrics_summary.csv`
- `query_results.jsonl`
- `summary.md`
- `paper_report/paper_table.md`
- `paper_report/reproducibility.json`
- `quality_gate_result.json`
- `experiment_manifest.json`
- `paper_metric_intervals/paper_metric_intervals.json`
- `paper_readiness/paper_readiness_audit.json`
- `paper_claims/claim_evidence_matrix.json`
- `paper_artifact_registry.json`
- `paper_bundle_result.json`

## Privacy Checklist

Before committing any full-run output, inspect JSON, JSONL, CSV, and Markdown files
for raw query text, answer text, transcript content, candidate evidence text, private
eval values, absolute local paths, and raw private identifiers. Generated public
artifacts should contain aggregate metrics, hashed refs, relative artifact filenames,
stage statuses, and coarse claim/readiness statuses only.

## Next Issue Candidate

If this folder still contains only skeleton files after the indexing fix lands, open a
follow-up issue to execute the full MIT paper bundle locally, replace the completeness
gate with final paper thresholds, and commit only privacy-checked aggregate artifacts.
