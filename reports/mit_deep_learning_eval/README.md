# MIT Deep Learning Evaluation Reports

This directory contains public-safe MIT Deep Learning evaluation report entry points.
Committed files here must stay aggregate-only: no raw query text, answer text, transcript
content, candidate evidence text, private eval values, or absolute local paths.

## Report Sets

- `seed_v1/`: seed retrieval benchmark aggregates and failure-analysis summaries.
- `paper_matrix_v1/`: runbook and skeleton configuration for the MIT paper bundle
  retrieval-answer ablation matrix and the dynamic concept graph smoke matrix.

## Paper Bundle

Run the MIT paper bundle from the repository root after the MIT project artifacts,
segment index, visual entity index, and shared window index are available:

```bash
PYTHONPATH=src python3 -m oarag run-paper-bundle \
  --manifest eval/mit_deep_learning_stt/paper_bundle_manifest.json \
  --output-dir reports/mit_deep_learning_eval/paper_matrix_v1 \
  --gate-config reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json \
  --baseline-variant-id segment_lexical \
  --sample-count 1000 \
  --seed 139
```

The gate in `paper_matrix_v1/retrieval_quality_gate.json` is a regression guard
derived from the #267 48-query provider-backed run. Run it immediately after every
benchmark or paper bundle generation, using the `semantic_smoke.json` produced by the
same run:

```bash
PYTHONPATH=src python3 -m oarag check-retrieval-gate \
  --metrics reports/mit_deep_learning_eval/full_provider_matrix_v1/metrics.json \
  --config reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json \
  --semantic-smoke reports/mit_deep_learning_eval/full_provider_matrix_v1/semantic_smoke.json
```

Treat a bundle as paper-ready only when the generated registry, readiness audit,
claim matrix, robustness intervals, and retrieval gate all support the intended
claim.

For dynamic concept graph diagnostics, use `cross-lecture-retrieval-smoke` and keep
Meili-only, Graph-only, Meili+Graph, and graph-aware rerank results separate from the
retrieval-answer bundle. Those rows support candidate recall, source contribution,
rerank delta, and skip-reason diagnostics only.

## Current Claim Boundary

As of the current repository state, this directory should be read as a public-safe
runbook and aggregate-report area, not as proof of final MIT paper performance. The
claim boundary is maintained in `../paper/current_claim_status.md`.

Do not claim provider-backed hybrid retrieval gains, VLM-over-OCR improvements, frame
coverage improvement, or aggregate timestamp-only link reduction from the skeleton
files alone. Do not claim that Graph DB availability proves verified object alignment.
Those claims require a fresh local bundle or cross-lecture smoke run and
privacy-checked aggregate artifacts.
