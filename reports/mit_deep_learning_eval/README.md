# MIT Deep Learning Evaluation Reports

This directory contains public-safe MIT Deep Learning evaluation report entry points.
Committed files here must stay aggregate-only: no raw query text, answer text, transcript
content, candidate evidence text, private eval values, or absolute local paths.

## Report Sets

- `seed_v1/`: seed retrieval benchmark aggregates and failure-analysis summaries.
- `paper_matrix_v1/`: runbook and skeleton configuration for the MIT paper bundle
  retrieval-answer ablation matrix.

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

The default gate in `paper_matrix_v1/retrieval_quality_gate.json` is a completeness
guard for bundle continuity, not a final paper performance threshold. Treat a bundle
as paper-ready only when the generated registry, readiness audit, claim matrix, and
robustness intervals all support the intended claim.
