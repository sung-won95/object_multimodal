# Next Steps

## Mapping Quality Report

`oarag mapping-quality-report` generates a public-safe aggregate report across multiple lecture project artifact directories or a manifest. It writes `mapping_quality_metrics.json` and `mapping_quality_summary.md`, with optional `mapping_quality_artifacts.csv`.

Use the report to check whether lecture loading, segment/window mapping, visual-state/entity alignment, entity-link verification, and concept clustering are healthy before using the artifacts in paper-facing analysis. Missing optional artifacts are reported as `missing_artifact` or `not_configured` instead of failing the run.

Interpret `timestamp_only_counted_as_verified_count` as a regression guard: it should stay `0`, because timestamp-only fallback links must not be counted as verified object alignment. Cross-lecture concept quality is summarized with lecture-local vs canonical concept counts, alias merge ratio, singleton ratio, hub counts, relation support counts, and over-merge reason codes.

The output is public-safe by construction: raw transcripts, local paths, raw queries, raw evidence text, visual labels, and raw candidate IDs are excluded. Project and lecture references are hashed.
