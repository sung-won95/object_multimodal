# Issue 280 strict deterministic verifier smoke

This public-safe sample records the expected shape of the strict deterministic
entity-link verifier report. It uses synthetic fixture-style counts only and
does not expose raw transcripts, visual text, local paths, raw link IDs, or
private lecture content.

Smoke command shape:

```bash
PYTHONPATH=src python -m oarag verify-entity-links-strict \
  --project-dir <project_dir> \
  --entity-links manifests/entity_links.jsonl \
  --output manifests/entity_links.strict.jsonl \
  --report reports/strict_deterministic_entity_link_verifier.json \
  --lexical-overlap-threshold 0.5 \
  --sweep-threshold 0.25 \
  --sweep-threshold 0.5 \
  --sweep-threshold 0.75

PYTHONPATH=src python -m oarag build-project-evidence-units \
  --project-dir <project_dir> \
  --entity-links manifests/entity_links.strict.jsonl
```

Expected smoke properties:

- `promoted_links > 0`
- `time_overlap_only_promoted = false`
- evidence-unit rebuild has `verified_object_alignment_ratio > 0`
- evidence-unit rebuild has
  `verified_link_source_counts.strict_deterministic_rule > 0`
- timestamp-fallback/time-overlap-only links remain unverified

The repository test suite fixes those properties with synthetic artifacts in
`tests/test_strict_deterministic_verifier.py`.
