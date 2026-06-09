# Issue 256 Remaining Miss Audit

## Scope

- lecture slice: `1-4`
- target variant: `evidence_unit_candidate`
- miss filter: `target_rank_bucket == not_found`
- Meili depths: `[5, 10, 30, 50, 100]`

## Public-Safe Status

- query results input: `synthetic_fixture`
- depth diagnostics input: `loaded`
- unrun reason: `1-4 lecture benchmark query_results were not available in this isolated worktree; wrote a synthetic fixture smoke report instead.`
- raw query text: `excluded`
- raw answers/transcripts/evidence text: `excluded`
- raw candidate ids and local paths: `excluded`

## Aggregate

- candidate rows audited: `2`
- remaining miss seed count: `2`
- cause counts: `{"candidate_depth_issue": 2, "evaluation_match_issue": 0, "missing_search_field": 1, "query_cleaning_issue": 1, "verified_coverage_issue": 1}`
- Meili depth found counts: `{"10": 0, "100": 1, "30": 1, "5": 0, "50": 1}`
- target search field coverage buckets: `{"full": 1, "minimal": 1}`
- quality bucket deltas: `{"same": 1, "target_stronger": 1}`

## Miss Rows

- seed: `seed:54ef50ceeada`
  - causes: `["candidate_depth_issue", "query_cleaning_issue"]`
  - top/target quality: `visual_candidate` -> `verified_visual`
  - target search coverage: `full`
  - Meili depth found: `{"10": false, "100": true, "30": true, "5": false, "50": true}`
- seed: `seed:d198cbe5b593`
  - causes: `["candidate_depth_issue", "missing_search_field", "verified_coverage_issue"]`
  - top/target quality: `transcript_only` -> `transcript_only`
  - target search coverage: `minimal`
  - Meili depth found: `{"10": false, "100": false, "30": false, "5": false, "50": false}`

Public note: This audit is public-safe: raw query text, raw answers, transcript excerpts, raw candidate ids, evidence text, and local absolute paths are excluded. Gold labels are consumed only through benchmark post-processing diagnostics, not as search input.
