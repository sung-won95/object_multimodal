# Issue 255 Window-Aware Target-Rank Smoke

schema_version: public-safe-evaluation-smoke-v1

## Scope

- Synthetic retrieval-answer matrix fixture only.
- Public-safe diagnostics only: no raw query text, raw transcript text, vectors, or local absolute paths.
- Ranking inputs were not changed; this smoke covers evaluation post-processing.

## Before

- Evidence-unit target-rank diagnostics exposed one target rank/bucket.
- A hit through `source_segment_ids` was not reported as a separate window-aware match bucket in public metrics.
- Time overlap against gold ranges was not reported as a separate target-rank match bucket.

## After

- Target-rank diagnostics distinguish `exact`, `window`, and `time_overlap` matches.
- `source_segment_ids` containing the gold segment is counted as a window-aware target found.
- Variant aggregate metrics include `target_match_type_counts` and `target_match_rank_bucket_counts`.

## Smoke Result

- Command: `python3 -m pytest tests/test_benchmark.py -q`
- Result: `13 passed`

## Larger Matrix Note

The 1-4 lecture retrieval-answer matrix was not run in this isolated worktree because it requires external local Meilisearch/evidence-unit artifacts that are not available here. The PR body records this as a remaining validation gap.
