# Issue 188 Evidence Unit Target-Rank Diagnostics

Date: 2026-06-01
Runner: `evidence-unit-smoke`

## Scope

This public-safe run adds target-rank diagnostics on top of the evidence-unit
retrieval smoke path. It uses two MIT Deep Learning lecture artifact copies and
records whether configured expected targets appear within the top 50 evidence-unit
results.

The private manifest and copied lecture artifacts stayed outside the repository.
Committed outputs redact raw query text, transcripts, local paths, and raw
segment/entity/evidence-unit IDs.

Timestamp-only overlap is reported as fallback/candidate evidence only. It is not
counted as verified object alignment.

## Runtime

Meilisearch was available through the repository Docker Compose service during this
run, so the build/index/query path was exercised.

```bash
docker compose up -d meilisearch
PYTHONPATH=src python3 -m oarag evidence-unit-smoke \
  --manifest <tmp-private-manifest> \
  --output-dir <repo>/reports/paper/evidence_units_retrieval_smoke/issue_188_mit_target_rank
```

## Aggregate Results

| Metric | Value |
| --- | ---: |
| suites | 2 |
| queries | 4 |
| queried | 4 |
| skipped | 0 |
| indexed evidence units | 2474 |
| inspectable top hits | 4 |
| top hits with visual state | 4 |
| configured targets found in top 50 | 2 |
| configured targets not found in top 50 | 2 |

## Target Rank Buckets

| Suite | configured | found top-k | top1 | top5 | top10 | top50 | not_found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | 2 | 1 | 0 | 0 | 0 | 1 | 1 |
| mit_lec04 | 2 | 1 | 0 | 0 | 1 | 0 | 1 |
| total | 4 | 2 | 0 | 0 | 1 | 1 | 2 |

## Quality Delta Notes

- All top hits and found targets remained candidate-level; no explicit verified
  object links were present.
- One found target had more visual entities than its top hit.
- One found target carried timestamp-fallback/candidate link flags, but this is not
  counted as verified object alignment.

## Artifacts

- `metrics.json`: public-safe aggregate metrics and suite summaries.
- `query_results.jsonl`: one sanitized row per query with target-rank diagnostics.
- `summary.md`: runner-generated public summary.

## Interpretation

The top-hit mismatch from the earlier smoke was partly a ranking problem and partly
a deeper retrieval/storage problem. Two configured targets appeared within top 50
but not at rank 1, while two configured targets were not found within top 50. This
keeps the next step focused: improve evidence-unit quality and ranking separately,
without treating timestamp-only overlap as verified object alignment.
