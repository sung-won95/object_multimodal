# Issue 186 Evidence Unit Retrieval Smoke

Date: 2026-06-01
Runner: `evidence-unit-smoke`

## Scope

This run applied the public-safe evidence unit retrieval smoke runner to two local MIT
Deep Learning lecture artifacts. The committed outputs redact raw query text,
transcripts, local paths, and raw segment/entity/evidence-unit IDs.

Timestamp-only overlap is reported as fallback/candidate evidence only. It is not
counted as verified object alignment.

## Runtime

Meilisearch was available through the repository Docker Compose service during this
run, so the available build/index/query path was exercised.

```bash
docker compose up -d meilisearch
PYTHONPATH=src python3 -c 'from oarag.cli import main; main()' \
  evidence-unit-smoke \
  --manifest <tmp-private-manifest> \
  --output-dir <repo>/reports/paper/evidence_units_retrieval_smoke/issue_186_mit
```

The private manifest was kept outside the repository because it contains raw query
text and expected raw segment IDs.

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
| top hits matching configured target | 0 |

## Per-Lecture Storage Counts

| Suite | evidence_units | candidate | transcript_only | verified | visual_state units | visual_entity units | vlm_entity units | verified_link units | candidate_links | timestamp_fallback_links | verified_links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | 1023 | 1011 | 12 | 0 | 1011 | 774 | 0 | 0 | 126 | 1946 | 0 |
| mit_lec04 | 1451 | 1431 | 20 | 0 | 1431 | 831 | 0 | 0 | 144 | 2256 | 0 |

## Top-Hit/Miss Memo

All four queries returned inspectable evidence-unit top hits, but every top hit missed
the configured seed target. Each top hit was candidate-level and had no verified
object link. This is useful as a storage/query smoke, but it is not evidence for an
object-aligned paper claim yet.

The top hits exposed the expected RAG input metadata:

- evidence and semantic text availability flags;
- hashed evidence-unit and target refs;
- source segment count;
- visual state/entity counts;
- candidate/verified alignment status;
- source quality flags.

Raw evidence text and semantic text are intentionally not included in this public
report.

## Artifacts

- `metrics.json`: public-safe aggregate metrics and suite summaries.
- `query_results.jsonl`: one sanitized row per query with top-hit/miss memo.
- `summary.md`: runner-generated short public summary.

## Interpretation

The runner successfully exercises the evidence-unit available path against
Meilisearch for two lecture artifacts. The retrieved rows are inspectable as RAG
inputs, but retrieval quality is weak on this slice: top hits miss the configured seed
targets, VLM entity counts are zero, and verified links are zero. The next iteration
should improve visual/entity evidence and ranking before making stronger object
alignment claims.
