# Issue 192 Evidence Unit Semantic Coverage Smoke

Date: 2026-06-01
Runner: `evidence-unit-smoke --quality-rerank`

## Scope

This public-safe smoke reran the two MIT lecture evidence-unit retrieval slice after
adding content coverage diagnostics. The private manifest and shadow artifact paths
were kept outside the repository. Committed outputs redact raw query text, raw
transcripts, local paths, and raw segment/entity/evidence-unit IDs.

Timestamp-only overlap is reported only as fallback/candidate evidence. It is not
counted as verified object alignment.

## Runtime

Meilisearch was available through the repository Docker Compose service.

```bash
docker compose up -d meilisearch
PYTHONPATH=src python3 -c 'from oarag.cli import main; main()' \
  evidence-unit-smoke \
  --manifest <tmp-private-manifest> \
  --output-dir <repo>/reports/paper/evidence_units_retrieval_smoke/issue_192_semantic_coverage \
  --quality-rerank
```

## Aggregate Results

| Metric | Value |
| --- | ---: |
| suites | 2 |
| queries | 4 |
| queried | 4 |
| skipped | 0 |
| indexed evidence units | 2474 |
| base top-hit target match | 0/4 |
| reranked top-hit target match | 0/4 |
| target found within top-k | 2/4 |
| target not found within top-k | 2/4 |
| top result changed by quality rerank | 0/4 |

Target rank buckets:

| Bucket | Count |
| --- | ---: |
| top10 | 1 |
| top50 | 1 |
| not_found | 2 |

## Content Coverage Diagnosis

The two targets found within the search depth had stronger content coverage than the
top hits:

| Signal | Count |
| --- | ---: |
| target has more query-term matches | 2 |
| target has longer semantic text | 2 |
| target has more visual entities | 1 |

For found targets, query-term coverage buckets were high or very high. The two
not-found targets could not be compared directly because they were outside the
configured search depth.

Top-hit query-term buckets:

| Bucket | Count |
| --- | ---: |
| medium | 2 |
| high | 1 |
| very_high | 1 |

Interpretation: the #191 0/4 top-hit failure does not look like a pure metadata
ranking problem. At least two configured targets have enough query-term/content signal
to be plausible answers but are still below the top result. The remaining two targets
are not found within the configured depth, which points to retrieval/index content or
query matching gaps rather than only rerank weighting.

## Storage Coverage

| Suite | evidence_units | candidate | transcript_only | verified | detected_text units | visual_description units | VLM units | verified links | timestamp fallback links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | 1023 | 1011 | 12 | 0 | 0 | 0 | 0 | 0 | 1946 |
| mit_lec04 | 1451 | 1431 | 20 | 0 | 0 | 0 | 0 | 0 | 2256 |

The local MIT slice still has no VLM entity coverage, no verified links, no
detected-text field coverage, and no visual-description field coverage in the
generated evidence units. The builder now preserves detected-text fields when present,
but this smoke confirms the current local artifacts do not provide that signal.

## RAG Input Inspectability

The public query rows now expose, without raw text:

- evidence/semantic/transcript char counts and buckets;
- query-term match counts, ratios, and buckets;
- top-vs-target content deltas when the target is found;
- visual state/entity/link counts;
- explicit verified/candidate/timestamp-fallback link counts and flags.

## Follow-Up Candidates

1. Add VLM-first visual entities or visual descriptions for the same two-lecture slice.
2. Investigate the two not-found targets by inspecting private raw evidence locally,
   then convert the finding into public-safe aggregate diagnostics.
3. Add retrieval/index ablations that compare transcript-only semantic text versus
   visual-context-augmented semantic text without using expected labels in scoring.
