# full_provider_matrix_v2 Go/No-Go Report

## Decision

**Result: no-go.** Continue as a benchmark/evaluation-protocol paper instead of a performance-improvement paper.

Follow-up benchmark transition issue: https://github.com/sung-won95/object_multimodal/issues/285

## Fixed Criteria

- Go requires `evidence_unit_verified` citation precision to increase significantly over `evidence_unit_candidate` by paired bootstrap 95% CI.
- Go also requires unsupported claim ratio to decrease and Hit@10s loss to stay within absolute 0.10.
- Bootstrap settings: query-level pairs, n=48, 1000 samples, seed 139.

## Matrix Completion

- 24 retrieval-answer suites completed.
- 9 variants completed per suite.
- 432 query-result rows were produced.
- Skipped variant count: 0.
- Semantic live smoke passed with provider-backed query vectors: provider-backed query vector count 96, local hash query vector count 0.

## Candidate vs Verified

| Metric | Candidate mean | Verified mean | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| citation_precision | 0.277777 | 0.298613 | +0.020835 | [0.000000, 0.048612] |
| expected_citation_hit | 0.458333 | 0.479167 | +0.020833 | [0.000000, 0.062500] |
| unsupported_claim_ratio | 0.291667 | 0.270833 | -0.020833 | [-0.062500, 0.000000] |
| hit_at_10s | 0.520833 | 0.541667 | +0.020833 | [0.000000, 0.062500] |

Citation precision did not show a strictly positive 95% CI lower bound, so the primary go criterion is not met. Unsupported claim ratio moved lower and Hit@10s did not regress, but those are insufficient without significant citation precision lift.

## Verifier Before/After

| Metric | v1 verified mean | v2 verified mean | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| citation_precision | 0.298613 | 0.298613 | +0.000000 | [0.000000, 0.000000] |
| expected_citation_hit | 0.479167 | 0.479167 | +0.000000 | [0.000000, 0.000000] |
| unsupported_claim_ratio | 0.270833 | 0.270833 | +0.000000 | [0.000000, 0.000000] |
| hit_at_10s | 0.541667 | 0.541667 | +0.000000 | [0.000000, 0.000000] |

The deterministic verifier rebuild did not change aggregate `evidence_unit_verified` query outcomes on the locked 48-query matrix.

## Quality Gate

The bundle completed, but the regression gate failed:

| Suite | Variant | Metric | Value | Threshold |
| --- | --- | --- | ---: | ---: |
| mitdl_lec05 | window_hybrid | hit_at_10s | 0.5 | 0.9 |
| mitdl_lec05 | window_hybrid | mrr_at_max_delta | 0.5 | 0.9 |
| mitdl_lec05 | window_hybrid | expected_citation_hit_ratio | 0.5 | 0.9 |

Paper readiness and claim generation remain blocked by the gate failure.

## Limitations

- #281 human audit and real VLM smoke are not complete; this is a deterministic-first v2 decision.
- The expanded query set from #270 was not available, so this report uses the locked 48-query matrix.
- The no-go result turns the verifier, matrix, and annotation outputs into benchmark baseline and evaluation-protocol assets.
