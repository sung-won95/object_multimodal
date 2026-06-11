# Paper Metric Intervals: mit_deep_learning_stt_full_provider_matrix_v2

Aggregate-only bootstrap confidence intervals and paired deltas for paper metrics.
Raw queries, answers, transcripts, candidate evidence, query IDs, and local paths are excluded.

## Robustness Status

- Status: `passed`
- Paired delta rows: 48
- Caveat count: 0
- Interpretation: Paired baseline deltas are available without generated caveats.

## Caveats

- No caveats were produced for the configured summaries.

## Aggregate Metrics

| variant | metric | n | mean | 95% CI | direction | caveats |
| --- | --- | ---: | ---: | --- | --- | --- |
| segment_lexical | hit_at_10s | 48 | 0.1458 | [0.0625, 0.25] | higher_is_better |  |
| segment_lexical | mrr | 48 | 0.1667 | [0.0625, 0.2708] | higher_is_better |  |
| segment_lexical | top1_abs_error | 48 | 1162.5988 | [864.8765, 1469.3688] | lower_is_better |  |
| segment_lexical | grounded_answer | 48 | 0.125 | [0.0417, 0.2292] | higher_is_better |  |
| segment_lexical | expected_citation_hit | 48 | 0.0833 | [0.0208, 0.1667] | higher_is_better |  |
| segment_lexical | unsupported_claim_count | 48 | 0.0625 | [0, 0.1458] | lower_is_better |  |
| domain_lexicon | hit_at_10s | 48 | 0.2292 | [0.125, 0.3542] | higher_is_better |  |
| domain_lexicon | mrr | 48 | 0.2073 | [0.1042, 0.3261] | higher_is_better |  |
| domain_lexicon | top1_abs_error | 48 | 1103.7385 | [739.9353, 1471.7469] | lower_is_better |  |
| domain_lexicon | grounded_answer | 48 | 0.2917 | [0.1667, 0.4167] | higher_is_better |  |
| domain_lexicon | expected_citation_hit | 48 | 0.125 | [0.0417, 0.2292] | higher_is_better |  |
| domain_lexicon | unsupported_claim_count | 48 | 0.25 | [0.125, 0.3958] | lower_is_better |  |
| hybrid | hit_at_10s | 48 | 0.875 | [0.7708, 0.9583] | higher_is_better |  |
| hybrid | mrr | 48 | 0.7333 | [0.6218, 0.8372] | higher_is_better |  |
| hybrid | top1_abs_error | 48 | 466.9177 | [224.2132, 743.3254] | lower_is_better |  |
| hybrid | grounded_answer | 48 | 0.7917 | [0.6667, 0.8958] | higher_is_better |  |
| hybrid | expected_citation_hit | 48 | 0.5625 | [0.4167, 0.7083] | higher_is_better |  |
| hybrid | unsupported_claim_count | 48 | 0.2917 | [0.1458, 0.4583] | lower_is_better |  |
| window | hit_at_10s | 48 | 0.1667 | [0.0833, 0.2708] | higher_is_better |  |
| window | mrr | 48 | 0.1788 | [0.0833, 0.2795] | higher_is_better |  |
| window | top1_abs_error | 48 | 1102.0358 | [794.8451, 1440.6139] | lower_is_better |  |
| window | grounded_answer | 48 | 0.1042 | [0.0208, 0.2083] | higher_is_better |  |
| window | expected_citation_hit | 48 | 0.1667 | [0.0625, 0.2708] | higher_is_better |  |
| window | unsupported_claim_count | 48 | 0.0833 | [0, 0.1875] | lower_is_better |  |
| window_hybrid | hit_at_10s | 48 | 0.6667 | [0.5208, 0.7917] | higher_is_better |  |
| window_hybrid | mrr | 48 | 0.6149 | [0.4941, 0.732] | higher_is_better |  |
| window_hybrid | top1_abs_error | 48 | 396.7383 | [189.1285, 666.6424] | lower_is_better |  |
| window_hybrid | grounded_answer | 48 | 0.2917 | [0.1667, 0.4375] | higher_is_better |  |
| window_hybrid | expected_citation_hit | 48 | 0.5208 | [0.375, 0.6667] | higher_is_better |  |
| window_hybrid | unsupported_claim_count | 48 | 0.1042 | [0.0208, 0.2292] | lower_is_better |  |
| rerank | hit_at_10s | 48 | 0.1667 | [0.0833, 0.2708] | higher_is_better |  |
| rerank | mrr | 48 | 0.1823 | [0.0833, 0.2917] | higher_is_better |  |
| rerank | top1_abs_error | 48 | 1170.5392 | [861.6196, 1490.1781] | lower_is_better |  |
| rerank | grounded_answer | 48 | 0.125 | [0.0417, 0.2292] | higher_is_better |  |
| rerank | expected_citation_hit | 48 | 0.0833 | [0.0208, 0.1667] | higher_is_better |  |
| rerank | unsupported_claim_count | 48 | 0.0625 | [0, 0.1458] | lower_is_better |  |
| evidence_unit_candidate | hit_at_10s | 48 | 0.5208 | [0.375, 0.6458] | higher_is_better |  |
| evidence_unit_candidate | mrr | 48 | 0.4087 | [0.2889, 0.525] | higher_is_better |  |
| evidence_unit_candidate | top1_abs_error | 48 | 863.5237 | [590.6192, 1154.2577] | lower_is_better |  |
| evidence_unit_candidate | grounded_answer | 48 | 0.4792 | [0.3333, 0.625] | higher_is_better |  |
| evidence_unit_candidate | expected_citation_hit | 48 | 0.4583 | [0.3328, 0.5833] | higher_is_better |  |
| evidence_unit_candidate | unsupported_claim_count | 48 | 0.2917 | [0.1667, 0.4167] | lower_is_better |  |
| evidence_unit_verified | hit_at_10s | 48 | 0.5417 | [0.3958, 0.6875] | higher_is_better |  |
| evidence_unit_verified | mrr | 48 | 0.4097 | [0.3003, 0.5226] | higher_is_better |  |
| evidence_unit_verified | top1_abs_error | 48 | 848.4846 | [597.6678, 1149.235] | lower_is_better |  |
| evidence_unit_verified | grounded_answer | 48 | 0.4583 | [0.3333, 0.6042] | higher_is_better |  |
| evidence_unit_verified | expected_citation_hit | 48 | 0.4792 | [0.3536, 0.625] | higher_is_better |  |
| evidence_unit_verified | unsupported_claim_count | 48 | 0.2708 | [0.1458, 0.3958] | lower_is_better |  |
| evidence_unit_quality_rerank | hit_at_10s | 48 | 0.5208 | [0.3958, 0.6667] | higher_is_better |  |
| evidence_unit_quality_rerank | mrr | 48 | 0.4632 | [0.3451, 0.5847] | higher_is_better |  |
| evidence_unit_quality_rerank | top1_abs_error | 48 | 984.6256 | [696.6752, 1312.747] | lower_is_better |  |
| evidence_unit_quality_rerank | grounded_answer | 48 | 0.5833 | [0.4583, 0.7089] | higher_is_better |  |
| evidence_unit_quality_rerank | expected_citation_hit | 48 | 0.4583 | [0.3125, 0.6042] | higher_is_better |  |
| evidence_unit_quality_rerank | unsupported_claim_count | 48 | 0.2292 | [0.1042, 0.3542] | lower_is_better |  |

## Paired Deltas

| baseline | variant | metric | pairs | mean delta | 95% CI | mean improvement | direction | caveats |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |
| segment_lexical | domain_lexicon | hit_at_10s | 48 | 0.0833 | [-0.0625, 0.2292] | 0.0833 | higher_is_better |  |
| segment_lexical | domain_lexicon | mrr | 48 | 0.0406 | [-0.0802, 0.1657] | 0.0406 | higher_is_better |  |
| segment_lexical | domain_lexicon | top1_abs_error | 48 | -58.8602 | [-392.1169, 302.1649] | 58.8602 | lower_is_better |  |
| segment_lexical | domain_lexicon | grounded_answer | 48 | 0.1667 | [0.062, 0.2917] | 0.1667 | higher_is_better |  |
| segment_lexical | domain_lexicon | expected_citation_hit | 48 | 0.0417 | [-0.063, 0.1667] | 0.0417 | higher_is_better |  |
| segment_lexical | domain_lexicon | unsupported_claim_count | 48 | 0.1875 | [0.0833, 0.3333] | -0.1875 | lower_is_better |  |
| segment_lexical | hybrid | hit_at_10s | 48 | 0.7292 | [0.6042, 0.8542] | 0.7292 | higher_is_better |  |
| segment_lexical | hybrid | mrr | 48 | 0.5667 | [0.4295, 0.7039] | 0.5667 | higher_is_better |  |
| segment_lexical | hybrid | top1_abs_error | 48 | -695.681 | [-1054.8089, -326.3306] | 695.681 | lower_is_better |  |
| segment_lexical | hybrid | grounded_answer | 48 | 0.6667 | [0.5208, 0.8125] | 0.6667 | higher_is_better |  |
| segment_lexical | hybrid | expected_citation_hit | 48 | 0.4792 | [0.3542, 0.625] | 0.4792 | higher_is_better |  |
| segment_lexical | hybrid | unsupported_claim_count | 48 | 0.2292 | [0.0833, 0.3958] | -0.2292 | lower_is_better |  |
| segment_lexical | window | hit_at_10s | 48 | 0.0208 | [-0.0625, 0.1042] | 0.0208 | higher_is_better |  |
| segment_lexical | window | mrr | 48 | 0.0122 | [-0.0783, 0.1094] | 0.0122 | higher_is_better |  |
| segment_lexical | window | top1_abs_error | 48 | -60.5629 | [-343.0432, 254.0438] | 60.5629 | lower_is_better |  |
| segment_lexical | window | grounded_answer | 48 | -0.0208 | [-0.125, 0.0625] | -0.0208 | higher_is_better |  |
| segment_lexical | window | expected_citation_hit | 48 | 0.0833 | [0.0208, 0.1667] | 0.0833 | higher_is_better |  |
| segment_lexical | window | unsupported_claim_count | 48 | 0.0208 | [-0.0625, 0.125] | -0.0208 | lower_is_better |  |
| segment_lexical | window_hybrid | hit_at_10s | 48 | 0.5208 | [0.3542, 0.6667] | 0.5208 | higher_is_better |  |
| segment_lexical | window_hybrid | mrr | 48 | 0.4483 | [0.3083, 0.5938] | 0.4483 | higher_is_better |  |
| segment_lexical | window_hybrid | top1_abs_error | 48 | -765.8604 | [-1085.6032, -454.1575] | 765.8604 | lower_is_better |  |
| segment_lexical | window_hybrid | grounded_answer | 48 | 0.1667 | [0, 0.3125] | 0.1667 | higher_is_better |  |
| segment_lexical | window_hybrid | expected_citation_hit | 48 | 0.4375 | [0.2917, 0.5833] | 0.4375 | higher_is_better |  |
| segment_lexical | window_hybrid | unsupported_claim_count | 48 | 0.0417 | [-0.0833, 0.1875] | -0.0417 | lower_is_better |  |
| segment_lexical | rerank | hit_at_10s | 48 | 0.0208 | [0, 0.0625] | 0.0208 | higher_is_better |  |
| segment_lexical | rerank | mrr | 48 | 0.0156 | [0, 0.0417] | 0.0156 | higher_is_better |  |
| segment_lexical | rerank | top1_abs_error | 48 | 7.9404 | [-35.3814, 62.3255] | -7.9404 | lower_is_better |  |
| segment_lexical | rerank | grounded_answer | 48 | 0 | [0, 0] | 0 | higher_is_better |  |
| segment_lexical | rerank | expected_citation_hit | 48 | 0 | [0, 0] | 0 | higher_is_better |  |
| segment_lexical | rerank | unsupported_claim_count | 48 | 0 | [0, 0] | 0 | lower_is_better |  |
| segment_lexical | evidence_unit_candidate | hit_at_10s | 48 | 0.375 | [0.2292, 0.5208] | 0.375 | higher_is_better |  |
| segment_lexical | evidence_unit_candidate | mrr | 48 | 0.242 | [0.1176, 0.3698] | 0.242 | higher_is_better |  |
| segment_lexical | evidence_unit_candidate | top1_abs_error | 48 | -299.075 | [-671.6007, 69.6747] | 299.075 | lower_is_better |  |
| segment_lexical | evidence_unit_candidate | grounded_answer | 48 | 0.3542 | [0.2292, 0.4792] | 0.3542 | higher_is_better |  |
| segment_lexical | evidence_unit_candidate | expected_citation_hit | 48 | 0.375 | [0.25, 0.5208] | 0.375 | higher_is_better |  |
| segment_lexical | evidence_unit_candidate | unsupported_claim_count | 48 | 0.2292 | [0.125, 0.3542] | -0.2292 | lower_is_better |  |
| segment_lexical | evidence_unit_verified | hit_at_10s | 48 | 0.3958 | [0.2292, 0.5417] | 0.3958 | higher_is_better |  |
| segment_lexical | evidence_unit_verified | mrr | 48 | 0.2431 | [0.1059, 0.3802] | 0.2431 | higher_is_better |  |
| segment_lexical | evidence_unit_verified | top1_abs_error | 48 | -314.1142 | [-682.1133, 63.0475] | 314.1142 | lower_is_better |  |
| segment_lexical | evidence_unit_verified | grounded_answer | 48 | 0.3333 | [0.2083, 0.4792] | 0.3333 | higher_is_better |  |
| segment_lexical | evidence_unit_verified | expected_citation_hit | 48 | 0.3958 | [0.25, 0.5417] | 0.3958 | higher_is_better |  |
| segment_lexical | evidence_unit_verified | unsupported_claim_count | 48 | 0.2083 | [0.1042, 0.3333] | -0.2083 | lower_is_better |  |
| segment_lexical | evidence_unit_quality_rerank | hit_at_10s | 48 | 0.375 | [0.2292, 0.5208] | 0.375 | higher_is_better |  |
| segment_lexical | evidence_unit_quality_rerank | mrr | 48 | 0.2965 | [0.1406, 0.4556] | 0.2965 | higher_is_better |  |
| segment_lexical | evidence_unit_quality_rerank | top1_abs_error | 48 | -177.9731 | [-591.5254, 215.7334] | 177.9731 | lower_is_better |  |
| segment_lexical | evidence_unit_quality_rerank | grounded_answer | 48 | 0.4583 | [0.2917, 0.6042] | 0.4583 | higher_is_better |  |
| segment_lexical | evidence_unit_quality_rerank | expected_citation_hit | 48 | 0.375 | [0.2292, 0.5417] | 0.375 | higher_is_better |  |
| segment_lexical | evidence_unit_quality_rerank | unsupported_claim_count | 48 | 0.1667 | [0.0411, 0.2917] | -0.1667 | lower_is_better |  |

## Stable Schema

- Schema: `paper-metric-intervals-v1`
- Variant aggregate rows live under `variants[].metrics`.
- Robustness comparisons live under `paired_deltas[]` and use `variant_minus_baseline`.
- Privacy policy is recorded under `privacy`.
