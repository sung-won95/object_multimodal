# MIT Seed Retrieval Failure Analysis: mit_deep_learning_stt_seed_v1

## Headline

- Baseline aggregate: mean Hit@10s=0.1458, mean MRR=0.1667, mean Hit@30s=0.1667.
- Coverage: 24 suites, 48 queries, 1 domain.
- Query-level detail: loaded (48 records).

## Failure Buckets

| bucket | level | affected queries | query share | suite variants | interpretation | next action |
| --- | --- | ---: | ---: | ---: | --- | --- |
| miss_at_10s | query | 41 | 0.8542 | 23 | The stricter 10 second window still misses most expected timestamps. | Use this as the headline precision-localization risk for seed results. |
| no_hit_at_30s | query | 40 | 0.8333 | 17 | The expected timestamp was not retrieved within the 30 second tolerance. | Treat this as the primary recall gap before making paper claims. |
| rerank_candidate_needed | query | 37 | 0.7708 | 20 | Reranking is disabled while top-1 localization error is large. | Run a reranker or candidate-generation ablation before paper-level reporting. |
| high_top1_error | query | 29 | 0.6042 | 18 | The top ranked candidate is far from the expected timestamp. | Audit ranking signals and candidate scoring before relying on top-1 evidence. |
| low_linked_entity_backed | query | 17 | 0.3542 | 12 | A material share of results lacks linked visual-entity evidence. | Audit entity linking and visual label coverage before grounding claims. |
| candidate_label_audit_needed | query | 15 | 0.3125 | 9 | Misses coincide with weak frame or linked-entity backing. | Inspect visual candidate labeling and entity links with private inputs only. |
| low_frame_backed | query | 12 | 0.25 | 9 | A material share of results is not backed by frame evidence. | Improve frame coverage before comparing visual grounding claims. |

## Domain Aggregate

| domain | suites | queries | Hit@10s | Hit@30s | MRR | top1 err s | frame-backed | linked-backed | top buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mit_deep_learning_stt_seed | 24 | 48 | 0.1458 | 0.1667 | 0.1667 | 1146.2098 | 0.75 | 0.6458 | miss_at_10s, no_hit_at_30s, rerank_candidate_needed |

## Suite Variant Aggregate

| suite | variant | queries | Hit@10s | Hit@30s | MRR | top1 err s | frame | linked | buckets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mitdl_lec01 | suite | 2 | 0 | 0 | 0 | 632.285 | 0.5 | 0.5 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec02 | suite | 2 | 0 | 0.5 | 0.5 | 458.85 | 1 | 0.5 | miss_at_10s, low_linked_entity_backed, rerank_candidate_needed |
| mitdl_lec03 | suite | 2 | 0 | 0 | 0 | 1203.49 | 0 | 0 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec04 | suite | 2 | 0.5 | 0.5 | 0.5 | 1066.66 | 1 | 1 | miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec05 | suite | 2 | 0.5 | 0.5 | 0.5 | 139.11 | 0.5 | 0.5 | miss_at_10s, low_frame_backed, low_linked_entity_backed |
| mitdl_lec06 | suite | 2 | 0 | 0 | 0 | 2435.095 | 1 | 0.5 | no_hit_at_30s, miss_at_10s, high_top1_error, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec07 | suite | 2 | 0 | 0 | 0 | 2104.395 | 0.5 | 0 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec08 | suite | 2 | 0 | 0 | 0 | 288.74 | 1 | 1 | no_hit_at_30s, miss_at_10s |
| mitdl_lec09 | suite | 2 | 0 | 0 | 0 | 2572.49 | 0.5 | 0.5 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec10 | suite | 2 | 0 | 0 | 0 | 1361.525 | 1 | 1 | no_hit_at_30s, miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec11 | suite | 2 | 0 | 0 | 0 | 479.82 | 1 | 1 | no_hit_at_30s, miss_at_10s, rerank_candidate_needed |
| mitdl_lec12 | suite | 2 | 1 | 1 | 1 | 2.045 | 1 | 1 | - |
| mitdl_lec13 | suite | 2 | 0 | 0 | 0 | 933.27 | 0 | 0 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec14 | suite | 2 | 0.5 | 0.5 | 0.5 | 1236.9 | 1 | 1 | miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec15 | suite | 2 | 0 | 0 | 0 | 992.665 | 0 | 0 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec16 | suite | 2 | 0 | 0 | 0 | 1599.205 | 1 | 1 | no_hit_at_30s, miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec17 | suite | 2 | 0 | 0 | 0 | 1117.85 | 1 | 1 | no_hit_at_30s, miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec18 | suite | 2 | 0.5 | 0.5 | 0.5 | 287.695 | 1 | 0.5 | miss_at_10s, low_linked_entity_backed |
| mitdl_lec19 | suite | 2 | 0 | 0 | 0 | 3072.985 | 1 | 1 | no_hit_at_30s, miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec20 | suite | 2 | 0 | 0 | 0 | 825.15 | 0.5 | 0.5 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |
| mitdl_lec21 | suite | 2 | 0.5 | 0.5 | 0.5 | 1368.68 | 1 | 1 | miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec23 | suite | 2 | 0 | 0 | 0 | 1028.08 | 1 | 1 | no_hit_at_30s, miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_lec24 | suite | 2 | 0 | 0 | 0 | 1128.825 | 1 | 1 | no_hit_at_30s, miss_at_10s, high_top1_error, rerank_candidate_needed |
| mitdl_review | suite | 2 | 0 | 0 | 0 | 1173.225 | 0.5 | 0 | no_hit_at_30s, miss_at_10s, high_top1_error, low_frame_backed, low_linked_entity_backed, candidate_label_audit_needed, rerank_candidate_needed |

## Public Output Policy

This artifact is aggregate-only. It excludes raw query strings, raw transcripts, candidate evidence text, local filesystem paths, frame file references, and raw candidate identifiers.

## Limitations

- This is an aggregate diagnostic, not a claim that the seed baseline is paper-ready.
- Buckets can overlap, so counts should not be summed into a unique failure total.
- Top-1 error thresholds are diagnostic cutoffs for triage, not statistical significance tests.
- Query-level records were used only for counts; raw text, evidence snippets, paths, and candidate identifiers were discarded.
