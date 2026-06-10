# Retrieval Benchmark: mit_deep_learning_stt_full_provider_matrix_v1

## Suites

| suite | type | domain | lexicon | queries | Hit@10s | MRR | latency ms | frame-backed | linked-backed | rerank |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mitdl_lec01 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.2778 | 0.1944 | 7.1111 | 0.8889 | 0.5 | off |
| mitdl_lec02 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.2778 | 0.5556 | 5.6667 | 0.9444 | 0.3889 | off |
| mitdl_lec03 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.5556 | 0.5139 | 6.1111 | 0.5556 | 0.2222 | off |
| mitdl_lec04 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.6111 | 0.4111 | 8.5 | 1 | 0.6667 | off |
| mitdl_lec05 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.7222 | 0.6111 | 8.5 | 0.8333 | 0.5 | off |
| mitdl_lec06 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.5 | 0.3352 | 7.7222 | 0.9444 | 0.4444 | off |
| mitdl_lec07 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.0556 | 0.0111 | 6.7778 | 0.6667 | 0.2222 | off |
| mitdl_lec08 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.3889 | 0.2917 | 8.2222 | 1 | 0.6667 | off |
| mitdl_lec09 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.2222 | 0.1778 | 5.8889 | 0.8333 | 0.4444 | off |
| mitdl_lec10 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.3333 | 0.3241 | 6 | 1 | 0.6667 | off |
| mitdl_lec11 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.3333 | 0.2426 | 6.6667 | 1 | 0.6667 | off |
| mitdl_lec12 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.6111 | 0.4602 | 8.3333 | 1 | 0.6667 | off |
| mitdl_lec13 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.4444 | 0.4444 | 4.7778 | 0.5556 | 0.2222 | off |
| mitdl_lec14 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.5 | 0.5694 | 7.7778 | 1 | 0.6667 | off |
| mitdl_lec15 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.2778 | 0.1102 | 7.1667 | 0.7222 | 0.3333 | off |
| mitdl_lec16 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.5556 | 0.3695 | 7.3889 | 1 | 0.6667 | off |
| mitdl_lec17 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.4444 | 0.4 | 7.3889 | 1 | 0.6667 | off |
| mitdl_lec18 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.4444 | 0.5185 | 7.1667 | 0.9444 | 0.5 | off |
| mitdl_lec19 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.5556 | 0.4907 | 7.2778 | 1 | 0.6667 | off |
| mitdl_lec20 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.4444 | 0.4722 | 4.9444 | 0.8333 | 0.5 | off |
| mitdl_lec21 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.5556 | 0.5556 | 6.8889 | 1 | 0.6667 | off |
| mitdl_lec23 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.4444 | 0.4278 | 6.5 | 1 | 0.6667 | off |
| mitdl_lec24 | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.3889 | 0.3074 | 8.0556 | 1 | 0.6667 | off |
| mitdl_review | retrieval_answer_matrix | mit_deep_learning_stt_paper_matrix | off | 2 | 0.3333 | 0.2241 | 4.8889 | 0.9444 | 0.3333 | off |

## Retrieval/Answer Matrix

Matrix outputs are aggregate and public-safe: raw queries, answer text, candidate evidence, transcript excerpts, local paths, and raw candidate IDs are omitted.
Grounding proxy metrics use deterministic expected hint overlap for regression tracking; they do not replace full LLM answer quality review.
Grounding gap counts separate answer policy decisions from retrieval localization misses in the JSON metrics.
Evidence-unit rows may be queried or skipped; skipped variants report zero hit/MRR and a public-safe skip reason in JSON metrics.

| suite | variant | queries | skipped | pool | Hit@10s | MRR | target in pool | frame-backed | linked-backed | candidate support | verified align | VLM | VLM text | OCR engine | grounded | cite P | cite R | expected hit | unsupported | abstain | primary failures | latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mitdl_lec01 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec01 | domain_lexicon | 2 | 0 | - | 0.5 | 0.125 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec01 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0.5 | 0.5 | 0 | 1 | 0.3333 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 3.5 |
| mitdl_lec01 | window | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 2 |
| mitdl_lec01 | window_hybrid | 2 | 0 | - | 1 | 0.625 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 3.5 |
| mitdl_lec01 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec01 | evidence_unit_candidate | 2 | 0 | 50 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0.5 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | retrieval_miss:2 | 18.5 |
| mitdl_lec01 | evidence_unit_verified | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 12.5 |
| mitdl_lec01 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | retrieval_miss:2 | 21 |
| mitdl_lec02 | segment_lexical | 2 | 0 | - | 0 | 0.5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 18.5791 |
| mitdl_lec02 | domain_lexicon | 2 | 0 | - | 0.5 | 1 | 0.5 | 1 | 0.5 | 0.5 | 0 | 0.5 | 0.5 | 0 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 26.4113 |
| mitdl_lec02 | hybrid | 2 | 0 | - | 1 | 0.625 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 2 |
| mitdl_lec02 | window | 2 | 0 | - | 0 | 0.5 | 0 | 0.5 | 0 | 0.5 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec02 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0.5 | 0.5 | 0 | 0.5 | 0.3333 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 3 |
| mitdl_lec02 | rerank | 2 | 0 | 30 | 0 | 0.5 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec02 | evidence_unit_candidate | 2 | 0 | 50 | 0 | 0.5 | 0 | 1 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 14.5 |
| mitdl_lec02 | evidence_unit_verified | 2 | 0 | 30 | 0 | 0.25 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 11 |
| mitdl_lec02 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0.125 | 0 | 1 | 0 | 1 | 0 | 1 | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 19 |
| mitdl_lec03 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec03 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 16.9158 |
| mitdl_lec03 | hybrid | 2 | 0 | - | 1 | 0.625 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 2 |
| mitdl_lec03 | window | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec03 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.3333 | 1 | 1 | 0 | 1 | insufficient_evidence:2 | 4 |
| mitdl_lec03 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec03 | evidence_unit_candidate | 2 | 0 | 50 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0.8334 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 16.5 |
| mitdl_lec03 | evidence_unit_verified | 2 | 0 | 30 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0.8334 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 12 |
| mitdl_lec03 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 18 |
| mitdl_lec04 | segment_lexical | 2 | 0 | - | 0.5 | 0.5 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 25.2691 |
| mitdl_lec04 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 30.86 |
| mitdl_lec04 | hybrid | 2 | 0 | - | 1 | 0.6667 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 2.5 |
| mitdl_lec04 | window | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 2 |
| mitdl_lec04 | window_hybrid | 2 | 0 | - | 1 | 0.6 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 4 |
| mitdl_lec04 | rerank | 2 | 0 | 30 | 0.5 | 0.5 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec04 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.1667 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 24 |
| mitdl_lec04 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.1667 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 16 |
| mitdl_lec04 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.1 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 26.5 |
| mitdl_lec05 | segment_lexical | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 0.5 |
| mitdl_lec05 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec05 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0.5 | 0.3333 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 1.5 |
| mitdl_lec05 | window | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 3.5 |
| mitdl_lec05 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0.5 | 1 | 0.6666 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 4.5 |
| mitdl_lec05 | rerank | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 1.5 |
| mitdl_lec05 | evidence_unit_candidate | 2 | 0 | 50 | 1 | 0.75 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 1 | 0.8334 | 1 | 1 | 0.5 | 0 | grounded_expected_citation:1, unsupported_claims:1 | 23.5 |
| mitdl_lec05 | evidence_unit_verified | 2 | 0 | 30 | 1 | 0.75 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 1 | 0.8334 | 1 | 1 | 0.5 | 0 | grounded_expected_citation:1, unsupported_claims:1 | 16.5 |
| mitdl_lec05 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 0.5 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0.5 | insufficient_evidence:1, unsupported_claims:1 | 24.5 |
| mitdl_lec06 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec06 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec06 | hybrid | 2 | 0 | - | 1 | 0.6 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 2 |
| mitdl_lec06 | window | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 2 |
| mitdl_lec06 | window_hybrid | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 5 |
| mitdl_lec06 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec06 | evidence_unit_candidate | 2 | 0 | 50 | 1 | 0.4167 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.3333 | 1 | 1 | 0.5 | 0.5 | insufficient_evidence:1, unsupported_claims:1 | 24.5 |
| mitdl_lec06 | evidence_unit_verified | 2 | 0 | 30 | 1 | 0.5 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0.5 | insufficient_evidence:1, unsupported_claims:1 | 14.5 |
| mitdl_lec06 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 18.5 |
| mitdl_lec07 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec07 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 22.1805 |
| mitdl_lec07 | hybrid | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 2.5 |
| mitdl_lec07 | window | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec07 | window_hybrid | 2 | 0 | - | 0.5 | 0.1 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 3.5 |
| mitdl_lec07 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec07 | evidence_unit_candidate | 2 | 0 | 50 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 19 |
| mitdl_lec07 | evidence_unit_verified | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 12 |
| mitdl_lec07 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 21.5 |
| mitdl_lec08 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 1 |
| mitdl_lec08 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 0.5 |
| mitdl_lec08 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 1 | 0.3333 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 2.5 |
| mitdl_lec08 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 2 |
| mitdl_lec08 | window_hybrid | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 3 |
| mitdl_lec08 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 1 |
| mitdl_lec08 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.25 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 22.5 |
| mitdl_lec08 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.25 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 15 |
| mitdl_lec08 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 0.625 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 26.5 |
| mitdl_lec09 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 31.142 |
| mitdl_lec09 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 40.3441 |
| mitdl_lec09 | hybrid | 2 | 0 | - | 1 | 0.6 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 1 |
| mitdl_lec09 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec09 | window_hybrid | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 3.5 |
| mitdl_lec09 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec09 | evidence_unit_candidate | 2 | 0 | 50 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 18.5 |
| mitdl_lec09 | evidence_unit_verified | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 12.5 |
| mitdl_lec09 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 15.5 |
| mitdl_lec10 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 22.9693 |
| mitdl_lec10 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 0.5 |
| mitdl_lec10 | hybrid | 2 | 0 | - | 1 | 0.6667 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 1 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 2.5 |
| mitdl_lec10 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec10 | window_hybrid | 2 | 0 | - | 1 | 1 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 4 |
| mitdl_lec10 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec10 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.2917 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 16 |
| mitdl_lec10 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.2917 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 13 |
| mitdl_lec10 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0.6667 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 16 |
| mitdl_lec11 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 0.5 |
| mitdl_lec11 | domain_lexicon | 2 | 0 | - | 0.5 | 0.1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 0.5 |
| mitdl_lec11 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 1 | 0.3333 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 1 |
| mitdl_lec11 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 1.5 |
| mitdl_lec11 | window_hybrid | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 4.5 |
| mitdl_lec11 | rerank | 2 | 0 | 30 | 0 | 0.25 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 1.5 |
| mitdl_lec11 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.1667 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 17 |
| mitdl_lec11 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.1667 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 12.5 |
| mitdl_lec11 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 21 |
| mitdl_lec12 | segment_lexical | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec12 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec12 | hybrid | 2 | 0 | - | 1 | 0.6667 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 2.5 |
| mitdl_lec12 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec12 | window_hybrid | 2 | 0 | - | 0 | 0.125 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 4.5 |
| mitdl_lec12 | rerank | 2 | 0 | 30 | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec12 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.125 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 24.5 |
| mitdl_lec12 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.125 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 15.5 |
| mitdl_lec12 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 0.6 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0 | 1 | 0.3333 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 24.5 |
| mitdl_lec13 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec13 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 16.9807 |
| mitdl_lec13 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 1 | 0.3333 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 2 |
| mitdl_lec13 | window | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.3333 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 1 |
| mitdl_lec13 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 3.5 |
| mitdl_lec13 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec13 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 12 |
| mitdl_lec13 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 9.5 |
| mitdl_lec13 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 13.5 |
| mitdl_lec14 | segment_lexical | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 1 |
| mitdl_lec14 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 1 |
| mitdl_lec14 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0.3333 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 2.5 |
| mitdl_lec14 | window | 2 | 0 | - | 0.5 | 0.625 | 0.5 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.3333 | 0.5 | 0.5 | 1 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 2.5 |
| mitdl_lec14 | window_hybrid | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0.1666 | 0.5 | 0.5 | 1 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 3.5 |
| mitdl_lec14 | rerank | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 1 |
| mitdl_lec14 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.75 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.3333 | 0.5 | 0.5 | 0.5 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 23 |
| mitdl_lec14 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.75 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.3333 | 0.5 | 0.5 | 0.5 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 15.5 |
| mitdl_lec14 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 20 |
| mitdl_lec15 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 20.5657 |
| mitdl_lec15 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec15 | hybrid | 2 | 0 | - | 1 | 0.6 | 1 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 2.5 |
| mitdl_lec15 | window | 2 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec15 | window_hybrid | 2 | 0 | - | 0.5 | 0.1 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 3.5 |
| mitdl_lec15 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec15 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.125 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 19 |
| mitdl_lec15 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.1667 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 13 |
| mitdl_lec15 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | retrieval_miss:2 | 24 |
| mitdl_lec16 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_lec16 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 31.6604 |
| mitdl_lec16 | hybrid | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 1.5 |
| mitdl_lec16 | window | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 1 |
| mitdl_lec16 | window_hybrid | 2 | 0 | - | 1 | 0.2667 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 5.5 |
| mitdl_lec16 | rerank | 2 | 0 | 30 | 0.5 | 0.125 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec16 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.6 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 21 |
| mitdl_lec16 | evidence_unit_verified | 2 | 0 | 30 | 1 | 0.6667 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0.3333 | 1 | 1 | 0 | 1 | insufficient_evidence:2 | 14 |
| mitdl_lec16 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 0.6667 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.6666 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 22 |
| mitdl_lec17 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 0.5 |
| mitdl_lec17 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | retrieval_miss:2 | 29.2403 |
| mitdl_lec17 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 2 |
| mitdl_lec17 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 2 |
| mitdl_lec17 | window_hybrid | 2 | 0 | - | 0.5 | 0.6 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 5 |
| mitdl_lec17 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 1 |
| mitdl_lec17 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.25 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 1 | 0.1666 | 0.5 | 0.5 | 1 | 0 | retrieval_miss:1, unsupported_claims:1 | 20 |
| mitdl_lec17 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.25 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 1 | 0.1666 | 0.5 | 0.5 | 1 | 0 | retrieval_miss:1, unsupported_claims:1 | 14 |
| mitdl_lec17 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 1 | 1 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 1 | 0.3333 | 0.75 | 1 | 0 | 0 | grounded_expected_citation:2 | 22 |
| mitdl_lec18 | segment_lexical | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 0.5 |
| mitdl_lec18 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 36.7784 |
| mitdl_lec18 | hybrid | 2 | 0 | - | 0.5 | 1 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 2 |
| mitdl_lec18 | window | 2 | 0 | - | 0.5 | 0.1667 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 1.5 |
| mitdl_lec18 | window_hybrid | 2 | 0 | - | 0.5 | 1 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.3333 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 4.5 |
| mitdl_lec18 | rerank | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 1 |
| mitdl_lec18 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.5 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 1 | 0.3333 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 21 |
| mitdl_lec18 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 1 | 0.3333 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 13 |
| mitdl_lec18 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.5 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 21 |
| mitdl_lec19 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec19 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 25.4931 |
| mitdl_lec19 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0.5 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 1.5 |
| mitdl_lec19 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec19 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0.5 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 4.5 |
| mitdl_lec19 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec19 | evidence_unit_candidate | 2 | 0 | 50 | 1 | 0.6667 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.6666 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 20 |
| mitdl_lec19 | evidence_unit_verified | 2 | 0 | 30 | 1 | 0.75 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.8334 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 14.5 |
| mitdl_lec19 | evidence_unit_quality_rerank | 2 | 0 | 100 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 21.5 |
| mitdl_lec20 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 19.7652 |
| mitdl_lec20 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 29.9273 |
| mitdl_lec20 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0.1666 | 0.5 | 0.5 | 0.5 | 0 | grounded_expected_citation:1, retrieval_miss:1 | 2.5 |
| mitdl_lec20 | window | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1.5 |
| mitdl_lec20 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.3333 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 3.5 |
| mitdl_lec20 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 24.5807 |
| mitdl_lec20 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 12.5 |
| mitdl_lec20 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 11.5 |
| mitdl_lec20 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.75 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 13 |
| mitdl_lec21 | segment_lexical | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 0.5 |
| mitdl_lec21 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 0.5 |
| mitdl_lec21 | hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0.3333 | 1 | 1 | 0 | 0 | grounded_expected_citation:2 | 2.5 |
| mitdl_lec21 | window | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0.3333 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 2 |
| mitdl_lec21 | window_hybrid | 2 | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 | 0 | 1 | insufficient_evidence:2 | 4 |
| mitdl_lec21 | rerank | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 1 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 1.5 |
| mitdl_lec21 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 19 |
| mitdl_lec21 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 12 |
| mitdl_lec21 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 20 |
| mitdl_lec23 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 22.2523 |
| mitdl_lec23 | domain_lexicon | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 27.1812 |
| mitdl_lec23 | hybrid | 2 | 0 | - | 0.5 | 0.6 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 2 |
| mitdl_lec23 | window | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0.3333 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 2.5 |
| mitdl_lec23 | window_hybrid | 2 | 0 | - | 0.5 | 1 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 5.5 |
| mitdl_lec23 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec23 | evidence_unit_candidate | 2 | 0 | 50 | 1 | 0.75 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.6667 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 17 |
| mitdl_lec23 | evidence_unit_verified | 2 | 0 | 30 | 1 | 0.75 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.6667 | 1 | 1 | 0 | 0.5 | grounded_expected_citation:1, insufficient_evidence:1 | 13.5 |
| mitdl_lec23 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.25 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0.3333 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 17 |
| mitdl_lec24 | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec24 | domain_lexicon | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 1 |
| mitdl_lec24 | hybrid | 2 | 0 | - | 1 | 0.35 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | retrieval_miss:2 | 3 |
| mitdl_lec24 | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 2 |
| mitdl_lec24 | window_hybrid | 2 | 0 | - | 0.5 | 0.25 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 4.5 |
| mitdl_lec24 | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_lec24 | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 23.5 |
| mitdl_lec24 | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0.5 | 0.3333 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 14.5 |
| mitdl_lec24 | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.6667 | 0.5 | 1 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 0 | 0.5 | grounded_expected_citation:1, retrieval_miss:1 | 22 |
| mitdl_review | segment_lexical | 2 | 0 | - | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 10.0833 |
| mitdl_review | domain_lexicon | 2 | 0 | - | 0.5 | 0.25 | 0.5 | 1 | 0.5 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | 0.1666 | 0.5 | 0.5 | 1 | 0.5 | insufficient_evidence:1, retrieval_miss:1 | 24.6502 |
| mitdl_review | hybrid | 2 | 0 | - | 0.5 | 0.1 | 0.5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0 | 0 | 0 | 0.5 | 0.5 | retrieval_miss:2 | 2.5 |
| mitdl_review | window | 2 | 0 | - | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 1 |
| mitdl_review | window_hybrid | 2 | 0 | - | 0.5 | 0.5 | 0.5 | 1 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0.1666 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 3.5 |
| mitdl_review | rerank | 2 | 0 | 30 | 0 | 0 | 0 | 1 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | retrieval_miss:2 | 0.5 |
| mitdl_review | evidence_unit_candidate | 2 | 0 | 50 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0.5 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 12.5 |
| mitdl_review | evidence_unit_verified | 2 | 0 | 30 | 0.5 | 0.5 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0.5 | 0.5 | 0.5 | 0 | 1 | insufficient_evidence:1, retrieval_miss:1 | 10 |
| mitdl_review | evidence_unit_quality_rerank | 2 | 0 | 100 | 0.5 | 0.1667 | 0.5 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.1666 | 0.5 | 0.5 | 0.5 | 0.5 | retrieval_miss:1, unsupported_claims:1 | 14 |

### Verified Object Alignment Coverage

Verified coverage is reported separately from candidate/fallback visual support. Timestamp fallback is excluded from verified alignment and paper-claim eligible support.

| suite | variant | candidate support | candidate-only | verified align | timestamp fallback | missing reasons | verified failure stages |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| mitdl_lec01 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, vlm_object_without_verified_link:1 | - |
| mitdl_lec01 | evidence_unit_verified | 1 | 1 | 0 | 0 | vlm_object_without_verified_link:2 | retrieval:target_not_found:2 |
| mitdl_lec01 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | vlm_object_without_verified_link:2 | - |
| mitdl_lec02 | evidence_unit_candidate | 1 | 1 | 0 | 0.5 | candidate_visual_support_without_verified_link:1, vlm_object_without_verified_link:1 | - |
| mitdl_lec02 | evidence_unit_verified | 1 | 1 | 0 | 0 | vlm_object_without_verified_link:2 | retrieval:target_not_found:2 |
| mitdl_lec02 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | vlm_object_without_verified_link:1, vlm_visible_text_without_verified_link:1 | - |
| mitdl_lec03 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec03 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | ok:grounded_expected_citation:2 |
| mitdl_lec03 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec04 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec04 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | grounding:unsupported_claims:1, retrieval:target_not_found:1 |
| mitdl_lec04 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec05 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec05 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | grounding:unsupported_claims:1, ok:grounded_expected_citation:1 |
| mitdl_lec05 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec06 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec06 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:insufficient_evidence:1, grounding:unsupported_claims:1 |
| mitdl_lec06 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec07 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec07 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | retrieval:target_not_found:2 |
| mitdl_lec07 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec08 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec08 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:unsupported_claims:1, retrieval:target_not_found:1 |
| mitdl_lec08 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec09 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec09 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | retrieval:target_not_found:2 |
| mitdl_lec09 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec10 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec10 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:expected_citation_miss:1, retrieval:target_not_found:1 |
| mitdl_lec10 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec11 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec11 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | grounding:insufficient_evidence:1, retrieval:target_not_found:1 |
| mitdl_lec11 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec12 | evidence_unit_candidate | 0.5 | 0.5 | 0 | 0 | no_object_visual_evidence:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec12 | evidence_unit_verified | 0.5 | 0.5 | 0 | 0 | no_object_visual_evidence:1, ocr_engine_without_verified_link:1 | grounding:expected_citation_miss:1, retrieval:target_not_found:1 |
| mitdl_lec12 | evidence_unit_quality_rerank | 0.5 | 0.5 | 0 | 0 | candidate_visual_support_without_verified_link:1, no_object_visual_evidence:1 | - |
| mitdl_lec13 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec13 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | ok:grounded_expected_citation:1, retrieval:target_not_found:1 |
| mitdl_lec13 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec14 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec14 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:insufficient_evidence:1, retrieval:target_not_found:1 |
| mitdl_lec14 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec15 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec15 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | grounding:expected_citation_miss:1, retrieval:target_not_found:1 |
| mitdl_lec15 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec16 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec16 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:insufficient_evidence:2 |
| mitdl_lec16 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec17 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec17 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | grounding:unsupported_claims:1, retrieval:target_not_found:1 |
| mitdl_lec17 | evidence_unit_quality_rerank | 0.5 | 0.5 | 0 | 0 | no_object_visual_evidence:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec18 | evidence_unit_candidate | 0.5 | 0.5 | 0 | 0 | no_object_visual_evidence:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec18 | evidence_unit_verified | 0.5 | 0.5 | 0 | 0 | no_object_visual_evidence:1, ocr_engine_without_verified_link:1 | ok:grounded_expected_citation:1, retrieval:target_not_found:1 |
| mitdl_lec18 | evidence_unit_quality_rerank | 0.5 | 0.5 | 0 | 0 | candidate_visual_support_without_verified_link:1, no_object_visual_evidence:1 | - |
| mitdl_lec19 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec19 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:insufficient_evidence:1, ok:grounded_expected_citation:1 |
| mitdl_lec19 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec20 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec20 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | grounding:unsupported_claims:1, retrieval:target_not_found:1 |
| mitdl_lec20 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec21 | evidence_unit_candidate | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | - |
| mitdl_lec21 | evidence_unit_verified | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:1, ocr_engine_without_verified_link:1 | ok:grounded_expected_citation:1, retrieval:target_not_found:1 |
| mitdl_lec21 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec23 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec23 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:insufficient_evidence:1, ok:grounded_expected_citation:1 |
| mitdl_lec23 | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |
| mitdl_lec24 | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_lec24 | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | ok:grounded_expected_citation:1, retrieval:target_not_found:1 |
| mitdl_lec24 | evidence_unit_quality_rerank | 0.5 | 0.5 | 0 | 0 | no_object_visual_evidence:1, ocr_engine_without_verified_link:1 | - |
| mitdl_review | evidence_unit_candidate | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | - |
| mitdl_review | evidence_unit_verified | 1 | 1 | 0 | 0 | ocr_engine_without_verified_link:2 | grounding:insufficient_evidence:1, retrieval:target_not_found:1 |
| mitdl_review | evidence_unit_quality_rerank | 1 | 1 | 0 | 0 | candidate_visual_support_without_verified_link:2 | - |

## Anti-Overfit View

- `mit_deep_learning_stt_paper_matrix`: suites=24, queries=48, mean_hit_at_10s=0.4282, mean_mrr=0.3758

## 48-Query Lexical Delta Summary

This aggregate uses 24 suites x 2 queries = 48 queries. All required variants completed with queried_count=48 and skipped_count=0.

| variant | queried | skipped | Hit@10s | Hit@10s delta vs lexical | MRR | MRR delta vs lexical | latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| segment_lexical | 48 | 0 | 0.145833 | 0.000000 | 0.166667 | 0.000000 | 23.93 |
| domain_lexicon | 48 | 0 | 0.229167 | 0.083334 | 0.207292 | 0.040625 | 32.14 |
| hybrid | 48 | 0 | 0.875000 | 0.729167 | 0.733338 | 0.566671 | 34.11 |
| window | 48 | 0 | 0.166667 | 0.020834 | 0.178821 | 0.012154 | 23.03 |
| window_hybrid | 48 | 0 | 0.687500 | 0.541667 | 0.631946 | 0.465279 | 36.91 |
| rerank | 48 | 0 | 0.166667 | 0.020834 | 0.182292 | 0.015625 | 38.80 |
| evidence_unit_candidate | 48 | 0 | 0.520833 | 0.375000 | 0.408687 | 0.242020 | 32.04 |
| evidence_unit_verified | 48 | 0 | 0.541667 | 0.395834 | 0.409729 | 0.243062 | 23.01 |
| evidence_unit_quality_rerank | 48 | 0 | 0.520833 | 0.375000 | 0.463200 | 0.296533 | 40.00 |

Best Hit@10s variant: `hybrid` with Hit@10s 0.875000 (delta +0.729167 vs `segment_lexical`) and MRR 0.733338 (delta +0.566671 vs `segment_lexical`).
