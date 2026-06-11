# Benchmark Transition Plan v1

## Scope

This public-safe transition plan reframes the `full_provider_matrix_v2` no-go result as a benchmark and evaluation-protocol paper draft. The result should not be presented as a successful performance-improvement paper: the primary citation-precision lift from `evidence_unit_candidate` to `evidence_unit_verified` was positive but not statistically decisive under the locked paired-bootstrap criterion.

Recommended paper direction:

> We introduce a public-safe evaluation protocol for object-aligned multimodal retrieval-augmented generation, using a locked 48-query, 24-suite matrix to compare retrieval and answer-grounding variants, report uncertainty, and characterize failures in object-level citation grounding.

This direction is consistent with the no-go decision because it treats the verifier and matrix outputs as benchmark instrumentation and baseline evidence, not as proof that verified object alignment significantly improves answer quality.

## Decision Basis

- Decision: no-go for a performance-improvement paper; continue as a benchmark/evaluation-protocol paper.
- Matrix completion: 24 retrieval-answer suites, 9 variants per suite, 432 query-result rows, skipped variant count 0.
- Semantic live smoke: provider-backed query vectors present; local hash query vectors absent.
- Quality gate: failed on `mitdl_lec05` / `window_hybrid` for `hit_at_10s`, `mrr_at_max_delta`, and `expected_citation_hit_ratio`.
- Candidate-to-verified paired bootstrap: `citation_precision` delta +0.020835 with 95% CI [0.000000, 0.048612], so the primary go criterion was not met.
- Secondary metrics moved in the desired direction but are insufficient for a performance-improvement claim: `unsupported_claim_ratio` delta -0.020833 and `hit_at_10s` delta +0.020833.

## Claim Matrix Draft

| Claim ID | Performance-paper claim to avoid | Benchmark-paper replacement claim | Evidence role | Status |
| --- | --- | --- | --- | --- |
| `protocol_matrix_complete` | The method improves retrieval-answer performance across the benchmark. | The locked matrix executes a complete 24-suite x 9-variant evaluation protocol with no skipped variants. | Coverage and reproducibility evidence. | Supported as protocol evidence. |
| `verified_vs_candidate_delta` | Verified object evidence significantly improves citation precision over candidate evidence. | Candidate and verified evidence-unit variants are comparable baselines whose paired deltas quantify uncertainty and failure modes. | Baseline comparison and uncertainty evidence. | Supported for benchmarking; not a performance win claim. |
| `quality_gate_interpretation` | The system is paper-ready for aggregate performance claims. | The failed regression gate identifies a benchmark stress case and motivates failure-taxonomy reporting. | Negative result and stress-case evidence. | Supported as limitation/stress-test evidence. |
| `object_alignment_protocol` | Object-aligned verification solves grounding. | Object-aligned verification defines an auditable protocol for separating candidate visual support, verified links, and citation grounding. | Protocol definition and annotation evidence. | Draft; requires human audit follow-up. |
| `public_safe_release` | The paper can release full examples and evidence traces. | The benchmark report can release aggregate metrics, artifact names, schema metadata, reason-code counts, and exclusion rules without private payload text. | Public-safe reporting evidence. | Supported by privacy statements; keep enforced. |
| `human_vlm_limitation` | Deterministic verifier outputs are final human/VLM-validated labels. | Deterministic verifier outputs are first-pass benchmark labels; #281 human audit and real VLM smoke remain required before stronger annotation-quality claims. | Risk and limitation. | Blocked until #281 completes. |

## Reusable Artifact Inventory

| Artifact | Reuse role | Use in benchmark paper | Public-safe handling |
| --- | --- | --- | --- |
| `go_no_go_report.md` | Decision anchor | Cite the no-go decision, fixed criteria, quality-gate failure, and benchmark-transition rationale. | Use aggregate metrics and decision text only. |
| `go_no_go_report.json` | Machine-readable decision anchor | Preserve reason codes, thresholds, bootstrap settings, and follow-up issue link. | Use structured status fields only. |
| `paper_claims/claim_evidence_matrix.json` | Claim rewrite source | Convert blocked performance claims into benchmark/protocol claims. | Use claim IDs, statuses, blocker codes, and artifact names only. |
| `paper_readiness/paper_readiness_audit.json` | Readiness and gap source | Show matrix coverage, semantic smoke evidence, privacy coverage, and quality-gate gap. | Use pass/fail checks and counts only. |
| `paper_artifact_registry.json` | Registry source | List reusable artifact names and current statuses. | Use filenames and status metadata only. |
| `paper_metric_intervals/paper_metric_intervals.json` | Baseline table source | Report variant-level aggregate means and uncertainty protocol. | Use aggregate statistics only. |
| `quality_gate_result.json` | Stress-case source | Report failed metrics as benchmark stress cases. | Use suite ID, variant ID, metric names, values, and thresholds only. |
| `metrics.json` and `metrics_summary.csv` | Failure taxonomy source | Summarize reason-code counts and variant aggregate metrics. | Do not include raw query, answer, transcript, evidence, or path payloads. |
| `semantic_smoke.json` | Provider-backed smoke source | Support the claim that live semantic query vectors were used. | Use provider/local vector counts only. |
| `paper_report/reproducibility.json` | Reproducibility source | Document commands, dataset descriptor counts, and privacy policy. | Use metadata and privacy coverage only. |

## Baseline Table Summary

The table below is an aggregate benchmark table, not a performance-win claim. Each variant has 48 query-level observations.

| Variant | Query count | Hit@10s | MRR | Grounded answer | Expected citation hit | Unsupported claim count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `segment_lexical` | 48 | 0.145833 | 0.166667 | 0.125000 | 0.083333 | 0.062500 |
| `domain_lexicon` | 48 | 0.229167 | 0.207292 | 0.291667 | 0.125000 | 0.250000 |
| `hybrid` | 48 | 0.875000 | 0.733333 | 0.791667 | 0.562500 | 0.291667 |
| `window` | 48 | 0.166667 | 0.178819 | 0.104167 | 0.166667 | 0.083333 |
| `window_hybrid` | 48 | 0.666667 | 0.614931 | 0.291667 | 0.520833 | 0.104167 |
| `rerank` | 48 | 0.166667 | 0.182292 | 0.125000 | 0.083333 | 0.062500 |
| `evidence_unit_candidate` | 48 | 0.520833 | 0.408681 | 0.479167 | 0.458333 | 0.291667 |
| `evidence_unit_verified` | 48 | 0.541667 | 0.409722 | 0.458333 | 0.479167 | 0.270833 |
| `evidence_unit_quality_rerank` | 48 | 0.520833 | 0.463194 | 0.583333 | 0.458333 | 0.229167 |

## Failure Taxonomy Summary

Aggregate reason-code counts across the matrix:

| Taxonomy group | Reason code | Count | Benchmark interpretation |
| --- | --- | ---: | --- |
| Answer failure | `retrieval_miss` | 291 | Retrieval misses remain the dominant failure mode. |
| Answer failure | `grounded_expected_citation` | 81 | A subset reaches expected citation grounding. |
| Answer failure | `insufficient_evidence` | 45 | Retrieved context sometimes cannot support an answer. |
| Answer failure | `unsupported_claims` | 15 | Answer generation can still overstate evidence. |
| Stage failure | `retrieval:target_not_found` | 256 | Most failures originate before answer synthesis. |
| Stage failure | `grounding:expected_citation_miss` | 35 | Some candidates are retrieved but miss expected citation grounding. |
| Stage failure | `grounding:insufficient_evidence` | 45 | Grounding fails when available evidence is too weak. |
| Stage failure | `grounding:unsupported_claims` | 15 | Unsupported claims remain a separate answer-level risk. |

This taxonomy suggests a benchmark paper can make a stronger contribution by specifying where object-aligned RAG fails than by claiming the current verifier is a decisive performance improvement.

## Risks and Limitations

- #281 human audit and real VLM smoke are incomplete. Until they finish, deterministic verifier labels should be described as first-pass protocol labels rather than final human/VLM-validated annotations.
- The expanded query set from the follow-up data work was not available for the no-go decision; this plan uses the locked 48-query matrix.
- The regression gate failure must remain visible. It should be framed as a stress-case finding and readiness blocker, not quietly averaged away.
- Candidate-to-verified deltas are compatible with a benchmark baseline comparison, but not with a statistically significant improvement claim under the fixed criterion.
- Public-safe reporting must exclude raw query content, answer content, transcript content, candidate evidence content, and local filesystem paths.

## Public-Safe Release Rules

Allowed:

- Aggregate metrics, confidence intervals, reason-code counts, suite IDs, variant IDs, artifact filenames, schema versions, pass/fail statuses, and threshold metadata.

Excluded:

- Raw query content, answer content, transcript content, candidate evidence content, local filesystem paths, private example payloads, and any copied evidence snippets.

## Next Paper Tasks

1. Convert the claim matrix above into a paper outline with benchmark/protocol claims only.
2. Add a compact table in the paper draft showing the 9-variant aggregate baseline.
3. Add a failure-taxonomy section centered on retrieval miss, insufficient evidence, and unsupported claim behavior.
4. Link #281 completion to annotation-quality claims and keep those claims blocked until audit evidence exists.
5. Keep the public-safe scanner in the release checklist before PR or manuscript export.
