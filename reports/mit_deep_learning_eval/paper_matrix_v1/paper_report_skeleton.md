# MIT Deep Learning Paper Matrix v1

Status: skeleton only. Replace this file with generated aggregate outputs after a
full privacy-checked `run-paper-bundle` execution.

## Retrieval-answer Variant Matrix

| baseline | compared variants | suite count | status |
| --- | --- | ---: | --- |
| `segment_lexical` | `domain_lexicon`, `hybrid`, `window`, `window_hybrid`, `rerank`, `evidence_unit_candidate`, `evidence_unit_verified`, `evidence_unit_quality_rerank` | 24 | pending full run |

## Dynamic Concept Graph Matrix

| baseline | compared variants | implementation issues | status |
| --- | --- | --- | --- |
| `meili_only` | `graph_only`, `meili_graph`, `graph_aware_rerank` | #213, #214, #217, #219, #220, #221, #222 | pending public-safe cross-lecture smoke |

## Artifact Status

| artifact | expected path | status |
| --- | --- | --- |
| metrics | `metrics.json` | pending |
| summary | `summary.md` | pending |
| readiness | `paper_readiness/paper_readiness_audit.json` | pending |
| claims | `paper_claims/claim_evidence_matrix.json` | pending |
| registry | `paper_artifact_registry.json` | pending |

## Public-Safe Notes

This skeleton intentionally contains no raw query text, answer text, transcript content,
candidate evidence text, private eval values, absolute local paths, or raw private
identifiers.

Dynamic concept graph result tables may expose only `query_id`, hashed refs, source
types, ranks, counts, recall buckets, rerank deltas, and skip reasons. Timestamp-only
overlap is candidate/fallback evidence only and must not be counted as verified
object alignment.
