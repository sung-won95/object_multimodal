# MIT Paper Matrix v1 Runbook

This folder is the public-safe landing zone for the MIT Deep Learning paper bundle.
It is intentionally a skeleton until a full local run produces fresh artifacts.
Until then, use it as a reproducibility plan only. It is not evidence for final
performance, VLM quality, provider-backed hybrid quality, or aggregate entity-link
ratio claims.

## Matrix Contract

This folder tracks two public-safe paper-facing matrices:

- The `run-paper-bundle` retrieval-answer matrix, which remains the final aggregate
  bundle path for answer/citation metrics.
- The `cross-lecture-retrieval-smoke` dynamic concept graph matrix, which is the
  public-safe diagnostic path for Meili, graph, combined candidate recall, and
  graph-aware rerank behavior.

Do not merge these claims in paper prose. The graph matrix can support candidate
recall, source contribution, rerank delta, and skip-reason diagnostics. It does not
prove verified object alignment or final answer quality.

## Retrieval-answer Matrix

The bundle manifest compares these nine variants for every MIT lecture suite:

- `segment_lexical`
- `domain_lexicon`
- `hybrid`
- `window`
- `window_hybrid`
- `rerank`
- `evidence_unit_candidate`
- `evidence_unit_verified`
- `evidence_unit_quality_rerank`

Every suite references the shared `eval/mit_deep_learning_stt/domain_lexicon.json`
through the manifest-relative value `domain_lexicon.json`.
Every suite also declares `evidence_unit_index=mit_deep_learning_stt_evidence_units`
for the evidence-unit variants. If the local evidence-unit index or artifact is not
ready, those variants remain in the matrix with `status=skipped`,
`skipped_count>0`, zero hit/MRR, and a public-safe `skip_reason` instead of
inflating object-alignment metrics.

Issue #203 adds a public-safe object-link diagnostics contract for the generated
`metrics.json` and `query_results.jsonl` files. Matrix rows expose:

- `candidate_visual_support`: aggregate/count-only candidate evidence, including
  frame backing, visual entity count, candidate link count, timestamp fallback count,
  and `candidate_link_signal_counts` for temporal overlap, lexical overlap,
  mention/deictic hook, spatial/position match, visual text overlap,
  VLM/object visual-description overlap, semantic/domain hint, and timestamp fallback.
- `verified_object_alignment`: aggregate/count-only verified alignment evidence,
  including `verified_link_count`, `verified_link_source_counts`, and
  `timestamp_fallback_counted_as_verified=false`.
- `object_link_diagnostics`: the same split packaged with a public note.

Variant metrics also aggregate `candidate_visual_support_ratio`,
`verified_object_alignment_ratio`, `candidate_link_signal_counts`,
`verified_link_source_counts`, `timestamp_fallback_link_ratio`,
`visual_state_coverage_ratio`, `vlm_entity_coverage_ratio`,
`verified_link_coverage_ratio`, `target_found_in_top_k_ratio`, status counts,
and skip reason counts.

Only `verified_object_alignment.paper_claim_eligible=true` supports a verified
object-alignment paper claim. `candidate_visual_support` is useful for retrieval
diagnostics and #204 matrix slices, but timestamp fallback and candidate support do
not count as verified object alignment.

Use the matrix to separate bottlenecks:

- If `target_found_in_top_k_ratio` is high but Hit/MRR or top1 match is low, the
  likely bottleneck is retrieval ranking/reranking.
- If evidence-unit variants show low `candidate_visual_support_ratio`,
  `vlm_entity_coverage_ratio`, or `verified_object_alignment_ratio`, or are mostly
  skipped, the likely bottleneck is evidence coverage/index readiness rather than
  ranking.

## Dynamic Concept Graph Matrix

The dynamic concept graph smoke compares these variants from #222:

| Variant | Candidate source | Implementation issue | Paper-facing diagnostic |
| --- | --- | --- | --- |
| `meili_only` | Meilisearch lexical/semantic candidate retrieval | #220, #222 | Meili candidate recall and Meili skip reasons |
| `graph_only` | Graph traversal over dynamic concept graph evidence | #217, #219, #220, #222 | Graph candidate recall and graph skip reasons |
| `meili_graph` | Combined Meili and graph candidate pool | #213, #214, #217, #219, #220, #222 | Source contribution counts and combined recall buckets |
| `graph_aware_rerank` | Combined pool plus deterministic graph-aware rerank | #221, #222 | Rank deltas and rerank diagnostics |

The graph matrix is connected to the implementation issues as follows:

| Axis | Linked issues | Public-safe evidence |
| --- | --- | --- |
| Architecture direction | #213 | Method scope: dynamic concept graph guided candidate recall, not final performance |
| Concept-aware evidence units | #214 | Counts of concept/evidence-unit coverage, never raw transcript or evidence text |
| Graph ingest | #217 | Aggregate graph availability/status and skip reasons |
| Cross-lecture concept merge | #219 | Hashed concept/evidence refs and source-type counts |
| Dual Meili+Graph candidates | #220 | Source contribution counts, ranks, and recall buckets |
| Graph-aware rerank | #221 | Rerank deltas and deterministic rerank diagnostics |
| Cross-lecture smoke/report | #222 | Public-safe `metrics.json`, `query_results.jsonl`, and `summary.md` |

Public result tables for this matrix may contain only `query_id`, hashed refs,
source types, ranks, counts, recall buckets, rerank deltas, and skip reasons. Raw
query text, answer text, transcript content, candidate evidence text, private eval
values, raw vectors, local absolute paths, and raw private identifiers are excluded.
Timestamp-only overlap remains candidate/fallback evidence and must stay separate
from verified object alignment.

## Prerequisites

Build and index the shared window corpus before running the bundle:

```bash
PYTHONPATH=src python3 scripts/index_mit_deep_learning_windows.py --build-only
PYTHONPATH=src python3 scripts/index_mit_deep_learning_windows.py --index-only --reset
```

For `window_hybrid`, pass the same hybrid embedder profile and dimensions used by
the segment index when required by the local Meilisearch configuration.
For the evidence-unit variants, build and index `segments/evidence_units.jsonl`
into `mit_deep_learning_stt_evidence_units` before a full claim-bearing run.
For the dynamic concept graph matrix, prepare public-safe cross-lecture smoke
manifests with graph artifacts and Meili indexes available, then run:

```bash
PYTHONPATH=src python3 -m oarag cross-lecture-retrieval-smoke \
  --manifest <cross_lecture_smoke_manifest.json> \
  --output-dir <cross_lecture_output_dir>
```

## Full Bundle Command

```bash
PYTHONPATH=src python3 -m oarag run-paper-bundle \
  --manifest eval/mit_deep_learning_stt/paper_bundle_manifest.json \
  --output-dir reports/mit_deep_learning_eval/paper_matrix_v1 \
  --gate-config reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json \
  --baseline-variant-id segment_lexical \
  --sample-count 1000 \
  --seed 139
```

If the local Meilisearch state is still being prepared, run a 1 to 2 suite smoke by
copying the manifest locally and keeping only the target lecture suites. Do not commit
that local smoke manifest or generated outputs unless they pass the privacy checklist.

## Expected Generated Files

`run-paper-bundle` should generate:

- `metrics.json`
- `metrics_summary.csv`
- `query_results.jsonl`
- `summary.md`
- `paper_report/paper_table.md`
- `paper_report/reproducibility.json`
- `quality_gate_result.json`
- `experiment_manifest.json`
- `paper_metric_intervals/paper_metric_intervals.json`
- `paper_readiness/paper_readiness_audit.json`
- `paper_claims/claim_evidence_matrix.json`
- `paper_artifact_registry.json`
- `paper_bundle_result.json`

## Privacy Checklist

Before committing any full-run output, inspect JSON, JSONL, CSV, and Markdown files
for raw query text, answer text, transcript content, candidate evidence text, private
eval values, absolute local paths, and raw private identifiers. Generated public
artifacts should contain aggregate metrics, hashed refs, relative artifact filenames,
stage statuses, and coarse claim/readiness statuses only.

## Next Issue Candidate

If this folder still contains only skeleton files after the indexing fix lands, open a
follow-up issue to execute the full MIT paper bundle locally, replace the completeness
gate with final paper thresholds, and commit only privacy-checked aggregate artifacts.
Before opening that follow-up, check `../../paper/current_claim_status.md` so the run
targets the held claims that are still missing evidence.
