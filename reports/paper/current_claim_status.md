# Current Paper Claim Status

Verified on June 8, 2026 against `develop` commit `3f9d25f` after #222.

This file separates claims that are currently supported by reproducible code and
public-safe artifacts from claims that still need a fresh private/local experiment
run. Do not move a held claim into paper prose until the supporting bundle artifacts
exist and pass the privacy checklist in `reports/paper/private_safe_runbook.md`.

## Paper Claim Draft

Current supported claim:

> We implement an object-aligned evidence-unit retrieval pipeline augmented with a
> dynamic cross-lecture concept graph. The current system can use the graph to guide
> candidate recall, report source-contribution diagnostics for Meili and graph paths,
> apply deterministic graph-aware reranking, and emit public-safe smoke/report artifacts.

Scope boundary:

- This is an implementation and diagnostic-readiness claim, not a final performance
  claim.
- "Graph DB is available" must not be written as "object alignment is verified."
- Timestamp-only overlap is a candidate or fallback signal only. It is never verified
  object alignment.
- Verified object alignment requires explicit verified link evidence and
  `verified_object_alignment.paper_claim_eligible=true` in aggregate artifacts.

## Method Section Outline

1. Evidence-unit construction and object-alignment contract: describe transcript
   windows, visual states/entities, candidate links, verified links, and the rule that
   timestamp-only overlap remains a fallback/candidate signal.
2. Dynamic concept graph construction: extract concept-aware evidence fields (#214),
   ingest concept graph artifacts into the graph runtime (#217), and merge related
   concepts across lectures (#219) under the #213 architecture direction.
3. Dual-source candidate generation: compare Meili candidate retrieval, graph
   traversal candidates, and the combined Meili+Graph pool (#220). Report source
   contribution counts and skip reasons instead of raw private evidence.
4. Graph-aware deterministic reranking: apply the graph-aware rerank path (#221) to
   the combined candidate pool and report rank deltas, source support, and target
   recall buckets without using private text in public outputs.
5. Public-safe smoke/report flow: run the cross-lecture retrieval smoke (#222) and
   the MIT paper bundle skeleton as aggregate-only evidence. Treat smoke results as
   readiness diagnostics until a final privacy-checked run exists.

## Supported Claims

| Claim | Current support | Boundary |
| --- | --- | --- |
| The repository has a reproducible CLI path for ingest, frame alignment, VLM-first visual entity extraction, entity linking, indexing, query, benchmark, and paper bundle artifact generation. | CLI help and unit tests cover these commands. | This is an implementation-readiness claim, not a final performance claim. |
| `local_hash_v1` is treated as a deterministic smoke fallback, not semantic quality evidence. | Query/index/benchmark outputs record `purpose: local_reproducibility_smoke_fallback` and `quality_claim: none`. | Do not cite local hash results as embedding performance. |
| OCR is an explicit baseline/fallback behind the VLM-first visual entity path. | `extract-visual-entities` defaults to VLM-first fallback order and documents `local-ocr` as baseline/fallback. | OCR-heavy pilot outputs should be described as baseline evidence only. |
| Entity linking can produce evidence beyond timestamp proximity. | Public fixtures cover lexical, mention, semantic hint, domain lexicon, reference cue, visual description, position, and relation evidence. | Aggregate timestamp-only reduction still needs a regenerated private/local run. |
| Evidence units carry concept-aware fields for graph construction. | #214 adds concept source fields to the evidence-unit contract. | Concept fields are retrieval and graph signals; they do not by themselves prove verified object alignment. |
| The dynamic concept graph can guide cross-lecture candidate recall. | #217 ingests concept graph artifacts, #219 merges cross-lecture concepts, #220 combines Meili and graph candidates, and #222 adds the cross-lecture smoke report. | Supported as candidate-recall and diagnostics capability only until a privacy-checked paper run records aggregate results. |
| Graph-aware deterministic reranking is available for diagnostics. | #221 adds graph-aware rerank diagnostics and #222 records rerank deltas in the public-safe smoke flow. | Do not claim learned ranking quality or final retrieval gains without aggregate evidence. |
| Paper bundle and smoke artifacts are designed to be public-safe. | The runbook and artifact schemas exclude raw queries, transcripts, local paths, private eval values, raw vectors, and full answer/evidence text. | Generated artifacts still need manual privacy inspection before commit or publication. |

## Held Claims

| Held claim | Required evidence before use |
| --- | --- |
| Hybrid retrieval improves semantic quality on MIT Deep Learning lectures. | A provider-backed document/query vector run with source model metadata, plus passing registry/readiness/claim artifacts. |
| Dynamic concept graph retrieval improves MIT lecture recall or ranking. | A completed public-safe cross-lecture smoke/report showing Meili-only, Graph-only, Meili+Graph, and graph-aware rerank rows with aggregate recall buckets and rank deltas. |
| Graph-aware rerank improves final answer quality. | A completed matrix that ties rerank deltas to answer/citation metrics without leaking raw answer or evidence text. |
| VLM-first parsing improves object-aligned grounding over OCR on the target lecture set. | Fresh VLM parser artifacts and a comparison against OCR baseline/fallback outputs. |
| Frame coverage fixes materially improve multimodal recall on the full MIT suite. | Regenerated ingest/alignment summaries showing coverage and temporal-gap metrics across the suite. |
| Timestamp-only entity-link dependence is reduced at aggregate MIT scale. | Regenerated entity link summaries reporting evidence type distributions before and after the change. |
| The MIT paper matrix is ready for final paper tables. | A completed `run-paper-bundle` output whose quality gate, readiness audit, claim matrix, robustness intervals, registry, and privacy review support the intended claim. |

## Required Limitations In Paper Prose

- `local_hash_v1` is a dependency-free smoke fallback and carries no semantic quality claim.
- OCR-only visual entities are baseline/fallback artifacts, not the target VLM-first method.
- Existing seed labels and public fixtures are regression guards, not human-validated final evaluation labels.
- Historical pilot outputs may include low frame coverage and high timestamp-only link ratios until regenerated with the current frame sampling and entity-link code.
- Public docs and PRs must not include raw private video paths, raw transcripts, raw query text, raw vectors, full answer/evidence text, or private evaluation values.
- Dynamic concept graph results may be described as candidate recall, source contribution diagnostics, and deterministic rerank diagnostics only until final public-safe aggregate evidence exists.
