# Current Paper Claim Status

Verified on May 29, 2026 against `develop` commit `f1bc5c4`.

This file separates claims that are currently supported by reproducible code and
public-safe artifacts from claims that still need a fresh private/local experiment
run. Do not move a held claim into paper prose until the supporting bundle artifacts
exist and pass the privacy checklist in `reports/paper/private_safe_runbook.md`.

## Verification Snapshot

| Check | Result | Interpretation |
| --- | --- | --- |
| `python3 -m pytest -q` | 341 passed, 4 skipped | Local test baseline is currently green; skipped tests are environment-dependent. |
| `python3 -m oarag --help` | OK | Paper bundle, readiness, claim, VLM, entity-link, and benchmark commands are present. |
| `python3 -m pytest tests/test_entity_links.py tests/test_domain_lexicon.py tests/test_graph_document.py -q` | 19 passed | Stronger link evidence is covered by public fixtures. |
| `python3 -m pytest tests/test_vlm.py tests/test_visual_entities.py tests/test_vlm_alignment_pipeline.py tests/test_cli.py -q` | 76 passed | VLM-first parser contracts and OCR fallback policy are covered by public fixtures. |
| `python3 -m pytest tests/test_meili.py tests/test_project_index.py tests/test_project_query.py tests/test_benchmark.py -q` | 75 passed, 1 skipped | Vector metadata, local fallback warnings, and benchmark reporting are covered by public fixtures. |

## Supported Claims

| Claim | Current support | Boundary |
| --- | --- | --- |
| The repository has a reproducible CLI path for ingest, frame alignment, VLM-first visual entity extraction, entity linking, indexing, query, benchmark, and paper bundle artifact generation. | CLI help and unit tests cover these commands. | This is an implementation-readiness claim, not a final performance claim. |
| `local_hash_v1` is treated as a deterministic smoke fallback, not semantic quality evidence. | Query/index/benchmark outputs record `purpose: local_reproducibility_smoke_fallback` and `quality_claim: none`. | Do not cite local hash results as embedding performance. |
| OCR is an explicit baseline/fallback behind the VLM-first visual entity path. | `extract-visual-entities` defaults to VLM-first fallback order and documents `local-ocr` as baseline/fallback. | OCR-heavy pilot outputs should be described as baseline evidence only. |
| Entity linking can produce evidence beyond timestamp proximity. | Public fixtures cover lexical, mention, semantic hint, domain lexicon, reference cue, visual description, position, and relation evidence. | Aggregate timestamp-only reduction still needs a regenerated private/local run. |
| Paper bundle artifacts are designed to be public-safe. | The runbook and artifact schemas exclude raw queries, transcripts, local paths, private eval values, raw vectors, and full answer text. | Generated artifacts still need manual privacy inspection before commit or publication. |

## Held Claims

| Held claim | Required evidence before use |
| --- | --- |
| Hybrid retrieval improves semantic quality on MIT Deep Learning lectures. | A provider-backed document/query vector run with source model metadata, plus passing registry/readiness/claim artifacts. |
| VLM-first parsing improves object-aligned grounding over OCR on the target lecture set. | Fresh VLM parser artifacts and a comparison against OCR baseline/fallback outputs. |
| Frame coverage fixes materially improve multimodal recall on the full MIT suite. | Regenerated ingest/alignment summaries showing coverage and temporal-gap metrics across the suite. |
| Timestamp-only entity-link dependence is reduced at aggregate MIT scale. | Regenerated entity link summaries reporting evidence type distributions before and after the change. |
| The MIT paper matrix is ready for final paper tables. | A completed `run-paper-bundle` output whose quality gate, readiness audit, claim matrix, robustness intervals, registry, and privacy review support the intended claim. |

## Required Limitations In Paper Prose

- `local_hash_v1` is a dependency-free smoke fallback and carries no semantic quality claim.
- OCR-only visual entities are baseline/fallback artifacts, not the target VLM-first method.
- Existing seed labels and public fixtures are regression guards, not human-validated final evaluation labels.
- Historical pilot outputs may include low frame coverage and high timestamp-only link ratios until regenerated with the current frame sampling and entity-link code.
- Public docs and PRs must not include raw private video paths, raw transcripts, raw query text, raw vectors, or full answer/evidence text.
