# Issue 194 VLM Visual Coverage Preflight

Date: 2026-06-01
Runner: preflight only, no VLM-first rebuild

## Scope

This public-safe preflight checked the same two-lecture MIT evidence-unit smoke slice
used by the previous retrieval smokes. The goal was to determine whether real VLM
visual observations or parser output were already available for a VLM-first
`visual_entities` rebuild.

No raw queries, transcripts, local paths, frame images, raw segment IDs, raw entity
IDs, or raw evidence-unit IDs are included here. Timestamp-only overlap remains
candidate/fallback evidence only and is not counted as verified object alignment.

## Result

Real VLM input was not available for either checked lecture project, so the smoke did
not rebuild visual entities or rerun evidence-unit retrieval. Deterministic/mock VLM
output was not used and is not represented as object-level evidence.

| Metric | Value |
| --- | ---: |
| suites checked | 2 |
| suites with `vlm_visual_observations.jsonl` | 0 |
| suites with `vlm_parser_output.jsonl` | 0 |
| suites whose project manifest records a VLM artifact key | 0 |
| suites whose current visual entity backend is local OCR | 2 |
| current visual entities checked | 885 |
| current visual entities from OCR | 885 |
| current visual entities from VLM source/model | 0 |
| current visual entities with `visual_description` | 0 |
| current visual entities with `detected_text` | 0 |
| current verified-like links | 0 |

## Per-Lecture Preflight

| Suite | project ref | frames | visual entities | VLM observations file | VLM parser file | current backend | OCR entities | VLM entities | visual-description entities | detected-text entities | entity links | verified-like links |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | `a5b8cce4a4e4` | 35 | 499 | missing | missing | local-ocr | 499 | 0 | 0 | 0 | 710 | 0 |
| mit_lec04 | `ddbcbead3467` | 37 | 386 | missing | missing | local-ocr | 386 | 0 | 0 | 0 | 800 | 0 |

Entity-link preflight showed only time-overlap-derived link types in the checked
artifact slice:

| Suite | time overlap | time overlap + lexical match | time overlap + lexical match + mention candidate | time overlap + mention candidate |
| --- | ---: | ---: | ---: | ---: |
| mit_lec01 | 668 | 42 | 0 | 0 |
| mit_lec04 | 752 | 45 | 2 | 1 |

## Checked Artifact Types

- `manifests/vlm_visual_observations.jsonl`: missing in both suites.
- `manifests/vlm_parser_output.jsonl`: missing in both suites.
- `manifests/visual_entities.jsonl`: present in both suites, but OCR-only.
- `manifests/entity_links.jsonl`: present in both suites, but no verified-like link status/type.
- `manifests/frames_manifest.jsonl`: present in both suites.
- `manifests/project_manifest.json`: present in both suites, but records local OCR visual entity extraction and no VLM artifact key.

## Why Retrieval Smoke Was Not Re-Run

The requested VLM-first path requires real VLM observations or a real structured VLM
parser JSONL. In this slice, `extract-visual-entities --backend vlm-first` would fall
back to OCR, which would not address the #193 bottleneck and could make the report
look like a VLM coverage smoke when no VLM evidence was actually present.

## Storage Coverage Compared With #193

Coverage did not improve in this PR. The blocker is upstream artifact availability:
the checked two-lecture projects do not contain real VLM visual observations or real
VLM parser output that can be converted into first-class visual entities.

## Minimal Follow-Up

Generate real `manifests/vlm_visual_observations.jsonl` or
`manifests/vlm_parser_output.jsonl` for the same two lecture projects, preferably on
the sampled-frame or candidate-frame subset, then rerun:

```bash
PYTHONPATH=src python3 -m oarag extract-visual-entities \
  --project-dir <private-project-dir> \
  --backend vlm-observations

PYTHONPATH=src python3 -m oarag evidence-unit-smoke \
  --manifest <tmp-private-manifest> \
  --output-dir reports/paper/evidence_units_retrieval_smoke/issue_194_vlm_visual_coverage \
  --quality-rerank
```

The follow-up report should include nonzero `units_with_vlm_entity` or
`units_with_visual_description` before any retrieval-quality interpretation is made.

## Checks Run

```bash
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["extract-visual-entities", "--help"])'
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["run-vlm", "--help"])'
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["run-vlm-alignment", "--help"])'
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["evidence-unit-smoke", "--help"])'
python3 -m pytest tests/test_visual_entities.py tests/test_vlm_alignment_pipeline.py tests/test_evidence_units.py tests/test_evidence_unit_smoke.py -q
```

Focused tests passed: `30 passed`.
