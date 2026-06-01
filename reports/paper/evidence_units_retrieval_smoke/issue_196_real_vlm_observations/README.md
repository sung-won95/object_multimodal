# Issue 196 Real VLM Observation Artifact Blocker

Date: 2026-06-01
Runner: VLM backend/config preflight plus candidate-frame dry-run

## Scope

This public-safe report covers the same two-lecture MIT evidence-unit smoke slice
checked in Issue 194 / PR 195. The goal was to create real VLM visual observation
artifacts for sampled/candidate frames, then rerun VLM-first visual entity extraction
and evidence-unit smoke if a real backend or real structured VLM JSONL input was
available.

No raw queries, transcripts, local paths, frame images, raw frame IDs, raw segment IDs,
raw entity IDs, raw evidence-unit IDs, or raw VLM responses are included here.
Timestamp-only overlap remains candidate/fallback evidence only and is not counted as
verified object alignment.

## Result

Real VLM artifact generation is blocked by missing real backend configuration or real
structured VLM input. No mock or deterministic VLM output was generated or presented as
paper-quality object evidence.

| Metric | Value |
| --- | ---: |
| suites checked | 2 |
| sampled frames checked | 72 |
| suites with existing `vlm_visual_observations.jsonl` | 0 |
| suites with existing `vlm_parser_output.jsonl` | 0 |
| existing visual entities checked | 885 |
| existing OCR visual entities | 885 |
| existing VLM visual entities | 0 |
| existing visual-description entities | 0 |
| existing detected-text entities | 0 |
| verified-like links | 0 |
| real VLM artifacts generated | 0 |
| VLM-first extraction rerun | 0 |
| evidence-unit smoke rerun | 0 |

Coverage did not improve over PR 195. The remaining blocker is a configured real VLM
command backend or a user-provided real structured VLM parser JSONL for the same two
lecture projects.

## Backend Availability Checked

| Backend path | Status | Action |
| --- | --- | --- |
| `command` | Available in code, but no real command contract/credential was configured for this run. | Not executed. |
| `jsonl` / `vlm-jsonl` | Available in code, but no real JSONL backend input or parser artifact was present for either suite. | Not executed. |
| `vlm-observations` | Extraction path is available, but source `vlm_visual_observations.jsonl` is missing in both suites. | Not executed. |
| `vlm-first` | Would fall back to OCR on this slice because VLM artifacts are absent. | Not used as VLM coverage evidence. |
| `deterministic` / `mock` | Available dependency-free test backends. | Intentionally not used for paper-quality evidence. |

Only one generic API CLI was found on PATH, but no repo-level adapter command and no
relevant VLM/API credential environment variable name was present in the current
process. Since the repository requires the `command` backend to be explicitly supplied
through `--vlm-options`, there was no safe real VLM command to execute.

## Candidate-Frame Dry Run

A dry-run with the real-backend command shape was used only to plan bounded VLM work.
It did not execute the command backend and did not write artifacts.

| Suite | sampled frames | planned candidate frames | candidate reduction ratio | segments | segments with frame coverage | segments with candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | 35 | 12 | 0.6571 | 1023 | 51 | 15 |
| mit_lec04 | 37 | 12 | 0.6757 | 1451 | 62 | 21 |

The planned cap is deliberately small: at most 12 candidate frames per lecture, one
candidate per segment, and four candidates per 60-second window. Recall risk is high,
but this is appropriate for a bounded real-backend smoke before scaling.

## Required Command Contract

The next run needs one of the following real inputs.

### Option A: Real command backend

Run once per private project directory:

```bash
PYTHONPATH=src python3 -m oarag run-vlm-alignment \
  --project-dir <private-project-dir> \
  --vlm-backend command \
  --vlm-model <real-vlm-model-id> \
  --vlm-options '{"command":["<vlm-adapter>","--frame","{frame_path}"],"input_mode":"json-stdin","prompt_template_version":"vlm-visual-parser-v1"}' \
  --max-vlm-frames 12 \
  --candidate-max-per-segment 1 \
  --candidate-max-per-window 4 \
  --resume
```

When `input_mode` is `json-stdin`, the adapter receives a `vlm-command-request-v1`
JSON object on stdin. The adapter must print JSON to stdout. The simplest accepted
shape is:

```json
{
  "observations": [
    {
      "observation_type": "diagram",
      "visual_description": "Public-safe description of the visible visual object or state.",
      "detected_text": "Optional detected slide text",
      "confidence": 0.87,
      "position": {"region": "center"},
      "relations": [{"type": "contains", "target": "label"}],
      "parser_version": "real-vlm-parser-v1"
    }
  ]
}
```

Each successful observation must include `visual_description` or `detected_text`.
Confidence, if present, must be between 0 and 1.

### Option B: Real structured parser JSONL

If a real VLM parser has already produced `manifests/vlm_parser_output.jsonl`, run:

```bash
PYTHONPATH=src python3 -m oarag extract-visual-entities \
  --project-dir <private-project-dir> \
  --backend vlm-jsonl \
  --vlm-jsonl manifests/vlm_parser_output.jsonl
```

### Follow-up after real VLM input exists

After either real VLM observation or parser artifacts exist for both suites:

```bash
PYTHONPATH=src python3 -m oarag extract-visual-entities \
  --project-dir <private-project-dir> \
  --backend vlm-observations

PYTHONPATH=src python3 -m oarag evidence-unit-smoke \
  --manifest <tmp-private-manifest> \
  --output-dir reports/paper/evidence_units_retrieval_smoke/issue_196_real_vlm_observations \
  --quality-rerank
```

The evidence-unit report should not be interpreted as VLM-improved until
`units_with_vlm_entity`, `units_with_visual_description`, or `units_with_detected_text`
becomes nonzero.

## Why Mock Output Was Not Used

The deterministic and mock backends are useful for CI and parser contract tests, but
they synthesize visual descriptions from frame metadata. Using them here would inflate
VLM coverage without real visual observation, contaminate the paper evidence trail, and
violate the issue constraint that deterministic/mock VLM output must not be presented
as real object evidence.

## Checks Run

```bash
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["run-vlm", "--help"])'
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["run-vlm-alignment", "--help"])'
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["extract-visual-entities", "--help"])'
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["evidence-unit-smoke", "--help"])'
python3 -m pytest tests/test_vlm.py tests/test_vlm_alignment_pipeline.py tests/test_visual_entities.py tests/test_evidence_units.py tests/test_evidence_unit_smoke.py -q
```

Focused tests passed: `42 passed`.
