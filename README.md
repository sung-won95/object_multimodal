# Object-Aligned RAG Pilot

Python-first implementation scaffold for the Meilisearch/object-aligned lecture video QA pilot.

Current goal:

- Index EDUVIDQA normalized JSONL as `lecture_segments`.
- Ingest local lecture videos into a reproducible artifact folder.
- Query Meilisearch for transcript evidence candidates.
- Evaluate timestamp proximity with Hit@5s/10s/15s.
- Keep schemas open for `visual_entities`, `entity_links`, and `evidence_windows`.

## Quick Start

Start Meilisearch:

```bash
docker compose up -d meilisearch
```

Check health:

```bash
PYTHONPATH=src python -m oarag health
```

Index the EDUVIDQA test split:

```bash
PYTHONPATH=src python -m oarag index-eduvidqa \
  --input ../data/normalized_links/mathsc_timestamp_test.jsonl \
  --index eduvidqa_mathsc_test_segments \
  --reset
```

Run one query:

```bash
PYTHONPATH=src python -m oarag query \
  --index eduvidqa_mathsc_test_segments \
  --query "why is N the number of molecules" \
  --limit 3
```

Run timestamp baseline evaluation:

```bash
PYTHONPATH=src python -m oarag eval-eduvidqa \
  --input ../data/normalized_links/mathsc_timestamp_test.jsonl \
  --index eduvidqa_mathsc_test_segments \
  --output artifacts/eduvidqa_test_eval.jsonl
```

## Development

Run tests:

```bash
python -m pytest
```

The pytest configuration adds `src` to the test import path, so `PYTHONPATH=src` is not required for tests.

## Retrieval Benchmark

Run cross-domain retrieval benchmarks from a JSON manifest:

```bash
PYTHONPATH=src python -m oarag benchmark-retrieval \
  --manifest benchmarks/retrieval_benchmark.json
```

Manifest shape:

```json
{
  "run_id": "baseline_001",
  "output_dir": "reports/perf_runs/baseline_001",
  "deltas": [5, 10, 15],
  "suites": [
    {
      "suite_id": "eduvidqa_test",
      "type": "eduvidqa",
      "domain": "public_eduvidqa",
      "input": "../data/normalized_links/mathsc_timestamp_test.jsonl",
      "index": "eduvidqa_mathsc_test_segments",
      "limit": 5
    },
    {
      "suite_id": "pilot_matrices",
      "type": "local_project",
      "domain": "local_pilot",
      "project_id": "pilot_issue8_matrices_safe",
      "queries": "reports/pilot_smoke/pilot_queries.csv",
      "index": "pilot_issue8_matrices_segments_safe",
      "video_id": "2_Matrices",
      "limit": 3
    }
  ]
}
```

The command writes `metrics.json`, `query_results.jsonl`, and `summary.md`. Reports include per-domain metrics so retrieval changes can be checked for overfitting instead of only improving one pilot video.

## Local Video Ingest

For a local video with a sibling `.srt` file:

```bash
PYTHONPATH=src python -m oarag ingest-video \
  --video "../data/업스윙 포커/2. 매트릭스(V)/2. Matrices.mp4" \
  --project-id upswing_matrices \
  --frame-rate 1 \
  --max-frames 120
```

The command writes:

- `artifacts/projects/{project_id}/source/`
- `artifacts/projects/{project_id}/frames/`
- `artifacts/projects/{project_id}/segments/lecture_segments.jsonl`
- `artifacts/projects/{project_id}/manifests/project_manifest.json`
- `artifacts/projects/{project_id}/manifests/frames_manifest.jsonl`

By default the source video is symlinked, not copied. Use `--copy-source` only when you really want a duplicate video file.

Frame cap behavior:

- `--frame-sampling uniform` is the default and spreads capped frames across the full video duration.
- `--frame-sampling prefix` preserves the older front-loaded smoke-test behavior.
- `--frame-selection none` is the default and keeps all sampled frames.
- `--frame-selection representative` post-processes sampled frames with domain-agnostic
  image-diff and low-information checks to reduce near-duplicate or blank frames before OCR.
- `--max-frames 0` disables the cap and samples at `--frame-rate` through the whole video.
- `project_manifest.json` records `frame_sampling`, including selected frame count, timestamp span, and temporal coverage ratio.
- `project_manifest.json` also records `frame_selection`, including dropped-frame reasons,
  quality signals, and the estimated OCR cost/recall tradeoff.

Ingest an entire local lecture folder recursively:

```bash
PYTHONPATH=src python -m oarag batch-ingest \
  --root "../data/업스윙 포커" \
  --project-prefix upswing \
  --frame-rate 0.5 \
  --frame-sampling uniform \
  --max-frames 120 \
  --summary-json artifacts/upswing_batch_summary.json \
  --summary-jsonl artifacts/upswing_batch_results.jsonl \
  --summary-csv artifacts/upswing_batch_results.csv
```

Behavior:

- Discovers supported video files under `--root` (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`).
- Generates stable `project_id` values from each video's relative path (plus a deterministic hash suffix).
- Reuses the same transcript/STT options as `ingest-video`, including `OARAG_STT_LANGUAGE` defaults.
- Resumes by default: if `manifests/project_manifest.json` already exists for that generated `project_id`, the video is skipped.
- Use `--force` to re-ingest already ingested videos.
- Per-video failures are recorded and the batch continues by default; use `--strict` to stop at the first failure.
- Prints a full JSON summary to stdout, with optional JSON/JSONL/CSV files via summary flags.

Transcript source behavior:

- `--transcript-source auto` uses a sibling `.srt` file when present, otherwise runs local STT with `mlx-whisper`.
- `--transcript-source srt` requires `--srt` or a sibling `.srt` file.
- `--transcript-source stt` forces STT even when an `.srt` exists.
- `--transcript-source none` creates frame/source artifacts without transcript segments.

Install local Apple Silicon STT support:

```bash
python -m pip install ".[stt]"
```

Run STT segment generation on an M-series Mac:

```bash
export OARAG_STT_LANGUAGE=en

PYTHONPATH=src python -m oarag ingest-video \
  --video "../data/업스윙 포커/2. 매트릭스(V)/2. Matrices.mp4" \
  --project-id upswing_matrices_stt \
  --transcript-source stt \
  --stt-model mlx-community/whisper-large-v3-mlx \
  --frame-rate 0.2 \
  --max-frames 0
```

You can also put the language in a local `.env` file:

```bash
OARAG_STT_LANGUAGE=en
```

The default STT model is `mlx-community/whisper-large-v3-mlx`, which is slower than Turbo but worked better on the pilot lecture. Use `--stt-model mlx-community/whisper-large-v3-turbo` when speed matters more than transcript quality. Set `OARAG_STT_LANGUAGE=en` for English lectures or `OARAG_STT_LANGUAGE=ko` for Korean lectures; `--stt-language` overrides `.env` and the environment variable for a single command. If neither is set, Whisper auto-detects the language. Do not force `ko` for English audio, because Whisper will decode Korean-looking text. The first STT run downloads the MLX Whisper model into the Hugging Face cache. STT artifacts are written to:

- `artifacts/projects/{project_id}/audio/`
- `artifacts/projects/{project_id}/transcripts/mlx_whisper_raw.json`
- `artifacts/projects/{project_id}/segments/lecture_segments.jsonl`

## Segment-to-Frame Alignment

Attach frame references to each transcript segment using timestamp overlap:

```bash
PYTHONPATH=src python -m oarag align-frames \
  --project-id upswing_matrices \
  --margin-seconds 0.5
```

Defaults:

- Reads `artifacts/projects/{project_id}/segments/lecture_segments.jsonl`
- Reads `artifacts/projects/{project_id}/manifests/frames_manifest.jsonl`
- Writes `artifacts/projects/{project_id}/segments/lecture_segments_aligned.jsonl`
- Updates `artifacts/projects/{project_id}/manifests/project_manifest.json` with
  - `artifacts.lecture_segments_aligned`
  - `alignment` summary counts
  - `counts.lecture_segments_aligned`

## Visual Entity Extraction

Extract frame-level visual entities from sampled frames:

```bash
PYTHONPATH=src python -m oarag extract-visual-entities \
  --project-id upswing_matrices \
  --backend auto
```

Defaults:

- Reads `artifacts/projects/{project_id}/manifests/frames_manifest.jsonl`
- Writes `artifacts/projects/{project_id}/manifests/visual_entities.jsonl`
- Updates `artifacts/projects/{project_id}/manifests/project_manifest.json` with
  - `artifacts.visual_entities`
  - `counts.visual_entities`
  - `visual_entity_extraction` summary

Backends:

- `--backend stub` always emits zero entities (safe test/default fallback).
- `--backend local-ocr` uses local `tesseract` command and emits `ocr_text` entities. This is the OCR-only baseline/fallback.
- `--backend vlm-jsonl` loads deterministic, precomputed VLM/MLLM parser output from JSONL. It does not call an external model.
- `--backend auto` uses `local-ocr` when `tesseract` is available, otherwise `stub`.

`vlm-jsonl` expects one JSON object per frame with a `frame_id` and an `entities` array. The parser output is validated against the frame manifest and converted into the shared `VisualEntity` schema:

```json
{
  "frame_id": "frame_000001",
  "parser_version": "vlm-jsonl-v1",
  "source_model": "stub-vlm",
  "entities": [
    {
      "visual_description": "A blue matrix diagram with one highlighted row",
      "entity_type": "diagram",
      "confidence": 0.93,
      "position": {"region": "center", "x": 0.5, "y": 0.45},
      "relations": [{"type": "points_to", "target": "row_label"}]
    }
  ]
}
```

Run it with an explicit JSONL path:

```bash
PYTHONPATH=src python -m oarag extract-visual-entities \
  --project-id upswing_matrices \
  --backend vlm-jsonl \
  --vlm-jsonl manifests/vlm_parser_output.jsonl
```

`VisualEntity` records can now carry VLM-first metadata:

- `visual_description`
- `position`
- `relations`
- `parser_version`
- `source_model`

Visual entity output is filtered before it is written:

- low-confidence text below `0.40` is dropped
- punctuation-only text is dropped
- single-character text is kept only when confidence is high enough
- exact duplicate text at the same frame/bounding box is removed
- `visual_entity_extraction` records raw, dropped, final, and reason-by-reason counts

Optional local OCR language hint:

```bash
PYTHONPATH=src python -m oarag extract-visual-entities \
  --project-id upswing_matrices \
  --backend local-ocr \
  --ocr-language eng
```

## VLM Alignment Pipeline

Run the end-to-end VLM smoke path after frame alignment:

```bash
PYTHONPATH=src python -m oarag run-vlm-alignment \
  --project-id upswing_matrices \
  --vlm-backend deterministic \
  --vlm-model stub-vlm \
  --max-vlm-frames 24 \
  --candidate-max-per-segment 2 \
  --resume
```

The deterministic backend is the safe default for CI and smoke reports. It emits stable
visual observations without calling an external model, then runs audio-visual
consistency checks over the aligned transcript windows.

Defaults:

- Reads `artifacts/projects/{project_id}/manifests/frames_manifest.jsonl`
- Reads `artifacts/projects/{project_id}/segments/lecture_segments_aligned.jsonl`
- Writes `artifacts/projects/{project_id}/manifests/vlm_frame_candidates.jsonl`
- Writes `artifacts/projects/{project_id}/manifests/vlm_visual_observations.jsonl`
- Writes `artifacts/projects/{project_id}/manifests/audio_visual_consistency.jsonl`
- Updates `artifacts/projects/{project_id}/manifests/project_manifest.json`

Use `--dry-run` to plan the VLM candidate count and output paths without writing
artifacts:

```bash
PYTHONPATH=src python -m oarag run-vlm-alignment \
  --project-id upswing_matrices \
  --vlm-model stub-vlm \
  --max-vlm-frames 24 \
  --dry-run
```

The command prints a JSON summary with a compact `smoke_metrics` block for reporting:

- candidate frame reduction ratio
- selected candidate, processed, failed, and resumed frame counts
- average visual-observation frame latency when frames were processed
- audio-visual consistency status and label distributions

For a local VLM adapter, use the `command` backend and pass a generic command template
through `--vlm-options`:

```bash
PYTHONPATH=src python -m oarag run-vlm-alignment \
  --project-id upswing_matrices \
  --vlm-backend command \
  --vlm-model local-vlm \
  --vlm-options '{"command":["vlm-adapter","--frame","{frame_path}"]}'
```

## Entity Linking

Link transcript segments to nearby visual entities:

```bash
PYTHONPATH=src python -m oarag link-entities \
  --project-id upswing_matrices
```

Defaults:

- Reads aligned segments when `segments/lecture_segments_aligned.jsonl` exists, otherwise falls back to `segments/lecture_segments.jsonl`
- Reads `artifacts/projects/{project_id}/manifests/visual_entities.jsonl`
- Writes `artifacts/projects/{project_id}/manifests/entity_links.jsonl`
- Updates `artifacts/projects/{project_id}/manifests/project_manifest.json` with
  - `artifacts.entity_links`
  - `counts.entity_links`
  - `entity_linking` summary

Current linking behavior:

- Requires segment/entity time overlap (or matching `frame_refs`) to create a weak link.
- Adds `lexical_match` evidence when normalized transcript terms overlap OCR text.
- Keeps domain aliases off by default. Core linking only uses direct normalized term overlap unless a project lexicon is configured.
- Adds `mention_candidate` evidence when transcript mention hooks or configured aliases match visual text.
- Records evidence type counts in `entity_linking.evidence_type_counts`.

Optional project domain lexicon:

```json
{
  "aliases": {
    "canonical_term": ["alias one", "alias two"]
  }
}
```

Save the file as `artifacts/projects/{project_id}/domain_lexicon.json` to enable it for that project. You can also pass an explicit path:

```bash
PYTHONPATH=src python -m oarag link-entities \
  --project-id upswing_matrices \
  --domain-lexicon domain_lexicon.json
```

The same project lexicon is used by `query-project` for query expansion when present:

```bash
PYTHONPATH=src python -m oarag query-project \
  --project-id upswing_matrices \
  --index upswing_matrices_segments \
  --query "alias-heavy question" \
  --domain-lexicon domain_lexicon.json
```

Project manifests and query/benchmark outputs record sanitized lexicon metadata (`enabled`, `source_path`, and counts), but do not copy the alias terms into metadata. To compare lexicon on/off with the retrieval benchmark, run matching `local_project` suites or runs: one without `domain_lexicon.json`, and one with the project file or a suite field such as `"domain_lexicon": "domain_lexicon.json"`. The benchmark summary marks the suite lexicon state, while `metrics.json` keeps the source path for reproducibility.

## Index Local Project Segments

Index one local project artifact folder into Meilisearch:

```bash
PYTHONPATH=src python -m oarag index-project \
  --project-id upswing_matrices \
  --index upswing_matrices_segments \
  --settings-profile lecture_segments_default_v1 \
  --batch-size 500 \
  --reset
```

You can also index by explicit path:

```bash
PYTHONPATH=src python -m oarag index-project \
  --project-dir artifacts/projects/upswing_matrices \
  --index upswing_matrices_segments
```

Segment artifact selection order:

- `segments/lecture_segments_aligned.jsonl` (preferred when present)
- `segments/lecture_segments.jsonl` (fallback)

Use `--segments` to override the default segment file path.

### Meilisearch Settings Policy

The default `lecture_segments` settings profile is domain-agnostic:

- Search prioritizes transcript text, normalized text, mention candidates, and low-priority video/slide identifiers.
- Filter/sort fields cover stable project, dataset, video, sample, and timestamp metadata.
- Displayed attributes are explicit so accidental extra fields are not returned by default.
- Core settings keep `synonyms` empty and `stopWords` empty. Domain-specific aliases belong in optional project lexicons, not the shared Meilisearch profile.
- Typo tolerance remains enabled, but is conservative for short tokens and disabled on identifier attributes to avoid surprising multilingual transcript matches.

`index-project` prints `settings_profile`, `settings_hash`, and a `settings_snapshot` in its JSON summary. Keep those values with benchmark artifacts so retrieval changes can be compared or rolled back with `--settings-profile lecture_segments_legacy_v0` if a cross-domain regression appears.

When validating settings changes with `benchmark-retrieval`, report aggregate `metrics.json` / `summary.md` numbers by domain and latency only. Do not paste private query text, transcript excerpts, frame paths, or local project paths into public PRs.

## Evidence Windows

Build a transcript and frame evidence bundle around one or more retrieved segment IDs:

```bash
PYTHONPATH=src python -m oarag evidence-window \
  --project-id upswing_matrices \
  --segment-id seg_12._Betsize_000003 \
  --query "what affects bet size" \
  --neighbor-count 1
```

Defaults:

- Reads `segments/lecture_segments_aligned.jsonl` when present
- Falls back to `segments/lecture_segments.jsonl`
- Reads `manifests/frames_manifest.jsonl`
- Includes the target segment, neighboring transcript context, and frame metadata for `frame_refs`

Use `--window-seconds` instead of `--neighbor-count` to select timestamp-overlapping context around the target segment.

## Multimodal Project Query

Search a local project index and assemble transcript, frame, visual, and link evidence for each top hit:

```bash
PYTHONPATH=src python -m oarag query-project \
  --project-id upswing_matrices \
  --index upswing_matrices_segments \
  --query "what affects bet size" \
  --limit 3 \
  --neighbor-count 1
```

Behavior:

- Searches the provided Meilisearch index and converts each hit into a local multimodal evidence bundle.
- Loads transcript context and frame metadata from the project artifact folder.
- Loads `visual_entities.jsonl` and `entity_links.jsonl` when present, but still works when those artifacts do not exist yet.
- Prints short human-readable summary lines to stderr and a JSON bundle to stdout.
- Keeps resolved artifact paths in the JSON output for reproducibility and later figure/report generation.

Optional:

- Use `--window-seconds` instead of `--neighbor-count` for timestamp-based context windows.
- Use `--output` to save the assembled JSON bundle inside or outside the project directory.

## Notes

This first baseline is a pipeline smoke test, not the final object-aligned evaluation. EDUVIDQA provides transcript/timestamp data but not visual entity or entity-link labels. Local video pilots will add `visual_entities` and weak `entity_links` next.
