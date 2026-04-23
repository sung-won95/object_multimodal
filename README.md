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

Ingest an entire local lecture folder recursively:

```bash
PYTHONPATH=src python -m oarag batch-ingest \
  --root "../data/업스윙 포커" \
  --project-prefix upswing \
  --frame-rate 0.5 \
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

## Visual Entity Extraction (OCR-first)

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
- `--backend local-ocr` uses local `tesseract` command and emits `ocr_text` entities.
- `--backend auto` uses `local-ocr` when `tesseract` is available, otherwise `stub`.

Optional local OCR language hint:

```bash
PYTHONPATH=src python -m oarag extract-visual-entities \
  --project-id upswing_matrices \
  --backend local-ocr \
  --ocr-language eng
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
- Adds `lexical_match` evidence when transcript terms overlap OCR text.
- Adds `mention_candidate` evidence when transcript mention hooks such as `this`, `board`, `stack`, or `bet size` appear in the OCR text.

## Index Local Project Segments

Index one local project artifact folder into Meilisearch:

```bash
PYTHONPATH=src python -m oarag index-project \
  --project-id upswing_matrices \
  --index upswing_matrices_segments \
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

## Notes

This first baseline is a pipeline smoke test, not the final object-aligned evaluation. EDUVIDQA provides transcript/timestamp data but not visual entity or entity-link labels. Local video pilots will add `visual_entities` and weak `entity_links` next.
