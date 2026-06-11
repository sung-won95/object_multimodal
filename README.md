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

When verifying from multiple git worktrees, pass a unique Compose project name
to keep Docker resources separate:

```bash
docker compose -p oarag_issue_82 up -d meilisearch
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

Current paper-claim status is tracked in
`reports/paper/current_claim_status.md`. Treat that file as the boundary between
implementation-readiness claims, smoke/fallback claims, and held paper-performance
claims.

Run the public-safe RAG readiness suite:

```bash
python -m pytest tests/test_rag_readiness_smoke.py -q -rs
```

The suite always checks `query-project` candidate/evidence plumbing with a
synthetic local fixture. It also checks `ask-project` when that command exists
on the current branch, and reports a clear skip when Meilisearch is unavailable
for the optional live smoke.

Run the public synthetic Meilisearch semantic retrieval smoke:

```bash
docker compose up -d meilisearch

PYTHONPATH=src python -m oarag index-project \
  --project-dir tests/fixtures/public_lecture_semantic_project \
  --index public_lecture_semantic_fixture \
  --reset

PYTHONPATH=src python -m oarag query-project \
  --project-dir tests/fixtures/public_lecture_semantic_project \
  --index public_lecture_semantic_fixture \
  --query "slope information lower loss" \
  --limit 2 \
  --neighbor-count 0
```

The same path is covered by `python -m pytest tests/test_meili_semantic_smoke.py -q -rs`.
Set `OARAG_MEILI_URL` or `OARAG_MEILI_API_KEY` when using a non-default local server.
If Meilisearch is not running, the pytest smoke reports a clear skip instead of a failure.
For concurrent worktrees, keep the Compose project name unique; host port
`7700` is still shared, so only one default-port Meilisearch instance can run
at a time.

For hybrid retrieval with a Meilisearch `userProvided` embedder, enable the local
vector store feature before indexing and provide provider-backed vector manifests
for indexed documents and queries. Document indexing fails closed without
`--vector-manifest`; use `--allow-local-hash-vectors` only for dependency-free
smoke fixtures. If a query also uses `--visual-index`, index visual entities
with the same embedder profile as the primary segment/window index:

```bash
curl -X PATCH 'http://127.0.0.1:7700/experimental-features/' \
  -H 'Authorization: Bearer dev-master-key' \
  -H 'Content-Type: application/json' \
  --data-binary '{"vectorStore": true}'

PYTHONPATH=src python -m oarag index-project \
  --project-dir tests/fixtures/public_lecture_semantic_project \
  --index public_lecture_semantic_fixture \
  --reset \
  --hybrid-embedder-profile manual_user_provided_v1 \
  --hybrid-embedder-dimensions 384 \
  --allow-local-hash-vectors

PYTHONPATH=src python -m oarag index-project-visual-entities \
  --project-dir tests/fixtures/public_lecture_semantic_project \
  --index public_lecture_semantic_visual_fixture \
  --reset \
  --hybrid-embedder-profile manual_user_provided_v1 \
  --hybrid-embedder-dimensions 384 \
  --allow-local-hash-vectors

PYTHONPATH=src python -m oarag query-project \
  --project-dir tests/fixtures/public_lecture_semantic_project \
  --index public_lecture_semantic_fixture \
  --query "slope information lower loss" \
  --limit 2 \
  --neighbor-count 0 \
  --hybrid-retrieval \
  --hybrid-query-vector-embedder default \
  --hybrid-query-vector-dimensions 384
```

When `--allow-local-hash-vectors` or only `--hybrid-query-vector-dimensions` is
supplied, OARAG creates deterministic hash vectors with `local_hash_v1`. This
path is a dependency-free local reproducibility and smoke-test fallback only. It
is recorded in metadata with `purpose: local_reproducibility_smoke_fallback` and
`quality_claim: none`; do not cite it as semantic retrieval quality evidence.

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
      "suite_id": "local_sample",
      "type": "local_project",
      "domain": "local_pilot",
      "project_id": "sample_lecture",
      "queries": "reports/pilot_smoke/pilot_queries.csv",
      "index": "sample_lecture_segments",
      "video_id": "sample_lecture",
      "limit": 3
    }
  ]
}
```

The command writes `metrics.json`, `query_results.jsonl`, and `summary.md`. Reports include per-domain metrics so retrieval changes can be checked for overfitting instead of only improving one pilot video.
For public-safe object-aligned retrieval ablations, use a `retrieval_ablation` suite with `modes` such as `transcript-only`, `visual-only`, `time-aligned`, and `object-aligned`; the public outputs omit raw queries, transcript excerpts, visual labels, local paths, and raw candidate IDs.

Check the public fixture retrieval regression gate against aggregate metrics:

```bash
PYTHONPATH=src python -m oarag check-retrieval-gate \
  --metrics reports/perf_runs/public_fixture/metrics.json \
  --config tests/fixtures/public_retrieval_ablation_project/retrieval_quality_gate.json
```

The gate thresholds are only a public fixture regression guard, not paper claim thresholds. The payload is aggregate-only and excludes raw query, transcript, and evidence text.

For paper experiment quality artifacts, prefer the end-to-end bundle command:

```bash
PYTHONPATH=src python -m oarag run-paper-bundle \
  --manifest benchmarks/retrieval_benchmark.json \
  --output-dir reports/paper/run_001 \
  --gate-config tests/fixtures/public_retrieval_ablation_project/retrieval_quality_gate.json \
  --baseline-variant-id segment_lexical
```

The bundle writes the experiment manifest, quality gate result, metric intervals,
paper readiness audit, claim matrix, artifact registry, and bundle result in one
private-safe flow. The registry connects gate, readiness, claims, and robustness
statuses by artifact filename. See `reports/paper/current_claim_status.md` before
turning any bundle output into paper prose, and
`reports/paper/private_safe_runbook.md` for the manual fallback flow.

## Private-Safe Lecture Smoke

Run a manifest-driven smoke suite for private/local lectures:

```bash
PYTHONPATH=src python -m oarag lecture-smoke \
  --manifest reports/lecture_smoke/manifest.json \
  --output-dir reports/lecture_smoke/run_001
```

Minimal manifest shape:

```json
{
  "run_id": "run_001",
  "suites": [
    {
      "suite_id": "local_safe_suite",
      "type": "lecture_project",
      "project_id": "local_project_id",
      "index": "local_project_segments",
      "visual_index": "local_project_visual_entities",
      "limit": 3,
      "queries": [
        {"query_id": "q001", "query_text": "safe placeholder query"}
      ]
    }
  ]
}
```

Public outputs are `metrics.json`, `query_results.jsonl`, and `summary.md`. These files keep only IDs, hashes, counts, booleans, source labels, timestamp availability, latency, frame-backed/linked-entity/graph counts, transcript-only fallback, and semantic-source-field availability. They must not contain raw query text, transcript excerpts, local absolute paths, frame paths, visual entity text, or full retrieval responses.

Raw/private output is opt-in only:

```bash
PYTHONPATH=src python -m oarag lecture-smoke \
  --manifest reports/lecture_smoke/manifest.json \
  --output-dir reports/lecture_smoke/run_001 \
  --allow-private-output \
  --private-output-dir /tmp/lecture_smoke_private_run_001
```

Keep private outputs outside public PRs, docs, and tracked fixtures.

## Local Video Ingest

For a local video with a sibling `.srt` file:

```bash
PYTHONPATH=src python -m oarag ingest-video \
  --video "../data/local_lectures/sample_lecture.mp4" \
  --project-id sample_lecture \
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
- `--frame-selection scene-change` keeps frames when adjacent visual signals cross
  the scene-change distance threshold.
- `--max-frame-gap-seconds` adds optional temporal-gap policy timestamps before
  applying the frame cap.
- `--max-frames 0` disables the cap and samples at `--frame-rate` through the whole video.
- `project_manifest.json` records `frame_sampling`, including selected frame count,
  timestamp span, temporal coverage ratio, max temporal gap, and coverage warnings.
- `project_manifest.json` also records `frame_selection`, including dropped-frame reasons,
  quality signals, and the estimated OCR cost/recall tradeoff.

Ingest an entire local lecture folder recursively:

```bash
PYTHONPATH=src python -m oarag batch-ingest \
  --root "../data/local_lectures" \
  --project-prefix local \
  --frame-rate 0.5 \
  --frame-sampling uniform \
  --max-frames 120 \
  --summary-json artifacts/local_batch_summary.json \
  --summary-jsonl artifacts/local_batch_results.jsonl \
  --summary-csv artifacts/local_batch_results.csv
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
  --video "../data/local_lectures/sample_lecture.mp4" \
  --project-id sample_lecture_stt \
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
  --project-id sample_lecture \
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
  --project-id sample_lecture \
  --backend vlm-first
```

Defaults:

- Reads `artifacts/projects/{project_id}/manifests/frames_manifest.jsonl`
- Writes `artifacts/projects/{project_id}/manifests/visual_entities.jsonl`
- Updates `artifacts/projects/{project_id}/manifests/project_manifest.json` with
  - `artifacts.visual_entities`
  - `counts.visual_entities`
  - `visual_entity_extraction` summary

Backends:

- `--backend vlm-first` is the default. It reads VLM observations first, then
  structured VLM parser JSONL, then falls back to local OCR only when no VLM
  artifact is available.
- `--backend auto` is an alias for the same VLM-first fallback order.
- `--backend stub` always emits zero entities (safe empty test fallback).
- `--backend local-ocr` uses local `tesseract` command and emits `ocr_text` entities. This is the OCR-only baseline/fallback.
- `--backend vlm-jsonl` loads deterministic, precomputed VLM/MLLM parser output from JSONL. It does not call an external model.
- `--backend vlm-observations` converts `run-vlm` output from
  `manifests/vlm_visual_observations.jsonl` into the shared entity schema.

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
  --project-id sample_lecture \
  --backend vlm-jsonl \
  --vlm-jsonl manifests/vlm_parser_output.jsonl
```

`VisualEntity` records can now carry VLM-first metadata:

- `visual_description`
- `position`
- `relations`
- `parser_version`
- `source_model`
- `confidence`
- `source`

Visual entity output is filtered before it is written:

- low-confidence text below `0.40` is dropped
- punctuation-only text is dropped
- single-character text is kept only when confidence is high enough
- exact duplicate text at the same frame/bounding box is removed
- `visual_entity_extraction` records raw, dropped, final, and reason-by-reason counts

Optional local OCR language hint:

```bash
PYTHONPATH=src python -m oarag extract-visual-entities \
  --project-id sample_lecture \
  --backend local-ocr \
  --ocr-language eng
```

## VLM Alignment Pipeline

Run the end-to-end VLM smoke path after frame alignment:

```bash
PYTHONPATH=src python -m oarag run-vlm-alignment \
  --project-id sample_lecture \
  --vlm-backend deterministic \
  --vlm-model stub-vlm \
  --max-vlm-frames 24 \
  --candidate-max-per-segment 2 \
  --resume
```

The deterministic/mock backends are safe defaults for CI and smoke reports. They emit
stable visual observations without calling an external model. The `command` backend
shells out to a configured parser, and the `jsonl` backend replays fixture
observations from `--vlm-options jsonl_path=...`.

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
  --project-id sample_lecture \
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
  --project-id sample_lecture \
  --vlm-backend command \
  --vlm-model local-vlm \
  --vlm-options '{"command":["vlm-adapter","--frame","{frame_path}"]}'
```

## Entity Linking

Link transcript segments to nearby visual entities:

```bash
PYTHONPATH=src python -m oarag link-entities \
  --project-id sample_lecture
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

### Strict deterministic link verification

Existing candidate `entity_links.jsonl` artifacts can be post-processed without
rerunning ingest or visual extraction:

```bash
PYTHONPATH=src python -m oarag verify-entity-links-strict \
  --project-id sample_lecture \
  --entity-links manifests/entity_links.jsonl \
  --output manifests/entity_links.strict.jsonl \
  --report reports/strict_deterministic_entity_link_verifier.json \
  --lexical-overlap-threshold 0.5 \
  --sweep-threshold 0.25 \
  --sweep-threshold 0.5 \
  --sweep-threshold 0.75
```

The rule promotes only candidate links that satisfy all of these checks:

- The link is not already verified and is not timestamp-fallback/time-overlap-only.
- Transcript terms exactly match terms from visual text fields: `visible_text`,
  `detected_text`, or `text`.
- The lexical overlap score, defined as matched visual-text terms divided by
  total visual-text terms, is greater than the configured threshold.

Promoted links receive `alignment_status="verified"`,
`verification_status="verified"`, `verified_link_source="strict_deterministic_rule"`,
and `verification_source="strict_deterministic_rule"`. The duplicated source
field is intentional: downstream evidence-unit and benchmark consumers aggregate
verified sources from `verification_source`, `verified_source`, `verified_by`,
`verifier`, `source`, and `reason_metadata.*`.

The verifier records a public-safe reason block under
`reason_metadata.strict_deterministic_verifier`. It stores counts, score,
threshold, source fields, and the rule name, but not raw transcript or visual
text terms. The optional report is also public-safe: it contains promotion
counts, reason buckets, and threshold-sweep counts by hashed lecture reference.
Use the verifier output as the `--entity-links` input when rebuilding evidence
units.

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
  --project-id sample_lecture \
  --domain-lexicon domain_lexicon.json
```

The same project lexicon is used by `query-project` for query expansion when present:

```bash
PYTHONPATH=src python -m oarag query-project \
  --project-id sample_lecture \
  --index sample_lecture_segments \
  --query "alias-heavy question" \
  --domain-lexicon domain_lexicon.json
```

Project manifests and query/benchmark outputs record sanitized lexicon metadata (`enabled`, `source_path`, and counts), but do not copy the alias terms into metadata. To compare lexicon on/off with the retrieval benchmark, run matching `local_project` suites or runs: one without `domain_lexicon.json`, and one with the project file or a suite field such as `"domain_lexicon": "domain_lexicon.json"`. The benchmark summary marks the suite lexicon state, while `metrics.json` keeps the source path for reproducibility.

## Index Local Project Segments

Index one local project artifact folder into Meilisearch:

```bash
PYTHONPATH=src python -m oarag index-project \
  --project-id sample_lecture \
  --index sample_lecture_segments \
  --settings-profile lecture_segments_default_v1 \
  --batch-size 500 \
  --reset
```

You can also index by explicit path:

```bash
PYTHONPATH=src python -m oarag index-project \
  --project-dir artifacts/projects/sample_lecture \
  --index sample_lecture_segments
```

Segment artifact selection order:

- `segments/lecture_segments_aligned.jsonl` (preferred when present)
- `segments/lecture_segments.jsonl` (fallback)

Use `--segments` to override the default segment file path.

## Semantic Graph Workflow

Neo4j stores the semantic graph used for temporal/reference expansion. Start it
next to Meilisearch for graph ingest and graph query smoke runs:

```bash
docker compose up -d meilisearch neo4j

PYTHONPATH=src python -m oarag health
PYTHONPATH=src python -m oarag health-neo4j
```

Local development defaults:

- Neo4j Browser: `http://127.0.0.1:7474`
- Bolt URI: `bolt://127.0.0.1:7687`
- User/password: `neo4j` / `dev-password`
- Database: `neo4j`

Override those with `OARAG_NEO4J_URI`, `OARAG_NEO4J_USER`,
`OARAG_NEO4J_PASSWORD`, and `OARAG_NEO4J_DATABASE` when using a non-local
runtime.

### Meilisearch vs Neo4j

- Meilisearch is the lexical/vector-like retrieval entry point for transcript
  segments. `index-project`, `query`, and `query-project` use it to find the
  best candidate segment IDs for a natural-language question.
- Neo4j is the semantic graph expansion layer. `graph-ingest` writes project,
  segment, frame, visual entity, concept, and reference-resolution edges.
  `graph-query` first retrieves Meilisearch candidates, then follows graph edges
  such as previous segments, concept mentions, visual entities, frames, and
  resolved references.
- Keep both stores in sync for a project: re-run `index-project` when segment
  artifacts change, and re-run `graph-ingest` when aligned segments, visual
  entities, entity links, or domain lexicon artifacts change.

### Reproducible Smoke Sequence

Use generic local paths in docs, PRs, and reports. Do not paste private source
paths, transcript excerpts, or raw query text from private lectures.

```bash
# 1. Create raw project artifacts from a local video and sibling SRT.
PYTHONPATH=src python -m oarag ingest-video \
  --video "../data/local_lectures/sample_lecture.mp4" \
  --project-id sample_lecture \
  --frame-rate 0.5 \
  --frame-sampling uniform \
  --max-frames 120

# 2. Add frame references to transcript segments.
PYTHONPATH=src python -m oarag align-frames \
  --project-id sample_lecture \
  --margin-seconds 0.5

# 3. Produce optional visual/link artifacts for richer graph edges.
PYTHONPATH=src python -m oarag extract-visual-entities \
  --project-id sample_lecture \
  --backend auto

PYTHONPATH=src python -m oarag link-entities \
  --project-id sample_lecture

# 4. Index the selected segment artifact into Meilisearch.
PYTHONPATH=src python -m oarag index-project \
  --project-id sample_lecture \
  --index sample_lecture_segments \
  --settings-profile lecture_segments_default_v1 \
  --reset

# 5. Ingest the same project artifacts into Neo4j.
PYTHONPATH=src python -m oarag graph-ingest \
  --project-id sample_lecture
```

For a plan-only graph check that does not connect to Neo4j, add `--dry-run`:

```bash
PYTHONPATH=src python -m oarag graph-ingest \
  --project-id sample_lecture \
  --dry-run
```

The graph ingest summary reports node/relationship counts, skipped relationships,
missing optional artifacts, and executed statement counts. Missing optional
artifacts such as `visual_entities` or `entity_links` are acceptable for a
minimal smoke run; missing segments means the project has not been ingested yet.

### Graph Query Smoke Scenario

Use `graph-query` for questions that depend on earlier context or pronouns such
as "아까 말했던 개념" / "the concept mentioned earlier":

```bash
PYTHONPATH=src python -m oarag graph-query \
  --project-id sample_lecture \
  --index sample_lecture_segments \
  --query "아까 말했던 개념이 왜 여기서 다시 필요한가요?" \
  --limit 3 \
  --graph-lookback-segments 3 \
  --graph-limit 12 \
  --output artifacts/projects/sample_lecture/graph_query_smoke.json
```

Expected smoke signals:

- `counts.graph_evidence` is greater than `0` when Meilisearch finds candidates
  and Neo4j has related temporal/reference paths.
- `graph_availability.neo4j.status` is `queried` when Neo4j was reached.
- `traversal_summary.hint_detection.has_graph_hint` is `true` for reference
  phrases such as `아까`, `말했던`, `previous`, or `mentioned`.
- If the query has no temporal/reference phrase,
  `graph_availability.graph_evidence.status` is
  `skipped_no_temporal_or_reference_hint`; use `query-project` for ordinary
  segment retrieval.

### Neo4j Failure Mode

When Neo4j is stopped or unreachable, graph ingest exits non-zero with an
unavailable-runtime error:

```text
error: Neo4j is unavailable: ...
```

`health-neo4j` gives the direct local runtime hint, including
`docker compose up neo4j`, when the default local endpoint cannot be reached.

`graph-query` still returns the Meilisearch-backed response, but
`graph_availability.neo4j.status` becomes `unavailable` and
`graph_availability.graph_evidence.status` becomes
`skipped_neo4j_unavailable`.

Fix:

```bash
docker compose up -d neo4j
PYTHONPATH=src python -m oarag health-neo4j
PYTHONPATH=src python -m oarag graph-ingest --project-id sample_lecture
```

Useful verification commands for README-only changes:

```bash
python3 -m pytest
PYTHONPATH=src python3 -m oarag graph-ingest --help
PYTHONPATH=src python3 -m oarag graph-query --help
```

### Meilisearch Settings Policy

The default `lecture_segments` settings profile is domain-agnostic:

- Search prioritizes `semantic_text`, then transcript text, normalized text, mention candidates, and low-priority video/slide identifiers.
- `semantic_source_fields` records which segment fields were folded into `semantic_text`; future visual entity text should append to `semantic_text` rather than replacing transcript text.
- Filter/sort fields cover stable project, dataset, video, sample, and timestamp metadata.
- Displayed attributes are explicit so accidental extra fields are not returned by default.
- Core settings keep `synonyms` empty and `stopWords` empty. Domain-specific aliases belong in optional project lexicons, not the shared Meilisearch profile.
- Typo tolerance remains enabled, but is conservative for short tokens and disabled on identifier attributes to avoid surprising multilingual transcript matches.

`index-project` prints `settings_profile`, `settings_hash`, and a `settings_snapshot` in its JSON summary. Keep those values with benchmark artifacts so retrieval changes can be compared or rolled back with `--settings-profile lecture_segments_default_v1` or `--settings-profile lecture_segments_legacy_v0` if a cross-domain regression appears.

When validating settings changes with `benchmark-retrieval`, report aggregate `metrics.json` / `summary.md` numbers by domain and latency only. Do not paste private query text, transcript excerpts, frame paths, or local project paths into public PRs.

## Evidence Windows

Build a transcript and frame evidence bundle around one or more retrieved segment IDs:

```bash
PYTHONPATH=src python -m oarag evidence-window \
  --project-id sample_lecture \
  --segment-id seg_sample_000003 \
  --query "what concept was introduced before this step" \
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
  --project-id sample_lecture \
  --index sample_lecture_segments \
  --query "what concept was introduced before this step" \
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
