from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oarag.ingestion.alignment import align_segments_to_frames
from oarag.evaluation.benchmark import run_benchmark
from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL, ENV_STT_LANGUAGE, default_paths, env_default
from oarag.ingestion.eduvidqa import iter_lecture_segments, iter_records
from oarag.evaluation.eval import evaluate_query, summarize
from oarag.vision.entity_links import link_entities
from oarag.retrieval.evidence import build_evidence_response
from oarag.graph.graph_ingest import ingest_project_graph
from oarag.graph.graph_query import GraphTraversalConfig, graph_query
from oarag.ingestion.ingest import (
    BatchIngestConfig,
    VideoIngestConfig,
    batch_ingest_videos,
    ingest_video,
    make_video_id,
    write_batch_summary_csv,
    write_batch_summary_json,
    write_batch_summary_jsonl,
)
from oarag.core.io import write_json
from oarag.evaluation.lecture_smoke import run_lecture_smoke
from oarag.integrations.meili import (
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_SETTINGS,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    MeiliClient,
    lecture_segment_settings_profile_names,
    visual_entity_settings_profile_names,
)
from oarag.integrations.neo4j import check_neo4j_health
from oarag.retrieval.project_query import query_project
from oarag.retrieval.project_index import (
    index_project_segments,
    index_project_visual_entities,
    project_dir_from_args,
)
from oarag.core.schemas import SearchCandidate
from oarag.ingestion.stt import DEFAULT_MLX_WHISPER_MODEL
from oarag.vision.visual_entities import extract_visual_entities
from oarag.vision.vlm import DEFAULT_VLM_BACKEND, run_vlm
from oarag.vision.audio_visual_consistency import AudioVisualConsistencyConfig
from oarag.vision.vlm_alignment_pipeline import VLMAlignmentPipelineConfig, run_vlm_alignment_pipeline
from oarag.vision.vlm_frame_candidates import VLMFrameCandidateConfig


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def main_run_vlm_alignment(argv: list[str] | None = None) -> None:
    parser = build_vlm_alignment_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oarag", description="Object-aligned RAG pilot CLI")
    parser.add_argument("--url", default=DEFAULT_MEILI_URL, help="Meilisearch URL")
    parser.add_argument("--api-key", default=DEFAULT_MEILI_API_KEY, help="Meilisearch API key")
    subparsers = parser.add_subparsers(required=True)

    health = subparsers.add_parser("health", help="Check Meilisearch health")
    health.set_defaults(func=cmd_health)

    neo4j_health = subparsers.add_parser("health-neo4j", help="Check Neo4j runtime health")
    neo4j_health.set_defaults(func=cmd_health_neo4j)

    graph_ingest = subparsers.add_parser(
        "graph-ingest",
        help="Ingest a local project graph document into Neo4j",
    )
    location = graph_ingest.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    location.add_argument("--graph-document", type=Path, help="Prebuilt graph document JSON path")
    graph_ingest.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip idempotent Neo4j constraint/index creation.",
    )
    graph_ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize Cypher merge plan without connecting to Neo4j.",
    )
    graph_ingest.set_defaults(func=cmd_graph_ingest)

    index = subparsers.add_parser("index-eduvidqa", help="Index normalized EDUVIDQA JSONL")
    index.add_argument("--input", required=True, type=Path)
    index.add_argument("--index", required=True)
    index.add_argument("--project-id", default="eduvidqa")
    index.add_argument("--limit", type=int)
    index.add_argument("--batch-size", type=int, default=500)
    index.add_argument("--reset", action="store_true")
    index.set_defaults(func=cmd_index_eduvidqa)

    index_project = subparsers.add_parser("index-project", help="Index a local project segment JSONL")
    index_project.add_argument("--index", required=True)
    location = index_project.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    index_project.add_argument(
        "--segments",
        type=Path,
        help="Optional JSONL path. Relative paths are resolved from --project-dir.",
    )
    index_project.add_argument("--batch-size", type=int, default=500)
    index_project.add_argument("--reset", action="store_true")
    index_project.add_argument(
        "--settings-profile",
        choices=lecture_segment_settings_profile_names(),
        default=LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile to apply to lecture_segments.",
    )
    index_project.set_defaults(func=cmd_index_project)

    index_visual_entities = subparsers.add_parser(
        "index-project-visual-entities",
        help="Index a local project visual_entities JSONL",
    )
    index_visual_entities.add_argument("--index", required=True)
    location = index_visual_entities.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    index_visual_entities.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    index_visual_entities.add_argument("--batch-size", type=int, default=500)
    index_visual_entities.add_argument("--reset", action="store_true")
    index_visual_entities.add_argument(
        "--settings-profile",
        choices=visual_entity_settings_profile_names(),
        default=VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile to apply to visual_entities.",
    )
    index_visual_entities.set_defaults(func=cmd_index_project_visual_entities)

    ingest = subparsers.add_parser("ingest-video", help="Ingest a local lecture video")
    ingest.add_argument("--video", required=True, type=Path)
    ingest.add_argument("--project-id", required=True)
    ingest.add_argument("--video-id")
    ingest.add_argument("--srt", type=Path, help="Optional SRT path. Defaults to sibling .srt")
    ingest.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/projects"),
        help="Directory where project artifacts are written",
    )
    ingest.add_argument("--frame-rate", type=float, default=1.0)
    ingest.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Frame cap for smoke tests. Use 0 for no cap.",
    )
    ingest.add_argument(
        "--frame-sampling",
        choices=["uniform", "prefix"],
        default="uniform",
        help="How to apply --max-frames. uniform spreads capped frames across the video.",
    )
    ingest.add_argument(
        "--frame-selection",
        choices=["none", "representative"],
        default="none",
        help="Optional post-sampling selector that removes near-duplicate or low-information frames.",
    )
    ingest.add_argument("--skip-frames", action="store_true")
    ingest.add_argument(
        "--copy-source",
        action="store_true",
        help="Copy source video instead of symlinking it into the artifact folder.",
    )
    ingest.add_argument(
        "--transcript-source",
        choices=["auto", "srt", "stt", "none"],
        default="auto",
        help="Transcript source. auto uses SRT when present, otherwise mlx-whisper STT.",
    )
    ingest.add_argument(
        "--stt-model",
        default=DEFAULT_MLX_WHISPER_MODEL,
        help="MLX Whisper model used when --transcript-source resolves to stt.",
    )
    ingest.add_argument(
        "--stt-language",
        default=env_default(ENV_STT_LANGUAGE),
        help=f"Optional Whisper language hint, for example ko or en. Defaults to ${ENV_STT_LANGUAGE}.",
    )
    ingest.add_argument(
        "--stt-task",
        choices=["transcribe", "translate"],
        default="transcribe",
    )
    ingest.add_argument(
        "--stt-word-timestamps",
        action="store_true",
        help="Ask mlx-whisper for word-level timestamps in the raw transcript artifact.",
    )
    ingest.set_defaults(func=cmd_ingest_video)

    batch_ingest = subparsers.add_parser(
        "batch-ingest",
        aliases=["ingest-folder"],
        help="Ingest all supported video files under a local folder",
    )
    batch_ingest.add_argument("--root", required=True, type=Path, help="Root folder to scan recursively")
    batch_ingest.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/projects"),
        help="Directory where project artifacts are written",
    )
    batch_ingest.add_argument(
        "--project-prefix",
        help="Optional prefix for generated project IDs",
    )
    batch_ingest.add_argument("--frame-rate", type=float, default=1.0)
    batch_ingest.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Frame cap for smoke tests. Use 0 for no cap.",
    )
    batch_ingest.add_argument(
        "--frame-sampling",
        choices=["uniform", "prefix"],
        default="uniform",
        help="How to apply --max-frames. uniform spreads capped frames across each video.",
    )
    batch_ingest.add_argument(
        "--frame-selection",
        choices=["none", "representative"],
        default="none",
        help="Optional post-sampling selector that removes near-duplicate or low-information frames.",
    )
    batch_ingest.add_argument("--skip-frames", action="store_true")
    batch_ingest.add_argument(
        "--copy-source",
        action="store_true",
        help="Copy source video instead of symlinking it into the artifact folder.",
    )
    batch_ingest.add_argument(
        "--transcript-source",
        choices=["auto", "srt", "stt", "none"],
        default="auto",
        help="Transcript source. auto uses SRT when present, otherwise mlx-whisper STT.",
    )
    batch_ingest.add_argument(
        "--stt-model",
        default=DEFAULT_MLX_WHISPER_MODEL,
        help="MLX Whisper model used when --transcript-source resolves to stt.",
    )
    batch_ingest.add_argument(
        "--stt-language",
        default=env_default(ENV_STT_LANGUAGE),
        help=f"Optional Whisper language hint, for example ko or en. Defaults to ${ENV_STT_LANGUAGE}.",
    )
    batch_ingest.add_argument(
        "--stt-task",
        choices=["transcribe", "translate"],
        default="transcribe",
    )
    batch_ingest.add_argument(
        "--stt-word-timestamps",
        action="store_true",
        help="Ask mlx-whisper for word-level timestamps in the raw transcript artifact.",
    )
    batch_ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even when project_manifest.json already exists for a discovered video.",
    )
    batch_ingest.add_argument(
        "--strict",
        action="store_true",
        help="Stop at first per-video failure. Without this flag the batch continues.",
    )
    batch_ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan-only mode that discovers videos and generated IDs without running ingest.",
    )
    batch_ingest.add_argument("--summary-json", type=Path, help="Optional path to write full summary JSON.")
    batch_ingest.add_argument(
        "--summary-jsonl",
        type=Path,
        help="Optional path to write one result row per line.",
    )
    batch_ingest.add_argument("--summary-csv", type=Path, help="Optional path to write summary CSV.")
    batch_ingest.set_defaults(func=cmd_batch_ingest)

    align = subparsers.add_parser(
        "align-frames",
        help="Attach frame_refs to transcript segments using timestamp overlap",
    )
    align.add_argument("--project-id", required=True)
    align.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/projects"),
        help="Directory where project artifacts are written",
    )
    align.add_argument("--segments", type=Path, help="Input lecture_segments JSONL")
    align.add_argument("--frames-manifest", type=Path, help="Input frames_manifest JSONL")
    align.add_argument("--output", type=Path, help="Aligned output JSONL")
    align.add_argument("--manifest", type=Path, help="Project manifest JSON path")
    align.add_argument(
        "--margin-seconds",
        type=float,
        default=0.0,
        help="Expand each segment time window by this margin on both sides.",
    )
    align.set_defaults(func=cmd_align_frames)

    evidence = subparsers.add_parser(
        "evidence-window",
        help="Build transcript and frame evidence windows around retrieved segment IDs",
    )
    location = evidence.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    evidence.add_argument(
        "--segment-id",
        action="append",
        required=True,
        help="Retrieved segment ID. Repeat this option for multiple top segments.",
    )
    evidence.add_argument("--query", help="Optional original query string to include in output.")
    evidence.add_argument("--segments", type=Path, help="Optional segment JSONL path.")
    evidence.add_argument("--frames-manifest", type=Path, help="Optional frames_manifest JSONL path.")
    evidence.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around the target.",
    )
    evidence.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    evidence.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    evidence.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    evidence.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before the target when using time-window mode.",
    )
    evidence.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after the target when using time-window mode.",
    )
    evidence.add_argument("--output", type=Path, help="Optional JSON output path.")
    evidence.set_defaults(func=cmd_evidence_window)

    extract_visual = subparsers.add_parser(
        "extract-visual-entities",
        help="Extract visual entities from sampled frames",
    )
    location = extract_visual.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    extract_visual.add_argument(
        "--backend",
        choices=["auto", "stub", "local-ocr", "vlm-jsonl", "vlm-observations"],
        default="auto",
        help=(
            "auto uses local OCR when available, otherwise stub. vlm-jsonl loads "
            "structured parser output. vlm-observations loads VLM observation JSONL."
        ),
    )
    extract_visual.add_argument(
        "--vlm-jsonl",
        type=Path,
        help=(
            "Structured VLM parser output JSONL for --backend vlm-jsonl. "
            "Relative paths are resolved from project dir."
        ),
    )
    extract_visual.add_argument(
        "--vlm-observations",
        type=Path,
        help=(
            "VLM visual observations JSONL for --backend vlm-observations. "
            "Defaults to manifests/vlm_visual_observations.jsonl under project dir."
        ),
    )
    extract_visual.add_argument(
        "--frames-manifest",
        type=Path,
        help="Input frames manifest JSONL. Relative paths are resolved from project dir.",
    )
    extract_visual.add_argument(
        "--output",
        type=Path,
        help="Output visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    extract_visual.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    extract_visual.add_argument(
        "--ocr-language",
        help="Optional OCR language hint for the local-ocr backend (for example eng or kor).",
    )
    extract_visual.set_defaults(func=cmd_extract_visual_entities)

    vlm = subparsers.add_parser(
        "run-vlm",
        help="Run a selectable VLM backend against project frames or frame candidates",
    )
    location = vlm.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    vlm.add_argument(
        "--vlm-backend",
        default=DEFAULT_VLM_BACKEND,
        help="VLM backend to run. Built-in options include deterministic and command.",
    )
    vlm.add_argument(
        "--vlm-model",
        required=True,
        help="VLM model identifier recorded as source_model in VLM artifacts.",
    )
    vlm.add_argument("--vlm-device", help="Optional device hint recorded in run metadata.")
    vlm.add_argument(
        "--vlm-options",
        help="Backend options as a JSON object or comma-separated key=value pairs.",
    )
    vlm.add_argument(
        "--frames-manifest",
        type=Path,
        help="Input frames manifest JSONL. Relative paths are resolved from project dir.",
    )
    vlm.add_argument(
        "--vlm-frame-candidates",
        type=Path,
        help="Optional VLM frame candidate JSONL. Relative paths are resolved from project dir.",
    )
    vlm.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing VLM visual observations and skip already processed frames.",
    )
    vlm.add_argument(
        "--output",
        type=Path,
        help="Output VLM visual observations JSONL path. Relative paths resolve from project dir.",
    )
    vlm.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    vlm.set_defaults(func=cmd_run_vlm)

    vlm_alignment = subparsers.add_parser(
        "run-vlm-alignment",
        help="Run VLM frame candidate selection, visual observations, and audio-visual consistency",
    )
    add_vlm_alignment_arguments(vlm_alignment)
    vlm_alignment.set_defaults(func=cmd_run_vlm_alignment)

    entity_links = subparsers.add_parser(
        "link-entities",
        help="Link transcript segments to nearby visual entities with weak evidence",
    )
    location = entity_links.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    entity_links.add_argument(
        "--segments",
        type=Path,
        help="Input segment JSONL path. Relative paths are resolved from project dir.",
    )
    entity_links.add_argument(
        "--visual-entities",
        type=Path,
        help="Input visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    entity_links.add_argument(
        "--output",
        type=Path,
        help="Output entity_links JSONL path. Relative paths are resolved from project dir.",
    )
    entity_links.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    entity_links.add_argument(
        "--domain-lexicon",
        type=Path,
        help="Optional domain_lexicon.json path. Relative paths are resolved from project dir.",
    )
    entity_links.set_defaults(func=cmd_link_entities)

    query = subparsers.add_parser("query", help="Search one query")
    query.add_argument("--index", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=5)
    query.set_defaults(func=cmd_query)

    query_project = subparsers.add_parser(
        "query-project",
        help="Search a local project and return multimodal evidence bundles",
    )
    query_project.add_argument("--index", required=True)
    query_project.add_argument(
        "--visual-index",
        help="Optional Meilisearch index containing visual_entities documents.",
    )
    location = query_project.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    query_project.add_argument("--query", required=True)
    query_project.add_argument("--limit", type=int, default=5)
    query_project.add_argument(
        "--segments",
        type=Path,
        help="Optional segment JSONL path. Relative paths are resolved from project dir.",
    )
    query_project.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path. Relative paths are resolved from project dir.",
    )
    query_project.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    query_project.add_argument(
        "--entity-links",
        type=Path,
        help="Optional entity_links JSONL path. Relative paths are resolved from project dir.",
    )
    query_project.add_argument(
        "--domain-lexicon",
        type=Path,
        help="Optional domain_lexicon.json path. Relative paths are resolved from project dir.",
    )
    query_project.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around the matched hit.",
    )
    query_project.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    query_project.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    query_project.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    query_project.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before the target when using time-window mode.",
    )
    query_project.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after the target when using time-window mode.",
    )
    query_project.add_argument(
        "--rerank",
        action="store_true",
        help="Reorder evidence bundles with deterministic domain-agnostic evidence signals.",
    )
    query_project.add_argument(
        "--rerank-time-hint",
        help="Optional timestamp hint for reranking, for example '10-14s'.",
    )
    query_project.add_argument("--output", type=Path, help="Optional JSON output path.")
    query_project.set_defaults(func=cmd_query_project)

    graph_query_parser = subparsers.add_parser(
        "graph-query",
        help="Search a project, then expand temporal/reference hints through Neo4j graph traversal",
    )
    graph_query_parser.add_argument("--index", required=True)
    location = graph_query_parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    graph_query_parser.add_argument("--query", required=True)
    graph_query_parser.add_argument("--limit", type=int, default=5)
    graph_query_parser.add_argument(
        "--segments",
        type=Path,
        help="Optional segment JSONL path. Relative paths are resolved from project dir.",
    )
    graph_query_parser.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path. Relative paths are resolved from project dir.",
    )
    graph_query_parser.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    graph_query_parser.add_argument(
        "--entity-links",
        type=Path,
        help="Optional entity_links JSONL path. Relative paths are resolved from project dir.",
    )
    graph_query_parser.add_argument(
        "--domain-lexicon",
        type=Path,
        help="Optional domain_lexicon.json path. Relative paths are resolved from project dir.",
    )
    graph_query_parser.add_argument(
        "--graph-lookback-segments",
        type=int,
        default=3,
        help="Maximum previous NEXT_SEGMENT hops to inspect when a graph hint is present.",
    )
    graph_query_parser.add_argument(
        "--graph-limit",
        type=int,
        default=12,
        help="Maximum graph evidence rows to return per Meilisearch candidate.",
    )
    graph_query_parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    graph_query_parser.set_defaults(func=cmd_graph_query)

    evaluate = subparsers.add_parser("eval-eduvidqa", help="Run timestamp proximity eval")
    evaluate.add_argument("--input", required=True, type=Path)
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--limit", type=int, default=5)
    evaluate.add_argument("--record-limit", type=int)
    evaluate.add_argument("--deltas", default="5,10,15")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(func=cmd_eval_eduvidqa)

    benchmark = subparsers.add_parser(
        "benchmark-retrieval",
        help="Run cross-domain retrieval benchmarks from a manifest",
    )
    benchmark.add_argument("--manifest", required=True, type=Path)
    benchmark.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to reports/perf_runs/{run_id} near the manifest.",
    )
    benchmark.set_defaults(func=cmd_benchmark_retrieval)

    lecture_smoke = subparsers.add_parser(
        "lecture-smoke",
        help="Run a private-safe lecture smoke suite from a manifest",
    )
    lecture_smoke.add_argument("--manifest", required=True, type=Path)
    lecture_smoke.add_argument(
        "--output-dir",
        type=Path,
        help="Optional public output directory. Writes sanitized metrics.json, query_results.jsonl, and summary.md.",
    )
    lecture_smoke.add_argument(
        "--private-output-dir",
        type=Path,
        help="Optional raw output directory. Requires --allow-private-output.",
    )
    lecture_smoke.add_argument(
        "--allow-private-output",
        action="store_true",
        help="Allow raw private outputs that may contain queries, transcripts, and local paths.",
    )
    lecture_smoke.set_defaults(func=cmd_lecture_smoke)

    return parser


def build_vlm_alignment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-vlm-alignment",
        description=(
            "Run VLM frame candidate selection, visual observations, "
            "and audio-visual consistency"
        ),
    )
    add_vlm_alignment_arguments(parser)
    parser.set_defaults(func=cmd_run_vlm_alignment)
    return parser


def add_vlm_alignment_arguments(parser: argparse.ArgumentParser) -> None:
    candidate_defaults = VLMFrameCandidateConfig()
    consistency_defaults = AudioVisualConsistencyConfig()
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    parser.add_argument(
        "--vlm-backend",
        default=DEFAULT_VLM_BACKEND,
        help="VLM backend to run. Built-in options include deterministic and command.",
    )
    parser.add_argument(
        "--vlm-model",
        required=True,
        help="VLM model identifier recorded as source_model in VLM artifacts.",
    )
    parser.add_argument("--vlm-device", help="Optional device hint recorded in run metadata.")
    parser.add_argument(
        "--vlm-options",
        help="Backend options as a JSON object or comma-separated key=value pairs.",
    )
    parser.add_argument(
        "--max-vlm-frames",
        type=int,
        default=candidate_defaults.max_candidates,
        help="Maximum candidate frames to send to the VLM.",
    )
    parser.add_argument(
        "--candidate-max-per-segment",
        type=int,
        default=candidate_defaults.max_per_segment,
        help="Maximum VLM candidate frames selected per transcript segment.",
    )
    parser.add_argument(
        "--candidate-window-seconds",
        type=float,
        default=candidate_defaults.window_seconds,
        help="Temporal window size used by candidate selection.",
    )
    parser.add_argument(
        "--candidate-max-per-window",
        type=int,
        default=candidate_defaults.max_per_window,
        help="Maximum candidate frames selected per temporal window.",
    )
    parser.add_argument(
        "--candidate-min-time-gap-seconds",
        type=float,
        default=candidate_defaults.min_time_gap_seconds,
        help="Minimum timestamp gap between selected candidate frames.",
    )
    parser.add_argument(
        "--candidate-segment-margin-seconds",
        type=float,
        default=candidate_defaults.segment_margin_seconds,
        help="Margin added to segment windows before selecting candidates.",
    )
    parser.add_argument(
        "--candidate-duplicate-timestamp-epsilon-seconds",
        type=float,
        default=candidate_defaults.duplicate_timestamp_epsilon_seconds,
        help="Timestamp epsilon used to suppress duplicate candidate frames.",
    )
    parser.add_argument(
        "--candidate-duplicate-distance-threshold",
        type=float,
        default=candidate_defaults.duplicate_distance_threshold,
        help="Nearest-selected distance threshold used to suppress duplicate frames.",
    )
    parser.add_argument(
        "--candidate-low-information-min-contrast",
        type=float,
        default=candidate_defaults.low_information_min_contrast,
        help="Minimum frame contrast accepted by candidate selection.",
    )
    parser.add_argument(
        "--candidate-low-information-min-detail-score",
        type=float,
        default=candidate_defaults.low_information_min_detail_score,
        help="Minimum frame detail score accepted by candidate selection.",
    )
    parser.add_argument(
        "--consistency-window-margin-seconds",
        type=float,
        default=consistency_defaults.window_margin_seconds,
        help="Margin used when matching visual observations to transcript segment windows.",
    )
    parser.add_argument(
        "--frames-manifest",
        type=Path,
        help="Input frames manifest JSONL. Relative paths are resolved from project dir.",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        help="Input segment JSONL path. Relative paths are resolved from project dir.",
    )
    parser.add_argument(
        "--vlm-frame-candidates",
        type=Path,
        help="Output VLM frame candidate JSONL. Relative paths are resolved from project dir.",
    )
    parser.add_argument(
        "--vlm-visual-observations",
        type=Path,
        help="Output VLM visual observations JSONL. Relative paths are resolved from project dir.",
    )
    parser.add_argument(
        "--audio-visual-consistency",
        type=Path,
        help="Output audio-visual consistency JSONL. Relative paths are resolved from project dir.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing candidate artifacts and pass resume through to VLM visual "
            "observation generation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan candidate frames and artifact paths without writing artifacts.",
    )


def client_from_args(args: argparse.Namespace) -> MeiliClient:
    return MeiliClient(base_url=args.url, api_key=args.api_key)


def cmd_health(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))


def cmd_health_neo4j(args: argparse.Namespace) -> None:
    print(json.dumps(check_neo4j_health(), ensure_ascii=False, indent=2))


def cmd_graph_ingest(args: argparse.Namespace) -> None:
    project_dir = None
    if args.project_id is not None or args.project_dir is not None:
        project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = ingest_project_graph(
        project_dir=project_dir,
        graph_document_path=args.graph_document,
        create_schema=not args.skip_schema,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_index_eduvidqa(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    if args.reset:
        client.wait_task(client.delete_index(args.index))
    client.wait_task(client.create_index(args.index, primary_key="segment_id"))
    client.wait_task(client.update_settings(args.index, LECTURE_SEGMENT_SETTINGS))

    total = 0
    batch: list[dict] = []
    for segment in iter_lecture_segments(args.input, project_id=args.project_id, limit=args.limit):
        batch.append(segment.to_dict())
        if len(batch) >= args.batch_size:
            client.wait_task(client.add_documents(args.index, batch))
            total += len(batch)
            batch = []
            print(f"indexed {total} documents", file=sys.stderr)
    if batch:
        client.wait_task(client.add_documents(args.index, batch))
        total += len(batch)

    print(json.dumps({"index": args.index, "indexed_documents": total}, indent=2))


def cmd_index_project(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = index_project_segments(
        client,
        index_uid=args.index,
        project_dir=project_dir,
        batch_size=args.batch_size,
        reset=args.reset,
        segments=args.segments,
        settings_profile=args.settings_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_index_project_visual_entities(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = index_project_visual_entities(
        client,
        index_uid=args.index,
        project_dir=project_dir,
        batch_size=args.batch_size,
        reset=args.reset,
        visual_entities=args.visual_entities,
        settings_profile=args.settings_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_ingest_video(args: argparse.Namespace) -> None:
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = default_paths().repo_root / output_root
    max_frames = None if args.max_frames == 0 else args.max_frames
    video_id = args.video_id or make_video_id(args.video)
    manifest = ingest_video(
        VideoIngestConfig(
            video_path=args.video,
            project_id=args.project_id,
            video_id=video_id,
            output_root=output_root,
            srt_path=args.srt,
            frame_rate=args.frame_rate,
            max_frames=max_frames,
            frame_sampling=args.frame_sampling,
            frame_selection=args.frame_selection,
            skip_frames=args.skip_frames,
            copy_source=args.copy_source,
            transcript_source=args.transcript_source,
            stt_model=args.stt_model,
            stt_language=args.stt_language,
            stt_task=args.stt_task,
            stt_word_timestamps=args.stt_word_timestamps,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def cmd_batch_ingest(args: argparse.Namespace) -> None:
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = default_paths().repo_root / output_root
    max_frames = None if args.max_frames == 0 else args.max_frames
    summary = batch_ingest_videos(
        BatchIngestConfig(
            root_dir=args.root,
            output_root=output_root,
            frame_rate=args.frame_rate,
            max_frames=max_frames,
            frame_sampling=args.frame_sampling,
            frame_selection=args.frame_selection,
            skip_frames=args.skip_frames,
            copy_source=args.copy_source,
            transcript_source=args.transcript_source,
            stt_model=args.stt_model,
            stt_language=args.stt_language,
            stt_task=args.stt_task,
            stt_word_timestamps=args.stt_word_timestamps,
            force=args.force,
            strict=args.strict,
            dry_run=args.dry_run,
            project_prefix=args.project_prefix,
        )
    )
    if args.summary_json:
        write_batch_summary_json(args.summary_json, summary)
    if args.summary_jsonl:
        write_batch_summary_jsonl(args.summary_jsonl, summary)
    if args.summary_csv:
        write_batch_summary_csv(args.summary_csv, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and summary["counts"]["failed"] > 0:
        raise SystemExit(1)


def cmd_align_frames(args: argparse.Namespace) -> None:
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = default_paths().repo_root / output_root

    summary = align_segments_to_frames(
        project_id=args.project_id,
        output_root=output_root,
        margin_seconds=args.margin_seconds,
        segments_path=args.segments,
        frames_manifest_path=args.frames_manifest,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_evidence_window(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    response = build_evidence_response(
        project_dir=project_dir,
        segment_ids=args.segment_id,
        query=args.query,
        segments_path=args.segments,
        frames_manifest_path=args.frames_manifest,
        window_seconds=args.window_seconds,
        neighbor_count=args.neighbor_count,
        previous_neighbor_count=args.previous_neighbor_count,
        next_neighbor_count=args.next_neighbor_count,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_extract_visual_entities(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = extract_visual_entities(
        project_dir=project_dir,
        backend=args.backend,
        frames_manifest_path=args.frames_manifest,
        output_path=args.output,
        manifest_path=args.manifest,
        ocr_language=args.ocr_language,
        vlm_jsonl_path=args.vlm_jsonl,
        vlm_observations_path=args.vlm_observations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_run_vlm(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = run_vlm(
        project_dir=project_dir,
        backend=args.vlm_backend,
        model=args.vlm_model,
        device=args.vlm_device,
        options=parse_vlm_options(args.vlm_options),
        frames_manifest_path=args.frames_manifest,
        frame_candidates_path=args.vlm_frame_candidates,
        output_path=args.output,
        manifest_path=args.manifest,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_run_vlm_alignment(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = run_vlm_alignment_pipeline(
        VLMAlignmentPipelineConfig(
            project_dir=project_dir,
            vlm_backend=args.vlm_backend,
            vlm_model=args.vlm_model,
            vlm_device=args.vlm_device,
            vlm_options=parse_vlm_options(args.vlm_options),
            frames_manifest_path=args.frames_manifest,
            segments_path=args.segments,
            frame_candidates_path=args.vlm_frame_candidates,
            visual_observations_path=args.vlm_visual_observations,
            audio_visual_consistency_path=args.audio_visual_consistency,
            manifest_path=args.manifest,
            candidate_config=VLMFrameCandidateConfig(
                max_candidates=args.max_vlm_frames,
                max_per_segment=args.candidate_max_per_segment,
                window_seconds=args.candidate_window_seconds,
                max_per_window=args.candidate_max_per_window,
                min_time_gap_seconds=args.candidate_min_time_gap_seconds,
                segment_margin_seconds=args.candidate_segment_margin_seconds,
                duplicate_timestamp_epsilon_seconds=(
                    args.candidate_duplicate_timestamp_epsilon_seconds
                ),
                duplicate_distance_threshold=args.candidate_duplicate_distance_threshold,
                low_information_min_contrast=args.candidate_low_information_min_contrast,
                low_information_min_detail_score=(
                    args.candidate_low_information_min_detail_score
                ),
            ),
            consistency_config=AudioVisualConsistencyConfig(
                window_margin_seconds=args.consistency_window_margin_seconds
            ),
            resume=args.resume,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_link_entities(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = link_entities(
        project_dir=project_dir,
        segments_path=args.segments,
        visual_entities_path=args.visual_entities,
        output_path=args.output,
        manifest_path=args.manifest,
        domain_lexicon_path=args.domain_lexicon,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    response = client.search(args.index, args.query, limit=args.limit)
    candidates = [
        SearchCandidate.from_hit(rank=rank, hit=hit).to_dict()
        for rank, hit in enumerate(response.get("hits", []), start=1)
    ]
    print(
        json.dumps(
            {
                "query": args.query,
                "index": args.index,
                "processing_time_ms": response.get("processingTimeMs"),
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_query_project(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    response = query_project(
        client=client,
        index_uid=args.index,
        visual_index_uid=args.visual_index,
        project_dir=project_dir,
        query=args.query,
        limit=args.limit,
        segments_path=args.segments,
        frames_manifest_path=args.frames_manifest,
        visual_entities_path=args.visual_entities,
        entity_links_path=args.entity_links,
        domain_lexicon_path=args.domain_lexicon,
        window_seconds=args.window_seconds,
        neighbor_count=args.neighbor_count,
        previous_neighbor_count=args.previous_neighbor_count,
        next_neighbor_count=args.next_neighbor_count,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
        rerank=args.rerank,
        rerank_time_hint=args.rerank_time_hint,
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    for line in response.get("summary_lines", []):
        print(line, file=sys.stderr)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_graph_query(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    response = graph_query(
        client=client,
        index_uid=args.index,
        project_dir=project_dir,
        project_id=args.project_id,
        query=args.query,
        limit=args.limit,
        segments_path=args.segments,
        frames_manifest_path=args.frames_manifest,
        visual_entities_path=args.visual_entities,
        entity_links_path=args.entity_links,
        domain_lexicon_path=args.domain_lexicon,
        traversal_config=GraphTraversalConfig(
            lookback_segments=args.graph_lookback_segments,
            per_candidate_limit=args.graph_limit,
        ),
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    for line in response.get("summary_lines", []):
        print(line, file=sys.stderr)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_eval_eduvidqa(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    deltas = parse_deltas(args.deltas)
    output_path = args.output
    if output_path is not None and not output_path.is_absolute():
        output_path = default_paths().repo_root / output_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    evals = []
    output_handle = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for count, record in enumerate(iter_records(args.input, limit=args.record_limit), start=1):
            response = client.search(args.index, record.question, limit=args.limit)
            candidates = [
                SearchCandidate.from_hit(rank=rank, hit=hit)
                for rank, hit in enumerate(response.get("hits", []), start=1)
            ]
            query_eval = evaluate_query(record, candidates, deltas=deltas)
            evals.append(query_eval)
            if output_handle:
                output_handle.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "video_name": record.video_name,
                            "question": record.question,
                            "timestamp_points": record.timestamp_points,
                            "best_abs_error": query_eval.best_abs_error,
                            "hit_by_delta": query_eval.hit_by_delta,
                            "topk_candidates": [candidate.to_dict() for candidate in candidates],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if count % 100 == 0:
                print(f"evaluated {count} queries", file=sys.stderr)
    finally:
        if output_handle:
            output_handle.close()

    print(json.dumps(summarize(evals, deltas=deltas), ensure_ascii=False, indent=2))


def cmd_benchmark_retrieval(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    run = run_benchmark(
        client=client,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        repo_root=default_paths().repo_root,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "output_dir": str(run.output_dir),
                "metrics": str(run.metrics_path),
                "query_results": str(run.query_results_path),
                "summary": str(run.summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_lecture_smoke(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    run = run_lecture_smoke(
        client=client,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        repo_root=default_paths().repo_root,
        private_output_dir=args.private_output_dir,
        allow_private_output=args.allow_private_output,
    )
    payload = {
        "run_id": run.run_id,
        "output_dir": str(run.output_dir),
        "metrics": str(run.metrics_path),
        "query_results": str(run.query_results_path),
        "summary": str(run.summary_path),
    }
    if run.private_query_outputs_path is not None:
        payload["private_query_outputs"] = str(run.private_query_outputs_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_deltas(value: str) -> list[int]:
    deltas = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped:
            deltas.append(int(stripped))
    if not deltas:
        raise ValueError("At least one delta must be provided")
    return deltas


def parse_vlm_options(value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    stripped = value.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("--vlm-options JSON value must be an object")
        return parsed

    options: dict[str, object] = {}
    for item in stripped.split(","):
        pair = item.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError("--vlm-options must be a JSON object or comma-separated key=value pairs")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--vlm-options contains an empty key")
        options[key] = _parse_vlm_option_value(raw_value.strip())
    return options


def _parse_vlm_option_value(value: str) -> object:
    if value == "":
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
