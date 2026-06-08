from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oarag.ingestion.alignment import align_segments_to_frames
from oarag.evaluation.benchmark import run_benchmark
from oarag.embeddings.cache import JsonlEmbeddingCache
from oarag.embeddings.manifest import build_vector_manifest, load_input_records
from oarag.embeddings.providers import (
    DEFAULT_EMBEDDING_API_KEY_ENV,
    DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
    DeterministicFixtureEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
)
from oarag.evaluation.claims import build_paper_claims
from oarag.evaluation.experiment import run_paper_experiment
from oarag.evaluation.paper_bundle import run_paper_bundle
from oarag.evaluation.paper_registry import build_paper_artifact_registry
from oarag.evaluation.quality_gate import check_retrieval_quality_gate
from oarag.evaluation.readiness import audit_paper_readiness
from oarag.evaluation.reporting import generate_evaluation_report
from oarag.evaluation.metric_intervals import generate_metric_intervals
from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL, ENV_STT_LANGUAGE, default_paths, env_default
from oarag.ingestion.eduvidqa import iter_lecture_segments, iter_records
from oarag.evaluation.eval import candidate_diagnostics, evaluate_query, summarize
from oarag.vision.entity_links import link_entities
from oarag.retrieval.evidence import build_evidence_response
from oarag.graph.concept_extraction import extract_project_concept_candidates
from oarag.graph.concept_graph_ingest import ingest_concept_graph
from oarag.graph.cross_lecture_merge import merge_cross_lecture_concepts
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
from oarag.evaluation.evidence_unit_smoke import run_evidence_unit_smoke
from oarag.integrations.meili import (
    EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    LECTURE_SEGMENT_SETTINGS,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    MeiliClient,
    evidence_unit_settings_profile_names,
    hybrid_embedder_settings_profile_names,
    lecture_segment_settings_profile_names,
    lecture_window_settings_profile_names,
    load_hybrid_embedder_settings,
    visual_entity_settings_profile_names,
)
from oarag.integrations.neo4j import check_neo4j_health
from oarag.retrieval.answer import ask_project, format_answer_text
from oarag.retrieval.project_query import (
    DEFAULT_HYBRID_EMBEDDER,
    DEFAULT_HYBRID_SEMANTIC_RATIO,
    query_project,
)
from oarag.retrieval.rerank import DEFAULT_RERANK_BACKEND, RERANK_BACKEND_CHOICES
from oarag.retrieval.project_index import (
    build_project_windows,
    index_project_evidence_units,
    index_project_segments,
    index_project_visual_entities,
    index_project_windows,
    project_dir_from_args,
)
from oarag.retrieval.evidence_units import build_project_evidence_units, build_project_visual_states
from oarag.retrieval.dual_candidates import query_project_dual_candidates
from oarag.retrieval.evidence_unit_index import query_project_evidence_units
from oarag.core.schemas import SearchCandidate
from oarag.ingestion.stt import DEFAULT_MLX_WHISPER_MODEL
from oarag.vision.visual_entities import DEFAULT_VISUAL_ENTITY_BACKEND, extract_visual_entities
from oarag.vision.vlm import DEFAULT_VLM_BACKEND, available_vlm_backends, run_vlm
from oarag.vision.vlm_evidence_validator import validate_vlm_object_evidence
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


def main_paper_metric_intervals(argv: list[str] | None = None) -> None:
    parser = build_paper_metric_intervals_parser()
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

    embedding_vectors = subparsers.add_parser(
        "build-embedding-vectors",
        help="Build a private vector manifest with a real embedding provider.",
    )
    embedding_vectors.add_argument("--input", required=True, type=Path)
    embedding_vectors.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "json", "csv"],
        default="auto",
    )
    embedding_vectors.add_argument("--output", required=True, type=Path)
    embedding_vectors.add_argument(
        "--kind",
        choices=["document", "query"],
        default="document",
        help="Manifest shape to write. query is compatible with hybrid query vectors.",
    )
    embedding_vectors.add_argument("--id-field", required=True)
    embedding_vectors.add_argument(
        "--text-field",
        action="append",
        required=True,
        dest="text_fields",
        help="Text field to embed. Repeat to concatenate multiple fields.",
    )
    embedding_vectors.add_argument("--embedder-name", default="default")
    embedding_vectors.add_argument(
        "--provider",
        choices=["openai-compatible", "sentence-transformers", "deterministic-fixture"],
        default="openai-compatible",
    )
    embedding_vectors.add_argument(
        "--model",
        help="Embedding model. Required for openai-compatible unless OARAG_EMBEDDING_MODEL is set.",
    )
    embedding_vectors.add_argument(
        "--api-base",
        help="OpenAI-compatible API base URL.",
    )
    embedding_vectors.add_argument(
        "--api-key-env",
        default=DEFAULT_EMBEDDING_API_KEY_ENV,
        help="Environment variable containing the embedding API key.",
    )
    embedding_vectors.add_argument("--dimensions", type=int)
    embedding_vectors.add_argument(
        "--local-files-only",
        action="store_true",
        help="For sentence-transformers, load only locally cached model files.",
    )
    embedding_vectors.add_argument("--batch-size", type=int, default=64)
    embedding_vectors.add_argument("--cache", type=Path)
    embedding_vectors.add_argument(
        "--allow-empty-text",
        action="store_true",
        help="Embed empty text records instead of failing.",
    )
    embedding_vectors.add_argument(
        "--source-label",
        help="Private-safe label for the input source, for example mitdl_segments.",
    )
    embedding_vectors.set_defaults(func=cmd_build_embedding_vectors)

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

    concept_candidates = subparsers.add_parser(
        "extract-concept-candidates",
        help="Extract lecture-local concept candidate nodes from evidence units",
    )
    location = concept_candidates.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    concept_candidates.add_argument(
        "--evidence-units",
        type=Path,
        help="Optional evidence_units JSONL path. Relative paths are resolved from project dir.",
    )
    concept_candidates.add_argument(
        "--output",
        type=Path,
        help="Output concept_graph JSONL path. Relative paths are resolved from project dir.",
    )
    concept_candidates.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    concept_candidates.add_argument(
        "--lecture-id",
        help="Optional lecture id override. Defaults to manifest/video/project id.",
    )
    concept_candidates.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum concept candidate confidence to write.",
    )
    concept_candidates.set_defaults(func=cmd_extract_concept_candidates)

    concept_graph_ingest = subparsers.add_parser(
        "concept-graph-ingest",
        help="Ingest a concept graph artifact into Neo4j with idempotent merges.",
    )
    location = concept_graph_ingest.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    location.add_argument("--concept-graph", type=Path, help="Concept graph JSONL artifact path")
    concept_graph_ingest.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip idempotent Neo4j constraint/index creation.",
    )
    concept_graph_ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize Cypher merge plan without connecting to Neo4j.",
    )
    concept_graph_ingest.add_argument(
        "--fail-on-unavailable-runtime",
        action="store_true",
        help="Raise an error instead of returning an explicit skip summary when Neo4j is unavailable.",
    )
    concept_graph_ingest.set_defaults(func=cmd_concept_graph_ingest)

    cross_lecture_merge = subparsers.add_parser(
        "merge-cross-lecture-concepts",
        help="Merge multiple lecture-local concept graphs into a global graph document.",
    )
    cross_lecture_merge.add_argument(
        "--concept-graph",
        action="append",
        required=True,
        type=Path,
        dest="concept_graphs",
        help="Lecture-local concept graph JSONL path. Repeat for each lecture.",
    )
    cross_lecture_merge.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output global concept graph document JSON path.",
    )
    cross_lecture_merge.add_argument(
        "--summary",
        type=Path,
        help="Optional counts-only public-safe summary JSON path.",
    )
    cross_lecture_merge.add_argument(
        "--project-id",
        help="Optional project id to store in the global graph document.",
    )
    cross_lecture_merge.set_defaults(func=cmd_merge_cross_lecture_concepts)

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
        "--vector-manifest",
        type=Path,
        help="Optional real embedding vector manifest. Relative paths resolve from project dir.",
    )
    index_project.add_argument(
        "--settings-profile",
        choices=lecture_segment_settings_profile_names(),
        default=LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile to apply to lecture_segments.",
    )
    _add_hybrid_embedder_index_options(index_project)
    index_project.set_defaults(func=cmd_index_project)

    build_windows = subparsers.add_parser(
        "build-project-windows",
        help="Build a local project lecture_windows JSONL for window-level retrieval",
    )
    location = build_windows.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    build_windows.add_argument(
        "--segments",
        type=Path,
        help="Optional segment JSONL path. Relative paths are resolved from project dir.",
    )
    build_windows.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path. Relative paths are resolved from project dir.",
    )
    build_windows.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    build_windows.add_argument(
        "--output",
        type=Path,
        help="Output lecture_windows JSONL path. Relative paths are resolved from project dir.",
    )
    build_windows.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    build_windows.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around each target.",
    )
    build_windows.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    build_windows.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    build_windows.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    build_windows.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before each target when using time-window mode.",
    )
    build_windows.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after each target when using time-window mode.",
    )
    build_windows.set_defaults(func=cmd_build_project_windows)

    build_visual_states = subparsers.add_parser(
        "build-project-visual-states",
        help="Build a local project visual_states JSONL artifact from sampled frames",
    )
    location = build_visual_states.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    build_visual_states.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path. Relative paths are resolved from project dir.",
    )
    build_visual_states.add_argument(
        "--output",
        type=Path,
        help="Output visual_states JSONL path. Relative paths are resolved from project dir.",
    )
    build_visual_states.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    build_visual_states.add_argument(
        "--state-padding-seconds",
        type=float,
        default=15.0,
        help="Seconds to extend first/last sampled frames when building rough visual state intervals.",
    )
    build_visual_states.add_argument(
        "--min-visual-states",
        type=int,
        help="Optional minimum visual state count for the visual state coverage gate.",
    )
    build_visual_states.add_argument(
        "--fail-on-visual-state-gate",
        action="store_true",
        help="Exit non-zero if configured visual state coverage thresholds fail.",
    )
    build_visual_states.set_defaults(func=cmd_build_project_visual_states)

    build_evidence_units = subparsers.add_parser(
        "build-project-evidence-units",
        help="Build a local project evidence_units JSONL for evidence-first retrieval",
    )
    location = build_evidence_units.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    build_evidence_units.add_argument(
        "--segments",
        type=Path,
        help="Optional segment JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--visual-states",
        type=Path,
        help="Optional visual_states JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--visual-states-output",
        type=Path,
        help="Optional visual_states JSONL output path to write the normalized visual state artifact.",
    )
    build_evidence_units.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--entity-links",
        type=Path,
        help="Optional entity_links JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--concept-graph",
        type=Path,
        help="Optional concept_graph JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--output",
        type=Path,
        help="Output evidence_units JSONL path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--manifest",
        type=Path,
        help="Project manifest JSON path. Relative paths are resolved from project dir.",
    )
    build_evidence_units.add_argument(
        "--state-padding-seconds",
        type=float,
        default=15.0,
        help="Seconds to extend first/last sampled frames when building rough visual state intervals.",
    )
    build_evidence_units.add_argument(
        "--visual-state-min-coverage-ratio",
        "--visual-state-min-unit-coverage-ratio",
        dest="visual_state_min_coverage_ratio",
        type=float,
        help="Optional minimum fraction of evidence units with visual_state for the coverage gate.",
    )
    build_evidence_units.add_argument(
        "--visual-state-min-total",
        type=int,
        help="Optional minimum total visual states for the coverage gate.",
    )
    build_evidence_units.add_argument(
        "--fail-on-visual-state-gate",
        action="store_true",
        help="Exit non-zero if configured visual state coverage thresholds fail.",
    )
    build_evidence_units.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around each target.",
    )
    build_evidence_units.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    build_evidence_units.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    build_evidence_units.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    build_evidence_units.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before each target when using time-window mode.",
    )
    build_evidence_units.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after each target when using time-window mode.",
    )
    build_evidence_units.set_defaults(func=cmd_build_project_evidence_units)

    index_evidence_units = subparsers.add_parser(
        "index-project-evidence-units",
        help="Index local project evidence_units JSONL documents",
    )
    index_evidence_units.add_argument("--index", required=True)
    location = index_evidence_units.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    index_evidence_units.add_argument(
        "--evidence-units",
        type=Path,
        help="Optional evidence_units JSONL path. Relative paths are resolved from project dir.",
    )
    index_evidence_units.add_argument("--batch-size", type=int, default=500)
    index_evidence_units.add_argument("--reset", action="store_true")
    index_evidence_units.add_argument(
        "--settings-profile",
        choices=evidence_unit_settings_profile_names(),
        default=EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile to apply to evidence unit documents.",
    )
    index_evidence_units.set_defaults(func=cmd_index_project_evidence_units)

    index_windows = subparsers.add_parser(
        "index-project-windows",
        help="Index local project lecture window documents",
    )
    index_windows.add_argument("--index", required=True)
    location = index_windows.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    index_windows.add_argument(
        "--windows",
        type=Path,
        help=(
            "Optional lecture_windows JSONL path. Relative paths are resolved from project dir. "
            "When omitted, an existing default artifact is used unless build inputs or "
            "window options are provided."
        ),
    )
    index_windows.add_argument(
        "--segments",
        type=Path,
        help="Optional segment JSONL path used when building windows in memory.",
    )
    index_windows.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path used when building windows in memory.",
    )
    index_windows.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path used when building windows in memory.",
    )
    index_windows.add_argument("--batch-size", type=int, default=500)
    index_windows.add_argument("--reset", action="store_true")
    index_windows.add_argument(
        "--vector-manifest",
        type=Path,
        help="Optional real embedding vector manifest. Relative paths resolve from project dir.",
    )
    index_windows.add_argument(
        "--settings-profile",
        choices=lecture_window_settings_profile_names(),
        default=LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile to apply to lecture window documents.",
    )
    _add_hybrid_embedder_index_options(index_windows)
    index_windows.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around each target.",
    )
    index_windows.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    index_windows.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    index_windows.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    index_windows.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before each target when using time-window mode.",
    )
    index_windows.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after each target when using time-window mode.",
    )
    index_windows.set_defaults(func=cmd_index_project_windows)

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
        "--vector-manifest",
        type=Path,
        help="Optional real embedding vector manifest. Relative paths resolve from project dir.",
    )
    index_visual_entities.add_argument(
        "--settings-profile",
        choices=visual_entity_settings_profile_names(),
        default=VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile to apply to visual_entities.",
    )
    _add_hybrid_embedder_index_options(index_visual_entities)
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
        "--max-frame-gap-seconds",
        type=float,
        help="Optional policy target for maximum seconds between sampled evidence frames.",
    )
    ingest.add_argument(
        "--frame-sampling",
        choices=["uniform", "prefix"],
        default="uniform",
        help="How to apply --max-frames. uniform spreads capped frames across the video.",
    )
    ingest.add_argument(
        "--frame-selection",
        choices=["none", "representative", "scene-change"],
        default="none",
        help="Optional post-sampling selector for representative or scene-change frames.",
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
        "--max-frame-gap-seconds",
        type=float,
        help="Optional policy target for maximum seconds between sampled evidence frames.",
    )
    batch_ingest.add_argument(
        "--frame-sampling",
        choices=["uniform", "prefix"],
        default="uniform",
        help="How to apply --max-frames. uniform spreads capped frames across each video.",
    )
    batch_ingest.add_argument(
        "--frame-selection",
        choices=["none", "representative", "scene-change"],
        default="none",
        help="Optional post-sampling selector for representative or scene-change frames.",
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
        help="Build VLM-first visual_entities; OCR is an explicit baseline/fallback.",
    )
    location = extract_visual.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    extract_visual.add_argument(
        "--backend",
        choices=["vlm-first", "auto", "vlm-observations", "vlm-jsonl", "local-ocr", "stub"],
        default=DEFAULT_VISUAL_ENTITY_BACKEND,
        help=(
            "vlm-first/auto prefer VLM observations, then VLM parser JSONL; "
            "local-ocr is used only as an OCR fallback/baseline, otherwise stub."
        ),
    )
    extract_visual.add_argument(
        "--vlm-jsonl",
        type=Path,
        help=(
            "Structured VLM parser output JSONL for --backend vlm-jsonl or "
            "vlm-first. Defaults to manifests/vlm_parser_output.jsonl when that "
            "backend is selected. Relative paths resolve from project dir."
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
        help=(
            "Optional OCR language hint for the local-ocr baseline/fallback "
            "(for example eng or kor)."
        ),
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
        choices=available_vlm_backends(),
        default=DEFAULT_VLM_BACKEND,
        help=(
            "VLM parser backend. deterministic/mock are dependency-free, command "
            "shells out to an external parser, and jsonl replays fixture observations."
        ),
    )
    vlm.add_argument(
        "--vlm-model",
        required=True,
        help="VLM model identifier recorded as source_model in VLM artifacts.",
    )
    vlm.add_argument("--vlm-device", help="Optional device hint recorded in run metadata.")
    vlm.add_argument(
        "--vlm-options",
        help=(
            "Backend options as JSON or key=value pairs. command uses command/input_mode; "
            "jsonl uses jsonl_path."
        ),
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

    validate_vlm = subparsers.add_parser(
        "validate-vlm-evidence",
        help="Write a public-safe VLM object evidence coverage report",
    )
    location = validate_vlm.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    validate_vlm.add_argument(
        "--visual-entities",
        type=Path,
        help="visual_entities JSONL path. Defaults to manifests/visual_entities.jsonl.",
    )
    validate_vlm.add_argument(
        "--evidence-units",
        type=Path,
        help="evidence_units JSONL path. Defaults to segments/evidence_units.jsonl.",
    )
    validate_vlm.add_argument(
        "--vlm-observations",
        type=Path,
        help="VLM visual observations JSONL path used for artifact presence checks.",
    )
    validate_vlm.add_argument(
        "--vlm-jsonl",
        type=Path,
        help="Structured VLM parser JSONL path used for artifact presence checks.",
    )
    validate_vlm.add_argument("--output", type=Path, help="Optional public-safe JSON report path.")
    validate_vlm.add_argument(
        "--summary",
        type=Path,
        help="Optional public-safe Markdown summary path.",
    )
    validate_vlm.set_defaults(func=cmd_validate_vlm_evidence)

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
        "--index-kind",
        choices=["segment", "window"],
        default="segment",
        help="Interpret --index hits as segment documents or window-level documents.",
    )
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
        "--candidate-pool-limit",
        type=int,
        help=(
            "Retrieve this many candidates per search channel before final limiting. "
            "Useful with --rerank to let the reranker see deeper candidates."
        ),
    )
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
    query_project.add_argument(
        "--rerank-backend",
        choices=RERANK_BACKEND_CHOICES,
        default=DEFAULT_RERANK_BACKEND,
        help="Private-safe reranker backend used by --rerank.",
    )
    query_project.add_argument(
        "--hybrid-retrieval",
        action="store_true",
        help=(
            "Also query a semantic Meilisearch channel and merge it with lexical "
            "candidate hits."
        ),
    )
    query_project.add_argument(
        "--hybrid-embedder",
        default=DEFAULT_HYBRID_EMBEDDER,
        help="Meilisearch embedder name used by --hybrid-retrieval.",
    )
    query_project.add_argument(
        "--hybrid-semantic-ratio",
        type=float,
        default=DEFAULT_HYBRID_SEMANTIC_RATIO,
        help="Semantic ratio for the semantic channel used by --hybrid-retrieval.",
    )
    query_project.add_argument(
        "--hybrid-query-vector-manifest",
        type=Path,
        help=(
            "JSON manifest containing userProvided query vectors. Relative paths are "
            "resolved from the project dir."
        ),
    )
    query_project.add_argument(
        "--hybrid-query-vector-name",
        help="Query vector name in --hybrid-query-vector-manifest.",
    )
    query_project.add_argument(
        "--hybrid-query-vector-embedder",
        help="Expected Meilisearch userProvided embedder name for the query vector.",
    )
    query_project.add_argument(
        "--hybrid-query-vector-dimensions",
        type=int,
        help="Expected query vector dimensions for mismatch checks.",
    )
    query_project.add_argument("--output", type=Path, help="Optional JSON output path.")
    query_project.set_defaults(func=cmd_query_project)

    query_evidence_units = subparsers.add_parser(
        "query-project-evidence-units",
        help="Search a local project evidence unit index",
    )
    query_evidence_units.add_argument("--index", required=True)
    location = query_evidence_units.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    query_evidence_units.add_argument("--query", required=True)
    query_evidence_units.add_argument("--limit", type=int, default=5)
    query_evidence_units.add_argument(
        "--evidence-units",
        type=Path,
        help="Optional evidence_units JSONL path used to resolve project_id filter.",
    )
    query_evidence_units.add_argument("--output", type=Path, help="Optional JSON output path.")
    query_evidence_units.set_defaults(func=cmd_query_project_evidence_units)

    dual_candidates = subparsers.add_parser(
        "query-project-dual-candidates",
        help="Union Meilisearch raw/expanded evidence-unit candidates with graph traversal",
    )
    dual_candidates.add_argument("--index", required=True)
    location = dual_candidates.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    dual_candidates.add_argument("--query", required=True)
    dual_candidates.add_argument("--limit", type=int, default=5)
    dual_candidates.add_argument(
        "--candidate-pool-limit",
        type=int,
        help="Candidates to request per Meilisearch source before union and final limiting.",
    )
    dual_candidates.add_argument(
        "--graph-limit",
        type=int,
        help="Maximum graph traversal candidate rows to request.",
    )
    dual_candidates.add_argument(
        "--evidence-units",
        type=Path,
        help="Optional evidence_units JSONL path used for project_id and graph candidate enrichment.",
    )
    dual_candidates.add_argument(
        "--target-evidence-unit-id",
        action="append",
        dest="target_evidence_unit_ids",
        help="Optional target evidence_unit_id for public-safe source recall diagnostics.",
    )
    dual_candidates.add_argument(
        "--target-segment-id",
        action="append",
        dest="target_segment_ids",
        help="Optional target segment id for public-safe source recall diagnostics.",
    )
    dual_candidates.add_argument(
        "--disable-graph",
        action="store_true",
        help="Skip Graph DB traversal and return Meilisearch-only candidates.",
    )
    dual_candidates.add_argument(
        "--graph-aware-rerank",
        action="store_true",
        help="Rerank the union with deterministic graph relation and object-evidence signals.",
    )
    dual_candidates.add_argument("--output", type=Path, help="Optional JSON output path.")
    dual_candidates.set_defaults(func=cmd_query_project_dual_candidates)

    ask_project_parser = subparsers.add_parser(
        "ask-project",
        help="Search a local project and compose a grounded answer with citations",
    )
    ask_project_parser.add_argument("--index", required=True)
    ask_project_parser.add_argument(
        "--index-kind",
        choices=["segment", "window"],
        default="segment",
        help="Interpret --index hits as segment documents or window-level documents.",
    )
    ask_project_parser.add_argument(
        "--visual-index",
        help="Optional Meilisearch index containing visual_entities documents.",
    )
    location = ask_project_parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    ask_project_parser.add_argument("--query", required=True)
    ask_project_parser.add_argument("--limit", type=int, default=5)
    ask_project_parser.add_argument(
        "--candidate-pool-limit",
        type=int,
        help=(
            "Retrieve this many candidates per search channel before final limiting. "
            "Useful with --rerank to let the reranker see deeper candidates."
        ),
    )
    ask_project_parser.add_argument(
        "--segments",
        type=Path,
        help="Optional segment JSONL path. Relative paths are resolved from project dir.",
    )
    ask_project_parser.add_argument(
        "--frames-manifest",
        type=Path,
        help="Optional frames manifest JSONL path. Relative paths are resolved from project dir.",
    )
    ask_project_parser.add_argument(
        "--visual-entities",
        type=Path,
        help="Optional visual_entities JSONL path. Relative paths are resolved from project dir.",
    )
    ask_project_parser.add_argument(
        "--entity-links",
        type=Path,
        help="Optional entity_links JSONL path. Relative paths are resolved from project dir.",
    )
    ask_project_parser.add_argument(
        "--domain-lexicon",
        type=Path,
        help="Optional domain_lexicon.json path. Relative paths are resolved from project dir.",
    )
    ask_project_parser.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around the matched hit.",
    )
    ask_project_parser.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    ask_project_parser.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    ask_project_parser.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    ask_project_parser.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before the target when using time-window mode.",
    )
    ask_project_parser.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after the target when using time-window mode.",
    )
    ask_project_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Reorder evidence bundles with deterministic domain-agnostic evidence signals.",
    )
    ask_project_parser.add_argument(
        "--rerank-time-hint",
        help="Optional timestamp hint for reranking, for example '10-14s'.",
    )
    ask_project_parser.add_argument(
        "--rerank-backend",
        choices=RERANK_BACKEND_CHOICES,
        default=DEFAULT_RERANK_BACKEND,
        help="Private-safe reranker backend used by --rerank.",
    )
    ask_project_parser.add_argument(
        "--hybrid-retrieval",
        action="store_true",
        help=(
            "Also query a semantic Meilisearch channel and merge it with lexical "
            "candidate hits."
        ),
    )
    ask_project_parser.add_argument(
        "--hybrid-embedder",
        default=DEFAULT_HYBRID_EMBEDDER,
        help="Meilisearch embedder name used by --hybrid-retrieval.",
    )
    ask_project_parser.add_argument(
        "--hybrid-semantic-ratio",
        type=float,
        default=DEFAULT_HYBRID_SEMANTIC_RATIO,
        help="Semantic ratio for the semantic channel used by --hybrid-retrieval.",
    )
    ask_project_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        dest="output_format",
        help="Output format for the composed answer response.",
    )
    ask_project_parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    ask_project_parser.set_defaults(func=cmd_ask_project)

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
    evaluate.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=0,
        help="Write sanitized per-candidate retrieval diagnostics for the top K hits.",
    )
    evaluate.add_argument(
        "--allow-private-output",
        action="store_true",
        help="Allow raw private fields such as questions and transcript excerpts in eval output.",
    )
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
    benchmark.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=None,
        help="Write sanitized per-candidate diagnostics for EduVidQA benchmark suites.",
    )
    benchmark.set_defaults(func=cmd_benchmark_retrieval)

    report_eval = subparsers.add_parser(
        "report-evaluation",
        help="Generate paper tables and a reproducibility report from benchmark metrics",
    )
    report_eval.add_argument("--metrics", required=True, type=Path)
    report_eval.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to paper_report next to metrics.json.",
    )
    report_eval.set_defaults(func=cmd_report_evaluation)

    metric_intervals = subparsers.add_parser(
        "paper-metric-intervals",
        help="Generate aggregate bootstrap CIs and paired deltas from query_results.jsonl",
    )
    add_paper_metric_intervals_arguments(metric_intervals)
    metric_intervals.set_defaults(func=cmd_paper_metric_intervals)

    retrieval_gate = subparsers.add_parser(
        "check-retrieval-gate",
        help="Check aggregate retrieval benchmark metrics against a public fixture quality gate",
    )
    retrieval_gate.add_argument("--metrics", required=True, type=Path)
    retrieval_gate.add_argument("--config", required=True, type=Path)
    retrieval_gate.set_defaults(func=cmd_check_retrieval_gate)

    paper_experiment = subparsers.add_parser(
        "run-paper-experiment",
        help="Run benchmark, paper report generation, optional gate, and experiment manifest",
    )
    paper_experiment.add_argument("--manifest", required=True, type=Path)
    paper_experiment.add_argument("--output-dir", required=True, type=Path)
    paper_experiment.add_argument(
        "--run-id",
        help="Optional run ID override. Defaults to the manifest run_id or a timestamp.",
    )
    paper_experiment.add_argument(
        "--gate-config",
        type=Path,
        help="Optional retrieval quality gate config JSON.",
    )
    paper_experiment.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=None,
        help="Write sanitized per-candidate diagnostics for EduVidQA benchmark suites.",
    )
    paper_experiment.add_argument(
        "--no-fail-on-gate",
        action="store_true",
        help="Write gate results but keep the command successful when thresholds fail.",
    )
    paper_experiment.set_defaults(func=cmd_run_paper_experiment)

    paper_bundle = subparsers.add_parser(
        "run-paper-bundle",
        help="Run the full paper experiment, robustness, readiness, claims, and registry bundle",
    )
    paper_bundle.add_argument("--manifest", required=True, type=Path)
    paper_bundle.add_argument("--output-dir", required=True, type=Path)
    paper_bundle.add_argument(
        "--run-id",
        help="Optional run ID override. Defaults to the manifest run_id or a timestamp.",
    )
    paper_bundle.add_argument(
        "--gate-config",
        type=Path,
        help="Optional retrieval quality gate config JSON.",
    )
    paper_bundle.add_argument(
        "--baseline-variant-id",
        help="Baseline variant_id or mode for paired variant-minus-baseline deltas.",
    )
    paper_bundle.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric name to summarize. Repeat or pass comma-separated values.",
    )
    paper_bundle.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed.",
    )
    paper_bundle.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="Bootstrap resample count.",
    )
    paper_bundle.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Bootstrap percentile interval confidence level.",
    )
    paper_bundle.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=None,
        help="Write sanitized per-candidate diagnostics for EduVidQA benchmark suites.",
    )
    paper_bundle.add_argument(
        "--no-fail-on-gate",
        action="store_true",
        help="Write gate results and continue bundle generation when thresholds fail.",
    )
    paper_bundle.set_defaults(func=cmd_run_paper_bundle)

    paper_readiness = subparsers.add_parser(
        "audit-paper-readiness",
        help="Generate a private-safe paper readiness checklist and gap report",
    )
    paper_readiness.add_argument("--experiment-manifest", required=True, type=Path)
    paper_readiness.add_argument("--output-dir", required=True, type=Path)
    paper_readiness.add_argument(
        "--metrics",
        type=Path,
        help="Optional metrics.json override. Defaults to the experiment manifest artifact.",
    )
    paper_readiness.add_argument(
        "--query-results",
        type=Path,
        help="Optional query_results.jsonl override for semantic evidence checks.",
    )
    paper_readiness.add_argument(
        "--quality-gate-result",
        type=Path,
        help="Optional quality_gate_result.json override.",
    )
    paper_readiness.add_argument(
        "--reproducibility",
        type=Path,
        help="Optional reproducibility.json override.",
    )
    paper_readiness.add_argument(
        "--semantic-smoke",
        type=Path,
        help="Optional sanitized semantic live-smoke summary JSON.",
    )
    paper_readiness.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Exit non-zero after writing outputs when readiness gaps are present.",
    )
    paper_readiness.set_defaults(func=cmd_audit_paper_readiness)

    paper_claims = subparsers.add_parser(
        "build-paper-claims",
        help="Build a private-safe claim/evidence matrix from paper experiment artifacts",
    )
    paper_claims.add_argument("--metrics", required=True, type=Path)
    paper_claims.add_argument("--reproducibility", required=True, type=Path)
    paper_claims.add_argument("--quality-gate-result", required=True, type=Path)
    paper_claims.add_argument("--readiness-audit", required=True, type=Path)
    paper_claims.add_argument(
        "--robustness",
        type=Path,
        help="Optional private-safe robustness summary artifact.",
    )
    paper_claims.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to paper_claims next to metrics.json.",
    )
    paper_claims.set_defaults(func=cmd_build_paper_claims)

    paper_registry = subparsers.add_parser(
        "build-paper-registry",
        help="Build a private-safe registry of paper experiment artifact filenames and statuses",
    )
    paper_registry.add_argument("--experiment-manifest", required=True, type=Path)
    paper_registry.add_argument("--readiness-audit", required=True, type=Path)
    paper_registry.add_argument("--claim-matrix", required=True, type=Path)
    paper_registry.add_argument(
        "--quality-gate-result",
        type=Path,
        help="Optional quality gate result override. Defaults to the experiment manifest artifact.",
    )
    paper_registry.add_argument(
        "--robustness",
        type=Path,
        help="Optional private-safe robustness summary artifact.",
    )
    paper_registry.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path. When omitted, prints the registry only.",
    )
    paper_registry.set_defaults(func=cmd_build_paper_registry)

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

    evidence_unit_smoke = subparsers.add_parser(
        "evidence-unit-smoke",
        help="Build, optionally index, and query evidence_units with public-safe smoke output",
    )
    evidence_unit_smoke.add_argument("--manifest", required=True, type=Path)
    evidence_unit_smoke.add_argument(
        "--output-dir",
        type=Path,
        help="Optional public output directory. Writes sanitized metrics.json, query_results.jsonl, and summary.md.",
    )
    evidence_unit_smoke.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest/output wiring without building artifacts or contacting Meilisearch.",
    )
    evidence_unit_smoke.add_argument(
        "--quality-rerank",
        action="store_true",
        help="Add smoke-only deterministic quality-aware rerank diagnostics to public outputs.",
    )
    evidence_unit_smoke.add_argument(
        "--modality-aware-rerank",
        action="store_true",
        help=(
            "Opt into evidence-unit query-path modality-aware rerank. This is distinct "
            "from --quality-rerank, which remains smoke-only diagnostics."
        ),
    )
    evidence_unit_smoke.set_defaults(func=cmd_evidence_unit_smoke)

    return parser


def build_paper_metric_intervals_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-metric-intervals",
        description="Generate aggregate bootstrap CIs and paired deltas from query_results.jsonl",
    )
    add_paper_metric_intervals_arguments(parser)
    parser.set_defaults(func=cmd_paper_metric_intervals)
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


def add_paper_metric_intervals_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-results", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        "--metrics-path",
        dest="metrics_path",
        type=Path,
        help="Optional metrics.json path used for run metadata.",
    )
    parser.add_argument(
        "--report",
        "--report-path",
        dest="report_path",
        type=Path,
        help="Optional paper report artifact path recorded by artifact name only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to paper_metric_intervals next to query_results.jsonl.",
    )
    parser.add_argument(
        "--baseline-variant-id",
        help="Baseline variant_id or mode for paired variant-minus-baseline deltas.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric name to summarize. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="Bootstrap resample count.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Bootstrap percentile interval confidence level.",
    )


def add_vlm_alignment_arguments(parser: argparse.ArgumentParser) -> None:
    candidate_defaults = VLMFrameCandidateConfig()
    consistency_defaults = AudioVisualConsistencyConfig()
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    parser.add_argument(
        "--vlm-backend",
        choices=available_vlm_backends(),
        default=DEFAULT_VLM_BACKEND,
        help=(
            "VLM parser backend. deterministic/mock are dependency-free, command "
            "shells out to an external parser, and jsonl replays fixture observations."
        ),
    )
    parser.add_argument(
        "--vlm-model",
        required=True,
        help="VLM model identifier recorded as source_model in VLM artifacts.",
    )
    parser.add_argument("--vlm-device", help="Optional device hint recorded in run metadata.")
    parser.add_argument(
        "--vlm-options",
        help=(
            "Backend options as JSON or key=value pairs. command uses command/input_mode; "
            "jsonl uses jsonl_path."
        ),
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


def _add_hybrid_embedder_index_options(parser: argparse.ArgumentParser) -> None:
    hybrid = parser.add_mutually_exclusive_group()
    hybrid.add_argument(
        "--hybrid-embedder-profile",
        choices=hybrid_embedder_settings_profile_names(),
        help=(
            "Opt in to a built-in Meilisearch hybrid/vector embedder settings profile. "
            "Default indexing leaves embedders unchanged."
        ),
    )
    hybrid.add_argument(
        "--hybrid-embedder-config",
        type=Path,
        help=(
            "JSON file containing a Meilisearch embedders object or an object with "
            "'embedders'. Secrets are sent to Meilisearch but redacted from summaries."
        ),
    )
    parser.add_argument(
        "--hybrid-embedder-name",
        default=DEFAULT_HYBRID_EMBEDDER,
        help="Embedder name for --hybrid-embedder-profile.",
    )
    parser.add_argument(
        "--hybrid-embedder-dimensions",
        type=int,
        help="Override vector dimensions for the built-in userProvided profile.",
    )
    parser.add_argument(
        "--hybrid-embedder-live-smoke",
        action="store_true",
        help="After applying hybrid embedder settings, read them back from Meilisearch.",
    )


def _hybrid_embedder_config_from_args(args: argparse.Namespace) -> dict | None:
    if args.hybrid_embedder_config is None:
        return None
    return load_hybrid_embedder_settings(args.hybrid_embedder_config)


def client_from_args(args: argparse.Namespace) -> MeiliClient:
    return MeiliClient(base_url=args.url, api_key=args.api_key)


def cmd_health(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))


def cmd_health_neo4j(args: argparse.Namespace) -> None:
    print(json.dumps(check_neo4j_health(), ensure_ascii=False, indent=2))


def cmd_build_embedding_vectors(args: argparse.Namespace) -> None:
    provider = _embedding_provider_from_args(args)
    cache = JsonlEmbeddingCache(args.cache)
    records = load_input_records(args.input, input_format=args.input_format)
    summary = build_vector_manifest(
        records=records,
        provider=provider,
        output_path=args.output,
        id_field=args.id_field,
        text_fields=args.text_fields,
        kind=args.kind,
        embedder=args.embedder_name,
        batch_size=args.batch_size,
        cache=cache,
        fail_on_empty_text=not args.allow_empty_text,
        source_label=args.source_label,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _embedding_provider_from_args(args: argparse.Namespace):
    if args.provider == "deterministic-fixture":
        return DeterministicFixtureEmbeddingProvider(
            model=args.model or "deterministic_fixture_v1",
            dimensions=args.dimensions or 8,
        )
    if args.provider == "sentence-transformers":
        return SentenceTransformersEmbeddingProvider(
            model=args.model or DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
            dimensions=args.dimensions,
            local_files_only=args.local_files_only,
        )
    return OpenAICompatibleEmbeddingProvider.from_env(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        dimensions=args.dimensions,
    )


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


def cmd_concept_graph_ingest(args: argparse.Namespace) -> None:
    project_dir = None
    project_id = args.project_id
    if args.project_id is not None or args.project_dir is not None:
        project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = ingest_concept_graph(
        project_dir=project_dir,
        concept_graph_path=args.concept_graph,
        project_id=project_id,
        create_schema=not args.skip_schema,
        dry_run=args.dry_run,
        skip_unavailable_runtime=not args.fail_on_unavailable_runtime,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_merge_cross_lecture_concepts(args: argparse.Namespace) -> None:
    summary = merge_cross_lecture_concepts(
        concept_graph_paths=args.concept_graphs,
        output_path=args.output,
        summary_path=args.summary,
        project_id=args.project_id,
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
        hybrid_embedder_profile=args.hybrid_embedder_profile,
        hybrid_embedder_config=_hybrid_embedder_config_from_args(args),
        hybrid_embedder_name=args.hybrid_embedder_name,
        hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
        hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
        vector_manifest=args.vector_manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_project_windows(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = build_project_windows(
        project_dir=project_dir,
        output_path=args.output,
        segments=args.segments,
        frames_manifest=args.frames_manifest,
        visual_entities=args.visual_entities,
        manifest_path=args.manifest,
        window_seconds=args.window_seconds,
        neighbor_count=args.neighbor_count,
        previous_neighbor_count=args.previous_neighbor_count,
        next_neighbor_count=args.next_neighbor_count,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_project_visual_states(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = build_project_visual_states(
        project_dir=project_dir,
        output_path=args.output,
        frames_manifest=args.frames_manifest,
        manifest_path=args.manifest,
        state_padding_seconds=args.state_padding_seconds,
        min_visual_states=args.min_visual_states,
        fail_on_visual_state_gate=args.fail_on_visual_state_gate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_project_evidence_units(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = build_project_evidence_units(
        project_dir=project_dir,
        output_path=args.output,
        segments=args.segments,
        frames_manifest=args.frames_manifest,
        visual_states=args.visual_states,
        visual_states_output=args.visual_states_output,
        visual_entities=args.visual_entities,
        entity_links=args.entity_links,
        concept_graph=args.concept_graph,
        manifest_path=args.manifest,
        window_seconds=args.window_seconds,
        neighbor_count=args.neighbor_count,
        previous_neighbor_count=args.previous_neighbor_count,
        next_neighbor_count=args.next_neighbor_count,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
        state_padding_seconds=args.state_padding_seconds,
        visual_state_min_coverage_ratio=args.visual_state_min_coverage_ratio,
        visual_state_min_total=args.visual_state_min_total,
        fail_on_visual_state_gate=args.fail_on_visual_state_gate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_extract_concept_candidates(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = extract_project_concept_candidates(
        project_dir=project_dir,
        evidence_units=args.evidence_units,
        output_path=args.output,
        manifest_path=args.manifest,
        lecture_id=args.lecture_id,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_index_project_evidence_units(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = index_project_evidence_units(
        client,
        index_uid=args.index,
        project_dir=project_dir,
        batch_size=args.batch_size,
        reset=args.reset,
        evidence_units=args.evidence_units,
        settings_profile=args.settings_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_index_project_windows(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = index_project_windows(
        client,
        index_uid=args.index,
        project_dir=project_dir,
        batch_size=args.batch_size,
        reset=args.reset,
        windows=args.windows,
        segments=args.segments,
        frames_manifest=args.frames_manifest,
        visual_entities=args.visual_entities,
        settings_profile=args.settings_profile,
        window_seconds=args.window_seconds,
        neighbor_count=args.neighbor_count,
        previous_neighbor_count=args.previous_neighbor_count,
        next_neighbor_count=args.next_neighbor_count,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
        hybrid_embedder_profile=args.hybrid_embedder_profile,
        hybrid_embedder_config=_hybrid_embedder_config_from_args(args),
        hybrid_embedder_name=args.hybrid_embedder_name,
        hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
        hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
        vector_manifest=args.vector_manifest,
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
        hybrid_embedder_profile=args.hybrid_embedder_profile,
        hybrid_embedder_config=_hybrid_embedder_config_from_args(args),
        hybrid_embedder_name=args.hybrid_embedder_name,
        hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
        hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
        vector_manifest=args.vector_manifest,
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
            max_frame_gap_seconds=args.max_frame_gap_seconds,
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
            max_frame_gap_seconds=args.max_frame_gap_seconds,
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


def cmd_validate_vlm_evidence(args: argparse.Namespace) -> None:
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    summary = validate_vlm_object_evidence(
        project_dir=project_dir,
        visual_entities_path=args.visual_entities,
        evidence_units_path=args.evidence_units,
        vlm_observations_path=args.vlm_observations,
        vlm_jsonl_path=args.vlm_jsonl,
        output_path=args.output,
        summary_path=args.summary,
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
        retrieval_index_kind=args.index_kind,
        visual_index_uid=args.visual_index,
        project_dir=project_dir,
        query=args.query,
        limit=args.limit,
        candidate_pool_limit=args.candidate_pool_limit,
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
        rerank_backend=args.rerank_backend,
        hybrid_retrieval=args.hybrid_retrieval,
        hybrid_embedder=args.hybrid_embedder,
        hybrid_semantic_ratio=args.hybrid_semantic_ratio,
        hybrid_query_vector_embedder=args.hybrid_query_vector_embedder,
        hybrid_query_vector_name=args.hybrid_query_vector_name,
        hybrid_query_vector_dimensions=args.hybrid_query_vector_dimensions,
        hybrid_query_vector_manifest_path=args.hybrid_query_vector_manifest,
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    for line in response.get("summary_lines", []):
        print(line, file=sys.stderr)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_query_project_evidence_units(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    response = query_project_evidence_units(
        client=client,
        index_uid=args.index,
        project_dir=project_dir,
        query=args.query,
        limit=args.limit,
        evidence_units=args.evidence_units,
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_query_project_dual_candidates(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    response = query_project_dual_candidates(
        client=client,
        index_uid=args.index,
        project_dir=project_dir,
        query=args.query,
        limit=args.limit,
        candidate_pool_limit=args.candidate_pool_limit,
        graph_limit=args.graph_limit,
        evidence_units=args.evidence_units,
        target_evidence_unit_ids=args.target_evidence_unit_ids,
        target_segment_ids=args.target_segment_ids,
        enable_graph=not args.disable_graph,
        graph_aware_rerank=args.graph_aware_rerank,
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_ask_project(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    project_dir = project_dir_from_args(project_id=args.project_id, project_dir=args.project_dir)
    response = ask_project(
        client=client,
        index_uid=args.index,
        retrieval_index_kind=args.index_kind,
        visual_index_uid=args.visual_index,
        project_dir=project_dir,
        query=args.query,
        limit=args.limit,
        candidate_pool_limit=args.candidate_pool_limit,
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
        rerank_backend=args.rerank_backend,
        hybrid_retrieval=args.hybrid_retrieval,
        hybrid_embedder=args.hybrid_embedder,
        hybrid_semantic_ratio=args.hybrid_semantic_ratio,
    )
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        write_json(output_path, response)
    if args.output_format == "text":
        print(format_answer_text(response["answer"]), end="")
        return
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
    diagnostic_top_k = max(int(getattr(args, "diagnostic_top_k", 0) or 0), 0)
    allow_private_output = bool(getattr(args, "allow_private_output", False))
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
                row = {
                    "sample_id": record.sample_id,
                    "video_name": record.video_name,
                    "timestamp_points": record.timestamp_points,
                    "best_abs_error": query_eval.best_abs_error,
                    "hit_by_delta": query_eval.hit_by_delta,
                    "topk_recall": query_eval.recall_by_k,
                }
                if diagnostic_top_k > 0:
                    row["diagnostic_candidates"] = candidate_diagnostics(
                        record,
                        candidates,
                        top_k=diagnostic_top_k,
                        include_private_fields=allow_private_output,
                    )
                if allow_private_output:
                    row["question"] = record.question
                    row["topk_candidates"] = [candidate.to_dict() for candidate in candidates]
                output_handle.write(
                    json.dumps(
                        row,
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
        diagnostic_top_k=args.diagnostic_top_k,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "output_dir": str(run.output_dir),
                "metrics": str(run.metrics_path),
                "metrics_csv": str(run.metrics_csv_path),
                "query_results": str(run.query_results_path),
                "semantic_smoke": str(run.semantic_smoke_path),
                "summary": str(run.summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_report_evaluation(args: argparse.Namespace) -> None:
    report = generate_evaluation_report(
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        repo_root=default_paths().repo_root,
        command=["oarag", *sys.argv[1:]],
    )
    print(
        json.dumps(
            {
                "output_dir": str(report.output_dir),
                "paper_table_csv": str(report.paper_table_csv_path),
                "paper_table_markdown": str(report.paper_table_markdown_path),
                "reproducibility_json": str(report.reproducibility_json_path),
                "reproducibility_markdown": str(report.reproducibility_markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_paper_metric_intervals(args: argparse.Namespace) -> None:
    run = generate_metric_intervals(
        query_results_path=args.query_results,
        output_dir=args.output_dir,
        metrics_path=args.metrics_path,
        report_path=args.report_path,
        baseline_variant_id=args.baseline_variant_id,
        metrics=args.metrics,
        seed=args.seed,
        sample_count=args.sample_count,
        confidence_level=args.confidence_level,
    )
    print(
        json.dumps(
            {
                "run_id": run.payload.get("run_id"),
                "output_dir": str(run.output_dir),
                "json": str(run.json_path),
                "csv": str(run.csv_path),
                "markdown": str(run.markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_check_retrieval_gate(args: argparse.Namespace) -> None:
    result = check_retrieval_quality_gate(metrics_path=args.metrics, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def cmd_run_paper_experiment(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    run = run_paper_experiment(
        client=client,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        gate_config_path=args.gate_config,
        repo_root=default_paths().repo_root,
        diagnostic_top_k=args.diagnostic_top_k,
        fail_on_gate=not args.no_fail_on_gate,
        command=["oarag", *sys.argv[1:]],
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "output_dir": str(run.output_dir),
                "metrics": str(run.benchmark.metrics_path),
                "metrics_summary": str(run.benchmark.metrics_csv_path),
                "query_results": str(run.benchmark.query_results_path),
                "semantic_smoke": str(run.benchmark.semantic_smoke_path),
                "summary": str(run.benchmark.summary_path),
                "paper_report": str(run.report.output_dir),
                "quality_gate_result": str(run.quality_gate_result_path),
                "experiment_manifest": str(run.experiment_manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_run_paper_bundle(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    run = run_paper_bundle(
        client=client,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        gate_config_path=args.gate_config,
        baseline_variant_id=args.baseline_variant_id,
        metrics=args.metrics,
        seed=args.seed,
        sample_count=args.sample_count,
        confidence_level=args.confidence_level,
        repo_root=default_paths().repo_root,
        diagnostic_top_k=args.diagnostic_top_k,
        fail_on_gate=not args.no_fail_on_gate,
        command=["oarag", *sys.argv[1:]],
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "output_dir": str(run.output_dir),
                "paper_bundle_result": str(run.result_path),
                "experiment_manifest": str(run.experiment.experiment_manifest_path),
                "metric_intervals": str(run.metric_intervals.json_path),
                "readiness_audit": str(run.readiness.json_path),
                "claim_matrix": str(run.claims.json_path),
                "artifact_registry": str(run.registry.json_path),
                "status": run.registry.payload.get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_audit_paper_readiness(args: argparse.Namespace) -> None:
    audit = audit_paper_readiness(
        experiment_manifest_path=args.experiment_manifest,
        output_dir=args.output_dir,
        metrics_path=args.metrics,
        query_results_path=args.query_results,
        quality_gate_result_path=args.quality_gate_result,
        reproducibility_path=args.reproducibility,
        semantic_smoke_path=args.semantic_smoke,
    )
    print(
        json.dumps(
            {
                "run_id": audit.payload.get("run_id"),
                "ready": audit.payload.get("ready"),
                "gap_count": audit.payload.get("gap_count"),
                "json": str(audit.json_path) if audit.json_path is not None else None,
                "markdown": str(audit.markdown_path) if audit.markdown_path is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.fail_on_gap and not audit.payload.get("ready"):
        raise SystemExit(1)


def cmd_build_paper_claims(args: argparse.Namespace) -> None:
    run = build_paper_claims(
        metrics_path=args.metrics,
        reproducibility_path=args.reproducibility,
        quality_gate_result_path=args.quality_gate_result,
        readiness_audit_path=args.readiness_audit,
        robustness_path=args.robustness,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "run_id": run.payload.get("run_id"),
                "overall_status": (run.payload.get("summary") or {}).get("overall_status"),
                "json": str(run.json_path),
                "markdown": str(run.markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_build_paper_registry(args: argparse.Namespace) -> None:
    run = build_paper_artifact_registry(
        experiment_manifest_path=args.experiment_manifest,
        readiness_audit_path=args.readiness_audit,
        claim_matrix_path=args.claim_matrix,
        quality_gate_result_path=args.quality_gate_result,
        robustness_path=args.robustness,
        output_path=args.output,
    )
    print(json.dumps(run.payload, ensure_ascii=False, indent=2))


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


def cmd_evidence_unit_smoke(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    run = run_evidence_unit_smoke(
        client=client,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        repo_root=default_paths().repo_root,
        dry_run=args.dry_run,
        quality_rerank=args.quality_rerank,
        modality_aware_rerank=args.modality_aware_rerank,
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
