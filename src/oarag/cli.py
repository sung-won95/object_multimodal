from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .alignment import align_segments_to_frames
from .config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL, ENV_STT_LANGUAGE, default_paths, env_default
from .eduvidqa import iter_lecture_segments, iter_records
from .eval import evaluate_query, summarize
from .entity_links import link_entities
from .evidence import build_evidence_response
from .ingest import (
    BatchIngestConfig,
    VideoIngestConfig,
    batch_ingest_videos,
    ingest_video,
    make_video_id,
    write_batch_summary_csv,
    write_batch_summary_json,
    write_batch_summary_jsonl,
)
from .io import write_json
from .meili import LECTURE_SEGMENT_SETTINGS, MeiliClient
from .project_index import index_project_segments, project_dir_from_args
from .schemas import SearchCandidate
from .stt import DEFAULT_MLX_WHISPER_MODEL
from .visual_entities import extract_visual_entities


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oarag", description="Object-aligned RAG pilot CLI")
    parser.add_argument("--url", default=DEFAULT_MEILI_URL, help="Meilisearch URL")
    parser.add_argument("--api-key", default=DEFAULT_MEILI_API_KEY, help="Meilisearch API key")
    subparsers = parser.add_subparsers(required=True)

    health = subparsers.add_parser("health", help="Check Meilisearch health")
    health.set_defaults(func=cmd_health)

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
    index_project.set_defaults(func=cmd_index_project)

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
    evidence.add_argument("--output", type=Path, help="Optional JSON output path.")
    evidence.set_defaults(func=cmd_evidence_window)

    extract_visual = subparsers.add_parser(
        "extract-visual-entities",
        help="Extract OCR-first visual entities from sampled frames",
    )
    location = extract_visual.add_mutually_exclusive_group(required=True)
    location.add_argument("--project-id", help="Project ID under artifacts/projects/")
    location.add_argument("--project-dir", type=Path, help="Project artifact directory")
    extract_visual.add_argument(
        "--backend",
        choices=["auto", "stub", "local-ocr"],
        default="auto",
        help="auto uses local OCR when available, otherwise stub.",
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
    entity_links.set_defaults(func=cmd_link_entities)

    query = subparsers.add_parser("query", help="Search one query")
    query.add_argument("--index", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=5)
    query.set_defaults(func=cmd_query)

    evaluate = subparsers.add_parser("eval-eduvidqa", help="Run timestamp proximity eval")
    evaluate.add_argument("--input", required=True, type=Path)
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--limit", type=int, default=5)
    evaluate.add_argument("--record-limit", type=int)
    evaluate.add_argument("--deltas", default="5,10,15")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(func=cmd_eval_eduvidqa)

    return parser


def client_from_args(args: argparse.Namespace) -> MeiliClient:
    return MeiliClient(base_url=args.url, api_key=args.api_key)


def cmd_health(args: argparse.Namespace) -> None:
    client = client_from_args(args)
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))


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


def parse_deltas(value: str) -> list[int]:
    deltas = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped:
            deltas.append(int(stripped))
    if not deltas:
        raise ValueError("At least one delta must be provided")
    return deltas
