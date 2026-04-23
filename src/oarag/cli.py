from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL, ENV_STT_LANGUAGE, default_paths, env_default
from .eduvidqa import iter_lecture_segments, iter_records
from .eval import evaluate_query, summarize
from .ingest import VideoIngestConfig, ingest_video, make_video_id
from .meili import LECTURE_SEGMENT_SETTINGS, MeiliClient
from .schemas import SearchCandidate
from .stt import DEFAULT_MLX_WHISPER_MODEL


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
