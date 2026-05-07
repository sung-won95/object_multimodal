from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oarag.ingestion.frame_selection import (
    FRAME_SELECTION_STRATEGIES,
    FrameSelectionConfig,
    select_representative_frames,
)
from oarag.core.io import write_json, write_jsonl
from oarag.core.schemas import LectureSegment, slugify
from oarag.core.srt import iter_srt_cues
from oarag.ingestion.stt import (
    DEFAULT_MLX_WHISPER_MODEL,
    extract_audio,
    stt_segments_from_mlx_result,
    transcribe_audio_with_mlx_whisper,
)


TRANSCRIPT_SOURCES = {"auto", "srt", "stt", "none"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
FRAME_SAMPLING_STRATEGIES = {"prefix", "uniform"}


@dataclass(frozen=True)
class VideoIngestConfig:
    video_path: Path
    project_id: str
    video_id: str
    output_root: Path
    srt_path: Path | None = None
    frame_rate: float = 1.0
    max_frames: int | None = 120
    frame_sampling: str = "uniform"
    frame_selection: str = "none"
    skip_frames: bool = False
    copy_source: bool = False
    transcript_source: str = "auto"
    stt_model: str = DEFAULT_MLX_WHISPER_MODEL
    stt_language: str | None = None
    stt_task: str = "transcribe"
    stt_word_timestamps: bool = False


@dataclass(frozen=True)
class BatchIngestConfig:
    root_dir: Path
    output_root: Path
    frame_rate: float = 1.0
    max_frames: int | None = 120
    frame_sampling: str = "uniform"
    frame_selection: str = "none"
    skip_frames: bool = False
    copy_source: bool = False
    transcript_source: str = "auto"
    stt_model: str = DEFAULT_MLX_WHISPER_MODEL
    stt_language: str | None = None
    stt_task: str = "transcribe"
    stt_word_timestamps: bool = False
    force: bool = False
    strict: bool = False
    dry_run: bool = False
    project_prefix: str | None = None


def ingest_video(config: VideoIngestConfig) -> dict[str, Any]:
    video_path = config.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if config.transcript_source not in TRANSCRIPT_SOURCES:
        raise ValueError(f"transcript_source must be one of {sorted(TRANSCRIPT_SOURCES)}")
    if config.frame_sampling not in FRAME_SAMPLING_STRATEGIES:
        raise ValueError(f"frame_sampling must be one of {sorted(FRAME_SAMPLING_STRATEGIES)}")
    if config.frame_selection not in FRAME_SELECTION_STRATEGIES:
        raise ValueError(f"frame_selection must be one of {sorted(FRAME_SELECTION_STRATEGIES)}")

    srt_path = resolve_srt(video_path, config.srt_path)
    project_dir = config.output_root / config.project_id
    source_dir = project_dir / "source"
    audio_dir = project_dir / "audio"
    frames_dir = project_dir / "frames"
    segments_dir = project_dir / "segments"
    manifests_dir = project_dir / "manifests"
    transcripts_dir = project_dir / "transcripts"

    source_dir.mkdir(parents=True, exist_ok=True)
    registered_source = register_source(video_path, source_dir, copy_source=config.copy_source)
    probe = probe_video(video_path)

    segments_path = segments_dir / "lecture_segments.jsonl"
    transcript_artifacts = write_transcript_segments(
        video_path=video_path,
        srt_path=srt_path,
        segments_path=segments_path,
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        config=config,
    )

    frame_count = 0
    frame_sampling_summary = _empty_frame_sampling_summary(
        frame_rate=config.frame_rate,
        max_frames=config.max_frames,
        frame_sampling=config.frame_sampling,
        duration_sec=probe.get("duration_sec"),
        skipped=config.skip_frames,
    )
    frame_selection_summary = _empty_frame_selection_summary(
        frame_selection=config.frame_selection,
        skipped=config.skip_frames,
    )
    frames_manifest_path = manifests_dir / "frames_manifest.jsonl"
    if config.skip_frames:
        write_jsonl(frames_manifest_path, [])
    else:
        frame_result = sample_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            frame_rate=config.frame_rate,
            max_frames=config.max_frames,
            duration_sec=probe.get("duration_sec"),
            frame_sampling=config.frame_sampling,
            frame_selection=config.frame_selection,
        )
        frame_count = len(frame_result["frames"])
        write_frames_manifest(
            frames_manifest_path,
            frame_result["frames"],
        )
        frame_sampling_summary = frame_result["summary"]
        frame_selection_summary = frame_result["selection_summary"]

    manifest = {
        "project_id": config.project_id,
        "video_id": config.video_id,
        "source_video_path": str(video_path),
        "registered_source_path": str(registered_source),
        "srt_path": str(srt_path) if srt_path else None,
        "project_dir": str(project_dir),
        "transcript_source": transcript_artifacts["source"],
        "transcript_config": {
            "stt_model": config.stt_model if transcript_artifacts["source"] == "stt" else None,
            "stt_language": config.stt_language if transcript_artifacts["source"] == "stt" else None,
            "stt_task": config.stt_task if transcript_artifacts["source"] == "stt" else None,
            "stt_word_timestamps": (
                config.stt_word_timestamps if transcript_artifacts["source"] == "stt" else None
            ),
        },
        "probe": probe,
        "frame_sampling": frame_sampling_summary,
        "frame_selection": frame_selection_summary,
        "artifacts": {
            "lecture_segments": str(segments_path),
            "audio": transcript_artifacts.get("audio_path"),
            "stt_raw_transcript": transcript_artifacts.get("raw_transcript_path"),
            "frames_dir": str(frames_dir),
            "frames_manifest": str(frames_manifest_path),
        },
        "counts": {
            "lecture_segments": transcript_artifacts["segment_count"],
            "frames": frame_count,
        },
    }
    write_json(manifests_dir / "project_manifest.json", manifest)
    return manifest


def write_transcript_segments(
    *,
    video_path: Path,
    srt_path: Path | None,
    segments_path: Path,
    audio_dir: Path,
    transcripts_dir: Path,
    config: VideoIngestConfig,
) -> dict[str, Any]:
    source = choose_transcript_source(config.transcript_source, srt_path)
    if source == "none":
        return {
            "source": source,
            "segment_count": write_jsonl(segments_path, []),
        }

    if source == "srt":
        if srt_path is None:
            raise FileNotFoundError(
                "No SRT file found. Use --transcript-source stt or provide --srt."
            )
        segment_count = write_jsonl(
            segments_path,
            [
                LectureSegment.from_local_transcript(
                    project_id=config.project_id,
                    video_id=config.video_id,
                    seq_no=cue.seq_no,
                    start_time=cue.start_time,
                    end_time=cue.end_time,
                    text=cue.text,
                    source="srt",
                ).to_dict()
                for cue in iter_srt_cues(srt_path)
            ],
        )
        return {"source": source, "segment_count": segment_count}

    audio_path = audio_dir / f"{slugify(config.video_id)}.wav"
    raw_transcript_path = transcripts_dir / "mlx_whisper_raw.json"
    extract_audio(video_path, audio_path)
    raw_result = transcribe_audio_with_mlx_whisper(
        audio_path,
        model=config.stt_model,
        language=config.stt_language,
        task=config.stt_task,
        word_timestamps=config.stt_word_timestamps,
    )
    stt_segments = stt_segments_from_mlx_result(raw_result)
    write_json(raw_transcript_path, raw_result)
    segment_count = write_jsonl(
        segments_path,
        [
            LectureSegment.from_local_transcript(
                project_id=config.project_id,
                video_id=config.video_id,
                seq_no=segment.seq_no,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                source="stt:mlx-whisper",
            ).to_dict()
            for segment in stt_segments
        ],
    )
    return {
        "source": source,
        "segment_count": segment_count,
        "audio_path": str(audio_path),
        "raw_transcript_path": str(raw_transcript_path),
    }


def choose_transcript_source(requested: str, srt_path: Path | None) -> str:
    if requested == "auto":
        return "srt" if srt_path is not None else "stt"
    return requested


def resolve_srt(video_path: Path, explicit_srt: Path | None) -> Path | None:
    if explicit_srt is not None:
        resolved = explicit_srt.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"SRT not found: {resolved}")
        return resolved
    sibling = video_path.with_suffix(".srt")
    return sibling if sibling.exists() else None


def register_source(video_path: Path, source_dir: Path, copy_source: bool) -> Path:
    target = source_dir / video_path.name
    if target.exists() or target.is_symlink():
        return target
    if copy_source:
        shutil.copy2(video_path, target)
    else:
        target.symlink_to(video_path)
    return target


def probe_video(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = run(command)
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    return {
        "duration_sec": _optional_float(fmt.get("duration")),
        "width": _optional_int(stream.get("width")),
        "height": _optional_int(stream.get("height")),
        "r_frame_rate": stream.get("r_frame_rate"),
    }


def sample_frames(
    video_path: Path,
    frames_dir: Path,
    frame_rate: float,
    max_frames: int | None,
    *,
    duration_sec: float | None = None,
    frame_sampling: str = "uniform",
    frame_selection: str = "none",
) -> dict[str, Any]:
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than 0")
    if frame_sampling not in FRAME_SAMPLING_STRATEGIES:
        raise ValueError(f"frame_sampling must be one of {sorted(FRAME_SAMPLING_STRATEGIES)}")
    if frame_selection not in FRAME_SELECTION_STRATEGIES:
        raise ValueError(f"frame_selection must be one of {sorted(FRAME_SELECTION_STRATEGIES)}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    for existing in frames_dir.glob("frame_*.jpg"):
        existing.unlink()

    selected_timestamps = select_frame_timestamps(
        duration_sec=duration_sec,
        frame_rate=frame_rate,
        max_frames=max_frames,
        frame_sampling=frame_sampling,
    )

    if selected_timestamps is None:
        output_pattern = frames_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={frame_rate}",
        ]
        command.append(str(output_pattern))
        run(command)
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        frames = [
            {
                "frame_id": path.stem,
                "frame_path": str(path),
                "timestamp": round(index / frame_rate, 3),
            }
            for index, path in enumerate(frame_paths)
        ]
    elif frame_sampling == "prefix":
        output_pattern = frames_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={frame_rate}",
            "-frames:v",
            str(len(selected_timestamps)),
            str(output_pattern),
        ]
        run(command)
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        frames = [
            {
                "frame_id": path.stem,
                "frame_path": str(path),
                "timestamp": selected_timestamps[index],
            }
            for index, path in enumerate(frame_paths)
            if index < len(selected_timestamps)
        ]
    else:
        frames = []
        for index, timestamp in enumerate(selected_timestamps, start=1):
            frame_path = frames_dir / f"frame_{index:06d}.jpg"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(frame_path),
            ]
            run(command)
            frames.append(
                {
                    "frame_id": f"frame_{index:06d}",
                    "frame_path": str(frame_path),
                    "timestamp": timestamp,
                }
            )

    selection_result = select_representative_frames(
        frames,
        config=FrameSelectionConfig(strategy=frame_selection),
    )
    frames = selection_result["frames"]
    selection_summary = dict(selection_result["summary"])
    selection_summary["skipped"] = False
    for dropped_frame_path in selection_result["dropped_frame_paths"]:
        Path(dropped_frame_path).unlink(missing_ok=True)

    return {
        "frames": frames,
        "summary": summarize_frame_sampling(
            frames=frames,
            duration_sec=duration_sec,
            frame_rate=frame_rate,
            max_frames=max_frames,
            frame_sampling=frame_sampling,
        ),
        "selection_summary": selection_summary,
    }


def select_frame_timestamps(
    *,
    duration_sec: float | None,
    frame_rate: float,
    max_frames: int | None,
    frame_sampling: str,
) -> list[float] | None:
    if max_frames is None:
        return None
    if max_frames <= 0:
        return None
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than 0")
    if duration_sec is None or duration_sec <= 0:
        return [round(index / frame_rate, 3) for index in range(max_frames)]

    candidate_count = max(1, math.ceil(duration_sec * frame_rate))
    candidate_timestamps = [round(index / frame_rate, 3) for index in range(candidate_count)]
    if len(candidate_timestamps) <= max_frames or frame_sampling == "prefix":
        return candidate_timestamps[:max_frames]

    if max_frames == 1:
        return [candidate_timestamps[0]]

    max_index = len(candidate_timestamps) - 1
    selected_indices = [
        round(position * max_index / (max_frames - 1)) for position in range(max_frames)
    ]
    deduped_indices = sorted(dict.fromkeys(int(index) for index in selected_indices))
    return [candidate_timestamps[index] for index in deduped_indices]


def write_frames_manifest(
    path: Path,
    frames: list[dict[str, Any]],
) -> None:
    write_jsonl(path, frames)


def summarize_frame_sampling(
    *,
    frames: list[dict[str, Any]],
    duration_sec: float | None,
    frame_rate: float,
    max_frames: int | None,
    frame_sampling: str,
) -> dict[str, Any]:
    timestamps = [
        timestamp
        for timestamp in (_optional_float(frame.get("timestamp")) for frame in frames)
        if timestamp is not None
    ]
    first_timestamp = min(timestamps) if timestamps else None
    last_timestamp = max(timestamps) if timestamps else None
    timestamp_span = (
        round(last_timestamp - first_timestamp, 3)
        if first_timestamp is not None and last_timestamp is not None
        else None
    )
    covered_until = None
    coverage_ratio = None
    if duration_sec is not None and duration_sec > 0 and last_timestamp is not None:
        covered_until = round(min(duration_sec, last_timestamp + (1 / frame_rate)), 3)
        coverage_ratio = round(min(1.0, covered_until / duration_sec), 4)
    candidate_count = (
        math.ceil(duration_sec * frame_rate) if duration_sec and duration_sec > 0 else None
    )
    cap = max_frames if max_frames is not None and max_frames > 0 else None
    return {
        "strategy": frame_sampling,
        "frame_rate": frame_rate,
        "max_frames": cap,
        "duration_sec": duration_sec,
        "candidate_frame_count": candidate_count,
        "selected_frame_count": len(frames),
        "capped": bool(cap is not None and candidate_count is not None and candidate_count > cap),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "covered_until_sec": covered_until,
        "timestamp_span_sec": timestamp_span,
        "temporal_coverage_ratio": coverage_ratio,
        "skipped": False,
    }


def _empty_frame_sampling_summary(
    *,
    frame_rate: float,
    max_frames: int | None,
    frame_sampling: str,
    duration_sec: float | None,
    skipped: bool,
) -> dict[str, Any]:
    summary = summarize_frame_sampling(
        frames=[],
        duration_sec=duration_sec,
        frame_rate=frame_rate,
        max_frames=max_frames,
        frame_sampling=frame_sampling,
    )
    summary["skipped"] = skipped
    return summary


def _empty_frame_selection_summary(*, frame_selection: str, skipped: bool) -> dict[str, Any]:
    summary = select_representative_frames(
        [],
        config=FrameSelectionConfig(strategy=frame_selection),
    )["summary"]
    summary["skipped"] = skipped
    if skipped:
        summary["status"] = "skipped"
    return summary


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{stderr}") from exc


def make_video_id(video_path: Path) -> str:
    return slugify(video_path.stem)


def discover_video_files(root_dir: Path, *, extensions: set[str] | None = None) -> list[Path]:
    resolved_root = root_dir.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Root directory not found: {resolved_root}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {resolved_root}")

    allowed = {ext.lower() for ext in (extensions or VIDEO_EXTENSIONS)}
    return sorted(
        path
        for path in resolved_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def make_project_id_for_video(
    video_path: Path,
    *,
    root_dir: Path,
    prefix: str | None = None,
) -> str:
    resolved_video = video_path.expanduser().resolve()
    resolved_root = root_dir.expanduser().resolve()
    relative_path = resolved_video.relative_to(resolved_root)
    relative_no_suffix = relative_path.with_suffix("")
    normalized = relative_no_suffix.as_posix()
    digest = hashlib.blake2s(normalized.encode("utf-8"), digest_size=4).hexdigest()

    parts = [slugify(part) for part in relative_no_suffix.parts]
    base = "__".join(part for part in parts if part) or "video"
    if prefix:
        prefix_slug = slugify(prefix)
        if prefix_slug:
            base = f"{prefix_slug}__{base}"
    return f"{base}_{digest}"


def batch_ingest_videos(
    config: BatchIngestConfig,
    *,
    ingest_fn: Any = ingest_video,
) -> dict[str, Any]:
    resolved_root = config.root_dir.expanduser().resolve()
    resolved_output_root = config.output_root.expanduser().resolve()
    videos = discover_video_files(resolved_root)

    results: list[dict[str, Any]] = []
    counts = {"discovered": len(videos), "ingested": 0, "skipped_existing": 0, "failed": 0}
    aborted = False

    for video_path in videos:
        project_id = make_project_id_for_video(
            video_path,
            root_dir=resolved_root,
            prefix=config.project_prefix,
        )
        video_id = make_video_id(video_path)
        project_dir = resolved_output_root / project_id
        manifest_path = project_dir / "manifests" / "project_manifest.json"

        result = {
            "video_path": str(video_path),
            "project_id": project_id,
            "video_id": video_id,
            "project_dir": str(project_dir),
            "manifest_path": str(manifest_path),
        }

        if manifest_path.exists() and not config.force:
            counts["skipped_existing"] += 1
            result["status"] = "skipped_existing"
            results.append(result)
            continue

        if config.dry_run:
            counts["ingested"] += 1
            result["status"] = "dry_run"
            results.append(result)
            continue

        try:
            manifest = ingest_fn(
                VideoIngestConfig(
                    video_path=video_path,
                    project_id=project_id,
                    video_id=video_id,
                    output_root=resolved_output_root,
                    frame_rate=config.frame_rate,
                    max_frames=config.max_frames,
                    frame_sampling=config.frame_sampling,
                    frame_selection=config.frame_selection,
                    skip_frames=config.skip_frames,
                    copy_source=config.copy_source,
                    transcript_source=config.transcript_source,
                    stt_model=config.stt_model,
                    stt_language=config.stt_language,
                    stt_task=config.stt_task,
                    stt_word_timestamps=config.stt_word_timestamps,
                )
            )
            counts["ingested"] += 1
            result["status"] = "ingested"
            result["transcript_source"] = manifest.get("transcript_source")
            result["segments"] = _safe_nested_int(manifest, "counts", "lecture_segments")
            result["frames"] = _safe_nested_int(manifest, "counts", "frames")
        except Exception as exc:
            counts["failed"] += 1
            result["status"] = "failed"
            result["error"] = str(exc)
            results.append(result)
            if config.strict:
                aborted = True
                break
            continue

        results.append(result)

    return {
        "root_dir": str(resolved_root),
        "output_root": str(resolved_output_root),
        "force": config.force,
        "strict": config.strict,
        "dry_run": config.dry_run,
        "project_prefix": config.project_prefix,
        "aborted": aborted,
        "counts": counts,
        "results": results,
    }


def write_batch_summary_json(path: Path, summary: dict[str, Any]) -> None:
    write_json(path, summary)


def write_batch_summary_jsonl(path: Path, summary: dict[str, Any]) -> int:
    return write_jsonl(path, summary.get("results", []))


def write_batch_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("results", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "video_path",
        "project_id",
        "video_id",
        "project_dir",
        "manifest_path",
        "transcript_source",
        "segments",
        "frames",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _safe_nested_int(payload: dict[str, Any], key: str, nested_key: str) -> int | None:
    raw = payload.get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get(nested_key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
