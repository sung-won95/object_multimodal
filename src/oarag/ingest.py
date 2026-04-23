from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl
from .schemas import LectureSegment, slugify
from .srt import iter_srt_cues
from .stt import (
    DEFAULT_MLX_WHISPER_MODEL,
    extract_audio,
    stt_segments_from_mlx_result,
    transcribe_audio_with_mlx_whisper,
)


TRANSCRIPT_SOURCES = {"auto", "srt", "stt", "none"}


@dataclass(frozen=True)
class VideoIngestConfig:
    video_path: Path
    project_id: str
    video_id: str
    output_root: Path
    srt_path: Path | None = None
    frame_rate: float = 1.0
    max_frames: int | None = 120
    skip_frames: bool = False
    copy_source: bool = False
    transcript_source: str = "auto"
    stt_model: str = DEFAULT_MLX_WHISPER_MODEL
    stt_language: str | None = None
    stt_task: str = "transcribe"
    stt_word_timestamps: bool = False


def ingest_video(config: VideoIngestConfig) -> dict[str, Any]:
    video_path = config.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if config.transcript_source not in TRANSCRIPT_SOURCES:
        raise ValueError(f"transcript_source must be one of {sorted(TRANSCRIPT_SOURCES)}")

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
    frames_manifest_path = manifests_dir / "frames_manifest.jsonl"
    if config.skip_frames:
        write_jsonl(frames_manifest_path, [])
    else:
        frame_count = sample_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            frame_rate=config.frame_rate,
            max_frames=config.max_frames,
        )
        write_frames_manifest(
            frames_manifest_path,
            frames_dir=frames_dir,
            frame_rate=config.frame_rate,
            frame_count=frame_count,
        )

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
) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
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
    if max_frames is not None and max_frames > 0:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_pattern))
    run(command)
    return len(sorted(frames_dir.glob("frame_*.jpg")))


def write_frames_manifest(
    path: Path,
    frames_dir: Path,
    frame_rate: float,
    frame_count: int,
) -> None:
    rows = []
    for index in range(1, frame_count + 1):
        rows.append(
            {
                "frame_id": f"frame_{index:06d}",
                "frame_path": str(frames_dir / f"frame_{index:06d}.jpg"),
                "timestamp": round((index - 1) / frame_rate, 3) if frame_rate > 0 else None,
            }
        )
    write_jsonl(path, rows)


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
