from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


@dataclass(frozen=True)
class SttSegment:
    seq_no: int
    start_time: float
    end_time: float
    text: str


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    run(command)
    return audio_path


def transcribe_audio_with_mlx_whisper(
    audio_path: Path,
    *,
    model: str = DEFAULT_MLX_WHISPER_MODEL,
    language: str | None = None,
    task: str = "transcribe",
    word_timestamps: bool = False,
    temperature: float = 0.0,
    condition_on_previous_text: bool = False,
    hallucination_silence_threshold: float | None = 1.0,
) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper is not installed. Install it with "
            "`python -m pip install '.[stt]'` or `python -m pip install mlx-whisper`."
        ) from exc

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": word_timestamps,
        "temperature": temperature,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if hallucination_silence_threshold is not None:
        kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
    if language:
        kwargs["language"] = language
    if task:
        kwargs["task"] = task
    return mlx_whisper.transcribe(str(audio_path), **kwargs)


def stt_segments_from_mlx_result(result: dict[str, Any]) -> list[SttSegment]:
    segments: list[SttSegment] = []
    for raw_index, raw_segment in enumerate(result.get("segments") or [], start=1):
        text = normalize_transcript_text(str(raw_segment.get("text", "")))
        if not text:
            continue
        start_time = optional_float(raw_segment.get("start"), default=0.0)
        end_time = optional_float(raw_segment.get("end"), default=start_time)
        if end_time < start_time:
            end_time = start_time
        segments.append(
            SttSegment(
                seq_no=len(segments) + 1,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

    if segments:
        return segments

    text = normalize_transcript_text(str(result.get("text", "")))
    if not text:
        return []
    return [SttSegment(seq_no=1, start_time=0.0, end_time=0.0, text=text)]


def normalize_transcript_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def optional_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{stderr}") from exc
