from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


TIMING_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass(frozen=True)
class SrtCue:
    seq_no: int
    start_time: float
    end_time: float
    text: str


def parse_timestamp(value: str) -> float:
    normalized = value.replace(",", ".")
    hours, minutes, seconds = normalized.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def iter_srt_cues(path: Path) -> Iterator[SrtCue]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())
    cue_index = 0
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_idx = next((idx for idx, line in enumerate(lines) if TIMING_RE.search(line)), None)
        if timing_idx is None:
            continue
        match = TIMING_RE.search(lines[timing_idx])
        if match is None:
            continue
        text_lines = lines[timing_idx + 1 :]
        if not text_lines:
            continue
        cue_index += 1
        yield SrtCue(
            seq_no=cue_index,
            start_time=parse_timestamp(match.group("start")),
            end_time=parse_timestamp(match.group("end")),
            text=" ".join(text_lines),
        )

