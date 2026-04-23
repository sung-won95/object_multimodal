from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


DEICTIC_HINTS = (
    "this",
    "that",
    "here",
    "left",
    "right",
    "top",
    "bottom",
    "graph",
    "chart",
    "matrix",
    "equation",
    "formula",
    "table",
    "red",
    "blue",
    "box",
    "line",
    "point",
    "figure",
    "이것",
    "이거",
    "이 부분",
    "이 장면",
    "여기",
    "왼쪽",
    "오른쪽",
    "그래프",
    "행렬",
    "수식",
    "표",
    "박스",
)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    return cleaned.strip("_") or "empty"


def timestamp_center(points: list[float]) -> float | None:
    if not points:
        return None
    return float(mean(points))


def mention_candidates(text: str) -> list[str]:
    lower = text.lower()
    return [hint for hint in DEICTIC_HINTS if hint.lower() in lower]


@dataclass(frozen=True)
class EduVidQARecord:
    dataset_name: str
    subset_name: str
    split_name: str
    sample_index: int
    sample_id: str
    video_name: str
    question: str
    answer: str
    transcript_text: str
    timestamp_points: list[float]
    has_timestamp: bool
    qid: str | None
    raw_keys: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EduVidQARecord":
        points = [float(point) for point in payload.get("timestamp_points", [])]
        return cls(
            dataset_name=str(payload.get("dataset_name", "")),
            subset_name=str(payload.get("subset_name", "")),
            split_name=str(payload.get("split_name", "")),
            sample_index=int(payload.get("sample_index", 0)),
            sample_id=str(payload.get("sample_id", "")),
            video_name=str(payload.get("video_name", "")),
            question=str(payload.get("question", "")),
            answer=str(payload.get("answer", "")),
            transcript_text=str(payload.get("transcript_text", "")),
            timestamp_points=points,
            has_timestamp=bool(payload.get("has_timestamp", bool(points))),
            qid=str(payload["qid"]) if payload.get("qid") is not None else None,
            raw_keys=list(payload.get("raw_keys", [])),
        )


@dataclass(frozen=True)
class LectureSegment:
    segment_id: str
    project_id: str
    dataset_name: str
    subset_name: str
    split_name: str
    sample_id: str
    sample_index: int
    video_id: str
    video_name: str
    start_time: float | None
    end_time: float | None
    timestamp_center: float | None
    timestamp_points: list[float]
    transcript_text: str
    normalized_text: str
    slide_id: str | None = None
    frame_refs: list[str] = field(default_factory=list)
    mention_candidates: list[str] = field(default_factory=list)
    source: str = "eduvidqa"

    @classmethod
    def from_eduvidqa(
        cls,
        record: EduVidQARecord,
        project_id: str = "eduvidqa",
        context_window_seconds: float = 90.0,
    ) -> "LectureSegment":
        center = timestamp_center(record.timestamp_points)
        start_time = None if center is None else max(0.0, center - context_window_seconds)
        end_time = None if center is None else center + context_window_seconds
        sample_slug = slugify(record.sample_id)
        return cls(
            segment_id=f"seg_{sample_slug}",
            project_id=project_id,
            dataset_name=record.dataset_name,
            subset_name=record.subset_name,
            split_name=record.split_name,
            sample_id=record.sample_id,
            sample_index=record.sample_index,
            video_id=record.video_name,
            video_name=record.video_name,
            start_time=start_time,
            end_time=end_time,
            timestamp_center=center,
            timestamp_points=record.timestamp_points,
            transcript_text=record.transcript_text,
            normalized_text=record.transcript_text.lower(),
            mention_candidates=mention_candidates(record.question + " " + record.transcript_text),
        )

    @classmethod
    def from_local_transcript(
        cls,
        project_id: str,
        video_id: str,
        seq_no: int,
        start_time: float,
        end_time: float,
        text: str,
        source: str = "srt",
    ) -> "LectureSegment":
        segment_id = f"seg_{slugify(video_id)}_{seq_no:06d}"
        center = (start_time + end_time) / 2.0
        return cls(
            segment_id=segment_id,
            project_id=project_id,
            dataset_name="local_video",
            subset_name="pilot",
            split_name="ingest",
            sample_id=segment_id,
            sample_index=seq_no,
            video_id=video_id,
            video_name=video_id,
            start_time=start_time,
            end_time=end_time,
            timestamp_center=center,
            timestamp_points=[center],
            transcript_text=text,
            normalized_text=text.lower(),
            mention_candidates=mention_candidates(text),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchCandidate:
    rank: int
    segment_id: str
    sample_id: str
    video_id: str
    start_time: float | None
    end_time: float | None
    timestamp_center: float | None
    transcript_excerpt: str
    score: float | None

    @classmethod
    def from_hit(cls, rank: int, hit: dict[str, Any]) -> "SearchCandidate":
        transcript = str(hit.get("transcript_text", ""))
        excerpt = transcript[:360] + ("..." if len(transcript) > 360 else "")
        return cls(
            rank=rank,
            segment_id=str(hit.get("segment_id", "")),
            sample_id=str(hit.get("sample_id", "")),
            video_id=str(hit.get("video_id", hit.get("video_name", ""))),
            start_time=_optional_float(hit.get("start_time")),
            end_time=_optional_float(hit.get("end_time")),
            timestamp_center=_optional_float(hit.get("timestamp_center")),
            transcript_excerpt=excerpt,
            score=_optional_float(hit.get("_rankingScore")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
