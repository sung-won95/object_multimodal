from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Iterator


VLM_SCHEMA_VERSION = "vlm-consistency-v1"
VLM_PROJECT_MANIFEST_SECTION = "vlm_consistency"

VLM_FRAME_CANDIDATES_ARTIFACT = "vlm_frame_candidates"
VLM_VISUAL_OBSERVATIONS_ARTIFACT = "vlm_visual_observations"
AUDIO_VISUAL_CONSISTENCY_ARTIFACT = "audio_visual_consistency"

VLM_ARTIFACT_PATHS = {
    VLM_FRAME_CANDIDATES_ARTIFACT: "manifests/vlm_frame_candidates.jsonl",
    VLM_VISUAL_OBSERVATIONS_ARTIFACT: "manifests/vlm_visual_observations.jsonl",
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT: "manifests/audio_visual_consistency.jsonl",
}

VLM_COMMON_RECORD_FIELDS = (
    "schema_version",
    "project_id",
    "video_id",
    "frame_id",
    "timestamp",
    "segment_id",
    "backend",
    "source_model",
    "model_version",
    "confidence",
    "status",
)

VLM_JSONL_ARTIFACT_CONTRACT = {
    VLM_FRAME_CANDIDATES_ARTIFACT: {
        "path": VLM_ARTIFACT_PATHS[VLM_FRAME_CANDIDATES_ARTIFACT],
        "record_type": "VLMFrameCandidate",
        "fields": (*VLM_COMMON_RECORD_FIELDS, "frame_path", "selection_reason", "rank"),
    },
    VLM_VISUAL_OBSERVATIONS_ARTIFACT: {
        "path": VLM_ARTIFACT_PATHS[VLM_VISUAL_OBSERVATIONS_ARTIFACT],
        "record_type": "VLMVisualObservation",
        "fields": (
            *VLM_COMMON_RECORD_FIELDS,
            "observation_id",
            "observation_type",
            "visual_description",
            "detected_text",
            "bbox",
            "position",
            "attributes",
            "relations",
        ),
    },
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT: {
        "path": VLM_ARTIFACT_PATHS[AUDIO_VISUAL_CONSISTENCY_ARTIFACT],
        "record_type": "AudioVisualConsistencyRecord",
        "fields": (
            *VLM_COMMON_RECORD_FIELDS,
            "consistency_id",
            "consistency",
            "visual_observation_ids",
            "evidence_refs",
            "failure_reason",
            "skip_reason",
        ),
    },
}

VLM_COUNT_FIELDS = (
    VLM_FRAME_CANDIDATES_ARTIFACT,
    VLM_VISUAL_OBSERVATIONS_ARTIFACT,
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT,
)

LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD = "semantic_text"
LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD = "semantic_source_fields"

LECTURE_SEGMENT_TEXT_SEMANTIC_SOURCE_FIELDS = (
    "transcript_text",
    "mention_candidates",
)

LECTURE_SEGMENT_VISUAL_SEMANTIC_SOURCE_FIELDS = (
    "visual_entities.text",
    "visual_entities.visual_description",
)

LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELD_ORDER = (
    *LECTURE_SEGMENT_TEXT_SEMANTIC_SOURCE_FIELDS,
    *LECTURE_SEGMENT_VISUAL_SEMANTIC_SOURCE_FIELDS,
)


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
    "board",
    "line",
    "point",
    "figure",
    "stack",
    "bet size",
    "position",
    "range",
    "equity",
    "hand",
    "card",
    "offsuit",
    "suited",
    "button",
    "이것",
    "이거",
    "이 부분",
    "이 장면",
    "여기",
    "화면",
    "위",
    "아래",
    "왼쪽",
    "오른쪽",
    "그래프",
    "행렬",
    "수식",
    "표",
    "박스",
    "보드",
    "스택",
    "벳",
    "베팅",
    "사이즈",
    "레인지",
    "포지션",
    "에퀴티",
    "핸드",
    "카드",
    "오프수딧",
    "수딧",
)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "empty"


def timestamp_center(points: list[float]) -> float | None:
    if not points:
        return None
    return float(mean(points))


def mention_candidates(text: str) -> list[str]:
    lower = text.lower()
    return [hint for hint in DEICTIC_HINTS if hint.lower() in lower]


def build_lecture_segment_semantic_text(payload: dict[str, Any]) -> tuple[str, list[str]]:
    pieces: list[str] = []
    source_fields: list[str] = []
    for source_field in LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELD_ORDER:
        values = _semantic_source_values(payload, source_field)
        text_values = _unique_text_values(values)
        if not text_values:
            continue
        pieces.extend(text_values)
        source_fields.append(source_field)
    return _compact_text(" ".join(pieces)), source_fields


def ensure_lecture_segment_semantic_contract(payload: dict[str, Any]) -> dict[str, Any]:
    document = dict(payload)
    computed_text, computed_source_fields = build_lecture_segment_semantic_text(document)

    existing_text = _compact_text(str(document.get(LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD) or ""))
    document[LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD] = existing_text or computed_text

    existing_source_fields = _semantic_source_field_list(
        document.get(LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD)
    )
    document[LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD] = (
        existing_source_fields or computed_source_fields
    )
    return document


def _semantic_source_values(payload: dict[str, Any], source_field: str) -> list[Any]:
    values = list(_extract_path_values(payload, source_field.split(".")))
    flattened: list[Any] = []
    for value in values:
        flattened.extend(_flatten_semantic_value(value))
    return flattened


def _extract_path_values(value: Any, parts: list[str]) -> Iterator[Any]:
    if not parts:
        yield value
        return
    if isinstance(value, dict):
        child = value.get(parts[0])
        if child is not None:
            yield from _extract_path_values(child, parts[1:])
    elif isinstance(value, list):
        for item in value:
            yield from _extract_path_values(item, parts)


def _flatten_semantic_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_semantic_value(item))
        return flattened
    return [value]


def _unique_text_values(values: list[Any]) -> list[str]:
    text_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _compact_text(str(value))
        if not text or text in seen:
            continue
        seen.add(text)
        text_values.append(text)
    return text_values


def _semantic_source_field_list(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else ([] if value is None else [value])
    fields: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        field_name = _compact_text(str(item))
        if not field_name or field_name in seen:
            continue
        seen.add(field_name)
        fields.append(field_name)
    return fields


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


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
    semantic_text: str = ""
    semantic_source_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        semantic_fields = ensure_lecture_segment_semantic_contract(
            {
                "transcript_text": self.transcript_text,
                "mention_candidates": self.mention_candidates,
                LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD: self.semantic_text,
                LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD: self.semantic_source_fields,
            }
        )
        object.__setattr__(
            self,
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
            semantic_fields[LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD],
        )
        object.__setattr__(
            self,
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
            semantic_fields[LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD],
        )

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
class VisualEntity:
    entity_id: str
    project_id: str
    frame_id: str
    timestamp: float | None
    frame_path: str
    bbox: dict[str, float] | None
    text: str
    entity_type: str
    confidence: float | None
    source: str
    visual_description: str | None = None
    position: dict[str, Any] | None = None
    relations: list[dict[str, Any]] = field(default_factory=list)
    parser_version: str | None = None
    source_model: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VisualEntity":
        raw_bbox = payload.get("bbox")
        bbox: dict[str, float] | None = None
        if isinstance(raw_bbox, dict):
            bbox = {}
            for key, value in raw_bbox.items():
                parsed = _optional_float(value)
                if parsed is not None:
                    bbox[str(key)] = parsed
            if not bbox:
                bbox = None

        raw_position = payload.get("position")
        position = dict(raw_position) if isinstance(raw_position, dict) else None

        raw_relations = payload.get("relations")
        relations: list[dict[str, Any]] = []
        if isinstance(raw_relations, list):
            relations = [dict(item) for item in raw_relations if isinstance(item, dict)]

        visual_description = payload.get("visual_description")
        parser_version = payload.get("parser_version")
        source_model = payload.get("source_model")

        return cls(
            entity_id=str(payload.get("entity_id", "")),
            project_id=str(payload.get("project_id", "")),
            frame_id=str(payload.get("frame_id", "")),
            timestamp=_optional_float(payload.get("timestamp")),
            frame_path=str(payload.get("frame_path", "")),
            bbox=bbox,
            text=str(payload.get("text", "")),
            entity_type=str(payload.get("entity_type", "")),
            confidence=_optional_float(payload.get("confidence")),
            source=str(payload.get("source", "")),
            visual_description=(
                str(visual_description) if visual_description is not None else None
            ),
            position=position,
            relations=relations,
            parser_version=str(parser_version) if parser_version is not None else None,
            source_model=str(source_model) if source_model is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VLMFrameCandidate:
    project_id: str
    video_id: str
    frame_id: str
    timestamp: float | None
    segment_id: str | None
    backend: str
    source_model: str | None
    model_version: str | None
    confidence: float | None
    status: str
    frame_path: str | None = None
    selection_reason: str | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = VLM_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VLMFrameCandidate":
        return cls(
            project_id=str(payload.get("project_id", "")),
            video_id=str(payload.get("video_id", "")),
            frame_id=str(payload.get("frame_id", "")),
            timestamp=_optional_float(payload.get("timestamp")),
            segment_id=_optional_str(payload.get("segment_id")),
            backend=str(payload.get("backend", "")),
            source_model=_optional_str(payload.get("source_model")),
            model_version=_optional_str(payload.get("model_version")),
            confidence=_optional_float(payload.get("confidence")),
            status=str(payload.get("status", "")),
            frame_path=_optional_str(payload.get("frame_path")),
            selection_reason=_optional_str(payload.get("selection_reason")),
            rank=_optional_int(payload.get("rank")),
            metadata=_mapping(payload.get("metadata")),
            schema_version=str(payload.get("schema_version", VLM_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VLMVisualObservation:
    observation_id: str
    project_id: str
    video_id: str
    frame_id: str
    timestamp: float | None
    segment_id: str | None
    backend: str
    source_model: str | None
    model_version: str | None
    confidence: float | None
    status: str
    observation_type: str
    visual_description: str
    detected_text: str | None = None
    bbox: dict[str, float] | None = None
    position: dict[str, Any] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = VLM_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VLMVisualObservation":
        return cls(
            observation_id=str(payload.get("observation_id", "")),
            project_id=str(payload.get("project_id", "")),
            video_id=str(payload.get("video_id", "")),
            frame_id=str(payload.get("frame_id", "")),
            timestamp=_optional_float(payload.get("timestamp")),
            segment_id=_optional_str(payload.get("segment_id")),
            backend=str(payload.get("backend", "")),
            source_model=_optional_str(payload.get("source_model")),
            model_version=_optional_str(payload.get("model_version")),
            confidence=_optional_float(payload.get("confidence")),
            status=str(payload.get("status", "")),
            observation_type=str(payload.get("observation_type", "")),
            visual_description=str(payload.get("visual_description", "")),
            detected_text=_optional_str(payload.get("detected_text")),
            bbox=_float_mapping_or_none(payload.get("bbox")),
            position=_optional_mapping(payload.get("position")),
            attributes=_mapping(payload.get("attributes")),
            relations=_dict_list(payload.get("relations")),
            metadata=_mapping(payload.get("metadata")),
            schema_version=str(payload.get("schema_version", VLM_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioVisualConsistencyRecord:
    consistency_id: str
    project_id: str
    video_id: str
    frame_id: str
    timestamp: float | None
    segment_id: str | None
    backend: str
    source_model: str | None
    model_version: str | None
    confidence: float | None
    status: str
    consistency: str
    visual_observation_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = VLM_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AudioVisualConsistencyRecord":
        return cls(
            consistency_id=str(payload.get("consistency_id", "")),
            project_id=str(payload.get("project_id", "")),
            video_id=str(payload.get("video_id", "")),
            frame_id=str(payload.get("frame_id", "")),
            timestamp=_optional_float(payload.get("timestamp")),
            segment_id=_optional_str(payload.get("segment_id")),
            backend=str(payload.get("backend", "")),
            source_model=_optional_str(payload.get("source_model")),
            model_version=_optional_str(payload.get("model_version")),
            confidence=_optional_float(payload.get("confidence")),
            status=str(payload.get("status", "")),
            consistency=str(payload.get("consistency", "")),
            visual_observation_ids=_str_list(payload.get("visual_observation_ids")),
            evidence_refs=_str_list(payload.get("evidence_refs")),
            failure_reason=_optional_str(payload.get("failure_reason")),
            skip_reason=_optional_str(payload.get("skip_reason")),
            metadata=_mapping(payload.get("metadata")),
            schema_version=str(payload.get("schema_version", VLM_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityLink:
    link_id: str
    project_id: str
    segment_id: str
    entity_id: str
    frame_id: str
    link_type: str
    score: float
    evidence: list[str]
    time_overlap: bool
    lexical_match: list[str]
    mention_candidate: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityLink":
        return cls(
            link_id=str(payload.get("link_id", "")),
            project_id=str(payload.get("project_id", "")),
            segment_id=str(payload.get("segment_id", "")),
            entity_id=str(payload.get("entity_id", "")),
            frame_id=str(payload.get("frame_id", "")),
            link_type=str(payload.get("link_type", "")),
            score=float(payload.get("score", 0.0)),
            evidence=[str(item) for item in payload.get("evidence", [])],
            time_overlap=bool(payload.get("time_overlap", False)),
            lexical_match=[str(item) for item in payload.get("lexical_match", [])],
            mention_candidate=[str(item) for item in payload.get("mention_candidate", [])],
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


@dataclass(frozen=True)
class EvidenceWindow:
    target_segment_id: str
    project_id: str | None
    video_id: str | None
    start_time: float | None
    end_time: float | None
    transcript_segments: list[dict[str, Any]]
    frame_refs: list[dict[str, Any]]
    target_segment: dict[str, Any] = field(default_factory=dict)
    neighbor_segments: list[dict[str, Any]] = field(default_factory=list)
    window_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_vlm_project_manifest_fields(
    *,
    backend: str,
    source_model: str | None = None,
    model_version: str | None = None,
    status: str = "planned",
    settings: dict[str, Any] | None = None,
    artifact_paths: dict[str, str] | None = None,
    counts: dict[str, int] | None = None,
    failures: dict[str, Any] | None = None,
    skips: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = dict(VLM_ARTIFACT_PATHS)
    if artifact_paths:
        artifacts.update({str(key): str(value) for key, value in artifact_paths.items()})

    resolved_counts = {key: 0 for key in VLM_COUNT_FIELDS}
    if counts:
        resolved_counts.update({str(key): int(value) for key, value in counts.items()})

    section = {
        "schema_version": VLM_SCHEMA_VERSION,
        "status": status,
        "backend": backend,
        "source_model": source_model,
        "model_version": model_version,
        "settings": _mapping(settings),
        "artifacts": artifacts,
        "counts": resolved_counts,
        "failures": _reason_summary(failures),
        "skips": _reason_summary(skips),
    }
    return {
        "artifacts": artifacts,
        "counts": resolved_counts,
        VLM_PROJECT_MANIFEST_SECTION: section,
    }


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _float_mapping_or_none(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, float] = {}
    for key, raw_value in value.items():
        float_value = _optional_float(raw_value)
        if float_value is not None:
            parsed[str(key)] = float_value
    return parsed or None


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _reason_summary(value: dict[str, Any] | None) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    reasons = payload.get("reasons")
    return {
        "count": int(payload.get("count", 0) or 0),
        "reasons": dict(reasons) if isinstance(reasons, dict) else {},
    }
