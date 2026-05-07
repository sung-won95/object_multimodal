from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.project_index import segment_artifact_path
from oarag.core.schemas import (
    AUDIO_VISUAL_CONSISTENCY_ARTIFACT,
    VLM_ARTIFACT_PATHS,
    VLM_PROJECT_MANIFEST_SECTION,
    VLM_SCHEMA_VERSION,
    VLM_VISUAL_OBSERVATIONS_ARTIFACT,
    AudioVisualConsistencyRecord,
    VLMVisualObservation,
    slugify,
)


DEFAULT_CONSISTENCY_BACKEND = "deterministic-audio-visual-verifier"
DEFAULT_CONSISTENCY_MODEL = "lexical-overlap-stub"
DEFAULT_CONSISTENCY_MODEL_VERSION = "rule-v1"

AVC_SUCCESS_STATUS = "success"
AVC_SKIPPED_STATUS = "skipped"
AVC_SOURCE_FAILURE_STATUS = "source_failure"

CONSISTENCY_ALIGNED = "aligned"
CONSISTENCY_MISMATCH = "mismatch"
CONSISTENCY_UNCERTAIN = "uncertain"

DEFAULT_SUCCESS_OBSERVATION_STATUSES = ("success",)
DEFAULT_FAILURE_OBSERVATION_STATUSES = ("backend_failure", "parse_failure")

NO_FRAME_REFS = "no_frame_refs"
MISSING_VISUAL_OBSERVATION = "missing_visual_observation"
NO_SUCCESS_OBSERVATION = "no_success_observation"


@dataclass(frozen=True)
class AudioVisualConsistencyConfig:
    backend: str = DEFAULT_CONSISTENCY_BACKEND
    source_model: str = DEFAULT_CONSISTENCY_MODEL
    model_version: str = DEFAULT_CONSISTENCY_MODEL_VERSION
    window_margin_seconds: float = 0.0
    success_statuses: tuple[str, ...] = DEFAULT_SUCCESS_OBSERVATION_STATUSES
    failure_statuses: tuple[str, ...] = DEFAULT_FAILURE_OBSERVATION_STATUSES


@dataclass(frozen=True)
class _SegmentRow:
    segment_id: str
    index: int
    project_id: str | None
    video_id: str | None
    start_time: float | None
    end_time: float | None
    timestamp_center: float | None
    frame_refs: tuple[str, ...]
    transcript_text: str


@dataclass(frozen=True)
class _VerificationDecision:
    consistency: str
    confidence: float
    rationale: str
    transcript_terms: tuple[str, ...]
    visual_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]


def generate_audio_visual_consistency(
    *,
    project_dir: Path,
    segments_path: Path | None = None,
    observations_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    config: AudioVisualConsistencyConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or AudioVisualConsistencyConfig()
    _validate_config(resolved_config)

    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    resolved_segments_path = segment_artifact_path(resolved_project_dir, segments=segments_path)
    resolved_observations_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=observations_path,
        default=resolved_project_dir / VLM_ARTIFACT_PATHS[VLM_VISUAL_OBSERVATIONS_ARTIFACT],
    )
    resolved_output_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=output_path,
        default=resolved_project_dir / VLM_ARTIFACT_PATHS[AUDIO_VISUAL_CONSISTENCY_ARTIFACT],
    )

    manifest = _read_json_object(resolved_manifest_path) if resolved_manifest_path.exists() else {}
    segments = _read_jsonl(resolved_segments_path)
    observations = _read_jsonl(resolved_observations_path)
    project_id = _project_id_from_inputs(
        manifest=manifest,
        project_dir=resolved_project_dir,
        segments=segments,
        observations=observations,
    )
    video_id = _video_id_from_inputs(
        manifest=manifest,
        project_id=project_id,
        segments=segments,
        observations=observations,
    )

    result = evaluate_audio_visual_consistency(
        segments=segments,
        observations=observations,
        project_id=project_id,
        video_id=video_id,
        config=resolved_config,
    )
    records = result["records"]
    summary = result["summary"]

    write_jsonl(resolved_output_path, [record.to_dict() for record in records])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        segments_path=resolved_segments_path,
        observations_path=resolved_observations_path,
        output_path=resolved_output_path,
        config=resolved_config,
        summary=summary,
    )

    return {
        "project_id": project_id,
        "video_id": video_id,
        "backend": resolved_config.backend,
        "source_model": resolved_config.source_model,
        "model_version": resolved_config.model_version,
        "status": summary["status"],
        "paths": {
            "project_dir": str(resolved_project_dir),
            "segments": str(resolved_segments_path),
            "vlm_visual_observations": str(resolved_observations_path),
            "audio_visual_consistency": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "counts": summary["counts"],
        "status_counts": summary["status_counts"],
        "consistency_counts": summary["consistency_counts"],
    }


def evaluate_audio_visual_consistency(
    *,
    segments: list[dict[str, Any]],
    observations: list[dict[str, Any] | VLMVisualObservation],
    project_id: str,
    video_id: str,
    config: AudioVisualConsistencyConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or AudioVisualConsistencyConfig()
    _validate_config(resolved_config)

    segment_rows = _normalize_segments(segments)
    observation_rows = _normalize_observations(observations)
    observations_by_frame_id = _observations_by_frame_id(observation_rows)
    success_statuses = _normalized_statuses(resolved_config.success_statuses)
    failure_statuses = _normalized_statuses(resolved_config.failure_statuses)

    records: list[AudioVisualConsistencyRecord] = []
    for segment in segment_rows:
        frame_ids = _segment_frame_ids(
            segment=segment,
            observations=observation_rows,
            margin_seconds=resolved_config.window_margin_seconds,
        )
        if not frame_ids:
            records.append(
                _no_frame_record(
                    segment=segment,
                    project_id=project_id,
                    video_id=video_id,
                    config=resolved_config,
                )
            )
            continue

        for frame_id in frame_ids:
            frame_observations = observations_by_frame_id.get(frame_id, [])
            success_observations = [
                observation
                for observation in frame_observations
                if observation.status in success_statuses
            ]
            failure_observations = [
                observation
                for observation in frame_observations
                if observation.status in failure_statuses
            ]

            if success_observations:
                records.append(
                    _success_record(
                        segment=segment,
                        frame_id=frame_id,
                        frame_observations=frame_observations,
                        success_observations=success_observations,
                        project_id=project_id,
                        video_id=video_id,
                        config=resolved_config,
                    )
                )
            elif failure_observations:
                records.append(
                    _source_failure_record(
                        segment=segment,
                        frame_id=frame_id,
                        frame_observations=frame_observations,
                        failure_observations=failure_observations,
                        project_id=project_id,
                        video_id=video_id,
                        config=resolved_config,
                    )
                )
            else:
                records.append(
                    _no_success_record(
                        segment=segment,
                        frame_id=frame_id,
                        frame_observations=frame_observations,
                        project_id=project_id,
                        video_id=video_id,
                        config=resolved_config,
                    )
                )

    summary = _summary(
        records=records,
        segments=segment_rows,
        observations=observation_rows,
        config=resolved_config,
    )
    return {"records": records, "summary": summary}


def _success_record(
    *,
    segment: _SegmentRow,
    frame_id: str,
    frame_observations: list[VLMVisualObservation],
    success_observations: list[VLMVisualObservation],
    project_id: str,
    video_id: str,
    config: AudioVisualConsistencyConfig,
) -> AudioVisualConsistencyRecord:
    decision = _verify_segment_observations(segment=segment, observations=success_observations)
    return AudioVisualConsistencyRecord(
        consistency_id=_consistency_id(segment.segment_id, frame_id),
        project_id=_record_project_id(segment, success_observations, project_id),
        video_id=_record_video_id(segment, success_observations, video_id),
        frame_id=frame_id,
        timestamp=_record_timestamp(success_observations),
        segment_id=segment.segment_id,
        backend=config.backend,
        source_model=config.source_model,
        model_version=config.model_version,
        confidence=decision.confidence,
        status=AVC_SUCCESS_STATUS,
        consistency=decision.consistency,
        visual_observation_ids=_observation_ids(success_observations),
        evidence_refs=_evidence_refs(segment.segment_id, frame_id, success_observations),
        metadata={
            "rationale": decision.rationale,
            "matched_terms": list(decision.matched_terms),
            "transcript_terms": list(decision.transcript_terms),
            "visual_terms": list(decision.visual_terms),
            **_source_observation_metadata(frame_observations),
        },
    )


def _source_failure_record(
    *,
    segment: _SegmentRow,
    frame_id: str,
    frame_observations: list[VLMVisualObservation],
    failure_observations: list[VLMVisualObservation],
    project_id: str,
    video_id: str,
    config: AudioVisualConsistencyConfig,
) -> AudioVisualConsistencyRecord:
    failure_reason = _failure_reason(failure_observations)
    return AudioVisualConsistencyRecord(
        consistency_id=_consistency_id(segment.segment_id, frame_id),
        project_id=_record_project_id(segment, failure_observations, project_id),
        video_id=_record_video_id(segment, failure_observations, video_id),
        frame_id=frame_id,
        timestamp=_record_timestamp(failure_observations),
        segment_id=segment.segment_id,
        backend=config.backend,
        source_model=config.source_model,
        model_version=config.model_version,
        confidence=None,
        status=AVC_SOURCE_FAILURE_STATUS,
        consistency=CONSISTENCY_UNCERTAIN,
        visual_observation_ids=_observation_ids(failure_observations),
        evidence_refs=_evidence_refs(segment.segment_id, frame_id, failure_observations),
        failure_reason=failure_reason,
        metadata={
            "rationale": "No consistency judgment: source visual observation failed.",
            **_source_observation_metadata(frame_observations),
        },
    )


def _no_success_record(
    *,
    segment: _SegmentRow,
    frame_id: str,
    frame_observations: list[VLMVisualObservation],
    project_id: str,
    video_id: str,
    config: AudioVisualConsistencyConfig,
) -> AudioVisualConsistencyRecord:
    skip_reason = (
        NO_SUCCESS_OBSERVATION if frame_observations else MISSING_VISUAL_OBSERVATION
    )
    return AudioVisualConsistencyRecord(
        consistency_id=_consistency_id(segment.segment_id, frame_id),
        project_id=_record_project_id(segment, frame_observations, project_id),
        video_id=_record_video_id(segment, frame_observations, video_id),
        frame_id=frame_id,
        timestamp=_record_timestamp(frame_observations),
        segment_id=segment.segment_id,
        backend=config.backend,
        source_model=config.source_model,
        model_version=config.model_version,
        confidence=None,
        status=AVC_SKIPPED_STATUS,
        consistency=CONSISTENCY_UNCERTAIN,
        visual_observation_ids=_observation_ids(frame_observations),
        evidence_refs=_evidence_refs(segment.segment_id, frame_id, frame_observations),
        skip_reason=skip_reason,
        metadata={
            "rationale": "No consistency judgment: no successful visual observation.",
            **_source_observation_metadata(frame_observations),
        },
    )


def _no_frame_record(
    *,
    segment: _SegmentRow,
    project_id: str,
    video_id: str,
    config: AudioVisualConsistencyConfig,
) -> AudioVisualConsistencyRecord:
    return AudioVisualConsistencyRecord(
        consistency_id=_consistency_id(segment.segment_id, "no_frame"),
        project_id=segment.project_id or project_id,
        video_id=segment.video_id or video_id,
        frame_id="",
        timestamp=segment.timestamp_center,
        segment_id=segment.segment_id,
        backend=config.backend,
        source_model=config.source_model,
        model_version=config.model_version,
        confidence=None,
        status=AVC_SKIPPED_STATUS,
        consistency=CONSISTENCY_UNCERTAIN,
        visual_observation_ids=[],
        evidence_refs=[f"segment:{segment.segment_id}"],
        skip_reason=NO_FRAME_REFS,
        metadata={
            "rationale": "No consistency judgment: segment has no related frames.",
        },
    )


def _verify_segment_observations(
    *,
    segment: _SegmentRow,
    observations: list[VLMVisualObservation],
) -> _VerificationDecision:
    transcript_terms = sorted(_terms(segment.transcript_text))
    visual_terms = sorted(
        _terms(" ".join(_observation_text(observation) for observation in observations))
    )
    matched_terms = tuple(sorted(set(transcript_terms) & set(visual_terms)))

    if matched_terms:
        confidence = min(0.95, 0.75 + (0.05 * len(matched_terms)))
        rationale = "Shared transcript/visual terms: " + ", ".join(matched_terms[:5])
        return _VerificationDecision(
            consistency=CONSISTENCY_ALIGNED,
            confidence=round(confidence, 3),
            rationale=rationale,
            transcript_terms=tuple(transcript_terms),
            visual_terms=tuple(visual_terms),
            matched_terms=matched_terms,
        )

    if transcript_terms and visual_terms:
        return _VerificationDecision(
            consistency=CONSISTENCY_MISMATCH,
            confidence=0.7,
            rationale="Transcript and visual observations have informative terms but no overlap.",
            transcript_terms=tuple(transcript_terms),
            visual_terms=tuple(visual_terms),
            matched_terms=(),
        )

    return _VerificationDecision(
        consistency=CONSISTENCY_UNCERTAIN,
        confidence=0.35,
        rationale="Insufficient lexical evidence for deterministic consistency judgment.",
        transcript_terms=tuple(transcript_terms),
        visual_terms=tuple(visual_terms),
        matched_terms=(),
    )


def _normalize_segments(segments: list[dict[str, Any]]) -> list[_SegmentRow]:
    rows = []
    for index, segment in enumerate(segments):
        segment_id = _optional_str(segment.get("segment_id")) or f"seg_{index:06d}"
        start_time, end_time = _segment_window(segment, margin_seconds=0.0)
        rows.append(
            _SegmentRow(
                segment_id=segment_id,
                index=index,
                project_id=_optional_str(segment.get("project_id")),
                video_id=_optional_str(segment.get("video_id")),
                start_time=start_time,
                end_time=end_time,
                timestamp_center=_segment_center(segment, start_time=start_time, end_time=end_time),
                frame_refs=tuple(_unique_strs(segment.get("frame_refs"))),
                transcript_text=_segment_text(segment),
            )
        )
    return rows


def _normalize_observations(
    observations: list[dict[str, Any] | VLMVisualObservation],
) -> list[VLMVisualObservation]:
    rows = [
        observation
        if isinstance(observation, VLMVisualObservation)
        else VLMVisualObservation.from_dict(observation)
        for observation in observations
    ]
    return sorted(
        rows,
        key=lambda observation: (
            observation.timestamp is None,
            observation.timestamp or 0.0,
            observation.frame_id,
            observation.observation_id,
        ),
    )


def _observations_by_frame_id(
    observations: list[VLMVisualObservation],
) -> dict[str, list[VLMVisualObservation]]:
    rows: dict[str, list[VLMVisualObservation]] = {}
    for observation in observations:
        rows.setdefault(observation.frame_id, []).append(observation)
    return rows


def _segment_frame_ids(
    *,
    segment: _SegmentRow,
    observations: list[VLMVisualObservation],
    margin_seconds: float,
) -> list[str]:
    if segment.frame_refs:
        return list(segment.frame_refs)

    start_time, end_time = _segment_window(
        {
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "timestamp_center": segment.timestamp_center,
        },
        margin_seconds=margin_seconds,
    )
    if start_time is None or end_time is None:
        return []

    frame_ids = []
    for observation in observations:
        if observation.timestamp is None:
            continue
        if start_time <= observation.timestamp <= end_time:
            frame_ids.append(observation.frame_id)
    return _dedupe(frame_ids)


def _segment_window(
    segment: dict[str, Any],
    *,
    margin_seconds: float,
) -> tuple[float | None, float | None]:
    start = _optional_float(segment.get("start_time"))
    end = _optional_float(segment.get("end_time"))
    points = [
        value
        for value in (_optional_float(point) for point in segment.get("timestamp_points", []))
        if value is not None
    ]
    if start is None and points:
        start = min(points)
    if end is None and points:
        end = max(points)
    if start is None and end is None:
        center = _optional_float(segment.get("timestamp_center"))
        if center is not None:
            start = center
            end = center
    if start is None or end is None:
        return start, end
    if end < start:
        start, end = end, start
    margin = max(0.0, margin_seconds)
    return max(0.0, start - margin), end + margin


def _segment_center(
    segment: dict[str, Any],
    *,
    start_time: float | None,
    end_time: float | None,
) -> float | None:
    center = _optional_float(segment.get("timestamp_center"))
    if center is not None:
        return center
    points = [
        value
        for value in (_optional_float(point) for point in segment.get("timestamp_points", []))
        if value is not None
    ]
    if points:
        return sum(points) / len(points)
    if start_time is not None and end_time is not None:
        return (start_time + end_time) / 2.0
    return start_time if start_time is not None else end_time


def _segment_text(segment: dict[str, Any]) -> str:
    values = [
        _optional_str(segment.get("transcript_text")),
        _optional_str(segment.get("normalized_text")),
    ]
    mention_candidates = segment.get("mention_candidates")
    if isinstance(mention_candidates, list):
        values.extend(_optional_str(item) for item in mention_candidates)
    return " ".join(value for value in values if value)


def _observation_text(observation: VLMVisualObservation) -> str:
    return " ".join(
        value
        for value in [
            observation.visual_description,
            observation.detected_text,
            observation.observation_type,
        ]
        if value
    )


TERM_RE = re.compile(r"[0-9A-Za-z가-힣]+")
STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "here",
    "there",
    "look",
    "see",
    "shown",
    "shows",
    "visible",
    "screen",
    "slide",
    "frame",
    "image",
    "figure",
    "diagram",
    "chart",
    "table",
    "summary",
    "observation",
    "이것",
    "이거",
    "여기",
    "저기",
    "화면",
    "보이는",
    "보세요",
    "슬라이드",
    "그림",
    "도표",
    "표",
}


def _terms(text: str) -> set[str]:
    return {
        term
        for term in (match.group(0).casefold() for match in TERM_RE.finditer(text))
        if _is_informative_term(term)
    }


def _is_informative_term(term: str) -> bool:
    return len(term) >= 2 and not term.isdigit() and term not in STOP_TERMS


def _summary(
    *,
    records: list[AudioVisualConsistencyRecord],
    segments: list[_SegmentRow],
    observations: list[VLMVisualObservation],
    config: AudioVisualConsistencyConfig,
) -> dict[str, Any]:
    status_counts = Counter(record.status for record in records)
    consistency_counts = Counter(record.consistency for record in records)
    skip_reasons = Counter(record.skip_reason for record in records if record.skip_reason)
    failure_reasons = Counter(
        record.failure_reason for record in records if record.failure_reason
    )
    observation_status_counts = Counter(observation.status for observation in observations)
    source_failure_count = status_counts[AVC_SOURCE_FAILURE_STATUS]
    skipped_count = status_counts[AVC_SKIPPED_STATUS]
    status = "completed"
    if source_failure_count:
        status = "completed_with_errors"
    elif skipped_count:
        status = "completed_with_skips"

    return {
        "status": status,
        "counts": {
            "segments_total": len(segments),
            "segments_without_frames": skip_reasons[NO_FRAME_REFS],
            "vlm_visual_observations": len(observations),
            "success_visual_observations": sum(
                observation_status_counts[status] for status in config.success_statuses
            ),
            "failure_visual_observations": sum(
                observation_status_counts[status] for status in config.failure_statuses
            ),
            AUDIO_VISUAL_CONSISTENCY_ARTIFACT: len(records),
            "judged_records": status_counts[AVC_SUCCESS_STATUS],
            "source_failure_records": source_failure_count,
            "skipped_records": skipped_count,
        },
        "status_counts": dict(status_counts),
        "consistency_counts": dict(consistency_counts),
        "observation_status_counts": dict(observation_status_counts),
        "failures": {"count": source_failure_count, "reasons": dict(failure_reasons)},
        "skips": {"count": skipped_count, "reasons": dict(skip_reasons)},
    }


def _update_project_manifest(
    *,
    manifest_path: Path,
    segments_path: Path,
    observations_path: Path,
    output_path: Path,
    config: AudioVisualConsistencyConfig,
    summary: dict[str, Any],
) -> None:
    payload = _read_json_object(manifest_path) if manifest_path.exists() else {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts[AUDIO_VISUAL_CONSISTENCY_ARTIFACT] = str(output_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts[AUDIO_VISUAL_CONSISTENCY_ARTIFACT] = summary["counts"][
        AUDIO_VISUAL_CONSISTENCY_ARTIFACT
    ]

    vlm_section = payload.setdefault(VLM_PROJECT_MANIFEST_SECTION, {})
    if not isinstance(vlm_section, dict):
        vlm_section = {}
        payload[VLM_PROJECT_MANIFEST_SECTION] = vlm_section
    vlm_section.setdefault("schema_version", VLM_SCHEMA_VERSION)
    vlm_section["audio_visual_consistency"] = {
        "schema_version": VLM_SCHEMA_VERSION,
        "status": summary["status"],
        "backend": config.backend,
        "source_model": config.source_model,
        "model_version": config.model_version,
        "settings": {
            "window_margin_seconds": config.window_margin_seconds,
            "success_statuses": list(config.success_statuses),
            "failure_statuses": list(config.failure_statuses),
        },
        "artifacts": {
            "segments": str(segments_path),
            VLM_VISUAL_OBSERVATIONS_ARTIFACT: str(observations_path),
            AUDIO_VISUAL_CONSISTENCY_ARTIFACT: str(output_path),
        },
        "counts": summary["counts"],
        "status_counts": summary["status_counts"],
        "consistency_counts": summary["consistency_counts"],
        "observation_status_counts": summary["observation_status_counts"],
        "failures": summary["failures"],
        "skips": summary["skips"],
    }

    write_json(manifest_path, payload)


def _source_observation_metadata(
    observations: list[VLMVisualObservation],
) -> dict[str, Any]:
    return {
        "source_observation_status_counts": dict(
            Counter(observation.status for observation in observations)
        ),
        "source_backends": sorted(
            {observation.backend for observation in observations if observation.backend}
        ),
        "source_models": sorted(
            {
                observation.source_model
                for observation in observations
                if observation.source_model is not None
            }
        ),
        "source_model_versions": sorted(
            {
                observation.model_version
                for observation in observations
                if observation.model_version is not None
            }
        ),
    }


def _failure_reason(observations: list[VLMVisualObservation]) -> str:
    reasons = []
    for observation in observations:
        reason = _optional_str(observation.metadata.get("failure_reason"))
        if reason is None:
            reason = observation.status
        reasons.append(reason)
    return "; ".join(_dedupe(reasons))


def _record_project_id(
    segment: _SegmentRow,
    observations: list[VLMVisualObservation],
    default_project_id: str,
) -> str:
    if segment.project_id is not None:
        return segment.project_id
    for observation in observations:
        if observation.project_id:
            return observation.project_id
    return default_project_id


def _record_video_id(
    segment: _SegmentRow,
    observations: list[VLMVisualObservation],
    default_video_id: str,
) -> str:
    if segment.video_id is not None:
        return segment.video_id
    for observation in observations:
        if observation.video_id:
            return observation.video_id
    return default_video_id


def _record_timestamp(observations: list[VLMVisualObservation]) -> float | None:
    for observation in observations:
        if observation.timestamp is not None:
            return observation.timestamp
    return None


def _observation_ids(observations: list[VLMVisualObservation]) -> list[str]:
    return [
        observation.observation_id
        for observation in observations
        if observation.observation_id
    ]


def _evidence_refs(
    segment_id: str,
    frame_id: str,
    observations: list[VLMVisualObservation],
) -> list[str]:
    refs = [f"segment:{segment_id}", f"frame:{frame_id}"]
    refs.extend(
        f"observation:{observation.observation_id}"
        for observation in observations
        if observation.observation_id
    )
    return refs


def _consistency_id(segment_id: str, frame_id: str) -> str:
    return f"avc_{slugify(segment_id)}_{slugify(frame_id)}"


def _project_id_from_inputs(
    *,
    manifest: dict[str, Any],
    project_dir: Path,
    segments: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    manifest_project_id = _optional_str(manifest.get("project_id"))
    if manifest_project_id is not None:
        return manifest_project_id
    for segment in segments:
        segment_project_id = _optional_str(segment.get("project_id"))
        if segment_project_id is not None:
            return segment_project_id
    for observation in observations:
        observation_project_id = _optional_str(observation.get("project_id"))
        if observation_project_id is not None:
            return observation_project_id
    return project_dir.name


def _video_id_from_inputs(
    *,
    manifest: dict[str, Any],
    project_id: str,
    segments: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    manifest_video_id = _optional_str(manifest.get("video_id"))
    if manifest_video_id is not None:
        return manifest_video_id
    for segment in segments:
        segment_video_id = _optional_str(segment.get("video_id"))
        if segment_video_id is not None:
            return segment_video_id
    for observation in observations:
        observation_video_id = _optional_str(observation.get("video_id"))
        if observation_video_id is not None:
            return observation_video_id
    return project_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL not found: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            rows.append(payload)
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _normalized_statuses(statuses: Iterable[str]) -> set[str]:
    return {str(status).strip() for status in statuses if str(status).strip()}


def _validate_config(config: AudioVisualConsistencyConfig) -> None:
    if config.window_margin_seconds < 0:
        raise ValueError("window_margin_seconds must be >= 0")
    if not _normalized_statuses(config.success_statuses):
        raise ValueError("At least one success observation status is required")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_strs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe(str(item) for item in value if str(item).strip())


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
