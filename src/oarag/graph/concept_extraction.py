from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from oarag.core.io import write_json
from oarag.core.schemas import slugify
from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    CONCEPT_GRAPH_SCHEMA_VERSION,
    ConceptGraphConceptNode,
    ConceptGraphEvidenceSource,
    write_concept_graph_artifact,
)
from oarag.retrieval.project_index import evidence_unit_artifact_path, iter_jsonl_documents


CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION = "oarag-concept-candidates-v1"
CONCEPT_CANDIDATE_PUBLIC_NOTE = (
    "Concept candidates are lecture-local nodes extracted from evidence unit source "
    "signals. Public summaries report counts and coverage only; raw transcripts, raw "
    "frame paths, and local absolute paths are not copied into the summary."
)

SOURCE_SIGNAL_TRANSCRIPT_MENTION = "transcript_mention"
SOURCE_SIGNAL_SLIDE_TEXT = "slide_text"
SOURCE_SIGNAL_DETECTED_TEXT = "detected_text"
SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION = "vlm_visual_observation"

DEFAULT_SOURCE_CONFIDENCE = {
    SOURCE_SIGNAL_TRANSCRIPT_MENTION: 0.64,
    SOURCE_SIGNAL_SLIDE_TEXT: 0.62,
    SOURCE_SIGNAL_DETECTED_TEXT: 0.58,
    SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION: 0.56,
}

_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z0-9가-힣]*|[0-9]+[A-Za-z가-힣]+|[0-9]+")
_MAX_CANDIDATE_TOKENS = 4

_STOP_TERMS = {
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
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "these",
    "those",
    "to",
    "we",
    "with",
}

_ACTION_TERMS = {
    "adjust",
    "adjusts",
    "describe",
    "describes",
    "enter",
    "enters",
    "explain",
    "explains",
    "highlight",
    "highlights",
    "indicate",
    "indicates",
    "measure",
    "measures",
    "move",
    "moves",
    "moving",
    "show",
    "showing",
    "shown",
    "shows",
    "transform",
    "transforms",
    "update",
    "updates",
    "use",
    "uses",
    "using",
}

_GENERIC_TERMS = {
    "annotation",
    "candidate",
    "caption",
    "chart",
    "concept",
    "course",
    "data",
    "description",
    "diagram",
    "example",
    "figure",
    "frame",
    "graph",
    "image",
    "lecture",
    "model",
    "object",
    "overview",
    "parameter",
    "parameters",
    "picture",
    "rule",
    "sample",
    "slide",
    "source",
    "summary",
    "term",
    "text",
    "thing",
    "title",
    "topic",
    "video",
    "visual",
}

_BOUNDARY_TERMS = _STOP_TERMS | _ACTION_TERMS | {
    "arrow",
    "board",
    "box",
    "caption",
    "diagram",
    "example",
    "figure",
    "image",
    "lower",
    "rule",
    "slide",
    "toward",
    "visual",
}

_UPPERCASE_TERMS = {
    "api",
    "cnn",
    "l1",
    "l2",
    "ocr",
    "relu",
    "rnn",
    "vlm",
}


@dataclass(frozen=True)
class ConceptCandidateExtractionResult:
    lecture_id: str
    concepts: tuple[ConceptGraphConceptNode, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _SourceText:
    evidence_unit_id: str
    source_type: str
    source_signal: str
    text: str
    source_id: str | None = None
    confidence: float | None = None
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class _ConceptAggregate:
    label: str
    first_seen: int
    occurrence_count: int = 0
    evidence_sources: dict[tuple[str, str, str | None], ConceptGraphEvidenceSource] = field(
        default_factory=dict
    )
    source_unit_ids: list[str] = field(default_factory=list)
    source_signal_counts: Counter[str] = field(default_factory=Counter)
    source_type_counts: Counter[str] = field(default_factory=Counter)

    def add(self, source: _SourceText) -> None:
        self.occurrence_count += 1
        self.source_signal_counts[source.source_signal] += 1
        self.source_type_counts[source.source_type] += 1
        if source.evidence_unit_id not in self.source_unit_ids:
            self.source_unit_ids.append(source.evidence_unit_id)
        key = (source.evidence_unit_id, source.source_signal, source.source_id)
        existing = self.evidence_sources.get(key)
        evidence_source = ConceptGraphEvidenceSource(
            evidence_unit_id=source.evidence_unit_id,
            source_type=source.source_type,
            source_signal=source.source_signal,
            source_id=source.source_id,
            confidence=source.confidence,
            start_time=source.start_time,
            end_time=source.end_time,
            is_verified_object_alignment=False,
            metadata={"raw_text_redacted": True},
        )
        if existing is None:
            self.evidence_sources[key] = evidence_source
            return
        if (evidence_source.confidence or 0.0) > (existing.confidence or 0.0):
            self.evidence_sources[key] = evidence_source


def extract_project_concept_candidates(
    *,
    project_dir: Path,
    evidence_units: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    lecture_id: str | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    evidence_units_path = evidence_unit_artifact_path(resolved_project_dir, evidence_units)
    resolved_output_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    )
    resolved_manifest_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    rows = list(iter_jsonl_documents(evidence_units_path))
    manifest = _load_json_object(resolved_manifest_path)
    resolved_lecture_id = _lecture_id(
        explicit=lecture_id,
        manifest=manifest,
        evidence_units=rows,
        project_dir=resolved_project_dir,
    )
    confidence_threshold = _confidence_threshold(min_confidence)
    result = extract_concept_candidates(
        rows,
        lecture_id=resolved_lecture_id,
        min_confidence=confidence_threshold,
    )
    concepts = list(result.concepts)
    write_concept_graph_artifact(resolved_output_path, concepts)
    summary = _summary(
        project_dir=resolved_project_dir,
        evidence_units_path=evidence_units_path,
        output_path=resolved_output_path,
        manifest_path=resolved_manifest_path,
        result=result,
    )
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        project_dir=resolved_project_dir,
        concept_graph_path=resolved_output_path,
        summary=summary,
    )
    return summary


def extract_concept_candidates(
    evidence_units: Iterable[dict[str, Any]],
    *,
    lecture_id: str,
    min_confidence: float = 0.0,
) -> ConceptCandidateExtractionResult:
    aggregates: dict[str, _ConceptAggregate] = {}
    source_texts_total = 0
    extracted_mentions_total = 0
    pruned_mentions_total = 0
    evidence_units_total = 0
    evidence_units_with_candidates: set[str] = set()
    source_text_counts: Counter[str] = Counter()

    for evidence_unit in evidence_units:
        evidence_units_total += 1
        unit_candidate_keys: set[str] = set()
        for source_text in _source_texts_for_unit(evidence_unit):
            source_texts_total += 1
            source_text_counts[source_text.source_signal] += 1
            labels, pruned_count = _candidate_labels_from_text(source_text.text)
            pruned_mentions_total += pruned_count
            for label in labels:
                key = _candidate_key(label)
                aggregate = aggregates.get(key)
                if aggregate is None:
                    aggregate = _ConceptAggregate(
                        label=label,
                        first_seen=len(aggregates) + 1,
                    )
                    aggregates[key] = aggregate
                aggregate.add(source_text)
                unit_candidate_keys.add(key)
                extracted_mentions_total += 1
        if unit_candidate_keys:
            evidence_unit_id = _text(evidence_unit.get("evidence_unit_id"))
            if evidence_unit_id:
                evidence_units_with_candidates.add(evidence_unit_id)

    concepts = [
        _concept_node(aggregate, lecture_id=lecture_id)
        for aggregate in aggregates.values()
    ]
    concepts = [
        concept
        for concept in concepts
        if concept.confidence >= _confidence_threshold(min_confidence)
    ]
    concepts.sort(
        key=lambda concept: (
            -concept.confidence,
            int(concept.metadata.get("first_seen") or 0),
            concept.label.casefold(),
        )
    )
    concepts = [_with_rank(concept, rank=index + 1) for index, concept in enumerate(concepts)]
    source_signal_counts = Counter()
    for concept in concepts:
        source_signal_counts.update(concept.metadata.get("source_signal_counts", {}))
    diagnostics = {
        "schema_version": CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION,
        "evidence_units_total": evidence_units_total,
        "evidence_units_with_candidates": len(evidence_units_with_candidates),
        "source_texts_total": source_texts_total,
        "source_text_counts": dict(sorted(source_text_counts.items())),
        "extracted_mentions_total": extracted_mentions_total,
        "pruned_mentions_total": pruned_mentions_total,
        "concept_candidates_total": len(concepts),
        "source_evidence_total": sum(len(concept.evidence_sources) for concept in concepts),
        "source_signal_counts": dict(sorted(source_signal_counts.items())),
        "raw_text_redacted": True,
    }
    return ConceptCandidateExtractionResult(
        lecture_id=lecture_id,
        concepts=tuple(concepts),
        diagnostics=diagnostics,
    )


def _concept_node(
    aggregate: _ConceptAggregate,
    *,
    lecture_id: str,
) -> ConceptGraphConceptNode:
    source_signals = dict(sorted(aggregate.source_signal_counts.items()))
    confidence = _aggregate_confidence(aggregate)
    return ConceptGraphConceptNode(
        concept_id=_concept_id(aggregate.label),
        lecture_id=lecture_id,
        label=aggregate.label,
        aliases=(),
        concept_type="candidate_concept",
        source_evidence_unit_ids=tuple(aggregate.source_unit_ids),
        evidence_sources=tuple(aggregate.evidence_sources.values()),
        confidence=confidence,
        metadata={
            "candidate_extractor_schema_version": CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION,
            "first_seen": aggregate.first_seen,
            "occurrence_count": aggregate.occurrence_count,
            "source_unit_count": len(aggregate.source_unit_ids),
            "source_signal_counts": source_signals,
            "source_type_counts": dict(sorted(aggregate.source_type_counts.items())),
            "generic_pruning": "minimal_v1",
            "raw_text_redacted": True,
        },
    )


def _with_rank(
    concept: ConceptGraphConceptNode,
    *,
    rank: int,
) -> ConceptGraphConceptNode:
    metadata = dict(concept.metadata)
    metadata["candidate_rank"] = rank
    return ConceptGraphConceptNode(
        concept_id=concept.concept_id,
        lecture_id=concept.lecture_id,
        label=concept.label,
        concept_type=concept.concept_type,
        source_evidence_unit_ids=concept.source_evidence_unit_ids,
        evidence_sources=concept.evidence_sources,
        confidence=concept.confidence,
        aliases=concept.aliases,
        description=concept.description,
        metadata=metadata,
    )


def _source_texts_for_unit(unit: dict[str, Any]) -> list[_SourceText]:
    evidence_unit_id = _text(unit.get("evidence_unit_id"))
    if not evidence_unit_id:
        return []
    start_time = _optional_float(unit.get("start_time"))
    end_time = _optional_float(unit.get("end_time"))
    sources: list[_SourceText] = []

    transcript = _text(unit.get("transcript_window_text"))
    if transcript:
        sources.append(
            _SourceText(
                evidence_unit_id=evidence_unit_id,
                source_type="transcript",
                source_signal=SOURCE_SIGNAL_TRANSCRIPT_MENTION,
                text=transcript,
                source_id=_transcript_source_id(unit),
                confidence=DEFAULT_SOURCE_CONFIDENCE[SOURCE_SIGNAL_TRANSCRIPT_MENTION],
                start_time=start_time,
                end_time=end_time,
            )
        )

    for state in _dict_items(unit.get("visual_states")):
        state_id = _text(state.get("visual_state_id")) or None
        for text in _text_values(
            [
                state.get("slide_text"),
                state.get("slide_texts"),
                state.get("title"),
                state.get("heading"),
                state.get("state_summary"),
            ]
        ):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="slide_text",
                    source_signal=SOURCE_SIGNAL_SLIDE_TEXT,
                    text=text,
                    source_id=state_id,
                    confidence=DEFAULT_SOURCE_CONFIDENCE[SOURCE_SIGNAL_SLIDE_TEXT],
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        for text in _text_values([state.get("detected_text")]):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="detected_text",
                    source_signal=SOURCE_SIGNAL_DETECTED_TEXT,
                    text=text,
                    source_id=state_id,
                    confidence=DEFAULT_SOURCE_CONFIDENCE[SOURCE_SIGNAL_DETECTED_TEXT],
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    for entity in _dict_items(unit.get("visual_entities")):
        entity_id = _text(entity.get("entity_id")) or None
        if _is_ocr_like_entity(entity):
            for text in _text_values([entity.get("text")]):
                sources.append(
                    _SourceText(
                        evidence_unit_id=evidence_unit_id,
                        source_type="detected_text",
                        source_signal=SOURCE_SIGNAL_DETECTED_TEXT,
                        text=text,
                        source_id=entity_id,
                        confidence=_entity_confidence(entity, SOURCE_SIGNAL_DETECTED_TEXT),
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
        for text in _text_values([entity.get("detected_text")]):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="detected_text",
                    source_signal=SOURCE_SIGNAL_DETECTED_TEXT,
                    text=text,
                    source_id=entity_id,
                    confidence=_entity_confidence(entity, SOURCE_SIGNAL_DETECTED_TEXT),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        for text in _text_values([entity.get("visual_description"), entity.get("description")]):
            sources.append(
                _SourceText(
                    evidence_unit_id=evidence_unit_id,
                    source_type="vlm_visual_description",
                    source_signal=SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION,
                    text=text,
                    source_id=entity_id,
                    confidence=_entity_confidence(entity, SOURCE_SIGNAL_VLM_VISUAL_DESCRIPTION),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
    return sources


def _candidate_labels_from_text(text: str) -> tuple[list[str], int]:
    chunks = _content_chunks(text)
    labels: list[str] = []
    pruned = 0
    for chunk in chunks:
        spans = _candidate_spans(chunk)
        if not spans:
            pruned += 1
            continue
        for span in spans:
            if not _valid_candidate_tokens(span):
                pruned += 1
                continue
            labels.append(_label_from_tokens(span))
    return _unique_labels(labels), pruned


def _content_chunks(text: str) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for token in _tokens(text):
        if token in _BOUNDARY_TERMS:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(token)
    if current:
        chunks.append(current)
    return chunks


def _candidate_spans(tokens: list[str]) -> list[list[str]]:
    trimmed = _trim_generic_edges(tokens)
    if not trimmed:
        return []
    if len(trimmed) <= _MAX_CANDIDATE_TOKENS:
        spans = [trimmed]
        if len(trimmed) > 2:
            spans.extend(trimmed[index : index + 2] for index in range(len(trimmed) - 1))
        return spans

    spans = []
    for width in (4, 3, 2):
        for index in range(0, len(trimmed) - width + 1):
            spans.append(trimmed[index : index + width])
    return spans


def _trim_generic_edges(tokens: list[str]) -> list[str]:
    start = 0
    end = len(tokens)
    while start < end and tokens[start] in _GENERIC_TERMS:
        start += 1
    while end > start and tokens[end - 1] in _GENERIC_TERMS:
        end -= 1
    return tokens[start:end]


def _valid_candidate_tokens(tokens: list[str]) -> bool:
    if not tokens or len(tokens) > _MAX_CANDIDATE_TOKENS:
        return False
    if not any(_has_letter(token) for token in tokens):
        return False
    specific = [
        token
        for token in tokens
        if token not in _STOP_TERMS and token not in _GENERIC_TERMS and _has_letter(token)
    ]
    if not specific:
        return False
    if len(tokens) == 1:
        token = tokens[0]
        return len(token) >= 3 and token not in _GENERIC_TERMS
    return tokens[0] not in _STOP_TERMS and tokens[-1] not in _STOP_TERMS


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _label_from_tokens(tokens: list[str]) -> str:
    formatted = [_format_token(token) for token in tokens]
    label = " ".join(formatted).strip()
    return label[:1].upper() + label[1:] if label else ""


def _format_token(token: str) -> str:
    if token in _UPPERCASE_TERMS:
        return token.upper()
    return token


def _unique_labels(labels: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        key = _candidate_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(label)
    return unique


def _candidate_key(label: str) -> str:
    return " ".join(_tokens(label)).casefold()


def _concept_id(label: str) -> str:
    base = slugify(_candidate_key(label).replace(" ", "_")).casefold()
    if not base or base == "empty":
        digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
        base = f"non_ascii_{digest}"
    return f"concept_{base}"


def _aggregate_confidence(aggregate: _ConceptAggregate) -> float:
    base = max((source.confidence or 0.5) for source in aggregate.evidence_sources.values())
    signal_bonus = min(0.16, 0.04 * max(len(aggregate.source_signal_counts) - 1, 0))
    unit_bonus = min(0.12, 0.04 * max(len(aggregate.source_unit_ids) - 1, 0))
    occurrence_bonus = min(0.08, 0.01 * max(aggregate.occurrence_count - 1, 0))
    return round(min(0.95, base + signal_bonus + unit_bonus + occurrence_bonus), 3)


def _summary(
    *,
    project_dir: Path,
    evidence_units_path: Path,
    output_path: Path,
    manifest_path: Path,
    result: ConceptCandidateExtractionResult,
) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics)
    evidence_units_total = int(diagnostics.get("evidence_units_total") or 0)
    units_with_candidates = int(diagnostics.get("evidence_units_with_candidates") or 0)
    coverage_ratio = round(units_with_candidates / evidence_units_total, 6) if evidence_units_total else None
    return {
        "schema_version": CONCEPT_CANDIDATE_EXTRACTION_SCHEMA_VERSION,
        "concept_graph_schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "lecture_id": result.lecture_id,
        "paths": {
            "evidence_units": _public_project_path(project_dir, evidence_units_path),
            "concept_graph": _public_project_path(project_dir, output_path),
            "project_manifest": _public_project_path(project_dir, manifest_path),
        },
        "counts": {
            "evidence_units_total": evidence_units_total,
            "evidence_units_with_candidates": units_with_candidates,
            "concept_candidates_total": diagnostics["concept_candidates_total"],
            "source_evidence_total": diagnostics["source_evidence_total"],
            "source_texts_total": diagnostics["source_texts_total"],
            "extracted_mentions_total": diagnostics["extracted_mentions_total"],
            "pruned_mentions_total": diagnostics["pruned_mentions_total"],
        },
        "coverage": {
            "unit_coverage_ratio": coverage_ratio,
            "source_signal_counts": diagnostics["source_signal_counts"],
            "source_text_counts": diagnostics["source_text_counts"],
        },
        "public_safety": {
            "raw_text_redacted": True,
            "raw_frame_paths_redacted": True,
            "local_absolute_paths_redacted": True,
        },
        "public_note": CONCEPT_CANDIDATE_PUBLIC_NOTE,
    }


def _update_project_manifest(
    *,
    manifest_path: Path,
    project_dir: Path,
    concept_graph_path: Path,
    summary: dict[str, Any],
) -> None:
    payload = _load_json_object(manifest_path)
    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["concept_graph"] = _public_project_path(project_dir, concept_graph_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["concept_candidates"] = summary["counts"]["concept_candidates_total"]

    payload["concept_candidate_extraction"] = {
        "schema_version": summary["schema_version"],
        "lecture_id": summary["lecture_id"],
        "counts": summary["counts"],
        "coverage": summary["coverage"],
        "public_safety": summary["public_safety"],
        "public_note": summary["public_note"],
    }
    write_json(manifest_path, payload)


def _lecture_id(
    *,
    explicit: str | None,
    manifest: dict[str, Any],
    evidence_units: list[dict[str, Any]],
    project_dir: Path,
) -> str:
    if _text(explicit):
        return _text(explicit)
    for key in ("lecture_id", "video_id", "project_id"):
        value = _text(manifest.get(key))
        if value:
            return value
    for row in evidence_units:
        for key in ("lecture_id", "video_id", "project_id"):
            value = _text(row.get(key))
            if value:
                return value
    return project_dir.name


def _transcript_source_id(unit: dict[str, Any]) -> str | None:
    source_segment_ids = _string_list(unit.get("source_segment_ids"))
    if source_segment_ids:
        return source_segment_ids[0]
    return _text(unit.get("target_segment_id")) or None


def _entity_confidence(entity: dict[str, Any], source_signal: str) -> float:
    confidence = _optional_float(entity.get("confidence"))
    if confidence is None:
        return DEFAULT_SOURCE_CONFIDENCE[source_signal]
    return max(0.0, min(1.0, confidence))


def _is_ocr_like_entity(entity: dict[str, Any]) -> bool:
    source = _text(entity.get("source")).casefold()
    entity_type = _text(entity.get("entity_type")).casefold()
    return source.startswith("ocr") or entity_type == "ocr_text"


def _resolve_project_output_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _public_project_path(project_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _confidence_threshold(value: float) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    return parsed


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text_values(values: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            texts.extend(_text_values(value.values()))
        elif isinstance(value, (list, tuple, set)):
            texts.extend(_text_values(value))
        else:
            text = _text(value)
            if text:
                texts.append(text)
    return texts


def _string_list(value: Any) -> list[str]:
    return _text_values([value])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_letter(token: str) -> bool:
    return any(character.isalpha() for character in token)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text
