from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from oarag.core.domain_lexicon import DomainLexicon
from oarag.core.schemas import EntityLink, VisualEntity, slugify

REFERENCE_RESOLUTION_SOURCE = "heuristic_time_lookback"
REFERENCE_LOOKBACK_SEGMENT_COUNT = 3
REFERENCE_LOOKBACK_SECONDS = 180.0

_REFERENCE_HINT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"아까\s*말(?:씀)?했(?:던|던\s*그|던\s*이)?\s*개념",
        r"앞에서\s*(?:설명|말)(?:한|했던)\s*(?:내용|개념|부분)",
        r"이\s*개념",
        r"그\s*개념",
        r"이\s*부분",
        r"그\s*부분",
        r"\b(?:this|that)\s+concept\b",
        r"\b(?:this|that)\s+part\b",
        r"\bprevious(?:ly)?\s+(?:concept|point|idea|content|section)\b",
        r"\b(?:earlier|before)\s+(?:concept|point|idea|content|section)\b",
        r"\bwhat\s+(?:I|we)\s+(?:said|explained|covered)\s+(?:earlier|before)\b",
    )
)

_STOP_TERMS = {
    "and",
    "are",
    "for",
    "is",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "with",
    "개념",
    "내용",
    "부분",
    "아까",
    "앞에서",
}


@dataclass(frozen=True)
class ReferenceTarget:
    node_namespace: str
    local_id: str
    target_type: str
    score: float
    reason: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceResolution:
    reference_id: str
    segment_id: str
    hint_text: str
    lookback_segment_id: str
    lookback_segment_count: int
    lookback_seconds: float | None
    score: float
    reason: str
    evidence: tuple[str, ...]
    targets: tuple[ReferenceTarget, ...] = field(default_factory=tuple)


def resolve_references(
    *,
    segments: list[dict[str, Any]],
    domain_lexicon: DomainLexicon,
    entities: list[VisualEntity] | None = None,
    links: list[EntityLink] | None = None,
    lookback_segment_count: int = REFERENCE_LOOKBACK_SEGMENT_COUNT,
    lookback_seconds: float = REFERENCE_LOOKBACK_SECONDS,
) -> list[ReferenceResolution]:
    """Resolve lightweight lecture reference mentions against earlier context."""
    entities = entities or []
    links = links or []
    entities_by_frame = _entities_by_frame(entities)
    links_by_segment = _links_by_segment(links)
    resolutions: list[ReferenceResolution] = []

    for video_id, ordered in _segments_by_video(segments).items():
        for index, segment in enumerate(ordered):
            segment_id = str(segment.get("segment_id", ""))
            if not segment_id:
                continue
            hint_text = _first_reference_hint(str(segment.get("transcript_text", "")))
            if not hint_text:
                continue
            context = _lookback_context(
                ordered=ordered,
                current_index=index,
                lookback_segment_count=lookback_segment_count,
                lookback_seconds=lookback_seconds,
            )
            if not context:
                continue
            resolution = _resolve_against_context(
                video_id=video_id,
                segment=segment,
                hint_text=hint_text,
                context=context,
                domain_lexicon=domain_lexicon,
                entities_by_frame=entities_by_frame,
                links_by_segment=links_by_segment,
            )
            if resolution is not None:
                resolutions.append(resolution)

    return sorted(resolutions, key=lambda item: item.reference_id)


def _resolve_against_context(
    *,
    video_id: str,
    segment: dict[str, Any],
    hint_text: str,
    context: list[tuple[int, dict[str, Any]]],
    domain_lexicon: DomainLexicon,
    entities_by_frame: dict[str, list[VisualEntity]],
    links_by_segment: dict[str, list[EntityLink]],
) -> ReferenceResolution | None:
    for distance, previous in context:
        previous_segment_id = str(previous.get("segment_id", ""))
        if not previous_segment_id:
            continue

        concepts = _concepts_from_segment(previous, domain_lexicon)
        linked_entities = _linked_entities(
            segment=previous,
            entities_by_frame=entities_by_frame,
            links_by_segment=links_by_segment,
        )
        if not concepts and not linked_entities:
            continue

        lookback_gap = _time_gap_seconds(segment, previous)
        score = _score(distance=distance, concepts=concepts, linked_entities=linked_entities)
        evidence = _base_evidence(previous, concepts=concepts, linked_entities=linked_entities)
        reason = _reason(
            hint_text=hint_text,
            segment_id=str(segment.get("segment_id", "")),
            previous_segment_id=previous_segment_id,
            concepts=concepts,
            lookback_segment_count=distance,
            lookback_seconds=lookback_gap,
        )
        targets = [
            ReferenceTarget(
                node_namespace="segment",
                local_id=previous_segment_id,
                target_type="segment",
                score=score,
                reason=reason,
                evidence=evidence,
            )
        ]
        if concepts:
            concept = concepts[0]
            targets.append(
                ReferenceTarget(
                    node_namespace="concept",
                    local_id=concept,
                    target_type="concept",
                    score=score,
                    reason=reason,
                    evidence=evidence,
                )
            )
        if linked_entities:
            entity = linked_entities[0]
            entity_evidence = (*evidence, _entity_evidence(entity))
            targets.append(
                ReferenceTarget(
                    node_namespace="visual_entity",
                    local_id=entity.entity_id,
                    target_type="visual_entity",
                    score=round(score + 0.05, 3),
                    reason=f"{reason} Visual entity evidence is available from aligned frames or links.",
                    evidence=entity_evidence,
                )
            )

        segment_id = str(segment.get("segment_id", ""))
        reference_id = f"{segment_id}:reference:{slugify(hint_text)}"
        return ReferenceResolution(
            reference_id=reference_id,
            segment_id=segment_id,
            hint_text=hint_text,
            lookback_segment_id=previous_segment_id,
            lookback_segment_count=distance,
            lookback_seconds=lookback_gap,
            score=score,
            reason=reason,
            evidence=evidence,
            targets=tuple(targets),
        )

    return None


def _segments_by_video(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        grouped.setdefault(_video_id_from_segment(segment), []).append(segment)
    return {
        video_id: sorted(
            video_segments,
            key=lambda item: (
                _segment_time(item) is None,
                _segment_time(item) or 0.0,
                str(item.get("segment_id", "")),
            ),
        )
        for video_id, video_segments in sorted(grouped.items())
    }


def _lookback_context(
    *,
    ordered: list[dict[str, Any]],
    current_index: int,
    lookback_segment_count: int,
    lookback_seconds: float,
) -> list[tuple[int, dict[str, Any]]]:
    current = ordered[current_index]
    context: list[tuple[int, dict[str, Any]]] = []
    for previous_index in range(current_index - 1, -1, -1):
        distance = current_index - previous_index
        if distance > lookback_segment_count:
            break
        previous = ordered[previous_index]
        gap = _time_gap_seconds(current, previous)
        if gap is not None and gap > lookback_seconds:
            continue
        context.append((distance, previous))
    return context


def _first_reference_hint(text: str) -> str | None:
    for pattern in _REFERENCE_HINT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return " ".join(match.group(0).split())
    return None


def _concepts_from_segment(segment: dict[str, Any], domain_lexicon: DomainLexicon) -> list[str]:
    terms = _candidate_terms(str(segment.get("transcript_text", "")))
    prioritized_terms: list[str] = []
    candidates = segment.get("mention_candidates")
    if isinstance(candidates, list):
        prioritized_terms = [str(candidate) for candidate in candidates]
        terms.update(prioritized_terms)
    prioritized_concepts = [
        domain_lexicon.canonicalize(term)
        for term in prioritized_terms
        if _is_concept_term(domain_lexicon.canonicalize(term))
    ]
    concepts = [
        domain_lexicon.canonicalize(term)
        for term in sorted(terms)
        if _is_concept_term(domain_lexicon.canonicalize(term))
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for concept in [*prioritized_concepts, *concepts]:
        if concept not in seen:
            ordered.append(concept)
            seen.add(concept)
    return ordered


def _candidate_terms(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in re.finditer(r"[0-9a-zA-Z가-힣]+", text)
        if len(match.group(0)) >= 2
    }


def _is_concept_term(term: str) -> bool:
    return bool(term) and not term.isdigit() and term not in _STOP_TERMS


def _linked_entities(
    *,
    segment: dict[str, Any],
    entities_by_frame: dict[str, list[VisualEntity]],
    links_by_segment: dict[str, list[EntityLink]],
) -> list[VisualEntity]:
    segment_id = str(segment.get("segment_id", ""))
    linked_entity_ids = {
        link.entity_id for link in links_by_segment.get(segment_id, ()) if link.entity_id
    }
    frame_ids = _segment_frame_ids(segment)
    entities: dict[str, VisualEntity] = {}
    for frame_id in frame_ids:
        for entity in entities_by_frame.get(frame_id, ()):
            if entity.entity_id:
                entities[entity.entity_id] = entity
    for frame_entities in entities_by_frame.values():
        for entity in frame_entities:
            if entity.entity_id in linked_entity_ids:
                entities[entity.entity_id] = entity
    return sorted(entities.values(), key=lambda item: (item.frame_id, item.entity_id))


def _entities_by_frame(entities: list[VisualEntity]) -> dict[str, list[VisualEntity]]:
    grouped: dict[str, list[VisualEntity]] = {}
    for entity in entities:
        if entity.frame_id:
            grouped.setdefault(entity.frame_id, []).append(entity)
    return grouped


def _links_by_segment(links: list[EntityLink]) -> dict[str, list[EntityLink]]:
    grouped: dict[str, list[EntityLink]] = {}
    for link in links:
        if link.segment_id:
            grouped.setdefault(link.segment_id, []).append(link)
    return grouped


def _segment_frame_ids(segment: dict[str, Any]) -> set[str]:
    frame_refs = segment.get("frame_refs")
    if not isinstance(frame_refs, list):
        return set()
    return {str(frame_ref) for frame_ref in frame_refs if frame_ref is not None}


def _score(
    *,
    distance: int,
    concepts: list[str],
    linked_entities: list[VisualEntity],
) -> float:
    score = 1.0 - ((distance - 1) * 0.15)
    if concepts:
        score += 0.1
    if linked_entities:
        score += 0.15
    return round(min(score, 1.0), 3)


def _base_evidence(
    segment: dict[str, Any],
    *,
    concepts: list[str],
    linked_entities: list[VisualEntity],
) -> tuple[str, ...]:
    evidence = [
        f"lookback_segment_id={segment.get('segment_id')}",
        f"lookback_transcript={_snippet(str(segment.get('transcript_text', '')))}",
    ]
    if concepts:
        evidence.append(f"candidate_concepts={', '.join(concepts[:5])}")
    if linked_entities:
        entity_ids = ", ".join(entity.entity_id for entity in linked_entities[:5])
        evidence.append(f"visual_entity_candidates={entity_ids}")
    return tuple(evidence)


def _entity_evidence(entity: VisualEntity) -> str:
    parts = [f"entity_id={entity.entity_id}", f"frame_id={entity.frame_id}"]
    if entity.text:
        parts.append(f"text={_snippet(entity.text)}")
    if entity.visual_description:
        parts.append(f"visual_description={_snippet(entity.visual_description)}")
    return "; ".join(parts)


def _reason(
    *,
    hint_text: str,
    segment_id: str,
    previous_segment_id: str,
    concepts: list[str],
    lookback_segment_count: int,
    lookback_seconds: float | None,
) -> str:
    concept_text = f" concept '{concepts[0]}'" if concepts else " previous segment context"
    seconds_text = (
        f" and {lookback_seconds:.1f}s"
        if lookback_seconds is not None
        else ""
    )
    return (
        f"Detected reference hint '{hint_text}' in segment {segment_id}; "
        f"resolved to{concept_text} from segment {previous_segment_id} "
        f"within {lookback_segment_count} segment(s){seconds_text} of lookback."
    )


def _snippet(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _time_gap_seconds(current: dict[str, Any], previous: dict[str, Any]) -> float | None:
    current_time = _segment_time(current)
    previous_time = _segment_time(previous)
    if current_time is None or previous_time is None:
        return None
    return round(max(0.0, current_time - previous_time), 3)


def _segment_time(segment: dict[str, Any]) -> float | None:
    for time_field in ("start_time", "timestamp_center", "end_time"):
        value = _optional_float(segment.get(time_field))
        if value is not None:
            return value
    return None


def _video_id_from_segment(segment: dict[str, Any]) -> str:
    return str(segment.get("video_id") or segment.get("video_name") or "video")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
