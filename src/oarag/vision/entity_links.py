from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from oarag.core.domain_lexicon import DomainLexicon, load_domain_lexicon
from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.project_index import segment_artifact_path
from oarag.core.schemas import EntityLink, VisualEntity, mention_candidates, slugify

STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

TIME_OVERLAP_SCORE = 0.2
VISUAL_MATCH_FIELDS = (
    "text",
    "visual_description",
    "entity_type",
    "position",
    "relations",
)
VISUAL_FIELD_EVIDENCE = {
    "text": "visual_text_match",
    "visual_description": "visual_description_match",
    "entity_type": "entity_type_match",
    "position": "position_match",
    "relations": "relations_match",
}
VISUAL_FIELD_SCORE_WEIGHTS = {
    "text": (0.4, 0.05, 0.6),
    "visual_description": (0.42, 0.05, 0.65),
    "entity_type": (0.14, 0.02, 0.22),
    "position": (0.1, 0.02, 0.16),
    "relations": (0.16, 0.02, 0.26),
}
MENTION_SCORE_BASE = 0.45
MENTION_SCORE_PER_EXTRA_TERM = 0.05
MENTION_SCORE_CAP = 0.6


def link_entities(
    *,
    project_dir: Path,
    segments_path: Path | None = None,
    visual_entities_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    domain_lexicon = load_domain_lexicon(
        project_dir=resolved_project_dir,
        domain_lexicon_path=domain_lexicon_path,
    )
    resolved_segments_path = segment_artifact_path(resolved_project_dir, segments=segments_path)
    resolved_visual_entities_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=visual_entities_path,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
    )
    resolved_output_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=output_path,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
    )
    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )

    segments = _read_jsonl(resolved_segments_path)
    entities = [VisualEntity.from_dict(row) for row in _read_jsonl(resolved_visual_entities_path)]
    project_id = _project_id_from_manifest_or_dir(resolved_manifest_path, resolved_project_dir, segments, entities)

    links: list[EntityLink] = []
    for segment in _sorted_segments(segments):
        for entity in _sorted_entities(entities):
            time_overlap = _has_time_overlap(segment, entity)
            if not time_overlap:
                continue

            link_evidence = _link_evidence(
                segment,
                entity,
                domain_lexicon=domain_lexicon,
            )

            links.append(
                EntityLink(
                    link_id=f"link_{slugify(str(segment.get('segment_id', '')))}_{slugify(entity.entity_id)}",
                    project_id=project_id,
                    segment_id=str(segment.get("segment_id", "")),
                    entity_id=entity.entity_id,
                    frame_id=entity.frame_id,
                    link_type=link_evidence["link_type"],
                    score=link_evidence["score"],
                    evidence=link_evidence["evidence"],
                    time_overlap=True,
                    lexical_match=link_evidence["lexical_match"],
                    mention_candidate=link_evidence["mention_candidate"],
                    score_breakdown=link_evidence["score_breakdown"],
                    reason_metadata=link_evidence["reason_metadata"],
                )
            )

    evidence_type_counts = _evidence_type_counts(links)
    score_breakdown_totals = _score_breakdown_totals(links)

    write_jsonl(resolved_output_path, [link.to_dict() for link in links])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        entity_links_path=resolved_output_path,
        segments_total=len(segments),
        visual_entities_total=len(entities),
        links=links,
        evidence_type_counts=evidence_type_counts,
        score_breakdown_totals=score_breakdown_totals,
        domain_lexicon_metadata=domain_lexicon.metadata(),
    )

    return {
        "project_id": project_id,
        "paths": {
            "project_dir": str(resolved_project_dir),
            "segments": str(resolved_segments_path),
            "visual_entities": str(resolved_visual_entities_path),
            "entity_links": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
            "domain_lexicon": domain_lexicon.metadata()["source_path"],
        },
        "domain_lexicon": domain_lexicon.metadata(),
        "counts": {
            "segments_total": len(segments),
            "visual_entities_total": len(entities),
            "entity_links": len(links),
            "lexical_links": sum(1 for link in links if link.lexical_match),
            "mention_links": sum(1 for link in links if link.mention_candidate),
            "timestamp_only_links": sum(
                1 for link in links if not link.lexical_match and not link.mention_candidate
            ),
            "semantic_links": sum(
                1 for link in links if link.lexical_match or link.mention_candidate
            ),
            "evidence_type_counts": evidence_type_counts,
            "score_summary": _score_summary(links),
            "score_breakdown_totals": score_breakdown_totals,
        },
    }


def _sorted_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        segments,
        key=lambda segment: (
            _optional_float(segment.get("timestamp_center")) is None,
            _optional_float(segment.get("timestamp_center")) or _optional_float(segment.get("start_time")) or 0.0,
            str(segment.get("segment_id", "")),
        ),
    )


def _sorted_entities(entities: list[VisualEntity]) -> list[VisualEntity]:
    return sorted(
        entities,
        key=lambda entity: (
            entity.timestamp is None,
            entity.timestamp or 0.0,
            entity.frame_id,
            entity.entity_id,
        ),
    )


def _has_time_overlap(segment: dict[str, Any], entity: VisualEntity) -> bool:
    frame_refs = {str(frame_ref) for frame_ref in segment.get("frame_refs") or []}
    if entity.frame_id and entity.frame_id in frame_refs:
        return True

    entity_time = entity.timestamp
    if entity_time is None:
        return False
    start = _optional_float(segment.get("start_time"))
    end = _optional_float(segment.get("end_time"))
    center = _optional_float(segment.get("timestamp_center"))
    if start is None and end is None:
        return center is not None and entity_time == center
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return False
    return min(start, end) <= entity_time <= max(start, end)


def _link_evidence(
    segment: dict[str, Any],
    entity: VisualEntity,
    *,
    domain_lexicon: DomainLexicon,
) -> dict[str, Any]:
    lexical_matches_by_field = _lexical_matches_by_field(
        segment,
        entity,
        domain_lexicon=domain_lexicon,
    )
    mention_matches_by_field = _mention_matches_by_field(
        segment,
        entity,
        domain_lexicon=domain_lexicon,
    )
    lexical_match = _unique_sorted_terms(lexical_matches_by_field)
    mention_match = _unique_sorted_terms(mention_matches_by_field)
    evidence = _evidence_types(
        lexical_matches_by_field=lexical_matches_by_field,
        mention_matches_by_field=mention_matches_by_field,
    )
    score_breakdown = _score_breakdown(
        lexical_matches_by_field=lexical_matches_by_field,
        mention_match=mention_match,
    )
    return {
        "link_type": _link_type(
            lexical_match=lexical_match,
            mention_match=mention_match,
            evidence=evidence,
        ),
        "score": round(sum(score_breakdown.values()), 3),
        "evidence": evidence,
        "lexical_match": lexical_match,
        "mention_candidate": mention_match,
        "score_breakdown": score_breakdown,
        "reason_metadata": _reason_metadata(
            entity=entity,
            lexical_matches_by_field=lexical_matches_by_field,
            mention_matches_by_field=mention_matches_by_field,
        ),
    }


def _lexical_matches_by_field(
    segment: dict[str, Any],
    entity: VisualEntity,
    *,
    domain_lexicon: DomainLexicon,
) -> dict[str, list[str]]:
    transcript_terms = _expanded_terms(
        str(segment.get("transcript_text", "")),
        domain_lexicon=domain_lexicon,
    )
    matches_by_field: dict[str, list[str]] = {}
    for field_name, field_text in _visual_field_texts(entity).items():
        field_terms = _expanded_terms(field_text, domain_lexicon=domain_lexicon)
        matches_by_field[field_name] = sorted(
            term for term in transcript_terms & field_terms if _informative_term(term)
        )
    return matches_by_field


def _mention_matches_by_field(
    segment: dict[str, Any],
    entity: VisualEntity,
    *,
    domain_lexicon: DomainLexicon,
) -> dict[str, list[str]]:
    transcript_text = str(segment.get("transcript_text", ""))
    all_candidates = _candidate_mentions(segment=segment, transcript_text=transcript_text)

    candidate_terms: set[str] = set()
    for candidate in all_candidates:
        candidate_terms.update(_expanded_terms(candidate, domain_lexicon=domain_lexicon))

    matches_by_field: dict[str, list[str]] = {}
    for field_name, field_text in _visual_field_texts(entity).items():
        field_text_casefold = field_text.casefold()
        direct_matches = {
            candidate.strip()
            for candidate in all_candidates
            if candidate.strip() and candidate.strip().casefold() in field_text_casefold
        }
        field_terms = _expanded_terms(field_text, domain_lexicon=domain_lexicon)
        alias_matches = {
            term for term in candidate_terms & field_terms if _informative_term(term)
        }
        matches_by_field[field_name] = sorted(direct_matches | alias_matches)
    return matches_by_field


def _candidate_mentions(*, segment: dict[str, Any], transcript_text: str) -> list[str]:
    candidates = segment.get("mention_candidates")
    if not isinstance(candidates, list):
        candidates = []
    all_candidates = [str(candidate) for candidate in candidates]
    all_candidates.extend(mention_candidates(transcript_text))
    return _unique_nonempty(all_candidates)


def _visual_field_texts(entity: VisualEntity) -> dict[str, str]:
    return {
        "text": entity.text,
        "visual_description": entity.visual_description or "",
        "entity_type": entity.entity_type.replace("_", " "),
        "position": _visual_metadata_text(entity.position),
        "relations": _visual_metadata_text(entity.relations),
    }


def _visual_metadata_text(value: Any) -> str:
    return " ".join(_visual_metadata_values(value))


def _visual_metadata_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[str] = []
        for child in value.values():
            values.extend(_visual_metadata_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_visual_metadata_values(child))
        return values
    return [str(value)]


def _terms(text: str) -> set[str]:
    return {match.group(0) for match in re.finditer(r"[0-9a-zA-Z가-힣]+", text.casefold())}


def _expanded_terms(text: str, *, domain_lexicon: DomainLexicon) -> set[str]:
    return {domain_lexicon.canonicalize(term) for term in _terms(text)}


def _informative_term(term: str) -> bool:
    return len(term) >= 2 and not term.isdigit() and term not in STOP_TERMS


def _unique_sorted_terms(matches_by_field: dict[str, list[str]]) -> list[str]:
    terms = {
        term
        for matches in matches_by_field.values()
        for term in matches
        if _informative_term(term)
    }
    return sorted(terms)


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_values.append(cleaned)
    return unique_values


def _evidence_types(
    *,
    lexical_matches_by_field: dict[str, list[str]],
    mention_matches_by_field: dict[str, list[str]],
) -> list[str]:
    evidence = ["time_overlap"]
    if any(lexical_matches_by_field.values()):
        evidence.append("lexical_match")
    if any(mention_matches_by_field.values()):
        evidence.append("mention_candidate")
    for field_name in VISUAL_MATCH_FIELDS:
        if lexical_matches_by_field.get(field_name) or mention_matches_by_field.get(field_name):
            evidence.append(VISUAL_FIELD_EVIDENCE[field_name])
    if evidence == ["time_overlap"]:
        evidence.append("timestamp_fallback")
    return evidence


def _link_type(
    *,
    lexical_match: list[str],
    mention_match: list[str],
    evidence: list[str],
) -> str:
    if lexical_match and mention_match:
        parts = ["time_overlap", "lexical_match", "mention_candidate"]
    elif lexical_match:
        parts = ["time_overlap", "lexical_match"]
    elif mention_match:
        parts = ["time_overlap", "mention_candidate"]
    else:
        return "time_overlap"

    for evidence_type in (
        "visual_description_match",
        "entity_type_match",
        "position_match",
        "relations_match",
    ):
        if evidence_type in evidence:
            parts.append(evidence_type)
    return "+".join(parts)


def _score_breakdown(
    *,
    lexical_matches_by_field: dict[str, list[str]],
    mention_match: list[str],
) -> dict[str, float]:
    breakdown = {"time_overlap": TIME_OVERLAP_SCORE}
    for field_name in VISUAL_MATCH_FIELDS:
        matches = lexical_matches_by_field.get(field_name, [])
        if not matches:
            continue
        base, per_extra_term, cap = VISUAL_FIELD_SCORE_WEIGHTS[field_name]
        component = min(cap, base + (per_extra_term * max(0, len(matches) - 1)))
        breakdown[VISUAL_FIELD_EVIDENCE[field_name]] = round(component, 3)
    if mention_match:
        component = min(
            MENTION_SCORE_CAP,
            MENTION_SCORE_BASE + (MENTION_SCORE_PER_EXTRA_TERM * max(0, len(mention_match) - 1)),
        )
        breakdown["mention_candidate"] = round(component, 3)
    return breakdown


def _reason_metadata(
    *,
    entity: VisualEntity,
    lexical_matches_by_field: dict[str, list[str]],
    mention_matches_by_field: dict[str, list[str]],
) -> dict[str, Any]:
    lexical_details = _nonempty_matches(lexical_matches_by_field)
    mention_details = _nonempty_matches(mention_matches_by_field)
    matched_fields = sorted(set(lexical_details) | set(mention_details))
    metadata: dict[str, Any] = {
        "summary": _reason_summary(
            lexical_matches_by_field=lexical_matches_by_field,
            mention_matches_by_field=mention_matches_by_field,
        ),
        "matched_visual_fields": matched_fields,
        "visual_fields_considered": _visual_fields_considered(entity),
    }
    if lexical_details:
        metadata["lexical_matches_by_field"] = lexical_details
    if mention_details:
        metadata["mention_matches_by_field"] = mention_details
    if entity.source_model:
        metadata["source_model"] = entity.source_model
    if entity.source:
        metadata["source"] = entity.source
    return metadata


def _reason_summary(
    *,
    lexical_matches_by_field: dict[str, list[str]],
    mention_matches_by_field: dict[str, list[str]],
) -> str:
    if not any(lexical_matches_by_field.values()) and not any(mention_matches_by_field.values()):
        return "timestamp_fallback_only"
    if (
        lexical_matches_by_field.get("visual_description")
        or mention_matches_by_field.get("visual_description")
    ):
        if any(mention_matches_by_field.values()):
            return "mention_candidate_visual_description_match"
        return "visual_description_match"
    if any(mention_matches_by_field.values()):
        return "mention_candidate_visual_match"
    return "visual_lexical_match"


def _nonempty_matches(matches_by_field: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        field_name: matches
        for field_name, matches in matches_by_field.items()
        if matches
    }


def _visual_fields_considered(entity: VisualEntity) -> list[str]:
    considered = [
        field_name
        for field_name, text in _visual_field_texts(entity).items()
        if text.strip()
    ]
    if entity.source_model:
        considered.append("source_model")
    return considered


def _evidence_type_counts(links: list[EntityLink]) -> dict[str, int]:
    counts = {
        "time_overlap": sum(1 for link in links if link.time_overlap),
        "lexical_match": sum(1 for link in links if link.lexical_match),
        "mention_candidate": sum(1 for link in links if link.mention_candidate),
        "timestamp_only": sum(
            1 for link in links if not link.lexical_match and not link.mention_candidate
        ),
        "semantic_match": sum(
            1 for link in links if link.lexical_match or link.mention_candidate
        ),
    }
    for evidence_type in (
        "timestamp_fallback",
        "visual_text_match",
        "visual_description_match",
        "entity_type_match",
        "position_match",
        "relations_match",
    ):
        counts[evidence_type] = sum(1 for link in links if evidence_type in link.evidence)
    return counts


def _score_summary(links: list[EntityLink]) -> dict[str, float | None]:
    if not links:
        return {"min": None, "max": None, "avg": None}
    scores = [link.score for link in links]
    return {
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "avg": round(sum(scores) / len(scores), 3),
    }


def _score_breakdown_totals(links: list[EntityLink]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for link in links:
        for component, value in link.score_breakdown.items():
            totals[component] = totals.get(component, 0.0) + value
    return {component: round(value, 3) for component, value in sorted(totals.items())}


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


def _project_id_from_manifest_or_dir(
    manifest_path: Path,
    project_dir: Path,
    segments: list[dict[str, Any]],
    entities: list[VisualEntity],
) -> str:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("project_id") is not None:
            return str(loaded["project_id"])
    if segments and segments[0].get("project_id") is not None:
        return str(segments[0]["project_id"])
    if entities:
        return entities[0].project_id
    return project_dir.name


def _update_project_manifest(
    *,
    manifest_path: Path,
    entity_links_path: Path,
    segments_total: int,
    visual_entities_total: int,
    links: list[EntityLink],
    evidence_type_counts: dict[str, int],
    score_breakdown_totals: dict[str, float],
    domain_lexicon_metadata: dict[str, Any],
) -> None:
    payload: dict[str, Any]
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["entity_links"] = str(entity_links_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["entity_links"] = len(links)

    payload["entity_linking"] = {
        "segments_total": segments_total,
        "visual_entities_total": visual_entities_total,
        "entity_links": len(links),
        "lexical_links": sum(1 for link in links if link.lexical_match),
        "mention_links": sum(1 for link in links if link.mention_candidate),
        "timestamp_only_links": sum(
            1 for link in links if not link.lexical_match and not link.mention_candidate
        ),
        "semantic_links": sum(
            1 for link in links if link.lexical_match or link.mention_candidate
        ),
        "evidence_type_counts": evidence_type_counts,
        "score_summary": _score_summary(links),
        "score_breakdown_totals": score_breakdown_totals,
        "domain_lexicon": domain_lexicon_metadata,
    }
    write_json(manifest_path, payload)


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
