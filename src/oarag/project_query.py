from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .domain_lexicon import load_domain_lexicon
from .evidence import (
    frame_id,
    make_evidence_window,
    optional_float,
    read_jsonl,
    resolve_project_path,
    resolve_window_config,
    select_window_segments,
)
from .meili import MeiliClient
from .project_index import segment_artifact_path
from .schemas import EntityLink, SearchCandidate, VisualEntity


def query_project(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    limit: int = 5,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    domain_lexicon = load_domain_lexicon(
        project_dir=resolved_project_dir,
        domain_lexicon_path=domain_lexicon_path,
    )
    resolved_segments_path = segment_artifact_path(resolved_project_dir, segments=segments_path)
    resolved_frames_manifest_path = resolve_project_path(
        resolved_project_dir,
        frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )
    resolved_visual_entities_path = resolve_project_path(
        resolved_project_dir,
        visual_entities_path,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
    )
    resolved_entity_links_path = resolve_project_path(
        resolved_project_dir,
        entity_links_path,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
    )

    segments = read_jsonl(resolved_segments_path)
    frames = read_jsonl(resolved_frames_manifest_path) if resolved_frames_manifest_path.exists() else []
    visual_entities = [VisualEntity.from_dict(row) for row in _read_optional_jsonl(resolved_visual_entities_path)]
    entity_links = [EntityLink.from_dict(row) for row in _read_optional_jsonl(resolved_entity_links_path)]

    segment_lookup = {str(segment.get("segment_id")): segment for segment in segments}
    frame_lookup = {frame_id(frame): frame for frame in frames}
    window_config = resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )
    entity_lookup = {entity.entity_id: entity for entity in visual_entities}
    links_by_segment: dict[str, list[EntityLink]] = defaultdict(list)
    for link in entity_links:
        links_by_segment[link.segment_id].append(link)

    search_query = domain_lexicon.expand_query(query)
    search_response = client.search(index_uid, search_query, limit=limit)
    hits = search_response.get("hits", [])

    bundles: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    warnings: list[str] = []
    bundled_entity_total = 0
    bundled_link_total = 0

    for rank, hit in enumerate(hits, start=1):
        candidate = SearchCandidate.from_hit(rank=rank, hit=hit)
        candidates.append(candidate.to_dict())

        target = segment_lookup.get(candidate.segment_id)
        if target is None:
            warnings.append(
                f"Skipped search hit because the segment was not found in project artifacts: {candidate.segment_id}"
            )
            continue

        window_segments = select_window_segments(
            segments,
            target_segment_id=candidate.segment_id,
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        )
        evidence_window = make_evidence_window(
            target=target,
            window_segments=window_segments,
            frame_lookup=frame_lookup,
            window_config=window_config,
        ).to_dict()
        window_visual_entities = _window_visual_entities(
            evidence_window=evidence_window,
            visual_entities=visual_entities,
        )
        linked_entities = _linked_entities_for_segment(
            segment_id=candidate.segment_id,
            evidence_window=evidence_window,
            links_by_segment=links_by_segment,
            entity_lookup=entity_lookup,
        )

        window_entity_ids = {entity["entity_id"] for entity in window_visual_entities}
        for linked in linked_entities:
            entity = linked.get("entity")
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("entity_id", ""))
            if entity_id and entity_id not in window_entity_ids:
                window_visual_entities.append(entity)
                window_entity_ids.add(entity_id)
        window_visual_entities.sort(key=_serialized_entity_sort_key)

        summary = _bundle_summary(
            candidate=candidate,
            evidence_window=evidence_window,
            linked_entities=linked_entities,
        )
        bundles.append(
            {
                "rank": candidate.rank,
                "candidate": candidate.to_dict(),
                "evidence_window": evidence_window,
                "visual_entities": window_visual_entities,
                "linked_entities": linked_entities,
                "summary": summary,
            }
        )
        summary_lines.append(summary["text"])
        bundled_entity_total += len(window_visual_entities)
        bundled_link_total += len(linked_entities)

    return {
        "query": query,
        "index": index_uid,
        "project_id": _project_id(segments=segments, project_dir=resolved_project_dir),
        "processing_time_ms": search_response.get("processingTimeMs"),
        "paths": {
            "project_dir": str(resolved_project_dir),
            "segments": str(resolved_segments_path),
            "frames_manifest": str(resolved_frames_manifest_path),
            "visual_entities": str(resolved_visual_entities_path),
            "entity_links": str(resolved_entity_links_path),
            "domain_lexicon": domain_lexicon.metadata()["source_path"],
        },
        "domain_lexicon": domain_lexicon.metadata(),
        "query_expansion": domain_lexicon.query_expansion_metadata(query),
        "artifact_availability": {
            "frames_manifest": resolved_frames_manifest_path.exists(),
            "visual_entities": resolved_visual_entities_path.exists(),
            "entity_links": resolved_entity_links_path.exists(),
            "domain_lexicon": domain_lexicon.enabled,
        },
        "retrieval_context": {
            "window_config": window_config,
        },
        "counts": {
            "search_hits": len(hits),
            "bundles": len(bundles),
            "skipped_hits": len(warnings),
            "project_segments": len(segments),
            "project_frames": len(frames),
            "project_visual_entities": len(visual_entities),
            "project_entity_links": len(entity_links),
            "bundled_visual_entities": bundled_entity_total,
            "bundled_linked_entities": bundled_link_total,
        },
        "summary_lines": summary_lines,
        "warnings": warnings,
        "candidates": candidates,
        "bundles": bundles,
    }


def _window_visual_entities(
    *,
    evidence_window: dict[str, Any],
    visual_entities: list[VisualEntity],
) -> list[dict[str, Any]]:
    frame_ids = {
        str(frame_ref.get("frame_id", ""))
        for frame_ref in evidence_window.get("frame_refs") or []
        if isinstance(frame_ref, dict) and frame_ref.get("frame_id") is not None
    }
    start_time = optional_float(evidence_window.get("start_time"))
    end_time = optional_float(evidence_window.get("end_time"))

    selected = [
        entity.to_dict()
        for entity in visual_entities
        if _entity_in_window(entity=entity, frame_ids=frame_ids, start_time=start_time, end_time=end_time)
    ]
    selected.sort(key=_serialized_entity_sort_key)
    return selected


def _linked_entities_for_segment(
    *,
    segment_id: str,
    evidence_window: dict[str, Any],
    links_by_segment: dict[str, list[EntityLink]],
    entity_lookup: dict[str, VisualEntity],
) -> list[dict[str, Any]]:
    frame_ids = {
        str(frame_ref.get("frame_id", ""))
        for frame_ref in evidence_window.get("frame_refs") or []
        if isinstance(frame_ref, dict) and frame_ref.get("frame_id") is not None
    }
    start_time = optional_float(evidence_window.get("start_time"))
    end_time = optional_float(evidence_window.get("end_time"))

    selected: list[dict[str, Any]] = []
    for link in sorted(links_by_segment.get(segment_id, []), key=_link_sort_key):
        entity = entity_lookup.get(link.entity_id)
        if entity is not None and not _entity_in_window(
            entity=entity,
            frame_ids=frame_ids,
            start_time=start_time,
            end_time=end_time,
        ):
            continue
        selected.append(
            {
                "link_id": link.link_id,
                "segment_id": link.segment_id,
                "entity_id": link.entity_id,
                "frame_id": link.frame_id,
                "link_type": link.link_type,
                "score": link.score,
                "evidence": link.evidence,
                "time_overlap": link.time_overlap,
                "lexical_match": link.lexical_match,
                "mention_candidate": link.mention_candidate,
                "explanation": _link_explanation(link),
                "entity": entity.to_dict() if entity is not None else None,
            }
        )
    return selected


def _entity_in_window(
    *,
    entity: VisualEntity,
    frame_ids: set[str],
    start_time: float | None,
    end_time: float | None,
) -> bool:
    if entity.frame_id and entity.frame_id in frame_ids:
        return True
    if entity.timestamp is None or start_time is None or end_time is None:
        return False
    return start_time <= entity.timestamp <= end_time


def _bundle_summary(
    *,
    candidate: SearchCandidate,
    evidence_window: dict[str, Any],
    linked_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    frame_paths = [
        str(frame_ref.get("frame_path", ""))
        for frame_ref in evidence_window.get("frame_refs") or []
        if isinstance(frame_ref, dict) and frame_ref.get("frame_path")
    ]
    linked_entity_labels = [
        _linked_entity_label(linked_entity)
        for linked_entity in linked_entities
        if _linked_entity_label(linked_entity) is not None
    ]
    transcript_excerpt = candidate.transcript_excerpt
    time_label = _time_label(candidate.start_time, candidate.end_time)

    summary_text = (
        f"#{candidate.rank} {candidate.video_id} {time_label}: "
        f"{transcript_excerpt or '(empty transcript)'} | "
        f"frames={len(evidence_window.get('frame_refs') or [])} | "
        f"linked_entities={', '.join(linked_entity_labels[:3]) or 'none'}"
    )
    return {
        "text": summary_text,
        "frame_paths": frame_paths,
        "linked_entity_labels": linked_entity_labels,
        "transcript_excerpt": transcript_excerpt,
    }


def _time_label(start_time: float | None, end_time: float | None) -> str:
    if start_time is None and end_time is None:
        return "@unknown"
    if start_time is None:
        return f"@{end_time:.1f}s"
    if end_time is None:
        return f"@{start_time:.1f}s"
    return f"@{start_time:.1f}-{end_time:.1f}s"


def _linked_entity_label(linked_entity: dict[str, Any]) -> str | None:
    entity = linked_entity.get("entity")
    if not isinstance(entity, dict):
        return None
    text = str(entity.get("text", "")).strip()
    if text:
        return text
    entity_id = str(entity.get("entity_id", "")).strip()
    return entity_id or None


def _link_explanation(link: EntityLink) -> str:
    parts = [f"time overlap evidence (score={link.score:.2f})"]
    if link.lexical_match:
        parts.append(f"shared terms: {', '.join(link.lexical_match)}")
    if link.mention_candidate:
        parts.append(f"mention/entity hint: {', '.join(link.mention_candidate)}")
    return "; ".join(parts)


def _serialized_entity_sort_key(entity: dict[str, Any]) -> tuple[bool, float, str, str]:
    timestamp = optional_float(entity.get("timestamp"))
    return (
        timestamp is None,
        timestamp or 0.0,
        str(entity.get("frame_id", "")),
        str(entity.get("entity_id", "")),
    )


def _link_sort_key(link: EntityLink) -> tuple[float, str, str]:
    return (-link.score, link.frame_id, link.entity_id)


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _project_id(*, segments: list[dict[str, Any]], project_dir: Path) -> str:
    if segments and segments[0].get("project_id") is not None:
        return str(segments[0]["project_id"])
    return project_dir.name
