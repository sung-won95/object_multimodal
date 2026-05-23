from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import urllib.error

from oarag.core.domain_lexicon import QueryExpansion, load_domain_lexicon, normalize_term
from oarag.retrieval.evidence import (
    frame_id,
    make_evidence_window,
    optional_float,
    read_jsonl,
    resolve_project_path,
    resolve_window_config,
    segment_sort_key,
    segment_window,
    select_window_segments,
)
from oarag.integrations.meili import MeiliClient
from oarag.retrieval.project_index import segment_artifact_path
from oarag.retrieval.rerank import rerank_bundles, rerank_metadata
from oarag.core.schemas import EntityLink, SearchCandidate, VisualEntity


SEGMENT_HIT_SOURCE = "segment"
WINDOW_HIT_SOURCE = "window"
VISUAL_ENTITY_HIT_SOURCE = "visual_entity"
SOURCE_PRIORITY = {
    SEGMENT_HIT_SOURCE: 0,
    WINDOW_HIT_SOURCE: 0,
    VISUAL_ENTITY_HIT_SOURCE: 1,
}
PRIMARY_INDEX_KINDS = {SEGMENT_HIT_SOURCE, WINDOW_HIT_SOURCE}


def query_project(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    retrieval_index_kind: str = SEGMENT_HIT_SOURCE,
    visual_index_uid: str | None = None,
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
    rerank: bool = False,
    rerank_time_hint: str | None = None,
) -> dict[str, Any]:
    if retrieval_index_kind not in PRIMARY_INDEX_KINDS:
        valid = ", ".join(sorted(PRIMARY_INDEX_KINDS))
        raise ValueError(f"Unknown retrieval_index_kind: {retrieval_index_kind}. Valid kinds: {valid}")

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
    links_by_entity: dict[str, list[EntityLink]] = defaultdict(list)
    for link in entity_links:
        links_by_segment[link.segment_id].append(link)
        links_by_entity[link.entity_id].append(link)

    query_expansion = domain_lexicon.expand_query_result(query)
    search_queries = _search_queries_for_expansion(query=query, query_expansion=query_expansion)
    search_response = _search_with_expanded_queries(
        client=client,
        index_uid=index_uid,
        queries=search_queries,
        limit=limit,
    )
    primary_hits = search_response.get("hits", [])

    bundles: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    warnings: list[str] = []
    bundled_entity_total = 0
    bundled_link_total = 0

    visual_search_response: dict[str, Any] | None = None
    visual_hits: list[dict[str, Any]] = []
    if visual_index_uid:
        try:
            visual_search_response = _search_with_expanded_queries(
                client=client,
                index_uid=visual_index_uid,
                queries=search_queries,
                limit=limit,
            )
            visual_hits = visual_search_response.get("hits", [])
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            warnings.append(
                f"Skipped visual entity search because the index was not found: {visual_index_uid}"
            )

    retrieval_entries: list[dict[str, Any]] = []
    for rank, hit in enumerate(primary_hits, start=1):
        candidate = _primary_candidate_from_hit(
            rank=rank,
            hit=hit,
            retrieval_index_kind=retrieval_index_kind,
        )
        candidate["source"] = retrieval_index_kind
        candidates.append(candidate)
        retrieval_entries.append(
            {
                "source": retrieval_index_kind,
                "index": index_uid,
                "rank": rank,
                "score": candidate.get("score"),
                "hit": hit,
                "candidate": candidate,
                "target_segment_id": _primary_target_segment_id(
                    hit=hit,
                    candidate=candidate,
                    retrieval_index_kind=retrieval_index_kind,
                ),
            }
        )

    for rank, hit in enumerate(visual_hits, start=1):
        entity = _visual_entity_from_hit(hit=hit, entity_lookup=entity_lookup)
        target_segment_id, target_resolution = _resolve_visual_entity_target_segment(
            hit=hit,
            entity=entity,
            segments=segments,
            segment_lookup=segment_lookup,
            links_by_entity=links_by_entity,
        )
        candidate = _visual_entity_candidate_from_hit(
            rank=rank,
            hit=hit,
            entity=entity,
            target_segment_id=target_segment_id,
            target=segment_lookup.get(target_segment_id or ""),
        )
        candidates.append(candidate)
        retrieval_entries.append(
            {
                "source": VISUAL_ENTITY_HIT_SOURCE,
                "index": visual_index_uid,
                "rank": rank,
                "score": candidate.get("score"),
                "hit": hit,
                "candidate": candidate,
                "entity": entity,
                "target_segment_id": target_segment_id,
                "target_resolution": target_resolution,
            }
        )

    entries_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in retrieval_entries:
        target_segment_id = str(entry.get("target_segment_id") or "")
        if not target_segment_id or target_segment_id not in segment_lookup:
            source = entry.get("source")
            if source == VISUAL_ENTITY_HIT_SOURCE:
                candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
                warnings.append(
                    "Skipped visual entity search hit because no nearby segment could be "
                    f"resolved: {candidate.get('entity_id') or candidate.get('frame_id') or '(unknown)'}"
                )
            else:
                warnings.append(
                    "Skipped search hit because the segment was not found in project "
                    f"artifacts: {target_segment_id}"
                )
            continue
        entries_by_target[target_segment_id].append(entry)

    merged_targets = sorted(
        entries_by_target.items(),
        key=lambda item: _target_merge_sort_key(item[0], item[1]),
    )[:limit]

    for merged_rank, (target_segment_id, entries) in enumerate(merged_targets, start=1):
        primary_entry = _primary_retrieval_entry(entries)
        candidate = copy.deepcopy(primary_entry["candidate"])
        candidate["merged_rank"] = merged_rank
        retrieval_sources = [
            _retrieval_source_summary(entry)
            for entry in sorted(entries, key=_retrieval_source_sort_key)
        ]
        candidate["merged_sources"] = [
            {
                "source": source["source"],
                "index": source["index"],
                "rank": source["rank"],
                "score": source["score"],
            }
            for source in retrieval_sources
        ]

        target = segment_lookup[target_segment_id]
        window_segments = select_window_segments(
            segments,
            target_segment_id=target_segment_id,
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
        _add_visual_hit_frames_to_window(
            evidence_window=evidence_window,
            entries=entries,
            frame_lookup=frame_lookup,
        )
        window_visual_entities = _window_visual_entities(
            evidence_window=evidence_window,
            visual_entities=visual_entities,
        )
        linked_entities = _linked_entities_for_segment(
            segment_id=target_segment_id,
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
        for entry in entries:
            if entry.get("source") != VISUAL_ENTITY_HIT_SOURCE:
                continue
            entity = entry.get("entity")
            if not isinstance(entity, VisualEntity):
                continue
            entity_id = entity.entity_id
            if entity_id and entity_id not in window_entity_ids:
                window_visual_entities.append(entity.to_dict())
                window_entity_ids.add(entity_id)
        window_visual_entities.sort(key=_serialized_entity_sort_key)

        summary = _bundle_summary_from_dict(
            candidate=candidate,
            evidence_window=evidence_window,
            linked_entities=linked_entities,
            display_rank=merged_rank,
        )
        bundles.append(
            {
                "rank": merged_rank,
                "candidate": candidate,
                "retrieval_sources": retrieval_sources,
                "merge": {
                    "target_key": f"segment:{target_segment_id}",
                    "target_segment_id": target_segment_id,
                    "source_count": len(retrieval_sources),
                    "selected_source": primary_entry["source"],
                    "selected_rank": primary_entry["rank"],
                    "deduplicated": len(retrieval_sources) > 1,
                },
                "evidence_window": evidence_window,
                "visual_entities": window_visual_entities,
                "linked_entities": linked_entities,
                "summary": summary,
            }
        )
        summary_lines.append(summary["text"])
        bundled_entity_total += len(window_visual_entities)
        bundled_link_total += len(linked_entities)

    rerank_context = rerank_metadata(enabled=False, query=query, timestamp_hint=rerank_time_hint)
    if rerank:
        bundles, rerank_context = rerank_bundles(
            bundles=bundles,
            query=query,
            timestamp_hint=rerank_time_hint,
        )
        summary_lines = _refresh_bundle_summaries(bundles)

    return {
        "query": query,
        "index": index_uid,
        "index_kind": retrieval_index_kind,
        "visual_index": visual_index_uid,
        "project_id": _project_id(segments=segments, project_dir=resolved_project_dir),
        "processing_time_ms": _combined_processing_time_ms(
            search_response,
            visual_search_response,
        ),
        "paths": {
            "project_dir": str(resolved_project_dir),
            "segments": str(resolved_segments_path),
            "frames_manifest": str(resolved_frames_manifest_path),
            "visual_entities": str(resolved_visual_entities_path),
            "entity_links": str(resolved_entity_links_path),
            "domain_lexicon": domain_lexicon.metadata()["source_path"],
        },
        "domain_lexicon": domain_lexicon.metadata(),
        "query_expansion": {
            **query_expansion.metadata(),
            "search_queries": search_queries,
        },
        "artifact_availability": {
            "frames_manifest": resolved_frames_manifest_path.exists(),
            "visual_entities": resolved_visual_entities_path.exists(),
            "entity_links": resolved_entity_links_path.exists(),
            "domain_lexicon": domain_lexicon.enabled,
        },
        "retrieval_context": {
            "indexes": {
                "segment": index_uid if retrieval_index_kind == SEGMENT_HIT_SOURCE else None,
                "window": index_uid if retrieval_index_kind == WINDOW_HIT_SOURCE else None,
                "visual_entity": visual_index_uid,
            },
            "searches": {
                "segment": _search_metadata(search_response, index_uid)
                if retrieval_index_kind == SEGMENT_HIT_SOURCE
                else None,
                "window": _search_metadata(search_response, index_uid)
                if retrieval_index_kind == WINDOW_HIT_SOURCE
                else None,
                "visual_entity": _search_metadata(visual_search_response, visual_index_uid)
                if visual_index_uid
                else None,
            },
            "window_config": window_config,
            "rerank": rerank_context,
        },
        "counts": {
            "search_hits": len(primary_hits) + len(visual_hits),
            "segment_search_hits": len(primary_hits)
            if retrieval_index_kind == SEGMENT_HIT_SOURCE
            else 0,
            "window_search_hits": len(primary_hits)
            if retrieval_index_kind == WINDOW_HIT_SOURCE
            else 0,
            "visual_entity_search_hits": len(visual_hits),
            "merged_candidate_targets": len(entries_by_target),
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


def _primary_candidate_from_hit(
    *,
    rank: int,
    hit: dict[str, Any],
    retrieval_index_kind: str,
) -> dict[str, Any]:
    if retrieval_index_kind == WINDOW_HIT_SOURCE:
        return _window_candidate_from_hit(rank=rank, hit=hit)
    return SearchCandidate.from_hit(rank=rank, hit=hit).to_dict()


def _primary_target_segment_id(
    *,
    hit: dict[str, Any],
    candidate: dict[str, Any],
    retrieval_index_kind: str,
) -> str:
    if retrieval_index_kind == WINDOW_HIT_SOURCE:
        return str(hit.get("target_segment_id") or candidate.get("segment_id") or "")
    return str(candidate.get("segment_id") or "")


def _window_candidate_from_hit(*, rank: int, hit: dict[str, Any]) -> dict[str, Any]:
    transcript = str(hit.get("transcript_window_text") or hit.get("transcript_text") or "")
    excerpt = transcript[:360] + ("..." if len(transcript) > 360 else "")
    target_segment_id = str(hit.get("target_segment_id") or hit.get("segment_id") or "")
    candidate = {
        "rank": rank,
        "segment_id": target_segment_id,
        "target_segment_id": target_segment_id,
        "window_id": str(hit.get("window_id", "")),
        "sample_id": str(hit.get("sample_id", "")),
        "video_id": str(hit.get("video_id", hit.get("video_name", ""))),
        "start_time": optional_float(hit.get("start_time")),
        "end_time": optional_float(hit.get("end_time")),
        "timestamp_center": optional_float(hit.get("timestamp_center")),
        "target_segment_start_time": optional_float(hit.get("target_start_time")),
        "target_segment_end_time": optional_float(hit.get("target_end_time")),
        "target_segment_timestamp_center": optional_float(hit.get("target_timestamp_center")),
        "transcript_excerpt": excerpt,
        "score": optional_float(hit.get("_rankingScore")),
        "semantic_source_fields": _semantic_source_fields(hit.get("semantic_source_fields")),
        "source_segment_ids": _semantic_source_fields(hit.get("source_segment_ids")),
    }
    return {key: value for key, value in candidate.items() if value not in (None, "", [])}


def _semantic_source_fields(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _visual_entity_from_hit(
    *,
    hit: dict[str, Any],
    entity_lookup: dict[str, VisualEntity],
) -> VisualEntity:
    entity_id = str(hit.get("entity_id", ""))
    existing = entity_lookup.get(entity_id)
    if existing is not None:
        return existing
    return VisualEntity.from_dict(hit)


def _visual_entity_candidate_from_hit(
    *,
    rank: int,
    hit: dict[str, Any],
    entity: VisualEntity,
    target_segment_id: str | None,
    target: dict[str, Any] | None,
) -> dict[str, Any]:
    target = target or {}
    transcript = str(target.get("transcript_text", ""))
    timestamp = entity.timestamp
    visual_text = _visual_entity_text(entity)
    return {
        "rank": rank,
        "source": VISUAL_ENTITY_HIT_SOURCE,
        "segment_id": target_segment_id or "",
        "sample_id": str(target.get("sample_id", "")),
        "video_id": str(target.get("video_id", hit.get("video_id", ""))),
        "start_time": timestamp,
        "end_time": timestamp,
        "timestamp_center": timestamp,
        "transcript_excerpt": transcript[:360] + ("..." if len(transcript) > 360 else ""),
        "score": optional_float(hit.get("_rankingScore")),
        "entity_id": entity.entity_id,
        "frame_id": entity.frame_id,
        "frame_path": entity.frame_path,
        "visual_entity_text": visual_text,
        "visual_description": entity.visual_description,
        "entity_type": entity.entity_type,
        "confidence": entity.confidence,
        "target_segment_start_time": optional_float(target.get("start_time")),
        "target_segment_end_time": optional_float(target.get("end_time")),
        "target_segment_timestamp_center": optional_float(target.get("timestamp_center")),
    }


def _resolve_visual_entity_target_segment(
    *,
    hit: dict[str, Any],
    entity: VisualEntity,
    segments: list[dict[str, Any]],
    segment_lookup: dict[str, dict[str, Any]],
    links_by_entity: dict[str, list[EntityLink]],
) -> tuple[str | None, dict[str, Any]]:
    hit_segment_id = str(hit.get("segment_id", "")).strip()
    if hit_segment_id and hit_segment_id in segment_lookup:
        return hit_segment_id, {"method": "hit_segment_id", "segment_id": hit_segment_id}

    link = _best_entity_link(
        entity_id=entity.entity_id,
        links_by_entity=links_by_entity,
        segment_lookup=segment_lookup,
    )
    if link is not None:
        return link.segment_id, {
            "method": "entity_link",
            "segment_id": link.segment_id,
            "link_id": link.link_id,
            "score": link.score,
        }

    frame_candidates = [
        segment
        for segment in segments
        if entity.frame_id and entity.frame_id in {str(ref) for ref in segment.get("frame_refs") or []}
    ]
    if frame_candidates:
        selected = _nearest_segment_to_timestamp(
            segments=frame_candidates,
            timestamp=entity.timestamp,
        )
        segment_id = str(selected.get("segment_id", ""))
        return segment_id, {
            "method": "frame_ref",
            "segment_id": segment_id,
            "frame_id": entity.frame_id,
        }

    if entity.timestamp is not None and segments:
        selected = _nearest_segment_to_timestamp(
            segments=segments,
            timestamp=entity.timestamp,
        )
        segment_id = str(selected.get("segment_id", ""))
        return segment_id, {
            "method": "timestamp_nearest",
            "segment_id": segment_id,
            "timestamp": entity.timestamp,
        }

    return None, {"method": "unresolved"}


def _best_entity_link(
    *,
    entity_id: str,
    links_by_entity: dict[str, list[EntityLink]],
    segment_lookup: dict[str, dict[str, Any]],
) -> EntityLink | None:
    links = [link for link in links_by_entity.get(entity_id, []) if link.segment_id in segment_lookup]
    if not links:
        return None
    return sorted(
        links,
        key=lambda link: (
            -link.score,
            _segment_lookup_sort_key(segment_lookup[link.segment_id]),
            link.segment_id,
            link.link_id,
        ),
    )[0]


def _nearest_segment_to_timestamp(
    *,
    segments: list[dict[str, Any]],
    timestamp: float | None,
) -> dict[str, Any]:
    if not segments:
        raise ValueError("segments must not be empty")
    return sorted(
        enumerate(segments),
        key=lambda item: (
            _segment_timestamp_distance(item[1], timestamp),
            segment_sort_key(item[1], item[0]),
            str(item[1].get("segment_id", "")),
        ),
    )[0][1]


def _segment_lookup_sort_key(segment: dict[str, Any]) -> tuple[float, int]:
    sample_index = _optional_int(segment.get("sample_index")) or 0
    return segment_sort_key(segment, sample_index)


def _segment_timestamp_distance(segment: dict[str, Any], timestamp: float | None) -> float:
    if timestamp is None:
        return float("inf")
    window = segment_window(segment)
    if window is None:
        center = optional_float(segment.get("timestamp_center"))
        if center is None:
            return float("inf")
        return abs(center - timestamp)
    if window[0] <= timestamp <= window[1]:
        return 0.0
    return min(abs(timestamp - window[0]), abs(timestamp - window[1]))


def _primary_retrieval_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(entries, key=_retrieval_entry_merge_key)[0]


def _target_merge_sort_key(
    target_segment_id: str,
    entries: list[dict[str, Any]],
) -> tuple[int, int, str, str]:
    primary = _primary_retrieval_entry(entries)
    rank, priority, discriminator = _retrieval_entry_merge_key(primary)
    return rank, priority, target_segment_id, discriminator


def _retrieval_entry_merge_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    source = str(entry.get("source", ""))
    rank = _optional_int(entry.get("rank")) or 0
    rank = rank if rank > 0 else 10**9
    priority = SOURCE_PRIORITY.get(source, 99)
    candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
    discriminator = str(candidate.get("entity_id") or candidate.get("segment_id") or "")
    return rank, priority, discriminator


def _retrieval_source_sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    source = str(entry.get("source", ""))
    rank = _optional_int(entry.get("rank")) or 0
    return SOURCE_PRIORITY.get(source, 99), rank, str(entry.get("target_segment_id", ""))


def _retrieval_source_summary(entry: dict[str, Any]) -> dict[str, Any]:
    candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
    summary = {
        "source": entry.get("source"),
        "index": entry.get("index"),
        "rank": entry.get("rank"),
        "score": entry.get("score"),
        "target_segment_id": entry.get("target_segment_id"),
    }
    if entry.get("source") == VISUAL_ENTITY_HIT_SOURCE:
        summary.update(
            {
                "entity_id": candidate.get("entity_id"),
                "frame_id": candidate.get("frame_id"),
                "timestamp": candidate.get("timestamp_center"),
                "visual_entity_text": candidate.get("visual_entity_text"),
                "target_resolution": entry.get("target_resolution"),
            }
        )
    if entry.get("source") == WINDOW_HIT_SOURCE:
        summary.update(
            {
                "window_id": candidate.get("window_id"),
                "source_segment_ids": candidate.get("source_segment_ids"),
            }
        )
    return summary


def _add_visual_hit_frames_to_window(
    *,
    evidence_window: dict[str, Any],
    entries: list[dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
) -> None:
    frame_refs = evidence_window.setdefault("frame_refs", [])
    if not isinstance(frame_refs, list):
        frame_refs = []
        evidence_window["frame_refs"] = frame_refs
    seen = {
        str(frame.get("frame_id", ""))
        for frame in frame_refs
        if isinstance(frame, dict) and frame.get("frame_id") is not None
    }
    for entry in entries:
        if entry.get("source") != VISUAL_ENTITY_HIT_SOURCE:
            continue
        entity = entry.get("entity")
        if not isinstance(entity, VisualEntity) or not entity.frame_id or entity.frame_id in seen:
            continue
        frame = dict(frame_lookup.get(entity.frame_id, {}))
        frame.setdefault("frame_id", entity.frame_id)
        if entity.timestamp is not None:
            frame.setdefault("timestamp", entity.timestamp)
        if entity.frame_path:
            frame.setdefault("frame_path", entity.frame_path)
        frame_refs.append(frame)
        seen.add(entity.frame_id)
    frame_refs.sort(key=lambda frame: (optional_float(frame.get("timestamp")) is None, optional_float(frame.get("timestamp")) or 0.0, str(frame.get("frame_id", ""))))


def _combined_processing_time_ms(*responses: dict[str, Any] | None) -> int | float | None:
    values = [
        optional_float(response.get("processingTimeMs"))
        for response in responses
        if isinstance(response, dict) and response.get("processingTimeMs") is not None
    ]
    if not values:
        return None
    total = sum(values)
    return int(total) if total.is_integer() else total


def _search_queries_for_expansion(*, query: str, query_expansion: QueryExpansion) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for candidate in [query, *(term.term for term in query_expansion.terms)]:
        normalized = normalize_term(candidate)
        if not normalized or normalized in seen:
            continue
        queries.append(candidate)
        seen.add(normalized)
    return queries


def _search_with_expanded_queries(
    *,
    client: MeiliClient,
    index_uid: str,
    queries: list[str],
    limit: int,
) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    hit_records: list[dict[str, Any]] = []
    hit_positions: dict[str, int] = {}

    for query_index, search_query in enumerate(queries):
        response = client.search(index_uid, search_query, limit=limit)
        responses.append(response)
        for rank, hit in enumerate(response.get("hits", []), start=1):
            hit_key = _search_hit_key(hit)
            score = _hit_score(hit)
            if hit_key in hit_positions:
                existing_index = hit_positions[hit_key]
                if score > hit_records[existing_index]["score"]:
                    hit_records[existing_index]["hit"] = hit
                    hit_records[existing_index]["score"] = score
                    hit_records[existing_index]["query_index"] = query_index
                    hit_records[existing_index]["rank"] = rank
                continue
            hit_positions[hit_key] = len(hit_records)
            hit_records.append(
                {
                    "hit": hit,
                    "score": score,
                    "query_index": query_index,
                    "rank": rank,
                    "first_seen": len(hit_records),
                }
            )

    aggregate = copy.deepcopy(responses[0]) if responses else {}
    ranked_hit_records = sorted(hit_records, key=_expanded_hit_record_sort_key)
    aggregate["hits"] = [record["hit"] for record in ranked_hit_records[:limit]]
    aggregate["processingTimeMs"] = _combined_processing_time_ms(*responses)
    aggregate["query"] = queries[0] if len(queries) == 1 else queries
    aggregate["queries"] = queries
    aggregate["hitCountBeforeLimit"] = len(hit_records)
    return aggregate


def _search_metadata(response: dict[str, Any] | None, index_uid: str | None) -> dict[str, Any] | None:
    if response is None:
        return None
    metadata = {
        "index": index_uid,
        "processing_time_ms": response.get("processingTimeMs"),
        "hit_count": len(response.get("hits", [])),
    }
    if "queries" in response:
        metadata["queries"] = response["queries"]
    return metadata


def _search_hit_key(hit: dict[str, Any]) -> str:
    for field in ("window_id", "target_segment_id", "segment_id", "sample_id", "entity_id", "id"):
        value = hit.get(field)
        if value is not None:
            return f"{field}:{value}"
    return json.dumps(hit, ensure_ascii=False, sort_keys=True)


def _hit_score(hit: dict[str, Any]) -> float:
    score = optional_float(hit.get("_rankingScore"))
    return score if score is not None else float("-inf")


def _expanded_hit_record_sort_key(record: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        -float(record["score"]),
        int(record["query_index"]),
        int(record["rank"]),
        int(record["first_seen"]),
    )


def _visual_entity_text(entity: VisualEntity) -> str:
    return str(entity.text or entity.visual_description or "").strip()


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
                "score_breakdown": link.score_breakdown,
                "reason_metadata": link.reason_metadata,
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
    display_rank: int | None = None,
) -> dict[str, Any]:
    return _bundle_summary_from_dict(
        candidate=candidate.to_dict(),
        evidence_window=evidence_window,
        linked_entities=linked_entities,
        display_rank=display_rank,
    )


def _bundle_summary_from_dict(
    *,
    candidate: dict[str, Any],
    evidence_window: dict[str, Any],
    linked_entities: list[dict[str, Any]],
    display_rank: int | None = None,
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
    transcript_excerpt = str(candidate.get("transcript_excerpt", ""))
    time_label = _time_label(
        optional_float(candidate.get("start_time")),
        optional_float(candidate.get("end_time")),
    )
    original_rank = _optional_int(candidate.get("rank"))
    rank = display_rank or original_rank or 0
    rank_label = f"#{rank}"
    if display_rank is not None and original_rank is not None and display_rank != original_rank:
        rank_label = f"#{display_rank} (orig #{original_rank})"

    visual_entity_text = str(candidate.get("visual_entity_text", "")).strip()
    if candidate.get("source") == VISUAL_ENTITY_HIT_SOURCE:
        main_excerpt = f"visual_entity={visual_entity_text or '(empty visual entity)'}"
        if transcript_excerpt:
            main_excerpt += f" | transcript={transcript_excerpt}"
    else:
        main_excerpt = transcript_excerpt or "(empty transcript)"

    summary_text = (
        f"{rank_label} {candidate.get('video_id')} {time_label}: "
        f"{main_excerpt} | "
        f"frames={len(evidence_window.get('frame_refs') or [])} | "
        f"linked_entities={', '.join(linked_entity_labels[:3]) or 'none'}"
    )
    return {
        "text": summary_text,
        "frame_paths": frame_paths,
        "linked_entity_labels": linked_entity_labels,
        "transcript_excerpt": transcript_excerpt,
        "visual_entity_text": visual_entity_text or None,
    }


def _refresh_bundle_summaries(bundles: list[dict[str, Any]]) -> list[str]:
    summary_lines: list[str] = []
    for display_rank, bundle in enumerate(bundles, start=1):
        candidate = bundle.get("candidate") if isinstance(bundle.get("candidate"), dict) else {}
        evidence_window = (
            bundle.get("evidence_window") if isinstance(bundle.get("evidence_window"), dict) else {}
        )
        linked_entities = (
            bundle.get("linked_entities") if isinstance(bundle.get("linked_entities"), list) else []
        )
        summary = _bundle_summary_from_dict(
            candidate=candidate,
            evidence_window=evidence_window,
            linked_entities=linked_entities,
            display_rank=display_rank,
        )
        rerank_info = bundle.get("rerank")
        if isinstance(rerank_info, dict):
            score = rerank_info.get("score")
            original_rank = rerank_info.get("original_rank")
            summary["text"] += f" | rerank_score={score} | original_rank={original_rank}"
            summary["rerank_explanation"] = rerank_info.get("explanation")
        bundle["summary"] = summary
        summary_lines.append(summary["text"])
    return summary_lines


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
    if "timestamp_fallback" in link.evidence:
        parts = [f"timestamp-only fallback (score={link.score:.2f})"]
    else:
        parts = [f"entity link evidence (score={link.score:.2f})"]
    if link.lexical_match:
        parts.append(f"shared terms: {', '.join(link.lexical_match)}")
    if link.mention_candidate:
        parts.append(f"mention/entity hint: {', '.join(link.mention_candidate)}")
    semantic_matches = _metadata_match_terms(link.reason_metadata, "semantic_matches_by_field")
    if semantic_matches:
        parts.append(f"semantic hint: {', '.join(semantic_matches)}")
    domain_matches = _metadata_match_terms(link.reason_metadata, "domain_lexicon_matches_by_field")
    if domain_matches:
        parts.append(f"domain alias: {', '.join(domain_matches)}")
    reference_cues = _metadata_match_terms(link.reason_metadata, "reference_cues_by_field")
    if reference_cues:
        parts.append(f"reference/position cue: {', '.join(reference_cues)}")
    if link.score_breakdown:
        parts.append(
            "score components: "
            + ", ".join(
                f"{key}={value:.2f}"
                for key, value in sorted(link.score_breakdown.items())
            )
        )
    return "; ".join(parts)


def _metadata_match_terms(metadata: dict[str, Any], key: str) -> list[str]:
    values = metadata.get(key)
    if not isinstance(values, dict):
        return []
    terms = {
        str(item)
        for matches in values.values()
        if isinstance(matches, list)
        for item in matches
        if str(item).strip()
    }
    return sorted(terms)


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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
