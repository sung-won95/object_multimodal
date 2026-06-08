from __future__ import annotations

import copy
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import urllib.error

from oarag.core.domain_lexicon import (
    DomainLexicon,
    QueryExpansion,
    load_domain_lexicon,
    normalize_term,
)
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
from oarag.integrations.meili import MeiliClient, normalize_query_vector
from oarag.embeddings.manifest import (
    NO_SEMANTIC_QUALITY_CLAIM,
    PROVIDER_EMBEDDING_QUALITY_CLAIM,
    embedding_backend_contract,
)
from oarag.retrieval.project_index import segment_artifact_path
from oarag.retrieval.rerank import DEFAULT_RERANK_BACKEND, rerank_bundles, rerank_metadata
from oarag.retrieval.vectors import LOCAL_HASH_VECTOR_SOURCE, deterministic_text_vector
from oarag.core.schemas import EntityLink, SearchCandidate, VisualEntity


SEGMENT_HIT_SOURCE = "segment"
WINDOW_HIT_SOURCE = "window"
VISUAL_ENTITY_HIT_SOURCE = "visual_entity"
LEXICAL_RETRIEVAL_MODE = "lexical"
SEMANTIC_RETRIEVAL_MODE = "semantic"
DEFAULT_HYBRID_EMBEDDER = "default"
DEFAULT_HYBRID_SEMANTIC_RATIO = 1.0
DEFAULT_QUERY_VECTOR_MANIFEST_RELATIVE_PATH = Path("manifests") / "query_vectors.json"
LOCAL_HASH_QUERY_VECTOR_WARNING = (
    "local_hash_v1 query vectors are deterministic smoke-test fallback only; "
    "do not report semantic embedding quality without provider-backed query vectors."
)
UNDECLARED_SEMANTIC_BACKEND_WARNING = (
    "Hybrid semantic retrieval has no provider-backed query embedding metadata in "
    "this response; semantic quality claims are disabled for this query result."
)
SOURCE_PRIORITY = {
    SEGMENT_HIT_SOURCE: 0,
    WINDOW_HIT_SOURCE: 0,
    VISUAL_ENTITY_HIT_SOURCE: 1,
}
RETRIEVAL_MODE_PRIORITY = {
    LEXICAL_RETRIEVAL_MODE: 0,
    SEMANTIC_RETRIEVAL_MODE: 1,
}
SOURCE_MODALITY = {
    SEGMENT_HIT_SOURCE: "transcript",
    WINDOW_HIT_SOURCE: "transcript",
    VISUAL_ENTITY_HIT_SOURCE: "visual",
}
HYBRID_RRF_RANK_CONSTANT = 60
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
    candidate_pool_limit: int | None = None,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
    disable_domain_lexicon: bool = False,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    rerank: bool = False,
    rerank_time_hint: str | None = None,
    rerank_backend: str = DEFAULT_RERANK_BACKEND,
    hybrid_retrieval: bool = False,
    hybrid_embedder: str | None = DEFAULT_HYBRID_EMBEDDER,
    hybrid_semantic_ratio: float = DEFAULT_HYBRID_SEMANTIC_RATIO,
    hybrid_query_vector: list[float] | None = None,
    hybrid_query_vector_embedder: str | None = None,
    hybrid_query_vector_name: str | None = None,
    hybrid_query_vector_dimensions: int | None = None,
    hybrid_query_vector_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if retrieval_index_kind not in PRIMARY_INDEX_KINDS:
        valid = ", ".join(sorted(PRIMARY_INDEX_KINDS))
        raise ValueError(f"Unknown retrieval_index_kind: {retrieval_index_kind}. Valid kinds: {valid}")

    result_limit = _positive_int(limit, field_name="limit")
    resolved_candidate_pool_limit = _candidate_pool_limit(
        result_limit=result_limit,
        candidate_pool_limit=candidate_pool_limit,
    )
    resolved_project_dir = project_dir.expanduser().resolve()
    domain_lexicon = (
        DomainLexicon()
        if disable_domain_lexicon
        else load_domain_lexicon(
            project_dir=resolved_project_dir,
            domain_lexicon_path=domain_lexicon_path,
        )
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
    project_filter = _project_filter(project_dir=resolved_project_dir, segments=segments)
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
    hybrid_query_vector_spec = _resolve_hybrid_query_vector(
        project_dir=resolved_project_dir,
        hybrid_retrieval=hybrid_retrieval,
        hybrid_embedder=hybrid_embedder,
        query_text=query,
        query_vector=hybrid_query_vector,
        query_vector_embedder=hybrid_query_vector_embedder,
        query_vector_name=hybrid_query_vector_name,
        query_vector_dimensions=hybrid_query_vector_dimensions,
        query_vector_manifest_path=hybrid_query_vector_manifest_path,
    )
    search_channels = _search_channels_for_retrieval(
        hybrid_retrieval=hybrid_retrieval,
        hybrid_embedder=hybrid_embedder,
        hybrid_semantic_ratio=hybrid_semantic_ratio,
        hybrid_query_vector_spec=hybrid_query_vector_spec,
    )
    search_response = _search_with_expanded_queries(
        client=client,
        index_uid=index_uid,
        queries=search_queries,
        limit=resolved_candidate_pool_limit,
        channels=search_channels,
        filter=project_filter,
    )
    primary_hit_records = _search_hit_records(search_response)
    primary_hits = [record["hit"] for record in primary_hit_records]

    bundles: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    warnings: list[str] = _hybrid_embedding_warnings(
        hybrid_retrieval=hybrid_retrieval,
        query_vector_spec=hybrid_query_vector_spec,
    )
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
                limit=resolved_candidate_pool_limit,
                channels=search_channels,
                filter=project_filter,
            )
            visual_hits = visual_search_response.get("hits", [])
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            warnings.append(
                f"Skipped visual entity search because the index was not found: {visual_index_uid}"
            )

    retrieval_entries: list[dict[str, Any]] = []
    for rank, record in enumerate(primary_hit_records, start=1):
        hit = record["hit"]
        candidate = _primary_candidate_from_hit(
            rank=rank,
            hit=hit,
            retrieval_index_kind=retrieval_index_kind,
        )
        _attach_retrieval_metadata(
            candidate,
            source=retrieval_index_kind,
            index_uid=index_uid,
            aggregate_rank=rank,
            record=record,
            include_match_details=hybrid_retrieval,
        )
        candidates.append(candidate)
        retrieval_entries.append(
            {
                "source": retrieval_index_kind,
                "modality": SOURCE_MODALITY[retrieval_index_kind],
                "index": index_uid,
                "rank": rank,
                "score": candidate.get("score"),
                "retrieval_mode": record.get("retrieval_mode"),
                "original_rank": record.get("rank"),
                "original_score": record.get("score"),
                "search_query": record.get("query"),
                "query_index": record.get("query_index"),
                "fusion_score": record.get("fusion_score"),
                "matches": _record_match_summaries(
                    record=record,
                    source=retrieval_index_kind,
                    index_uid=index_uid,
                    include_match_details=hybrid_retrieval,
                ),
                "hit": hit,
                "candidate": candidate,
                "target_segment_id": _primary_target_segment_id(
                    hit=hit,
                    candidate=candidate,
                    retrieval_index_kind=retrieval_index_kind,
                ),
            }
        )

    visual_hit_records = _search_hit_records(visual_search_response)
    for rank, record in enumerate(visual_hit_records, start=1):
        hit = record["hit"]
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
        _attach_retrieval_metadata(
            candidate,
            source=VISUAL_ENTITY_HIT_SOURCE,
            index_uid=visual_index_uid,
            aggregate_rank=rank,
            record=record,
            include_match_details=hybrid_retrieval,
        )
        candidates.append(candidate)
        retrieval_entries.append(
            {
                "source": VISUAL_ENTITY_HIT_SOURCE,
                "modality": SOURCE_MODALITY[VISUAL_ENTITY_HIT_SOURCE],
                "index": visual_index_uid,
                "rank": rank,
                "score": candidate.get("score"),
                "retrieval_mode": record.get("retrieval_mode"),
                "original_rank": record.get("rank"),
                "original_score": record.get("score"),
                "search_query": record.get("query"),
                "query_index": record.get("query_index"),
                "fusion_score": record.get("fusion_score"),
                "matches": _record_match_summaries(
                    record=record,
                    source=VISUAL_ENTITY_HIT_SOURCE,
                    index_uid=visual_index_uid,
                    include_match_details=hybrid_retrieval,
                ),
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
    )[: (resolved_candidate_pool_limit if rerank else result_limit)]

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
                "modality": source.get("modality"),
                "index": source["index"],
                "rank": source["rank"],
                "score": source["score"],
                "retrieval_mode": source.get("retrieval_mode"),
                "original_rank": source.get("original_rank"),
                "original_score": source.get("original_score"),
                "deduplicated": source.get("deduplicated", False),
                "matches": source.get("matches", []),
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
        merge_metadata = {
            "target_key": f"segment:{target_segment_id}",
            "target_segment_id": target_segment_id,
            "source_count": len(retrieval_sources),
            "selected_source": primary_entry["source"],
            "selected_rank": primary_entry["rank"],
            "deduplicated": len(retrieval_sources) > 1,
        }
        if hybrid_retrieval:
            merge_metadata.update(
                {
                    "retrieval_match_count": _retrieval_match_count(retrieval_sources),
                    "selected_retrieval_mode": primary_entry.get("retrieval_mode"),
                    "deduplicated": len(retrieval_sources) > 1
                    or any(source.get("deduplicated") for source in retrieval_sources),
                }
            )
        bundles.append(
            {
                "rank": merged_rank,
                "candidate": candidate,
                "retrieval_sources": retrieval_sources,
                "merge": merge_metadata,
                "evidence_window": evidence_window,
                "visual_entities": window_visual_entities,
                "linked_entities": linked_entities,
                "summary": summary,
            }
        )
        summary_lines.append(summary["text"])
        bundled_entity_total += len(window_visual_entities)
        bundled_link_total += len(linked_entities)

    rerank_context = rerank_metadata(
        enabled=False,
        query=query,
        timestamp_hint=rerank_time_hint,
        backend=rerank_backend,
    )
    if rerank:
        bundles, rerank_context = rerank_bundles(
            bundles=bundles,
            query=query,
            timestamp_hint=rerank_time_hint,
            backend=rerank_backend,
        )
        bundles = bundles[:result_limit]
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
            "hybrid_retrieval": _hybrid_retrieval_metadata(
                enabled=hybrid_retrieval,
                channels=search_channels,
            ),
            "window_config": window_config,
            "candidate_generation": {
                "result_limit": result_limit,
                "candidate_pool_limit": resolved_candidate_pool_limit,
                "pool_expanded": resolved_candidate_pool_limit > result_limit,
            },
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
            "result_limit": result_limit,
            "candidate_pool_limit": resolved_candidate_pool_limit,
            "candidate_pool_targets": len(merged_targets),
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
    local_entity_id = str(hit.get("local_entity_id") or entity_id)
    existing = entity_lookup.get(local_entity_id)
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
        entity_id=str(hit.get("local_entity_id") or entity.entity_id),
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
        if entity.frame_id and entity.frame_id in _segment_frame_ref_ids(segment)
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


def _segment_frame_ref_ids(segment: dict[str, Any]) -> set[str]:
    frame_ids: set[str] = set()
    frame_refs = segment.get("frame_refs")
    if not isinstance(frame_refs, list):
        return frame_ids
    for frame_ref in frame_refs:
        if isinstance(frame_ref, dict):
            value = frame_ref.get("frame_id")
        else:
            value = frame_ref
        if value not in (None, ""):
            frame_ids.add(str(value))
    return frame_ids


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


def _retrieval_source_sort_key(entry: dict[str, Any]) -> tuple[int, int, int, str]:
    source = str(entry.get("source", ""))
    rank = _optional_int(entry.get("rank")) or 0
    mode = str(entry.get("retrieval_mode") or "")
    return (
        SOURCE_PRIORITY.get(source, 99),
        RETRIEVAL_MODE_PRIORITY.get(mode, 99),
        rank,
        str(entry.get("target_segment_id", "")),
    )


def _retrieval_source_summary(entry: dict[str, Any]) -> dict[str, Any]:
    candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
    matches = entry.get("matches") if isinstance(entry.get("matches"), list) else []
    summary = {
        "source": entry.get("source"),
        "modality": entry.get("modality") or SOURCE_MODALITY.get(str(entry.get("source") or "")),
        "index": entry.get("index"),
        "retrieval_mode": entry.get("retrieval_mode"),
        "rank": entry.get("rank"),
        "score": entry.get("score"),
        "original_rank": entry.get("original_rank"),
        "original_score": entry.get("original_score"),
        "search_query": entry.get("search_query"),
        "query_index": entry.get("query_index"),
        "fusion_score": entry.get("fusion_score"),
        "deduplicated": len(matches) > 1,
        "matches": matches,
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


def _attach_retrieval_metadata(
    candidate: dict[str, Any],
    *,
    source: str,
    index_uid: str | None,
    aggregate_rank: int,
    record: dict[str, Any],
    include_match_details: bool,
) -> None:
    candidate["source"] = source
    candidate["modality"] = SOURCE_MODALITY.get(source)
    candidate["retrieval_mode"] = record.get("retrieval_mode")
    candidate["retrieval_rank"] = aggregate_rank
    candidate["original_rank"] = record.get("rank")
    candidate["original_score"] = record.get("score")
    candidate["search_query"] = record.get("query")
    candidate["query_index"] = record.get("query_index")
    candidate["fusion_score"] = record.get("fusion_score")
    candidate["retrieval_matches"] = _record_match_summaries(
        record=record,
        source=source,
        index_uid=index_uid,
        include_match_details=include_match_details,
    )


def _record_match_summaries(
    *,
    record: dict[str, Any],
    source: str,
    index_uid: str | None,
    include_match_details: bool,
) -> list[dict[str, Any]]:
    matches = record.get("matches") if isinstance(record.get("matches"), list) else []
    if not matches:
        matches = [record]
    summaries: list[dict[str, Any]] = []
    for match in sorted(matches, key=_match_summary_sort_key):
        summary: dict[str, Any] = {
            "source": source,
            "modality": SOURCE_MODALITY.get(source),
            "index": index_uid,
            "retrieval_mode": match.get("retrieval_mode"),
            "rank": match.get("rank"),
            "score": match.get("score"),
            "query": match.get("query"),
            "query_index": match.get("query_index"),
        }
        if include_match_details:
            summary["channel_index"] = match.get("channel_index")
            summary["first_seen"] = match.get("first_seen")
            if isinstance(match.get("query_vector"), dict):
                summary["query_vector"] = dict(match["query_vector"])
        summaries.append(summary)
    return summaries


def _match_summary_sort_key(match: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(match.get("query_index") or 0),
        RETRIEVAL_MODE_PRIORITY.get(str(match.get("retrieval_mode") or ""), 99),
        int(match.get("rank") or 10**9),
        int(match.get("first_seen") or 10**9),
    )


def _retrieval_match_count(retrieval_sources: list[dict[str, Any]]) -> int:
    total = 0
    for source in retrieval_sources:
        matches = source.get("matches") if isinstance(source.get("matches"), list) else []
        total += len(matches) if matches else 1
    return total


def _hybrid_retrieval_metadata(
    *,
    enabled: bool,
    channels: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "modes": [str(channel.get("mode")) for channel in channels],
    }
    semantic_channel = next(
        (
            channel
            for channel in channels
            if str(channel.get("mode")) == SEMANTIC_RETRIEVAL_MODE
            and isinstance(channel.get("hybrid"), dict)
        ),
        None,
    )
    if semantic_channel is not None:
        hybrid = semantic_channel["hybrid"]
        metadata.update(
            {
                "embedder": hybrid.get("embedder"),
                "semantic_ratio": hybrid.get("semanticRatio"),
                "query_vector": {"used": False},
                "fusion": {
                    "method": "reciprocal_rank_fusion",
                    "rank_constant": HYBRID_RRF_RANK_CONSTANT,
                },
            }
        )
        query_vector_metadata = semantic_channel.get("query_vector")
        if isinstance(query_vector_metadata, dict):
            metadata["query_vector"] = dict(query_vector_metadata)
    return metadata


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


def _search_channels_for_retrieval(
    *,
    hybrid_retrieval: bool,
    hybrid_embedder: str | None,
    hybrid_semantic_ratio: float,
    hybrid_query_vector_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = [
        {
            "mode": LEXICAL_RETRIEVAL_MODE,
            "hybrid": None,
        }
    ]
    if not hybrid_retrieval:
        return channels

    embedder = str(hybrid_embedder or "").strip()
    if not embedder:
        raise ValueError("hybrid_embedder is required when hybrid_retrieval is enabled")
    try:
        semantic_ratio = float(hybrid_semantic_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("hybrid_semantic_ratio must be > 0.0 and <= 1.0") from exc
    if not 0.0 < semantic_ratio <= 1.0:
        raise ValueError("hybrid_semantic_ratio must be > 0.0 and <= 1.0")
    semantic_channel = {
        "mode": SEMANTIC_RETRIEVAL_MODE,
        "hybrid": {
            "embedder": embedder,
            "semanticRatio": semantic_ratio,
        },
    }
    if hybrid_query_vector_spec is not None:
        semantic_channel["vector"] = hybrid_query_vector_spec["vector"]
        semantic_channel["query_vector"] = hybrid_query_vector_spec["metadata"]
    channels.append(semantic_channel)
    return channels


def _resolve_hybrid_query_vector(
    *,
    project_dir: Path,
    hybrid_retrieval: bool,
    hybrid_embedder: str | None,
    query_text: str,
    query_vector: list[float] | None,
    query_vector_embedder: str | None,
    query_vector_name: str | None,
    query_vector_dimensions: int | None,
    query_vector_manifest_path: Path | None,
) -> dict[str, Any] | None:
    has_vector_options = any(
        (
            query_vector is not None,
            _non_empty_string(query_vector_embedder) is not None,
            _non_empty_string(query_vector_name) is not None,
            query_vector_dimensions is not None,
            query_vector_manifest_path is not None,
        )
    )
    if not has_vector_options:
        return None
    if not hybrid_retrieval:
        raise ValueError("hybrid query vector options require hybrid_retrieval=True")

    embedder = str(hybrid_embedder or "").strip()
    if not embedder:
        raise ValueError("hybrid_embedder is required when hybrid_retrieval is enabled")

    vector_source = "argument"
    vector_name = _non_empty_string(query_vector_name)
    vector_embedder = _non_empty_string(query_vector_embedder) or embedder
    manifest_dimensions: int | None = None
    raw_vector: Any = query_vector

    if query_vector_manifest_path is not None or vector_name is not None:
        if query_vector is not None:
            raise ValueError(
                "Provide only one of hybrid_query_vector or "
                "hybrid_query_vector_manifest_path/name"
            )
        manifest_path = _resolve_query_vector_manifest_path(
            project_dir=project_dir,
            path=query_vector_manifest_path,
        )
        manifest_spec = _load_query_vector_manifest(manifest_path, vector_name=vector_name)
        raw_vector = manifest_spec["vector"]
        vector_name = manifest_spec["name"]
        vector_embedder = _non_empty_string(query_vector_embedder) or manifest_spec["embedder"]
        manifest_dimensions = manifest_spec.get("dimensions")
        vector_source = "manifest"

    if (
        raw_vector is None
        and query_vector_dimensions is not None
        and vector_source == "argument"
    ):
        raw_vector = deterministic_text_vector(
            query_text,
            dimensions=query_vector_dimensions,
        )
        manifest_dimensions = query_vector_dimensions
        vector_source = LOCAL_HASH_VECTOR_SOURCE

    if raw_vector is None:
        requested = vector_embedder or embedder
        raise ValueError(
            f"query vector is required for userProvided embedder '{requested}'; "
            "provide hybrid_query_vector or a hybrid_query_vector_manifest_path/name"
        )

    if vector_embedder != embedder:
        raise ValueError(
            "query vector embedder mismatch: "
            f"hybrid_embedder is '{embedder}' but query vector is for '{vector_embedder}'"
        )

    expected_dimensions = (
        query_vector_dimensions
        if query_vector_dimensions is not None
        else manifest_dimensions
    )
    if (
        query_vector_dimensions is not None
        and manifest_dimensions is not None
        and query_vector_dimensions != manifest_dimensions
    ):
        raise ValueError(
            f"query vector dimension mismatch for embedder '{vector_embedder}': "
            f"manifest declares {manifest_dimensions}, expected {query_vector_dimensions}"
        )

    vector = normalize_query_vector(
        raw_vector,
        dimensions=expected_dimensions,
        field_name=f"query vector for embedder '{vector_embedder}'",
    )
    metadata = {
        "used": True,
        "embedder": vector_embedder,
        "dimensions": len(vector),
        "source": vector_source,
    }
    if vector_source == LOCAL_HASH_VECTOR_SOURCE:
        metadata["purpose"] = "local_reproducibility_smoke_fallback"
        metadata["quality_claim"] = NO_SEMANTIC_QUALITY_CLAIM
    elif vector_source == "manifest":
        backend_contract = manifest_spec.get("backend_contract")
        if isinstance(backend_contract, dict):
            metadata["backend_contract"] = backend_contract
            metadata["provider"] = backend_contract.get("provider")
            metadata["source_model"] = backend_contract.get("source_model")
            metadata["quality_claim"] = backend_contract.get("quality_claim")
    else:
        metadata["quality_claim"] = "unverified"
    if vector_name is not None:
        metadata["name"] = vector_name
    return {
        "vector": vector,
        "metadata": metadata,
    }


def _resolve_query_vector_manifest_path(*, project_dir: Path, path: Path | None) -> Path:
    if path is None:
        return (project_dir / DEFAULT_QUERY_VECTOR_MANIFEST_RELATIVE_PATH).resolve()
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _load_query_vector_manifest(path: Path, *, vector_name: str | None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"query vector manifest not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse query vector manifest JSON at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("query vector manifest must be a JSON object")

    queries = loaded.get("queries")
    if not isinstance(queries, dict) or not queries:
        raise ValueError("query vector manifest must include a non-empty 'queries' object")
    selected_name = _select_query_vector_name(queries, vector_name=vector_name)
    entry = queries.get(selected_name)
    if not isinstance(entry, dict):
        raise ValueError(f"query vector manifest entry '{selected_name}' must be a JSON object")

    vector = entry.get("vector")
    embedder = _non_empty_string(entry.get("embedder")) or _non_empty_string(
        loaded.get("embedder")
    )
    if embedder is None:
        raise ValueError(f"query vector manifest entry '{selected_name}' is missing embedder")

    dimensions = _query_vector_manifest_dimensions(
        manifest=loaded,
        entry=entry,
        embedder=embedder,
    )
    provider = loaded.get("provider") if isinstance(loaded.get("provider"), dict) else {}
    return {
        "name": selected_name,
        "embedder": embedder,
        "dimensions": dimensions,
        "vector": vector,
        "backend_contract": embedding_backend_contract(
            source="query_vector_manifest",
            provider=provider,
            embedder_names=[embedder],
            dimensions_by_embedder={embedder: dimensions},
            vector_count=len(queries),
        ),
    }


def _hybrid_embedding_warnings(
    *,
    hybrid_retrieval: bool,
    query_vector_spec: dict[str, Any] | None,
) -> list[str]:
    if not hybrid_retrieval:
        return []
    metadata = (
        query_vector_spec.get("metadata")
        if isinstance(query_vector_spec, dict)
        and isinstance(query_vector_spec.get("metadata"), dict)
        else None
    )
    if metadata is None:
        return [UNDECLARED_SEMANTIC_BACKEND_WARNING]
    if metadata.get("source") == LOCAL_HASH_VECTOR_SOURCE:
        return [LOCAL_HASH_QUERY_VECTOR_WARNING]
    if metadata.get("quality_claim") != PROVIDER_EMBEDDING_QUALITY_CLAIM:
        return [UNDECLARED_SEMANTIC_BACKEND_WARNING]
    return []


def _select_query_vector_name(queries: dict[str, Any], *, vector_name: str | None) -> str:
    if vector_name is not None:
        if vector_name not in queries:
            available = ", ".join(sorted(str(name) for name in queries))
            raise ValueError(
                f"query vector manifest does not contain '{vector_name}'. "
                f"Available query vectors: {available}"
            )
        return vector_name
    if len(queries) == 1:
        return str(next(iter(queries)))
    available = ", ".join(sorted(str(name) for name in queries))
    raise ValueError(
        "hybrid_query_vector_name is required when the query vector manifest contains "
        f"multiple vectors. Available query vectors: {available}"
    )


def _query_vector_manifest_dimensions(
    *,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    embedder: str,
) -> int | None:
    dimensions = entry.get("dimensions")
    embedders = manifest.get("embedders")
    if embedders is None:
        return dimensions if isinstance(dimensions, int) else None
    if not isinstance(embedders, dict):
        raise ValueError(
            "query vector manifest embedder settings problem: 'embedders' must be an object"
        )
    embedder_settings = embedders.get(embedder)
    if not isinstance(embedder_settings, dict):
        raise ValueError(
            "query vector manifest embedder settings problem: "
            f"missing embedder '{embedder}'"
        )
    if embedder_settings.get("source") != "userProvided":
        raise ValueError(
            "query vector manifest embedder settings problem: "
            f"embedder '{embedder}' must use source 'userProvided'"
        )
    embedder_dimensions = embedder_settings.get("dimensions")
    if dimensions is not None and dimensions != embedder_dimensions:
        raise ValueError(
            f"query vector dimension mismatch for embedder '{embedder}': "
            f"entry declares {dimensions}, embedder declares {embedder_dimensions}"
        )
    return embedder_dimensions if isinstance(embedder_dimensions, int) else None


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _candidate_pool_limit(
    *,
    result_limit: int,
    candidate_pool_limit: int | None,
) -> int:
    if candidate_pool_limit is None:
        return result_limit
    parsed = _positive_int(candidate_pool_limit, field_name="candidate_pool_limit")
    return max(result_limit, parsed)


def _project_filter(*, project_dir: Path, segments: list[dict[str, Any]]) -> str:
    project_id = ""
    for segment in segments:
        candidate = str(segment.get("project_id") or "").strip()
        if candidate:
            project_id = candidate
            break
    project_id = project_id or project_dir.name
    return f'project_id = "{_escape_meili_filter_string(project_id)}"'


def _escape_meili_filter_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _search_with_expanded_queries(
    *,
    client: MeiliClient,
    index_uid: str,
    queries: list[str],
    limit: int,
    channels: list[dict[str, Any]] | None = None,
    filter: str | list[str] | None = None,
) -> dict[str, Any]:
    search_channels = channels or _search_channels_for_retrieval(
        hybrid_retrieval=False,
        hybrid_embedder=DEFAULT_HYBRID_EMBEDDER,
        hybrid_semantic_ratio=DEFAULT_HYBRID_SEMANTIC_RATIO,
    )
    responses: list[dict[str, Any]] = []
    hit_records: list[dict[str, Any]] = []
    hit_positions: dict[str, int] = {}
    search_calls: list[dict[str, Any]] = []

    for query_index, search_query in enumerate(queries):
        for channel_index, channel in enumerate(search_channels):
            response = _run_search_channel(
                client=client,
                index_uid=index_uid,
                search_query=search_query,
                limit=limit,
                channel=channel,
                filter=filter,
            )
            responses.append(response)
            hits = response.get("hits", [])
            query_vector_metadata = channel.get("query_vector")
            search_call = {
                "index": index_uid,
                "query": search_query,
                "query_index": query_index,
                "retrieval_mode": channel["mode"],
                "channel_index": channel_index,
                "limit": limit,
                "hit_count": len(hits),
                "processing_time_ms": response.get("processingTimeMs"),
            }
            if isinstance(query_vector_metadata, dict):
                search_call["query_vector"] = dict(query_vector_metadata)
            search_calls.append(search_call)
            for rank, hit in enumerate(hits, start=1):
                hit_key = _search_hit_key(hit)
                score = _hit_score(hit)
                occurrence = {
                    "hit": hit,
                    "score": score,
                    "query": search_query,
                    "query_index": query_index,
                    "retrieval_mode": channel["mode"],
                    "channel_index": channel_index,
                    "rank": rank,
                    "first_seen": sum(len(record.get("matches", [])) for record in hit_records),
                }
                if isinstance(query_vector_metadata, dict):
                    occurrence["query_vector"] = dict(query_vector_metadata)
                if hit_key in hit_positions:
                    existing_index = hit_positions[hit_key]
                    hit_records[existing_index]["matches"].append(occurrence)
                    if _is_better_hit_occurrence(occurrence, hit_records[existing_index]):
                        hit_records[existing_index].update(_selected_hit_fields(occurrence))
                    continue
                hit_positions[hit_key] = len(hit_records)
                hit_records.append(
                    {
                        **_selected_hit_fields(occurrence),
                        "hit_key": hit_key,
                        "matches": [occurrence],
                    }
                )

    aggregate = copy.deepcopy(responses[0]) if responses else {}
    for record in hit_records:
        record["fusion_score"] = _fusion_score(record.get("matches", []))
    ranked_hit_records = sorted(
        hit_records,
        key=_hybrid_hit_record_sort_key
        if len(search_channels) > 1
        else _expanded_hit_record_sort_key,
    )
    limited_records = ranked_hit_records[:limit]
    for aggregate_rank, record in enumerate(limited_records, start=1):
        record["aggregate_rank"] = aggregate_rank
    aggregate["hits"] = [record["hit"] for record in ranked_hit_records[:limit]]
    aggregate["hitRecords"] = limited_records
    aggregate["processingTimeMs"] = _combined_processing_time_ms(*responses)
    aggregate["query"] = queries[0] if len(queries) == 1 else queries
    aggregate["queries"] = queries
    aggregate["limit"] = limit
    aggregate["retrievalModes"] = [channel["mode"] for channel in search_channels]
    aggregate["searchCalls"] = search_calls
    aggregate["hitCountBeforeLimit"] = len(hit_records)
    aggregate["rawHitCountBeforeDedupe"] = sum(call["hit_count"] for call in search_calls)
    return aggregate


def _run_search_channel(
    *,
    client: MeiliClient,
    index_uid: str,
    search_query: str,
    limit: int,
    channel: dict[str, Any],
    filter: str | list[str] | None = None,
) -> dict[str, Any]:
    hybrid = channel.get("hybrid")
    kwargs: dict[str, Any] = {"limit": limit}
    if filter is not None:
        kwargs["filter"] = filter
    if isinstance(hybrid, dict):
        kwargs["hybrid"] = hybrid
        vector = channel.get("vector")
        if vector is not None:
            kwargs["vector"] = vector
    return _search_with_supported_kwargs(client, index_uid, search_query, kwargs)


def _search_with_supported_kwargs(
    client: MeiliClient,
    index_uid: str,
    query: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key != "filter" or _search_accepts_keyword(client, "filter")
    }
    try:
        return client.search(index_uid, query, **supported_kwargs)
    except TypeError as exc:
        if "filter" in supported_kwargs and _is_unexpected_keyword_error(exc, "filter"):
            fallback_kwargs = {
                key: value for key, value in supported_kwargs.items() if key != "filter"
            }
            return client.search(index_uid, query, **fallback_kwargs)
        raise


def _search_accepts_keyword(client: MeiliClient, keyword: str) -> bool:
    try:
        parameters = inspect.signature(client.search).parameters.values()
    except (TypeError, ValueError):
        return True
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return True
    return False


def _is_unexpected_keyword_error(exc: TypeError, keyword: str) -> bool:
    message = str(exc)
    return (
        f"unexpected keyword argument '{keyword}'" in message
        or f'unexpected keyword argument "{keyword}"' in message
    )


def _selected_hit_fields(occurrence: dict[str, Any]) -> dict[str, Any]:
    selected = {
        "hit": occurrence["hit"],
        "score": occurrence["score"],
        "query": occurrence["query"],
        "query_index": occurrence["query_index"],
        "retrieval_mode": occurrence["retrieval_mode"],
        "channel_index": occurrence["channel_index"],
        "rank": occurrence["rank"],
        "first_seen": occurrence["first_seen"],
    }
    if isinstance(occurrence.get("query_vector"), dict):
        selected["query_vector"] = dict(occurrence["query_vector"])
    return selected


def _is_better_hit_occurrence(occurrence: dict[str, Any], record: dict[str, Any]) -> bool:
    return _occurrence_selection_key(occurrence) < _occurrence_selection_key(record)


def _occurrence_selection_key(occurrence: dict[str, Any]) -> tuple[float, int, int, int, int]:
    return (
        -float(occurrence.get("score", float("-inf"))),
        int(occurrence.get("query_index") or 0),
        RETRIEVAL_MODE_PRIORITY.get(str(occurrence.get("retrieval_mode") or ""), 99),
        int(occurrence.get("rank") or 10**9),
        int(occurrence.get("first_seen") or 10**9),
    )


def _fusion_score(matches: list[dict[str, Any]]) -> float:
    score = 0.0
    for match in matches:
        rank = _optional_int(match.get("rank")) or 10**9
        score += 1.0 / (HYBRID_RRF_RANK_CONSTANT + rank)
    return round(score, 8)


def _search_hit_records(response: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    records = response.get("hitRecords")
    if isinstance(records, list):
        return [
            record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("hit"), dict)
        ]
    fallback_records: list[dict[str, Any]] = []
    for rank, hit in enumerate(response.get("hits", []), start=1):
        if not isinstance(hit, dict):
            continue
        fallback_records.append(
            {
                "hit": hit,
                "score": _hit_score(hit),
                "query": response.get("query"),
                "query_index": 0,
                "retrieval_mode": LEXICAL_RETRIEVAL_MODE,
                "channel_index": 0,
                "rank": rank,
                "aggregate_rank": rank,
                "first_seen": rank - 1,
                "fusion_score": _fusion_score([{"rank": rank}]),
                "matches": [
                    {
                        "hit": hit,
                        "score": _hit_score(hit),
                        "query": response.get("query"),
                        "query_index": 0,
                        "retrieval_mode": LEXICAL_RETRIEVAL_MODE,
                        "channel_index": 0,
                        "rank": rank,
                        "first_seen": rank - 1,
                    }
                ],
            }
        )
    return fallback_records


def _search_metadata(response: dict[str, Any] | None, index_uid: str | None) -> dict[str, Any] | None:
    if response is None:
        return None
    metadata = {
        "index": index_uid,
        "limit": response.get("limit"),
        "processing_time_ms": response.get("processingTimeMs"),
        "hit_count": len(response.get("hits", [])),
    }
    if "queries" in response:
        metadata["queries"] = response["queries"]
    if "retrievalModes" in response:
        metadata["retrieval_modes"] = response["retrievalModes"]
    if "searchCalls" in response:
        metadata["calls"] = [
            {
                key: call.get(key)
                for key in (
                    "index",
                    "query",
                    "query_index",
                    "retrieval_mode",
                    "channel_index",
                    "limit",
                    "hit_count",
                    "processing_time_ms",
                    "query_vector",
                )
                if key in call
            }
            for call in response["searchCalls"]
            if isinstance(call, dict)
        ]
    if "hitCountBeforeLimit" in response:
        metadata["hit_count_before_limit"] = response["hitCountBeforeLimit"]
    if "rawHitCountBeforeDedupe" in response:
        metadata["raw_hit_count_before_dedupe"] = response["rawHitCountBeforeDedupe"]
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


def _hybrid_hit_record_sort_key(record: dict[str, Any]) -> tuple[float, float, int, int, int, int]:
    score = optional_float(record.get("score"))
    return (
        -float(record.get("fusion_score") or 0.0),
        -(score if score is not None else float("-inf")),
        int(record.get("query_index") or 0),
        RETRIEVAL_MODE_PRIORITY.get(str(record.get("retrieval_mode") or ""), 99),
        int(record.get("rank") or 10**9),
        int(record.get("first_seen") or 10**9),
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
                "verified": link.verified,
                "verification_status": link.verification_status,
                "status": link.status,
                "alignment_status": link.alignment_status,
                "verification_source": link.verification_source,
                "verified_by": link.verified_by,
                "verifier": link.verifier,
                "source": link.source,
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
