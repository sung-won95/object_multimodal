from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from oarag.core.domain_lexicon import DomainLexicon, load_domain_lexicon
from oarag.retrieval.evidence import (
    frame_id,
    make_evidence_window,
    read_jsonl,
    resolve_project_path,
    resolve_window_config,
    segment_window,
    select_window_segments,
)
from oarag.retrieval.project_index import evidence_unit_artifact_path, segment_artifact_path
from oarag.retrieval.project_query import (
    DEFAULT_HYBRID_EMBEDDER,
    DEFAULT_HYBRID_SEMANTIC_RATIO,
    SEGMENT_HIT_SOURCE,
    WINDOW_HIT_SOURCE,
    query_project,
)
from oarag.retrieval.rerank import DEFAULT_RERANK_BACKEND


TEACHING_MOMENT_HIT_SOURCE = "teaching_moment_span"
TEACHING_MOMENT_DIAGNOSTICS_SCHEMA_VERSION = "teaching-moment-span-diagnostics-public-v1"
DEFAULT_SPAN_WINDOW_SECONDS = 90.0
DEFAULT_SPAN_CANDIDATE_POOL_LIMIT = 30
DEFAULT_SPAN_NEIGHBOR_FALLBACK = 2
CANDIDATE_LINK_SIGNAL_KEYS = (
    "temporal_overlap",
    "lexical_overlap",
    "mention_deictic_hook",
    "spatial_position",
    "visual_text_overlap",
    "vlm_object_visual_description_overlap",
    "semantic_domain_hint",
    "timestamp_fallback",
)
VERIFIED_LINK_SOURCE_KEYS = (
    "explicit_verified_flag",
    "explicit_verified_status",
    "human_gold",
    "vlm_verifier",
    "strict_deterministic_rule",
    "unspecified_verified",
)
PUBLIC_NOTE = (
    "Teaching-moment span diagnostics expose counts, buckets, hashed refs, and reason "
    "codes only. Raw query text, transcript text, answer text, evidence text, paths, "
    "and raw candidate ids are excluded from public benchmark outputs."
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")
_STOPWORDS = {
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
    "how",
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
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_DENSITY_BUCKETS = {
    "none": (0.0, 0.0),
    "low": (0.0, 0.25),
    "medium": (0.25, 0.5),
    "high": (0.5, 0.75),
    "very_high": (0.75, 1.0),
}


def query_project_teaching_moments(
    *,
    client: Any,
    index_uid: str,
    project_dir: Path,
    query: str,
    anchor_index_kind: str = SEGMENT_HIT_SOURCE,
    visual_index_uid: str | None = None,
    limit: int = 5,
    candidate_pool_limit: int | None = None,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    evidence_units_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
    disable_domain_lexicon: bool = True,
    span_window_seconds: float | None = DEFAULT_SPAN_WINDOW_SECONDS,
    span_window_before_seconds: float | None = None,
    span_window_after_seconds: float | None = None,
    span_neighbor_count: int = DEFAULT_SPAN_NEIGHBOR_FALLBACK,
    rerank_backend: str = DEFAULT_RERANK_BACKEND,
    hybrid_retrieval: bool = False,
    hybrid_embedder: str | None = DEFAULT_HYBRID_EMBEDDER,
    hybrid_semantic_ratio: float = DEFAULT_HYBRID_SEMANTIC_RATIO,
    hybrid_query_vector_embedder: str | None = None,
    hybrid_query_vector_name: str | None = None,
    hybrid_query_vector_dimensions: int | None = None,
    hybrid_query_vector_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Retrieve broad anchors, expand them into explanation spans, and rerank spans.

    This helper deliberately keeps concept graph/evidence-unit signals as rerank features
    rather than a primary retriever. The primary anchor retrieval remains segment/window
    search, optionally with visual and hybrid channels.
    """

    result_limit = _positive_int(limit, field_name="limit")
    resolved_pool_limit = max(
        result_limit,
        _positive_int(candidate_pool_limit, field_name="candidate_pool_limit")
        if candidate_pool_limit is not None
        else DEFAULT_SPAN_CANDIDATE_POOL_LIMIT,
    )
    resolved_project_dir = project_dir.expanduser().resolve()
    normalized_anchor_kind = _anchor_index_kind(anchor_index_kind)
    anchor_response = query_project(
        client=client,
        index_uid=index_uid,
        retrieval_index_kind=normalized_anchor_kind,
        visual_index_uid=visual_index_uid,
        project_dir=resolved_project_dir,
        query=query,
        limit=resolved_pool_limit,
        candidate_pool_limit=resolved_pool_limit,
        segments_path=segments_path,
        frames_manifest_path=frames_manifest_path,
        visual_entities_path=visual_entities_path,
        entity_links_path=entity_links_path,
        domain_lexicon_path=domain_lexicon_path,
        disable_domain_lexicon=disable_domain_lexicon,
        neighbor_count=0,
        rerank=False,
        rerank_backend=rerank_backend,
        hybrid_retrieval=hybrid_retrieval,
        hybrid_embedder=hybrid_embedder,
        hybrid_semantic_ratio=hybrid_semantic_ratio,
        hybrid_query_vector_embedder=hybrid_query_vector_embedder,
        hybrid_query_vector_name=hybrid_query_vector_name,
        hybrid_query_vector_dimensions=hybrid_query_vector_dimensions,
        hybrid_query_vector_manifest_path=hybrid_query_vector_manifest_path,
    )

    artifacts = _load_span_artifacts(
        project_dir=resolved_project_dir,
        segments_path=segments_path,
        frames_manifest_path=frames_manifest_path,
        visual_entities_path=visual_entities_path,
        entity_links_path=entity_links_path,
        evidence_units_path=evidence_units_path,
    )
    domain_lexicon = (
        DomainLexicon()
        if disable_domain_lexicon
        else load_domain_lexicon(
            project_dir=resolved_project_dir,
            domain_lexicon_path=domain_lexicon_path,
        )
    )
    span_config = _span_window_config(
        span_window_seconds=span_window_seconds,
        span_window_before_seconds=span_window_before_seconds,
        span_window_after_seconds=span_window_after_seconds,
        span_neighbor_count=span_neighbor_count,
    )
    query_context = _query_context(query=query, domain_lexicon=domain_lexicon)
    base_bundles = _list_of_dicts(anchor_response.get("bundles"))
    span_candidates = build_teaching_moment_spans(
        anchor_bundles=base_bundles,
        artifacts=artifacts,
        query_context=query_context,
        span_config=span_config,
    )
    reranked_bundles = rerank_teaching_moment_spans(
        span_candidates,
        query_context=query_context,
    )[:result_limit]
    context = teaching_moment_context(
        base_spans=span_candidates,
        reranked_spans=reranked_bundles,
        span_config=span_config,
    )
    retrieval_context = dict(anchor_response.get("retrieval_context") or {})
    retrieval_context["teaching_moment_span"] = context

    counts = dict(anchor_response.get("counts") or {})
    counts.update(
        {
            "anchor_bundle_count": len(base_bundles),
            "span_candidate_count": len(span_candidates),
            "span_bundles": len(reranked_bundles),
            "span_candidate_pool_limit": resolved_pool_limit,
            "span_result_limit": result_limit,
            "span_evidence_units_loaded": len(artifacts["evidence_units"]),
        }
    )
    return {
        "query": query,
        "index": index_uid,
        "index_kind": TEACHING_MOMENT_HIT_SOURCE,
        "anchor_index_kind": normalized_anchor_kind,
        "visual_index": visual_index_uid,
        "project_id": anchor_response.get("project_id") or _project_id(artifacts["segments"]),
        "processing_time_ms": anchor_response.get("processing_time_ms"),
        "domain_lexicon": anchor_response.get("domain_lexicon"),
        "query_expansion": anchor_response.get("query_expansion"),
        "artifact_availability": {
            **dict(anchor_response.get("artifact_availability") or {}),
            "evidence_units": bool(artifacts["evidence_units"]),
        },
        "retrieval_context": retrieval_context,
        "counts": counts,
        "summary_lines": _span_summary_lines(reranked_bundles),
        "warnings": list(anchor_response.get("warnings") or []),
        "candidates": [bundle["candidate"] for bundle in reranked_bundles],
        "bundles": reranked_bundles,
    }


def build_teaching_moment_spans(
    *,
    anchor_bundles: list[dict[str, Any]],
    artifacts: dict[str, Any],
    query_context: dict[str, Any],
    span_config: dict[str, Any],
) -> list[dict[str, Any]]:
    segments = _list_of_dicts(artifacts.get("segments"))
    segment_lookup = {str(segment.get("segment_id") or ""): segment for segment in segments}
    frame_lookup = {frame_id(frame): frame for frame in _list_of_dicts(artifacts.get("frames"))}
    visual_entities = _list_of_dicts(artifacts.get("visual_entities"))
    entity_lookup = {
        str(entity.get("entity_id") or ""): entity
        for entity in visual_entities
        if str(entity.get("entity_id") or "")
    }
    links = _list_of_dicts(artifacts.get("entity_links"))
    evidence_units = _list_of_dicts(artifacts.get("evidence_units"))

    bundles: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for anchor_rank, anchor_bundle in enumerate(anchor_bundles, start=1):
        anchor_candidate = _mapping(anchor_bundle.get("candidate"))
        target_segment_id = _target_segment_id(anchor_bundle)
        if not target_segment_id or target_segment_id not in segment_lookup:
            continue
        if target_segment_id in seen_targets:
            continue
        seen_targets.add(target_segment_id)

        target = segment_lookup[target_segment_id]
        window_segments = select_window_segments(
            segments,
            target_segment_id=target_segment_id,
            window_seconds=span_config.get("window_seconds"),
            neighbor_count=int(span_config.get("neighbor_count") or 0),
            previous_neighbor_count=_optional_int(span_config.get("previous_neighbor_count")),
            next_neighbor_count=_optional_int(span_config.get("next_neighbor_count")),
            window_before_seconds=_optional_float(span_config.get("window_before_seconds")),
            window_after_seconds=_optional_float(span_config.get("window_after_seconds")),
        )
        evidence_window = make_evidence_window(
            target=target,
            window_segments=window_segments,
            frame_lookup=frame_lookup,
            window_config=span_config,
        ).to_dict()
        source_segment_ids = _source_segment_ids(window_segments)
        start_time, end_time = _span_bounds(evidence_window, window_segments)
        span_links = _links_for_span(
            links=links,
            source_segment_ids=source_segment_ids,
            entity_lookup=entity_lookup,
        )
        span_visual_entities = _visual_entities_for_span(
            visual_entities=visual_entities,
            evidence_window=evidence_window,
            links=span_links,
            start_time=start_time,
            end_time=end_time,
            entity_lookup=entity_lookup,
        )
        span_evidence_units = _evidence_units_for_span(
            evidence_units=evidence_units,
            source_segment_ids=source_segment_ids,
            start_time=start_time,
            end_time=end_time,
        )
        source_quality = _span_source_quality(
            visual_entities=span_visual_entities,
            linked_entities=span_links,
            evidence_units=span_evidence_units,
        )
        transcript_text = _transcript_window_text(window_segments)
        evidence_text = _span_evidence_text(
            transcript_text=transcript_text,
            visual_entities=span_visual_entities,
            evidence_units=span_evidence_units,
        )
        score_input = _span_score_input(
            query_context=query_context,
            transcript_text=transcript_text,
            visual_entities=span_visual_entities,
            evidence_units=span_evidence_units,
            linked_entities=span_links,
            source_segment_ids=source_segment_ids,
            source_quality=source_quality,
            anchor_candidate=anchor_candidate,
            anchor_rank=anchor_rank,
        )
        span_id = f"tm_{_short_hash(target_segment_id)}_{anchor_rank}"
        candidate = {
            "source": TEACHING_MOMENT_HIT_SOURCE,
            "retrieval_mode": TEACHING_MOMENT_HIT_SOURCE,
            "span_id": span_id,
            "segment_id": target_segment_id,
            "target_segment_id": target_segment_id,
            "sample_id": target.get("sample_id") or target_segment_id,
            "video_id": target.get("video_id"),
            "source_segment_ids": source_segment_ids,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp_center": _midpoint(start_time, end_time)
            or _first_float(
                anchor_candidate.get("timestamp_center"),
                target.get("timestamp_center"),
                target.get("start_time"),
            ),
            "rank": anchor_rank,
            "original_rank": anchor_candidate.get("rank") or anchor_rank,
            "score": _optional_float(anchor_candidate.get("score")),
            "anchor_score": _optional_float(anchor_candidate.get("score")),
            "anchor_source": anchor_candidate.get("source") or SEGMENT_HIT_SOURCE,
            "transcript_window_text": transcript_text,
            "transcript_excerpt": transcript_text,
            "evidence_text": evidence_text,
            "semantic_text": _span_semantic_text(
                transcript_text=transcript_text,
                visual_entities=span_visual_entities,
                evidence_units=span_evidence_units,
            ),
            "visual_entity_ids": _unique_text_values(
                entity.get("entity_id") for entity in span_visual_entities
            ),
            "verified_entity_link_ids": _link_ids_by_status(span_links, status="verified"),
            "candidate_entity_link_ids": _candidate_link_ids(span_links),
            "candidate_entity_link_statuses": _candidate_link_statuses(span_links),
            "concept_ids": _unique_text_values(
                concept_id
                for unit in span_evidence_units
                for concept_id in _string_list(unit.get("concept_ids"))
            ),
            "concept_labels": _unique_text_values(
                label
                for unit in span_evidence_units
                for label in _string_list(unit.get("concept_labels"))
            ),
            "concept_aliases": _unique_text_values(
                alias
                for unit in span_evidence_units
                for alias in _string_list(unit.get("concept_aliases"))
            ),
            "source_quality": source_quality,
        }
        diagnostics = _span_score_diagnostics(score_input)
        candidate["teaching_moment_span"] = diagnostics
        bundles.append(
            {
                "rank": anchor_rank,
                "candidate": candidate,
                "evidence_window": evidence_window,
                "visual_entities": span_visual_entities,
                "linked_entities": span_links,
                "retrieval_sources": [
                    {
                        "source": TEACHING_MOMENT_HIT_SOURCE,
                        "modality": "multimodal"
                        if span_visual_entities or span_links or span_evidence_units
                        else "transcript",
                        "retrieval_mode": TEACHING_MOMENT_HIT_SOURCE,
                        "rank": anchor_rank,
                        "score": diagnostics["score"],
                        "target_segment_id": target_segment_id,
                        "source_segment_ids": source_segment_ids,
                        "span_ref": diagnostics["span_ref"],
                        "anchor_ref": diagnostics["anchor_ref"],
                    },
                    *_anchor_retrieval_sources(anchor_bundle),
                ],
                "merge": {
                    "source_count": 1 + len(_anchor_retrieval_sources(anchor_bundle)),
                    "sources": [TEACHING_MOMENT_HIT_SOURCE],
                    "target_segment_id": target_segment_id,
                },
                "summary": {
                    "rank": anchor_rank,
                    "text": _compact_text(transcript_text[:360]),
                    "source_count": 1 + len(_anchor_retrieval_sources(anchor_bundle)),
                },
            }
        )
    return bundles


def rerank_teaching_moment_spans(
    spans: list[dict[str, Any]],
    *,
    query_context: dict[str, Any],
) -> list[dict[str, Any]]:
    del query_context
    scored = []
    for fallback_rank, span in enumerate(spans, start=1):
        candidate = _mapping(span.get("candidate"))
        diagnostics = _mapping(candidate.get("teaching_moment_span"))
        score = _optional_float(diagnostics.get("score")) or 0.0
        original_rank = _optional_int(candidate.get("rank")) or fallback_rank
        target_ref = str(candidate.get("span_id") or candidate.get("target_segment_id") or "")
        scored.append((score, original_rank, target_ref, span))

    reranked: list[dict[str, Any]] = []
    for new_rank, (_, _, _, span) in enumerate(
        sorted(scored, key=lambda item: (-item[0], item[1], item[2])),
        start=1,
    ):
        updated = dict(span)
        candidate = dict(_mapping(updated.get("candidate")))
        diagnostics = dict(_mapping(candidate.get("teaching_moment_span")))
        diagnostics["reranked_rank"] = new_rank
        candidate["teaching_moment_span"] = diagnostics
        candidate["rank"] = new_rank
        updated["candidate"] = candidate
        updated["rank"] = new_rank
        retrieval_sources = []
        for source in _list_of_dicts(updated.get("retrieval_sources")):
            source = dict(source)
            if source.get("source") == TEACHING_MOMENT_HIT_SOURCE:
                source["rank"] = new_rank
            retrieval_sources.append(source)
        updated["retrieval_sources"] = retrieval_sources
        reranked.append(updated)
    return reranked


def teaching_moment_context(
    *,
    base_spans: list[dict[str, Any]],
    reranked_spans: list[dict[str, Any]],
    span_config: dict[str, Any],
) -> dict[str, Any]:
    base_top = _mapping(_mapping(base_spans[0]).get("candidate")) if base_spans else {}
    reranked_top = _mapping(_mapping(reranked_spans[0]).get("candidate")) if reranked_spans else {}
    reason_counts: Counter[str] = Counter()
    component_presence: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for span in reranked_spans:
        diagnostics = _mapping(_mapping(span.get("candidate")).get("teaching_moment_span"))
        for reason in _string_list(diagnostics.get("reason_codes")):
            reason_counts[reason] += 1
        for component, value in _mapping(diagnostics.get("components")).items():
            if _optional_float(value) not in (None, 0.0):
                component_presence[str(component)] += 1
        bucket = str(_mapping(diagnostics.get("buckets")).get("query_relevance") or "unknown")
        bucket_counts[bucket] += 1
    return {
        "schema_version": TEACHING_MOMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "enabled": True,
        "strategy": TEACHING_MOMENT_HIT_SOURCE,
        "candidate_count": len(reranked_spans),
        "base_candidate_count": len(base_spans),
        "top_changed": _span_identity(base_top) != _span_identity(reranked_top)
        if base_spans and reranked_spans
        else False,
        "base_top_ref": _span_ref(base_top),
        "reranked_top_ref": _span_ref(reranked_top),
        "reranked_top_score": _mapping(reranked_top.get("teaching_moment_span")).get("score"),
        "reranked_top_score_bucket": _score_bucket(
            _optional_float(_mapping(reranked_top.get("teaching_moment_span")).get("score"))
        ),
        "span_window": {
            "mode": span_config.get("mode"),
            "window_seconds": span_config.get("window_seconds"),
            "window_before_seconds": span_config.get("window_before_seconds"),
            "window_after_seconds": span_config.get("window_after_seconds"),
            "neighbor_count": span_config.get("neighbor_count"),
        },
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "score_component_presence_counts": dict(sorted(component_presence.items())),
        "query_relevance_bucket_counts": dict(sorted(bucket_counts.items())),
        "privacy": {
            "raw_queries": "excluded",
            "transcript_text": "excluded",
            "evidence_text": "excluded",
            "local_paths": "excluded",
            "raw_candidate_ids": "hashed",
        },
        "public_note": PUBLIC_NOTE,
    }


def public_teaching_moment_candidate_diagnostics(candidate: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = _mapping(candidate.get("teaching_moment_span"))
    if not diagnostics:
        return None
    return {
        "schema_version": TEACHING_MOMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "strategy": diagnostics.get("strategy"),
        "span_ref": diagnostics.get("span_ref"),
        "anchor_ref": diagnostics.get("anchor_ref"),
        "original_rank": diagnostics.get("original_rank"),
        "reranked_rank": diagnostics.get("reranked_rank"),
        "score": diagnostics.get("score"),
        "score_bucket": _score_bucket(_optional_float(diagnostics.get("score"))),
        "components": _mapping(diagnostics.get("components")),
        "buckets": _mapping(diagnostics.get("buckets")),
        "reason_codes": _string_list(diagnostics.get("reason_codes")),
        "aggregate_counts": _mapping(diagnostics.get("aggregate_counts")),
        "privacy": {
            "raw_queries": "excluded",
            "transcript_text": "excluded",
            "evidence_text": "excluded",
            "raw_candidate_ids": "hashed",
        },
    }


def _load_span_artifacts(
    *,
    project_dir: Path,
    segments_path: Path | None,
    frames_manifest_path: Path | None,
    visual_entities_path: Path | None,
    entity_links_path: Path | None,
    evidence_units_path: Path | None,
) -> dict[str, Any]:
    resolved_segments_path = segment_artifact_path(project_dir, segments=segments_path)
    resolved_frames_path = resolve_project_path(
        project_dir,
        frames_manifest_path,
        default=project_dir / "manifests" / "frames_manifest.jsonl",
    )
    resolved_visual_entities_path = resolve_project_path(
        project_dir,
        visual_entities_path,
        default=project_dir / "manifests" / "visual_entities.jsonl",
    )
    resolved_entity_links_path = resolve_project_path(
        project_dir,
        entity_links_path,
        default=project_dir / "manifests" / "entity_links.jsonl",
    )
    try:
        resolved_evidence_units_path = evidence_unit_artifact_path(
            project_dir,
            evidence_units=evidence_units_path,
        )
    except FileNotFoundError:
        resolved_evidence_units_path = None
    return {
        "segments": read_jsonl(resolved_segments_path),
        "frames": read_jsonl(resolved_frames_path) if resolved_frames_path.exists() else [],
        "visual_entities": read_jsonl(resolved_visual_entities_path)
        if resolved_visual_entities_path.exists()
        else [],
        "entity_links": read_jsonl(resolved_entity_links_path)
        if resolved_entity_links_path.exists()
        else [],
        "evidence_units": read_jsonl(resolved_evidence_units_path)
        if resolved_evidence_units_path is not None
        else [],
    }


def _span_window_config(
    *,
    span_window_seconds: float | None,
    span_window_before_seconds: float | None,
    span_window_after_seconds: float | None,
    span_neighbor_count: int,
) -> dict[str, Any]:
    use_time_window = (
        span_window_seconds is not None
        or span_window_before_seconds is not None
        or span_window_after_seconds is not None
    )
    if use_time_window:
        symmetric = _nonnegative_float(span_window_seconds, default=DEFAULT_SPAN_WINDOW_SECONDS)
        before = _nonnegative_float(span_window_before_seconds, default=symmetric)
        after = _nonnegative_float(span_window_after_seconds, default=symmetric)
        return resolve_window_config(
            window_seconds=None,
            neighbor_count=0,
            window_before_seconds=before,
            window_after_seconds=after,
        )
    return resolve_window_config(
        window_seconds=None,
        neighbor_count=max(0, int(span_neighbor_count)),
    )


def _query_context(*, query: str, domain_lexicon: DomainLexicon) -> dict[str, Any]:
    expansion = domain_lexicon.expand_query_result(query)
    query_texts = [query, *(term.term for term in expansion.terms)]
    terms = _coverage_terms(" ".join(query_texts))
    aliases = set()
    for term in terms:
        canonical = domain_lexicon.canonicalize(term)
        aliases.update(
            _coverage_terms(" ".join(domain_lexicon.aliases_by_canonical.get(canonical, ())))
        )
    return {
        "query_terms": terms,
        "alias_terms": aliases,
        "query_text_count": len([item for item in query_texts if str(item).strip()]),
    }


def _span_score_input(
    *,
    query_context: dict[str, Any],
    transcript_text: str,
    visual_entities: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
    linked_entities: list[dict[str, Any]],
    source_segment_ids: list[str],
    source_quality: dict[str, Any],
    anchor_candidate: dict[str, Any],
    anchor_rank: int,
) -> dict[str, Any]:
    evidence_text = _compact_text(
        " ".join(
            _unique_text_values(
                unit.get("evidence_text") for unit in evidence_units
            )
        )
    )
    semantic_text = _compact_text(
        " ".join(
            _unique_text_values(
                [
                    *(unit.get("semantic_text") for unit in evidence_units),
                    *(unit.get("transcript_window_text") for unit in evidence_units),
                ]
            )
        )
    )
    concept_text = _compact_text(
        " ".join(
            _unique_text_values(
                [
                    *(
                        label
                        for unit in evidence_units
                        for label in _string_list(unit.get("concept_labels"))
                    ),
                    *(
                        alias
                        for unit in evidence_units
                        for alias in _string_list(unit.get("concept_aliases"))
                    ),
                    *(unit.get("concept_relation_text") for unit in evidence_units),
                    *(unit.get("concept_search_text") for unit in evidence_units),
                ]
            )
        )
    )
    visual_text = _compact_text(
        " ".join(
            _unique_text_values(
                [
                    *(
                        entity.get(key)
                        for entity in visual_entities
                        for key in (
                            "text",
                            "detected_text",
                            "visual_description",
                            "entity_type",
                        )
                    ),
                    *(
                        _mapping(link.get("entity")).get(key)
                        for link in linked_entities
                        for key in ("text", "detected_text", "visual_description")
                    ),
                ]
            )
        )
    )
    return {
        "query_context": query_context,
        "transcript_overlap": _query_text_overlap(query_context, transcript_text),
        "evidence_overlap": _query_text_overlap(query_context, evidence_text),
        "semantic_overlap": _query_text_overlap(query_context, semantic_text),
        "concept_overlap": _query_text_overlap(query_context, concept_text),
        "visual_overlap": _query_text_overlap(query_context, visual_text),
        "source_segment_count": len(source_segment_ids),
        "evidence_unit_count": len(evidence_units),
        "visual_entity_count": len(visual_entities),
        "linked_entity_count": len(linked_entities),
        "source_quality": source_quality,
        "anchor_score": _optional_float(anchor_candidate.get("score")),
        "anchor_rank": anchor_rank,
        "anchor_identity": _candidate_identity(anchor_candidate),
    }


def _span_score_diagnostics(score_input: dict[str, Any]) -> dict[str, Any]:
    transcript = _mapping(score_input.get("transcript_overlap"))
    evidence = _mapping(score_input.get("evidence_overlap"))
    semantic = _mapping(score_input.get("semantic_overlap"))
    concept = _mapping(score_input.get("concept_overlap"))
    visual = _mapping(score_input.get("visual_overlap"))
    source_quality = _mapping(score_input.get("source_quality"))
    query_relevance = max(
        float(transcript.get("ratio") or 0.0),
        float(evidence.get("ratio") or 0.0),
        float(semantic.get("ratio") or 0.0),
    )
    concept_ratio = float(concept.get("ratio") or 0.0)
    visual_ratio = float(visual.get("ratio") or 0.0)
    verified_count = int(source_quality.get("verified_link_count") or 0)
    candidate_count = int(source_quality.get("candidate_link_count") or 0)
    timestamp_fallback_count = int(source_quality.get("timestamp_fallback_link_count") or 0)
    visual_density = min(
        1.0,
        (
            int(score_input.get("visual_entity_count") or 0)
            + int(source_quality.get("visual_state_count") or 0)
        )
        / 6.0,
    )
    source_density = min(1.0, float(score_input.get("source_segment_count") or 0) / 5.0)
    evidence_density = min(1.0, float(score_input.get("evidence_unit_count") or 0) / 4.0)
    concept_component = 0.28 * concept_ratio
    concept_capped = False
    if query_relevance < 0.2:
        concept_component = min(concept_component, 0.08)
        concept_capped = concept_ratio > 0
    visual_component = min(0.22, 0.10 * visual_density + 0.18 * visual_ratio)
    if query_relevance <= 0:
        visual_component = min(visual_component, 0.06)
    verified_component = 0.08 if verified_count else 0.0
    candidate_component = min(0.06, candidate_count * 0.015)
    timestamp_component = -0.08 if timestamp_fallback_count and not verified_count else 0.0
    anchor_score = _optional_float(score_input.get("anchor_score")) or 0.0
    anchor_rank = max(1, int(score_input.get("anchor_rank") or 1))
    components = {
        "anchor_prior": round(min(0.16, 0.08 / anchor_rank + 0.08 * min(anchor_score, 1.0)), 6),
        "query_text_overlap": round(1.15 * query_relevance, 6),
        "evidence_semantic_overlap": round(
            0.45
            * max(float(evidence.get("ratio") or 0.0), float(semantic.get("ratio") or 0.0)),
            6,
        ),
        "concept_support": round(concept_component, 6),
        "visual_support": round(visual_component, 6),
        "verified_object_alignment": round(verified_component, 6),
        "candidate_visual_signal": round(candidate_component, 6),
        "span_coherence": round(0.10 * source_density + 0.08 * evidence_density, 6),
        "timestamp_fallback_candidate_only": round(timestamp_component, 6),
    }
    score = round(sum(float(value) for value in components.values()), 6)
    reason_codes = []
    if query_relevance > 0:
        reason_codes.append("query_text_overlap")
    if evidence.get("match_count") or semantic.get("match_count"):
        reason_codes.append("evidence_semantic_overlap")
    if concept_ratio > 0:
        reason_codes.append("concept_support_capped" if concept_capped else "concept_support")
    if visual_component > 0:
        reason_codes.append("visual_support")
    if verified_count:
        reason_codes.append("verified_object_alignment")
    if candidate_count:
        reason_codes.append("candidate_visual_signal")
    if timestamp_fallback_count:
        reason_codes.append("timestamp_fallback_candidate_only")
    if source_density > 0 or evidence_density > 0:
        reason_codes.append("span_coherence")
    return {
        "schema_version": TEACHING_MOMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "strategy": TEACHING_MOMENT_HIT_SOURCE,
        "span_ref": f"span:{_short_hash(str(score_input.get('anchor_identity') or 'span'))}",
        "anchor_ref": f"anchor:{_short_hash(str(score_input.get('anchor_identity') or 'anchor'))}",
        "original_rank": score_input.get("anchor_rank"),
        "reranked_rank": None,
        "score": score,
        "components": components,
        "buckets": {
            "query_relevance": _ratio_bucket(query_relevance),
            "transcript_overlap": transcript.get("bucket"),
            "evidence_overlap": evidence.get("bucket"),
            "semantic_overlap": semantic.get("bucket"),
            "concept_overlap": concept.get("bucket"),
            "visual_overlap": visual.get("bucket"),
            "visual_density": _ratio_bucket(visual_density),
            "source_density": _ratio_bucket(source_density),
            "evidence_density": _ratio_bucket(evidence_density),
        },
        "reason_codes": reason_codes,
        "aggregate_counts": {
            "query_term_count": int(transcript.get("query_term_count") or 0),
            "transcript_match_count": int(transcript.get("match_count") or 0),
            "evidence_match_count": int(evidence.get("match_count") or 0),
            "semantic_match_count": int(semantic.get("match_count") or 0),
            "concept_match_count": int(concept.get("match_count") or 0),
            "visual_match_count": int(visual.get("match_count") or 0),
            "source_segment_count": int(score_input.get("source_segment_count") or 0),
            "evidence_unit_count": int(score_input.get("evidence_unit_count") or 0),
            "visual_entity_count": int(score_input.get("visual_entity_count") or 0),
            "linked_entity_count": int(score_input.get("linked_entity_count") or 0),
            "candidate_link_count": candidate_count,
            "verified_link_count": verified_count,
            "timestamp_fallback_link_count": timestamp_fallback_count,
        },
        "privacy": {
            "raw_queries": "excluded",
            "transcript_text": "excluded",
            "evidence_text": "excluded",
            "raw_candidate_ids": "hashed",
        },
    }


def _span_source_quality(
    *,
    visual_entities: list[dict[str, Any]],
    linked_entities: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_signal_counts = _zero_count_map(CANDIDATE_LINK_SIGNAL_KEYS)
    verified_source_counts = _zero_count_map(VERIFIED_LINK_SOURCE_KEYS)
    candidate_link_count = 0
    verified_link_count = 0
    timestamp_fallback_link_count = 0
    candidate_statuses = _candidate_link_statuses(linked_entities)
    for link in linked_entities:
        status = _link_alignment_status(link)
        if status == "verified":
            verified_link_count += 1
            _add_counts(verified_source_counts, _verified_link_source_counts(link))
        else:
            if status == "timestamp_fallback":
                timestamp_fallback_link_count += 1
            else:
                candidate_link_count += 1
            _add_counts(candidate_signal_counts, _candidate_link_signal_counts(link, status=status))

    concept_count = 0
    concept_relation_count = 0
    timestamp_only_concept_relation_count = 0
    visual_state_ids: set[str] = set()
    visual_entity_ids: set[str] = {str(entity.get("entity_id")) for entity in visual_entities if entity.get("entity_id")}
    for unit in evidence_units:
        quality = _mapping(unit.get("source_quality"))
        _add_counts(candidate_signal_counts, _mapping(quality.get("candidate_link_signal_counts")))
        _add_counts(verified_source_counts, _mapping(quality.get("verified_link_source_counts")))
        candidate_link_count += int(quality.get("candidate_link_count") or 0)
        verified_link_count += int(quality.get("verified_link_count") or 0)
        timestamp_fallback_link_count += int(quality.get("timestamp_fallback_link_count") or 0)
        concept_count += int(quality.get("concept_count") or len(_string_list(unit.get("concept_ids"))))
        concept_relation_count += int(
            quality.get("concept_relation_count")
            or len(_list_of_dicts(unit.get("concept_relations")))
        )
        timestamp_only_concept_relation_count += int(
            quality.get("timestamp_only_concept_relation_count") or 0
        )
        visual_state_ids.update(_string_list(unit.get("visual_state_ids")))
        visual_entity_ids.update(_string_list(unit.get("visual_entity_ids")))

    has_vlm_entity = any(_is_vlm_entity(entity) for entity in visual_entities) or any(
        _mapping(unit.get("source_quality")).get("has_vlm_entity") is True
        for unit in evidence_units
    )
    detected_text_count = sum(
        1
        for entity in visual_entities
        if str(entity.get("detected_text") or entity.get("text") or "").strip()
    )
    visual_description_count = sum(
        1 for entity in visual_entities if str(entity.get("visual_description") or "").strip()
    )
    visual_state_count = len(visual_state_ids)
    visual_entity_count = len(visual_entity_ids)
    if visual_entities and visual_entity_count == 0:
        visual_entity_count = len(visual_entities)
    has_candidate_visual_support = bool(
        visual_state_count
        or visual_entity_count
        or candidate_link_count
        or timestamp_fallback_link_count
    )
    has_verified_object_alignment = verified_link_count > 0
    return {
        "has_visual_state": visual_state_count > 0,
        "has_visual_entity": visual_entity_count > 0,
        "has_vlm_entity": has_vlm_entity,
        "uses_ocr_only": bool(visual_entity_count) and not has_vlm_entity,
        "has_detected_text": detected_text_count > 0,
        "has_visual_description": visual_description_count > 0,
        "has_concept": concept_count > 0,
        "has_concept_relation": concept_relation_count > 0,
        "has_concept_search_text": any(
            _has_concept_search_text(unit) for unit in evidence_units
        ),
        "has_verified_link": has_verified_object_alignment,
        "has_timestamp_fallback_link": timestamp_fallback_link_count > 0,
        "visual_state_count": visual_state_count,
        "visual_entity_count": visual_entity_count,
        "visual_state_detected_text_count": 0,
        "visual_entity_detected_text_count": detected_text_count,
        "visual_description_count": visual_description_count,
        "concept_count": concept_count,
        "concept_label_count": len(
            _unique_text_values(
                label
                for unit in evidence_units
                for label in _string_list(unit.get("concept_labels"))
            )
        ),
        "concept_alias_count": len(
            _unique_text_values(
                alias
                for unit in evidence_units
                for alias in _string_list(unit.get("concept_aliases"))
            )
        ),
        "concept_relation_count": concept_relation_count,
        "timestamp_only_concept_relation_count": timestamp_only_concept_relation_count,
        "candidate_link_count": candidate_link_count,
        "timestamp_fallback_link_count": timestamp_fallback_link_count,
        "verified_link_count": verified_link_count,
        "candidate_link_signal_counts": candidate_signal_counts,
        "verified_link_source_counts": verified_source_counts,
        "candidate_visual_support": {
            "has_candidate_visual_support": has_candidate_visual_support,
            "visual_state_count": visual_state_count,
            "visual_entity_count": visual_entity_count,
            "candidate_link_count": candidate_link_count,
            "timestamp_fallback_link_count": timestamp_fallback_link_count,
            "candidate_link_signal_counts": candidate_signal_counts,
            "paper_claim_eligible": False,
        },
        "verified_object_alignment": {
            "has_verified_object_alignment": has_verified_object_alignment,
            "verified_link_count": verified_link_count,
            "verified_link_source_counts": verified_source_counts,
            "timestamp_fallback_counted_as_verified": False,
            "paper_claim_eligible": has_verified_object_alignment,
        },
        "concept_field_coverage": {
            "has_concept_search_text": any(
                _has_concept_search_text(unit) for unit in evidence_units
            ),
            "concept_count": concept_count,
            "concept_relation_count": concept_relation_count,
            "timestamp_only_concept_relation_count": timestamp_only_concept_relation_count,
            "timestamp_only_counted_as_verified_object_alignment": False,
        },
        "candidate_entity_link_statuses": candidate_statuses,
    }


def _visual_entities_for_span(
    *,
    visual_entities: list[dict[str, Any]],
    evidence_window: dict[str, Any],
    links: list[dict[str, Any]],
    start_time: float | None,
    end_time: float | None,
    entity_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    frame_ids = {
        str(frame.get("frame_id") or "")
        for frame in _list_of_dicts(evidence_window.get("frame_refs"))
        if str(frame.get("frame_id") or "")
    }
    selected: dict[str, dict[str, Any]] = {}
    for entity in visual_entities:
        entity_id = str(entity.get("entity_id") or "")
        if not entity_id:
            continue
        timestamp = _optional_float(entity.get("timestamp"))
        in_time = (
            start_time is not None
            and end_time is not None
            and timestamp is not None
            and min(start_time, end_time) <= timestamp <= max(start_time, end_time)
        )
        in_frame = str(entity.get("frame_id") or "") in frame_ids
        if in_time or in_frame:
            selected[entity_id] = entity
    for link in links:
        entity_id = str(link.get("entity_id") or "")
        entity = entity_lookup.get(entity_id)
        if entity is not None:
            selected[entity_id] = entity
    return sorted(selected.values(), key=lambda entity: (float(_optional_float(entity.get("timestamp")) or 0.0), str(entity.get("entity_id") or "")))


def _links_for_span(
    *,
    links: list[dict[str, Any]],
    source_segment_ids: list[str],
    entity_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_set = set(source_segment_ids)
    selected = []
    for link in links:
        if str(link.get("segment_id") or "") not in source_set:
            continue
        row = dict(link)
        entity = entity_lookup.get(str(link.get("entity_id") or ""))
        if entity is not None:
            row["entity"] = entity
        selected.append(row)
    return sorted(selected, key=lambda link: str(link.get("link_id") or ""))


def _evidence_units_for_span(
    *,
    evidence_units: list[dict[str, Any]],
    source_segment_ids: list[str],
    start_time: float | None,
    end_time: float | None,
) -> list[dict[str, Any]]:
    source_set = set(source_segment_ids)
    selected = []
    for unit in evidence_units:
        unit_segments = set(_string_list(unit.get("source_segment_ids")))
        target_id = str(unit.get("target_segment_id") or "")
        shares_segment = bool(unit_segments & source_set or target_id in source_set)
        overlaps_time = _ranges_overlap(
            (_optional_float(unit.get("start_time")), _optional_float(unit.get("end_time"))),
            (start_time, end_time),
        )
        if shares_segment or overlaps_time:
            selected.append(unit)
    return sorted(
        selected,
        key=lambda unit: (
            _optional_float(unit.get("start_time")) is None,
            _optional_float(unit.get("start_time")) or 0.0,
            str(unit.get("evidence_unit_id") or ""),
        ),
    )


def _span_evidence_text(
    *,
    transcript_text: str,
    visual_entities: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
) -> str:
    return _compact_text(
        " ".join(
            _unique_text_values(
                [
                    transcript_text,
                    *(unit.get("evidence_text") for unit in evidence_units),
                    *(unit.get("semantic_text") for unit in evidence_units),
                    *(
                        entity.get(key)
                        for entity in visual_entities
                        for key in ("text", "detected_text", "visual_description")
                    ),
                ]
            )
        )
    )


def _span_semantic_text(
    *,
    transcript_text: str,
    visual_entities: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
) -> str:
    return _compact_text(
        " ".join(
            _unique_text_values(
                [
                    transcript_text,
                    *(unit.get("semantic_text") for unit in evidence_units),
                    *(unit.get("concept_search_text") for unit in evidence_units),
                    *(
                        entity.get(key)
                        for entity in visual_entities
                        for key in ("text", "detected_text", "visual_description", "entity_type")
                    ),
                ]
            )
        )
    )


def _anchor_retrieval_sources(anchor_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    sources = []
    for source in _list_of_dicts(anchor_bundle.get("retrieval_sources")):
        sanitized = {
            key: source.get(key)
            for key in (
                "source",
                "modality",
                "retrieval_mode",
                "rank",
                "score",
                "original_rank",
                "original_score",
                "fusion_score",
                "target_segment_id",
                "source_segment_ids",
                "window_id",
            )
            if key in source
        }
        sources.append(sanitized)
    return sources


def _candidate_link_signal_counts(link: dict[str, Any], *, status: str) -> dict[str, int]:
    counts = _zero_count_map(CANDIDATE_LINK_SIGNAL_KEYS)
    evidence = _link_evidence(link)
    if link.get("time_overlap") is True or "time_overlap" in evidence:
        counts["temporal_overlap"] += 1
    if _string_list(link.get("lexical_match")) or evidence & {"lexical_match", "visual_text_match"}:
        counts["lexical_overlap"] += 1
    if _string_list(link.get("mention_candidate")) or evidence & {"mention_candidate", "reference_cue"}:
        counts["mention_deictic_hook"] += 1
    if evidence & {"position_match", "relations_match"}:
        counts["spatial_position"] += 1
    if "visual_text_match" in evidence:
        counts["visual_text_overlap"] += 1
    if evidence & {"visual_description_match", "entity_type_match"}:
        counts["vlm_object_visual_description_overlap"] += 1
    if evidence & {"semantic_hint", "domain_lexicon_match"}:
        counts["semantic_domain_hint"] += 1
    if status == "timestamp_fallback" or "timestamp_fallback" in evidence:
        counts["timestamp_fallback"] += 1
    return counts


def _verified_link_source_counts(link: dict[str, Any]) -> dict[str, int]:
    counts = _zero_count_map(VERIFIED_LINK_SOURCE_KEYS)
    matched = False
    explicit_status = str(
        link.get("alignment_status")
        or link.get("verification_status")
        or link.get("status")
        or ""
    ).casefold()
    if link.get("verified") is True:
        counts["explicit_verified_flag"] += 1
        matched = True
    if explicit_status == "verified":
        counts["explicit_verified_status"] += 1
        matched = True
    source_text = _verified_source_text(link)
    if any(token in source_text for token in ("human", "gold", "annotator", "annotation")):
        counts["human_gold"] += 1
        matched = True
    if any(token in source_text for token in ("vlm", "vision", "verifier", "validator", "model")):
        counts["vlm_verifier"] += 1
        matched = True
    if any(token in source_text for token in ("strict", "deterministic", "rule")):
        counts["strict_deterministic_rule"] += 1
        matched = True
    if not matched:
        counts["unspecified_verified"] += 1
    return counts


def _link_alignment_status(link: dict[str, Any]) -> str:
    explicit_status = str(
        link.get("alignment_status")
        or link.get("verification_status")
        or link.get("status")
        or ""
    ).casefold()
    if explicit_status == "verified" or link.get("verified") is True:
        return "verified"
    if _timestamp_fallback_link(link):
        return "timestamp_fallback"
    return "candidate"


def _timestamp_fallback_link(link: dict[str, Any]) -> bool:
    evidence = _link_evidence(link)
    link_type = str(link.get("link_type") or "").casefold()
    reason_summary = str(_mapping(link.get("reason_metadata")).get("summary") or "").casefold()
    has_strong_signal = bool(evidence - {"time_overlap", "timestamp_fallback"})
    return (
        not has_strong_signal
        or link_type == "time_overlap"
        or reason_summary == "timestamp_fallback_only"
    )


def _link_evidence(link: dict[str, Any]) -> set[str]:
    return {str(item).casefold() for item in link.get("evidence") or [] if str(item).strip()}


def _verified_source_text(link: dict[str, Any]) -> str:
    metadata = _mapping(link.get("reason_metadata"))
    values = [
        link.get("verification_source"),
        link.get("verified_source"),
        link.get("verified_by"),
        link.get("verifier"),
        link.get("source"),
        link.get("source_model"),
        metadata.get("verification_source"),
        metadata.get("verified_by"),
        metadata.get("verifier"),
        metadata.get("source"),
        metadata.get("source_model"),
        metadata.get("summary"),
    ]
    return " ".join(str(value).strip() for value in values if str(value or "").strip()).casefold()


def _candidate_link_ids(links: list[dict[str, Any]]) -> list[str]:
    return [
        str(link.get("link_id"))
        for link in links
        if str(link.get("link_id") or "") and _link_alignment_status(link) != "verified"
    ]


def _link_ids_by_status(links: list[dict[str, Any]], *, status: str) -> list[str]:
    return [
        str(link.get("link_id"))
        for link in links
        if str(link.get("link_id") or "") and _link_alignment_status(link) == status
    ]


def _candidate_link_statuses(links: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(link.get("link_id")): _link_alignment_status(link)
        for link in links
        if str(link.get("link_id") or "") and _link_alignment_status(link) != "verified"
    }


def _query_text_overlap(query_context: dict[str, Any], candidate_text: str) -> dict[str, Any]:
    query_terms = set(query_context.get("query_terms") or set())
    alias_terms = set(query_context.get("alias_terms") or set())
    candidate_terms = _coverage_terms(candidate_text)
    matched = query_terms & candidate_terms
    alias_matched = alias_terms & candidate_terms
    effective_match_count = len(matched) + min(len(alias_matched), max(1, len(query_terms) // 2))
    ratio = round(effective_match_count / len(query_terms), 6) if query_terms else 0.0
    ratio = min(1.0, ratio)
    return {
        "query_term_count": len(query_terms),
        "match_count": len(matched),
        "alias_match_count": len(alias_matched),
        "ratio": ratio,
        "bucket": _ratio_bucket(ratio),
    }


def _coverage_terms(value: str) -> set[str]:
    return {
        term
        for term in (match.group(0).strip("_").casefold() for match in _TOKEN_RE.finditer(value))
        if term and term not in _STOPWORDS and not (len(term) == 1 and term.isascii())
    }


def _ratio_bucket(value: float | None) -> str:
    ratio = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    for bucket, (lower, upper) in _DENSITY_BUCKETS.items():
        if bucket == "none" and ratio == 0:
            return bucket
        if bucket != "none" and lower < ratio <= upper:
            return bucket
    return "very_high"


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "not_available"
    if score < 0:
        return "negative"
    if score < 0.5:
        return "low"
    if score < 1.0:
        return "medium"
    if score < 1.5:
        return "high"
    return "very_high"


def _span_summary_lines(bundles: list[dict[str, Any]]) -> list[str]:
    lines = []
    for bundle in bundles:
        candidate = _mapping(bundle.get("candidate"))
        diagnostics = _mapping(candidate.get("teaching_moment_span"))
        lines.append(
            "teaching_moment_span "
            f"rank={bundle.get('rank')} "
            f"score_bucket={_score_bucket(_optional_float(diagnostics.get('score')))} "
            f"relevance={_mapping(diagnostics.get('buckets')).get('query_relevance')}"
        )
    return lines


def _target_segment_id(bundle: dict[str, Any]) -> str:
    candidate = _mapping(bundle.get("candidate"))
    evidence_window = _mapping(bundle.get("evidence_window"))
    target = _mapping(evidence_window.get("target_segment"))
    return str(
        target.get("segment_id")
        or candidate.get("target_segment_id")
        or candidate.get("segment_id")
        or evidence_window.get("target_segment_id")
        or ""
    )


def _candidate_identity(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("span_id")
        or candidate.get("segment_id")
        or candidate.get("target_segment_id")
        or candidate.get("window_id")
        or candidate.get("sample_id")
        or candidate.get("rank")
        or ""
    )


def _span_identity(candidate: dict[str, Any]) -> str:
    return _candidate_identity(candidate)


def _span_ref(candidate: dict[str, Any]) -> str | None:
    identity = _span_identity(candidate)
    if not identity:
        return None
    return f"span:{_short_hash(identity)}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _source_segment_ids(segments: Iterable[dict[str, Any]]) -> list[str]:
    return [
        segment_id
        for segment_id in (str(segment.get("segment_id") or "") for segment in segments)
        if segment_id
    ]


def _span_bounds(
    evidence_window: dict[str, Any],
    window_segments: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    start = _optional_float(evidence_window.get("start_time"))
    end = _optional_float(evidence_window.get("end_time"))
    if start is not None or end is not None:
        return start, end
    bounds = [
        bound
        for segment in window_segments
        for bound in (segment_window(segment) or ())
        if bound is not None
    ]
    if not bounds:
        return None, None
    return min(bounds), max(bounds)


def _ranges_overlap(
    left: tuple[float | None, float | None],
    right: tuple[float | None, float | None],
) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return False
    return min(left_start, left_end) <= max(right_start, right_end) and min(
        right_start,
        right_end,
    ) <= max(left_start, left_end)


def _transcript_window_text(segments: list[dict[str, Any]]) -> str:
    return _compact_text(
        " ".join(_unique_text_values(segment.get("transcript_text") for segment in segments))
    )


def _has_concept_search_text(unit: dict[str, Any]) -> bool:
    return bool(
        str(unit.get("concept_search_text") or "").strip()
        or _string_list(unit.get("concept_labels"))
        or _string_list(unit.get("concept_aliases"))
        or str(unit.get("concept_relation_text") or "").strip()
    )


def _is_vlm_entity(entity: dict[str, Any]) -> bool:
    source_text = " ".join(
        str(entity.get(key) or "")
        for key in ("source", "source_model", "model", "entity_type")
    ).casefold()
    return bool(
        "vlm" in source_text
        or "vision" in source_text
        or str(entity.get("visual_description") or "").strip()
    )


def _anchor_index_kind(value: str) -> str:
    normalized = str(value or SEGMENT_HIT_SOURCE).strip().casefold().replace("_", "-")
    if normalized in {"window", "windows", "lecture-window"}:
        return WINDOW_HIT_SOURCE
    return SEGMENT_HIT_SOURCE


def _project_id(segments: list[dict[str, Any]]) -> str | None:
    for segment in segments:
        value = str(segment.get("project_id") or "").strip()
        if value:
            return value
    return None


def _add_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in target:
        target[key] += int(source.get(key) or 0)


def _zero_count_map(keys: Iterable[str]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item or "").strip()]


def _unique_text_values(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _unique_text_values(value.values())
        elif isinstance(value, (list, tuple, set)):
            nested = _unique_text_values(value)
        else:
            nested = [_compact_text(str(value))]
        for item in nested:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
    return unique


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _midpoint(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (start + end) / 2.0


def _nonnegative_float(value: Any, *, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        return default
    return max(0.0, parsed)


def _positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value
