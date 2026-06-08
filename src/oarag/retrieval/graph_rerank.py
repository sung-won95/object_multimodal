from __future__ import annotations

import hashlib
import re
from typing import Any


GRAPH_AWARE_RERANK = "graph_aware"

_GRAPH_TRAVERSAL_SOURCE = "graph_traversal"
_GRAPH_RELATED_MATCH_TYPES = {
    "related_concept_evidence",
    "global_concept_related_evidence",
}
_GRAPH_DIRECT_MATCH_TYPES = {
    "direct_mention",
    "concept_source_evidence",
}
_NON_RELATIONSHIP_TYPES = {
    "CONCEPT_SOURCE_EVIDENCE",
    "MENTIONS",
    "RELATED_SOURCE_EVIDENCE",
    "INSTANCE_OF_GLOBAL_CONCEPT",
}
_RELATION_QUERY_TERMS = {
    "alias": {"alias", "same_as"},
    "compare": {"compare", "contrasts_with", "related_to"},
    "contrast": {"contrast", "contrasts_with"},
    "define": {"defines", "definition", "is_a"},
    "depends": {"depends_on", "requires", "prerequisite_of"},
    "explains": {"explains", "supports", "related_to"},
    "prerequisite": {"prerequisite_of", "requires", "depends_on"},
    "related": {"related_to"},
    "uses": {"uses", "used_by"},
}
_STOPWORDS = {
    "about",
    "after",
    "before",
    "does",
    "during",
    "explain",
    "explains",
    "find",
    "from",
    "give",
    "how",
    "into",
    "lecture",
    "lecturer",
    "part",
    "show",
    "shows",
    "slide",
    "tell",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def graph_aware_rerank_candidates(
    candidates: list[dict[str, Any]],
    *,
    query: str,
    query_analysis: Any | None = None,
) -> list[dict[str, Any]]:
    """Rerank dual candidates with deterministic graph and object-evidence signals."""

    analysis = _query_analysis_mapping(query_analysis)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for fallback_rank, candidate in enumerate(candidates, start=1):
        original_rank = _optional_int(candidate.get("rank")) or fallback_rank
        breakdown = graph_aware_rerank_breakdown(
            candidate,
            query=query,
            query_analysis=analysis,
            original_rank=original_rank,
        )
        updated = dict(candidate)
        updated["graph_aware_rerank"] = {
            "strategy": GRAPH_AWARE_RERANK,
            "query_type": _query_type(analysis),
            "score": breakdown["score"],
            "original_rank": original_rank,
            "components": breakdown["components"],
            "flags": breakdown["flags"],
            "match_counts": breakdown["match_counts"],
            "source_summary": breakdown["source_summary"],
            "public_note": (
                "Graph-aware rerank diagnostics expose query type, source ids, feature flags, "
                "match counts, and numeric score components only; raw query and evidence text "
                "are omitted."
            ),
        }
        scored.append((float(breakdown["score"]), original_rank, updated))

    reranked: list[dict[str, Any]] = []
    for new_rank, (_, _, candidate) in enumerate(
        sorted(scored, key=lambda item: (-item[0], item[1])),
        start=1,
    ):
        updated = dict(candidate)
        rerank = dict(_mapping(updated.get("graph_aware_rerank")))
        rerank["reranked_rank"] = new_rank
        updated["graph_aware_rerank"] = rerank
        updated["rank"] = new_rank
        reranked.append(updated)
    return reranked


def graph_aware_rerank_breakdown(
    candidate: dict[str, Any],
    *,
    query: str,
    query_analysis: Any | None = None,
    original_rank: int | None = None,
) -> dict[str, Any]:
    analysis = _query_analysis_mapping(query_analysis)
    rank = original_rank or _optional_int(candidate.get("rank")) or 1
    source_summary = _source_summary(candidate)
    graph_features = _graph_features(candidate, query=query, query_analysis=analysis)
    concept_features = _concept_features(candidate, query=query, query_analysis=analysis)
    text_features = _text_features(candidate, query=query)
    quality_features = _quality_features(candidate)
    score = _candidate_score(candidate)

    components = {
        "rank_preservation": round(0.08 / max(rank, 1), 6),
        "retrieval_score": round(0.18 * max(0.0, min(float(score or 0.0), 1.0)), 6),
        "evidence_text_query_overlap": round(0.38 * text_features["evidence_ratio"], 6),
        "semantic_text_query_overlap": round(0.3 * text_features["semantic_ratio"], 6),
        "transcript_query_overlap": round(0.16 * text_features["transcript_ratio"], 6),
        "query_concept_direct_match": round(0.58 * concept_features["direct_ratio"], 6),
        "related_concept_graph_path": graph_features["related_path_score"],
        "graph_direct_match": graph_features["direct_graph_score"],
        "relation_type_match": graph_features["relation_type_score"],
        "visual_description": quality_features["visual_description_score"],
        "visual_object_support": quality_features["visual_object_support_score"],
        "vlm_visual_entity": quality_features["vlm_visual_entity_score"],
        "verified_alignment": quality_features["verified_alignment_score"],
        "candidate_link_quality": quality_features["candidate_link_score"],
        "ocr_only_penalty": quality_features["ocr_only_penalty"],
        "timestamp_fallback_penalty": quality_features["timestamp_fallback_penalty"],
    }
    score_total = round(sum(float(value) for value in components.values()), 6)
    return {
        "score": score_total,
        "components": components,
        "flags": {
            "query_type": _query_type(analysis),
            "has_graph_source": source_summary["has_graph_source"],
            "has_related_graph_path": graph_features["has_related_graph_path"],
            "has_relation_type_match": graph_features["has_relation_type_match"],
            "has_direct_concept_match": concept_features["direct_match_count"] > 0,
            "has_visual_description": quality_features["has_visual_description"],
            "has_visual_object_support": quality_features["has_visual_object_support"],
            "has_vlm_visual_entity": quality_features["has_vlm_visual_entity"],
            "has_verified_alignment": quality_features["has_verified_alignment"],
            "has_candidate_link": quality_features["candidate_link_count"] > 0,
            "uses_ocr_only": quality_features["uses_ocr_only"],
            "has_timestamp_fallback_link": quality_features["has_timestamp_fallback"],
            "timestamp_only_fallback": quality_features["timestamp_only_fallback"],
            "combined_query_overlap_bucket": _ratio_bucket(text_features["combined_ratio"]),
        },
        "match_counts": {
            "query_term_count": text_features["query_term_count"],
            "evidence_text_match_count": text_features["evidence_match_count"],
            "semantic_text_match_count": text_features["semantic_match_count"],
            "transcript_match_count": text_features["transcript_match_count"],
            "query_concept_count": concept_features["query_concept_count"],
            "direct_concept_match_count": concept_features["direct_match_count"],
            "graph_source_count": source_summary["graph_source_count"],
            "related_graph_path_count": graph_features["related_path_count"],
            "relation_type_match_count": graph_features["relation_type_match_count"],
            "verified_link_count": quality_features["verified_link_count"],
            "candidate_link_count": quality_features["candidate_link_count"],
            "timestamp_fallback_link_count": quality_features["timestamp_fallback_count"],
        },
        "source_summary": source_summary,
    }


def graph_aware_rerank_context(
    *,
    base_candidates: list[dict[str, Any]],
    reranked_candidates: list[dict[str, Any]],
    query_analysis: Any | None = None,
) -> dict[str, Any]:
    analysis = _query_analysis_mapping(query_analysis)
    base_top = base_candidates[0] if base_candidates else {}
    reranked_top = reranked_candidates[0] if reranked_candidates else {}
    top_rerank = _mapping(reranked_top.get("graph_aware_rerank"))
    return {
        "enabled": True,
        "strategy": GRAPH_AWARE_RERANK,
        "query_type": _query_type(analysis),
        "candidate_count": len(reranked_candidates),
        "top_changed": _candidate_identity(base_top) != _candidate_identity(reranked_top),
        "base_top_ref": _candidate_ref(base_top),
        "reranked_top_ref": _candidate_ref(reranked_top),
        "reranked_top_score": top_rerank.get("score"),
        "reranked_top_original_rank": top_rerank.get("original_rank"),
        "reranked_top_rank": top_rerank.get("reranked_rank"),
        "component_names": sorted(_mapping(top_rerank.get("components"))),
        "public_safe": True,
        "public_note": (
            "Graph-aware rerank context exposes hashed refs, query type, counts, flags, "
            "and score component names only; raw query text and evidence text are omitted."
        ),
    }


def graph_aware_rerank_diagnostics(
    *,
    base_candidates: list[dict[str, Any]],
    reranked_candidates: list[dict[str, Any]],
    returned_candidates: list[dict[str, Any]],
    query_analysis: Any | None = None,
) -> dict[str, Any]:
    context = graph_aware_rerank_context(
        base_candidates=base_candidates,
        reranked_candidates=reranked_candidates,
        query_analysis=query_analysis,
    )
    return {
        **context,
        "returned_breakdowns": [
            _public_candidate_breakdown(candidate) for candidate in returned_candidates
        ],
        "omits": ["raw_query", "raw_transcript", "evidence_text", "local_paths", "frame_paths"],
    }


def _public_candidate_breakdown(candidate: dict[str, Any]) -> dict[str, Any]:
    rerank = _mapping(candidate.get("graph_aware_rerank"))
    return {
        "candidate_ref": _candidate_ref(candidate),
        "source_ids": _candidate_source_ids(candidate),
        "source_types": _string_list(candidate.get("candidate_source_types")),
        "original_rank": rerank.get("original_rank"),
        "reranked_rank": rerank.get("reranked_rank"),
        "score": rerank.get("score"),
        "components": _mapping(rerank.get("components")),
        "flags": _mapping(rerank.get("flags")),
        "match_counts": _mapping(rerank.get("match_counts")),
    }


def _source_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    sources = _candidate_sources(candidate)
    graph_sources = [
        source
        for source in sources
        if str(source.get("source_type") or "") == _GRAPH_TRAVERSAL_SOURCE
    ]
    graph_match_types = sorted(
        {
            str(source.get("graph_match_type") or "")
            for source in graph_sources
            if source.get("graph_match_type")
        }
    )
    relationship_types = sorted(
        {
            relationship
            for source in graph_sources
            for relationship in _relationship_types(source.get("relationships"))
        }
    )
    return {
        "candidate_key": str(candidate.get("candidate_key") or _candidate_identity(candidate)),
        "evidence_unit_id": str(candidate.get("evidence_unit_id") or ""),
        "target_segment_id": str(candidate.get("target_segment_id") or ""),
        "source_types": _string_list(candidate.get("candidate_source_types")),
        "source_count": len(sources),
        "graph_source_count": len(graph_sources),
        "has_graph_source": bool(graph_sources),
        "graph_match_types": graph_match_types,
        "relationship_types": relationship_types,
    }


def _graph_features(
    candidate: dict[str, Any],
    *,
    query: str,
    query_analysis: dict[str, Any],
) -> dict[str, Any]:
    graph_sources = [
        source
        for source in _candidate_sources(candidate)
        if str(source.get("source_type") or "") == _GRAPH_TRAVERSAL_SOURCE
    ]
    related_scores: list[float] = []
    direct_scores: list[float] = []
    relation_match_count = 0
    has_relation_type_match = False
    for source in graph_sources:
        match_type = str(source.get("graph_match_type") or "")
        graph_score = max(0.0, min(float(_optional_float(source.get("score")) or 0.0), 1.0))
        path_length = _optional_int(source.get("path_length")) or len(
            _string_list(source.get("graph_path"))
        )
        path_factor = 1.0 / max(path_length, 1)
        if match_type in _GRAPH_RELATED_MATCH_TYPES or source.get("related_concept"):
            related_scores.append(0.45 + (0.18 * graph_score) + (0.18 * path_factor))
        elif match_type in _GRAPH_DIRECT_MATCH_TYPES:
            direct_scores.append(0.12 + (0.12 * graph_score))
        relationships = _relationship_types(source.get("relationships"))
        matched, generic = _relation_match(query=query, query_analysis=query_analysis, relationships=relationships)
        if matched or generic:
            relation_match_count += 1
            has_relation_type_match = True
    related_path_score = round(min(0.82, max(related_scores, default=0.0)), 6)
    direct_graph_score = round(min(0.26, max(direct_scores, default=0.0)), 6)
    relation_type_score = round(
        min(0.48, (0.42 if relation_match_count else 0.0) + (0.06 * max(relation_match_count - 1, 0))),
        6,
    )
    return {
        "related_path_score": related_path_score,
        "direct_graph_score": direct_graph_score,
        "relation_type_score": relation_type_score,
        "has_related_graph_path": bool(related_scores),
        "has_relation_type_match": has_relation_type_match,
        "related_path_count": len(related_scores),
        "relation_type_match_count": relation_match_count,
    }


def _concept_features(
    candidate: dict[str, Any],
    *,
    query: str,
    query_analysis: dict[str, Any],
) -> dict[str, Any]:
    query_concepts = _query_concept_terms(query=query, query_analysis=query_analysis)
    concept_texts = _candidate_concept_texts(candidate)
    match_count = 0
    partial_score = 0.0
    for query_concept in query_concepts:
        query_terms = _coverage_terms(query_concept)
        if not query_terms:
            continue
        best_ratio = 0.0
        normalized_query = _normalize_text(query_concept)
        for concept_text in concept_texts:
            normalized_concept = _normalize_text(concept_text)
            if not normalized_concept:
                continue
            if normalized_query and (
                normalized_query in normalized_concept or normalized_concept in normalized_query
            ):
                best_ratio = max(best_ratio, 1.0)
                continue
            concept_terms = _coverage_terms(normalized_concept)
            if concept_terms:
                best_ratio = max(best_ratio, len(query_terms & concept_terms) / len(query_terms))
        if best_ratio > 0:
            match_count += 1
            partial_score += min(1.0, best_ratio)
    direct_ratio = round(partial_score / len(query_concepts), 6) if query_concepts else 0.0
    return {
        "query_concept_count": len(query_concepts),
        "direct_match_count": match_count,
        "direct_ratio": direct_ratio,
    }


def _text_features(candidate: dict[str, Any], *, query: str) -> dict[str, Any]:
    evidence_overlap = _query_text_overlap(query, _text_for_coverage(candidate.get("evidence_text")))
    semantic_overlap = _query_text_overlap(query, _text_for_coverage(candidate.get("semantic_text")))
    transcript_overlap = _query_text_overlap(
        query,
        _text_for_coverage(candidate.get("transcript_window_text")),
    )
    combined_ratio = max(
        float(evidence_overlap["ratio"]),
        float(semantic_overlap["ratio"]),
        float(transcript_overlap["ratio"]),
    )
    return {
        "query_term_count": evidence_overlap["query_term_count"],
        "evidence_match_count": evidence_overlap["match_count"],
        "semantic_match_count": semantic_overlap["match_count"],
        "transcript_match_count": transcript_overlap["match_count"],
        "evidence_ratio": evidence_overlap["ratio"],
        "semantic_ratio": semantic_overlap["ratio"],
        "transcript_ratio": transcript_overlap["ratio"],
        "combined_ratio": combined_ratio,
    }


def _quality_features(candidate: dict[str, Any]) -> dict[str, Any]:
    quality = _mapping(candidate.get("source_quality"))
    candidate_visual_support = _mapping(quality.get("candidate_visual_support"))
    verified_object_alignment = _mapping(quality.get("verified_object_alignment"))
    visual_state_count = _quality_count(
        quality,
        "visual_state_count",
        fallback=len(_string_list(candidate.get("visual_state_ids"))),
    )
    visual_entity_count = _quality_count(
        quality,
        "visual_entity_count",
        fallback=len(_string_list(candidate.get("visual_entity_ids"))),
    )
    visual_description_count = _quality_count(quality, "visual_description_count", fallback=0)
    verified_count = _verified_link_count(candidate, quality)
    candidate_count = _candidate_link_count(candidate, quality)
    timestamp_fallback_count = _timestamp_fallback_count(candidate, quality)
    has_timestamp_fallback = (
        bool(quality.get("has_timestamp_fallback_link")) or timestamp_fallback_count > 0
    )
    timestamp_counted_as_verified = bool(
        verified_object_alignment.get("timestamp_fallback_counted_as_verified")
    )
    timestamp_only_fallback = bool(
        has_timestamp_fallback and verified_count == 0 or timestamp_counted_as_verified
    )
    has_vlm_visual_entity = bool(quality.get("has_vlm_entity"))
    uses_ocr_only = bool(quality.get("uses_ocr_only")) and not has_vlm_visual_entity
    has_visual_description = (
        bool(quality.get("has_visual_description")) or visual_description_count > 0
    )
    has_visual_object_support = bool(
        candidate_visual_support.get("has_candidate_visual_support")
        or visual_state_count > 0
        or visual_entity_count > 0
        or candidate_count > 0
    )
    has_verified_alignment = bool(
        not timestamp_only_fallback
        and (
            verified_count > 0
            or bool(quality.get("has_verified_link"))
            or bool(verified_object_alignment.get("has_verified_object_alignment"))
        )
    )
    return {
        "has_visual_description": has_visual_description,
        "has_visual_object_support": has_visual_object_support,
        "has_vlm_visual_entity": has_vlm_visual_entity,
        "has_verified_alignment": has_verified_alignment,
        "uses_ocr_only": uses_ocr_only,
        "has_timestamp_fallback": has_timestamp_fallback,
        "timestamp_only_fallback": timestamp_only_fallback,
        "verified_link_count": verified_count,
        "candidate_link_count": candidate_count,
        "timestamp_fallback_count": timestamp_fallback_count,
        "visual_description_score": round(0.24 if has_visual_description else 0.0, 6),
        "visual_object_support_score": round(
            0.18 + 0.02 * min(visual_state_count + visual_entity_count, 4)
            if has_visual_object_support
            else 0.0,
            6,
        ),
        "vlm_visual_entity_score": round(0.26 if has_vlm_visual_entity else 0.0, 6),
        "verified_alignment_score": round(0.46 if has_verified_alignment else 0.0, 6),
        "candidate_link_score": round(0.1 * min(candidate_count, 3), 6),
        "ocr_only_penalty": -0.38 if uses_ocr_only else 0.0,
        "timestamp_fallback_penalty": -0.32 if has_timestamp_fallback else 0.0,
    }


def _relation_match(
    *,
    query: str,
    query_analysis: dict[str, Any],
    relationships: set[str],
) -> tuple[bool, bool]:
    relation_types = {
        _normalize_relationship(value)
        for value in relationships
        if value and value not in _NON_RELATIONSHIP_TYPES
    }
    if not relation_types:
        return False, False
    query_terms = _coverage_terms(query)
    exact_terms: set[str] = set()
    for term in query_terms:
        exact_terms.update(
            _normalize_relationship(relationship)
            for relationship in _RELATION_QUERY_TERMS.get(term, set())
        )
    if relation_types & exact_terms:
        return True, False
    query_type = _query_type(query_analysis)
    intent_candidates = {
        str(value)
        for value in _list_value(query_analysis.get("intent_candidates"))
        if str(value)
    }
    generic_relation = "relation" in intent_candidates or query_type in {
        "relation",
        "visual_relation",
    }
    return False, generic_relation


def _candidate_concept_texts(candidate: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(_string_list(candidate.get("concept_ids")))
    values.extend(_string_list(candidate.get("concept_labels")))
    values.extend(_string_list(candidate.get("concept_aliases")))
    for concept in _list_of_dicts(candidate.get("concepts")):
        values.extend(_concept_payload_texts(concept))
    for relation in _list_of_dicts(candidate.get("concept_relations")):
        values.extend(_concept_payload_texts(relation))
    for source in _candidate_sources(candidate):
        values.extend(_concept_payload_texts(_mapping(source.get("matched_concept"))))
        values.extend(_concept_payload_texts(_mapping(source.get("related_concept"))))
    return _unique_strings(values)


def _concept_payload_texts(payload: dict[str, Any]) -> list[str]:
    values = [
        str(payload.get("concept_id") or ""),
        str(payload.get("label") or ""),
        str(payload.get("relation_type") or ""),
        str(payload.get("source_concept_id") or ""),
        str(payload.get("target_concept_id") or ""),
    ]
    values.extend(_string_list(payload.get("aliases")))
    return [value for value in values if value]


def _query_concept_terms(*, query: str, query_analysis: dict[str, Any]) -> list[str]:
    concepts = []
    for item in _list_value(query_analysis.get("concept_candidates")):
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                concepts.append(text)
    compact_tokens = [token for token in _coverage_terms(query) if token not in _STOPWORDS]
    if not concepts and compact_tokens:
        concepts.append(" ".join(sorted(compact_tokens)))
    return _unique_strings(concepts)[:12]


def _query_analysis_mapping(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, dict):
            return payload
    return {}


def _query_type(query_analysis: dict[str, Any]) -> str:
    return str(query_analysis.get("query_type") or "unknown")


def _candidate_score(candidate: dict[str, Any]) -> float | None:
    for key in ("score", "_rankingScore"):
        score = _optional_float(candidate.get(key))
        if score is not None:
            return score
    source_scores = [
        _optional_float(source.get("score"))
        for source in _candidate_sources(candidate)
        if _optional_float(source.get("score")) is not None
    ]
    if source_scores:
        return max(score for score in source_scores if score is not None)
    return None


def _candidate_sources(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    value = candidate.get("candidate_sources")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _candidate_source_ids(candidate: dict[str, Any]) -> list[str]:
    values = [
        str(candidate.get("candidate_key") or ""),
        str(candidate.get("evidence_unit_id") or ""),
        str(candidate.get("target_segment_id") or ""),
    ]
    return _unique_strings([value for value in values if value])


def _query_text_overlap(query: str, candidate_text: str) -> dict[str, Any]:
    query_terms = _coverage_terms(query)
    candidate_terms = _coverage_terms(candidate_text)
    match_count = len(query_terms & candidate_terms)
    ratio = round(match_count / len(query_terms), 6) if query_terms else 0.0
    return {
        "query_term_count": len(query_terms),
        "match_count": match_count,
        "ratio": ratio,
    }


def _relationship_types(value: Any) -> set[str]:
    return {_normalize_relationship(item) for item in _string_list(value) if item}


def _normalize_relationship(value: str) -> str:
    return str(value).strip().upper()


def _coverage_terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if len(term) >= 2
    }


def _normalize_text(value: str) -> str:
    return " ".join(sorted(_coverage_terms(value)))


def _ratio_bucket(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio < 0.25:
        return "low"
    if ratio < 0.5:
        return "medium"
    if ratio < 0.75:
        return "high"
    return "very_high"


def _text_for_coverage(value: Any) -> str:
    if value in (None, ""):
        return ""
    return " ".join(str(value).split())


def _quality_count(quality: dict[str, Any], key: str, *, fallback: int = 0) -> int:
    value = quality.get(key)
    if value is None:
        return fallback
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return fallback


def _verified_link_count(candidate: dict[str, Any], quality: dict[str, Any]) -> int:
    count = _quality_count(
        quality,
        "verified_link_count",
        fallback=len(_string_list(candidate.get("verified_entity_link_ids"))),
    )
    statuses = _mapping(candidate.get("candidate_entity_link_statuses"))
    status_count = sum(1 for status in statuses.values() if str(status) == "verified")
    return max(count, status_count)


def _candidate_link_count(candidate: dict[str, Any], quality: dict[str, Any]) -> int:
    count = _quality_count(quality, "candidate_link_count", fallback=0)
    statuses = _mapping(candidate.get("candidate_entity_link_statuses"))
    candidate_status_count = sum(1 for status in statuses.values() if str(status) == "candidate")
    fallback_count = _timestamp_fallback_count(candidate, quality)
    raw_candidate_count = max(
        0,
        len(_string_list(candidate.get("candidate_entity_link_ids")))
        - fallback_count
        - _verified_link_count(candidate, quality),
    )
    return max(count, candidate_status_count, raw_candidate_count)


def _timestamp_fallback_count(candidate: dict[str, Any], quality: dict[str, Any]) -> int:
    count = _quality_count(quality, "timestamp_fallback_link_count", fallback=0)
    statuses = _mapping(candidate.get("candidate_entity_link_statuses"))
    return max(
        count,
        sum(1 for status in statuses.values() if str(status) == "timestamp_fallback"),
    )


def _candidate_identity(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("evidence_unit_id")
        or candidate.get("target_segment_id")
        or candidate.get("candidate_key")
        or candidate.get("rank")
        or ""
    )


def _candidate_ref(candidate: dict[str, Any]) -> str | None:
    identity = _candidate_identity(candidate)
    if not identity:
        return None
    return f"evu:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
