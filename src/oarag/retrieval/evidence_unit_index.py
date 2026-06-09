from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from oarag.integrations.meili import MeiliClient
from oarag.retrieval.project_index import (
    evidence_unit_artifact_path,
    iter_jsonl_documents,
)


EVIDENCE_UNIT_RESULT_FIELDS = [
    "evidence_unit_id",
    "project_id",
    "video_id",
    "target_segment_id",
    "source_segment_ids",
    "start_time",
    "end_time",
    "visual_state_ids",
    "visual_entity_ids",
    "concept_ids",
    "concept_labels",
    "concept_aliases",
    "concept_relation_text",
    "concept_search_text",
    "concepts",
    "concept_relations",
    "transcript_keywords",
    "visual_state_text",
    "visual_entity_text",
    "verified_entity_link_ids",
    "candidate_entity_link_ids",
    "candidate_entity_link_statuses",
    "candidate_link_signal_summary",
    "verified_link_signal_summary",
    "alignment_status",
    "source_quality",
]
MODALITY_AWARE_RERANK = "modality_aware"
EVIDENCE_UNIT_QUERY_FUSION_STRATEGY = "evidence_unit_query_fusion"
EVIDENCE_UNIT_FUSION_RRF = "reciprocal_rank_fusion"
EVIDENCE_UNIT_FUSION_ROUND_ROBIN = "round_robin"
DEFAULT_EVIDENCE_UNIT_FUSION = EVIDENCE_UNIT_FUSION_RRF
DEFAULT_EVIDENCE_UNIT_CANDIDATE_DEPTH = 50
DEFAULT_EVIDENCE_UNIT_RRF_RANK_CONSTANT = 60
QUERY_TYPE_SPEECH_HEAVY = "speech-heavy"
QUERY_TYPE_VISUAL_HEAVY = "visual-heavy"
QUERY_TYPE_MIXED = "mixed"
QUERY_TYPE_UNKNOWN = "unknown"
MODALITY_QUERY_TYPES = {
    QUERY_TYPE_SPEECH_HEAVY,
    QUERY_TYPE_VISUAL_HEAVY,
    QUERY_TYPE_MIXED,
    QUERY_TYPE_UNKNOWN,
}
_VISUAL_QUERY_TERMS = {
    "arrow",
    "chart",
    "color",
    "diagram",
    "display",
    "displayed",
    "draw",
    "drawn",
    "equation",
    "figure",
    "graph",
    "image",
    "label",
    "labeled",
    "matrix",
    "picture",
    "plot",
    "screen",
    "shape",
    "shown",
    "slide",
    "table",
    "visual",
}
_SPEECH_QUERY_TERMS = {
    "according",
    "define",
    "definition",
    "discuss",
    "explain",
    "explained",
    "lecture",
    "lecturer",
    "mention",
    "mentioned",
    "said",
    "say",
    "says",
    "spoken",
    "statement",
    "transcript",
}
_QUERY_PLANNING_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "also",
    "and",
    "are",
    "can",
    "did",
    "does",
    "for",
    "from",
    "how",
    "into",
    "near",
    "not",
    "now",
    "the",
    "this",
    "that",
    "these",
    "those",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def query_project_evidence_units(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    limit: int = 5,
    evidence_units: Path | None = None,
    evidence_unit_rerank: str | None = None,
    modality_aware_rerank: bool = False,
    candidate_depth: int | None = None,
    raw_candidate_depth: int | None = None,
    broad_candidate_depth: int | None = None,
    concept_candidate_depth: int | None = None,
    candidate_fusion: str = DEFAULT_EVIDENCE_UNIT_FUSION,
) -> dict[str, Any]:
    result_limit = _positive_int(limit, field_name="limit")
    fusion_method = _evidence_unit_fusion_method(candidate_fusion)
    resolved_project_dir = project_dir.expanduser().resolve()
    project_id = _project_id_from_evidence_units(
        project_dir=resolved_project_dir,
        evidence_units=evidence_units,
    )
    query_plan = _evidence_unit_query_plan(
        query=query,
        result_limit=result_limit,
        candidate_depth=candidate_depth,
        raw_candidate_depth=raw_candidate_depth,
        broad_candidate_depth=broad_candidate_depth,
        concept_candidate_depth=concept_candidate_depth,
        fusion_method=fusion_method,
    )
    filter_value = f'project_id = "{_escape_meili_filter_string(project_id)}"'
    search_results = [
        _evidence_unit_variant_search(
            client=client,
            index_uid=index_uid,
            project_id=project_id,
            filter_value=filter_value,
            variant=variant,
        )
        for variant in query_plan["variants"]
    ]
    fused_records = _fuse_evidence_unit_records(search_results, fusion_method=fusion_method)
    candidates = [
        _candidate_from_fused_record(rank=rank, record=record, fusion_method=fusion_method)
        for rank, record in enumerate(fused_records[:result_limit], start=1)
    ]
    rerank_mode = _evidence_unit_rerank_mode(
        evidence_unit_rerank=evidence_unit_rerank,
        modality_aware_rerank=modality_aware_rerank,
    )
    base_candidates = [dict(candidate) for candidate in candidates]
    rerank_context = _evidence_unit_rerank_context(
        enabled=bool(rerank_mode),
        mode=rerank_mode,
        query=query,
        base_candidates=base_candidates,
        reranked_candidates=candidates,
    )
    if rerank_mode == MODALITY_AWARE_RERANK:
        candidates = modality_aware_rerank_candidates(candidates, query=query)
        rerank_context = _evidence_unit_rerank_context(
            enabled=True,
            mode=rerank_mode,
            query=query,
            base_candidates=base_candidates,
            reranked_candidates=candidates,
        )
    return {
        "query": query,
        "index": index_uid,
        "project_id": project_id,
        "processing_time_ms": _combined_processing_time_ms(search_results),
        "limit": result_limit,
        "candidates": candidates,
        "base_candidates": base_candidates if rerank_mode else [],
        "retrieval_context": {
            "query_planning": _evidence_unit_query_planning_context(
                query_plan=query_plan,
                search_results=search_results,
                fused_records=fused_records,
                returned_candidate_count=len(candidates),
            ),
            "evidence_unit_rerank": rerank_context,
        },
    }


def _evidence_unit_query_plan(
    *,
    query: str,
    result_limit: int,
    candidate_depth: int | None,
    raw_candidate_depth: int | None,
    broad_candidate_depth: int | None,
    concept_candidate_depth: int | None,
    fusion_method: str,
) -> dict[str, Any]:
    default_depth = _candidate_depth(candidate_depth, result_limit=result_limit)
    depths = {
        "raw": _candidate_depth(
            raw_candidate_depth,
            result_limit=result_limit,
            default=default_depth,
        ),
        "broad": _candidate_depth(
            broad_candidate_depth,
            result_limit=result_limit,
            default=default_depth,
        ),
        "concept": _candidate_depth(
            concept_candidate_depth,
            result_limit=result_limit,
            default=default_depth,
        ),
    }
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, variant_query in (
        ("raw", _raw_query_variant(query)),
        ("broad", _broad_query_variant(query)),
        ("concept", _concept_query_variant(query)),
    ):
        normalized = _normalize_query_variant(variant_query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append(
            {
                "role": role,
                "query": variant_query,
                "query_ref": _query_ref(role=role, query=variant_query),
                "depth": depths[role],
            }
        )
    return {
        "strategy": EVIDENCE_UNIT_QUERY_FUSION_STRATEGY,
        "fusion_method": fusion_method,
        "candidate_limit": result_limit,
        "depth": depths,
        "variants": variants,
    }


def _evidence_unit_variant_search(
    *,
    client: MeiliClient,
    index_uid: str,
    project_id: str,
    filter_value: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    depth = _positive_int(variant.get("depth"), field_name=f"{variant.get('role')}_depth")
    response = client.search(
        index_uid,
        str(variant.get("query") or ""),
        limit=depth,
        filter=filter_value,
    )
    hits = [hit for hit in response.get("hits", []) if isinstance(hit, dict)]
    return {
        "index": index_uid,
        "project_id": project_id,
        "role": str(variant.get("role") or ""),
        "query_ref": str(variant.get("query_ref") or ""),
        "depth": depth,
        "hits": hits,
        "processing_time_ms": response.get("processingTimeMs"),
    }


def _fuse_evidence_unit_records(
    search_results: list[dict[str, Any]],
    *,
    fusion_method: str,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    key_aliases: dict[str, str] = {}
    first_seen = 0
    for query_index, result in enumerate(search_results):
        for rank, hit in enumerate(result.get("hits", []), start=1):
            if not isinstance(hit, dict):
                continue
            candidate_keys = _candidate_dedupe_keys(hit)
            key = next(
                (
                    key_aliases[candidate_key]
                    for candidate_key in candidate_keys
                    if candidate_key in key_aliases
                ),
                candidate_keys[0],
            )
            score = _optional_float(hit.get("_rankingScore"))
            match = {
                "query_role": result.get("role"),
                "query_index": query_index,
                "query_ref": result.get("query_ref"),
                "rank": rank,
                "depth": result.get("depth"),
                "score": score,
            }
            if key not in records:
                records[key] = {
                    "key": key,
                    "hit": hit,
                    "matches": [match],
                    "first_seen": first_seen,
                    "best_rank": rank,
                    "best_query_index": query_index,
                    "best_score": score,
                    "dedupe_aliases": candidate_keys,
                }
                for candidate_key in candidate_keys:
                    key_aliases[candidate_key] = key
                first_seen += 1
                continue
            record = records[key]
            record["matches"].append(match)
            aliases = record.setdefault("dedupe_aliases", [])
            for candidate_key in candidate_keys:
                key_aliases[candidate_key] = key
                if candidate_key not in aliases:
                    aliases.append(candidate_key)
            if _is_better_evidence_unit_hit(
                score=score,
                rank=rank,
                query_index=query_index,
                record=record,
            ):
                record["hit"] = hit
                record["best_rank"] = rank
                record["best_query_index"] = query_index
                record["best_score"] = score

    for record in records.values():
        record["fusion_score"] = _evidence_unit_rrf_score(record.get("matches", []))

    if fusion_method == EVIDENCE_UNIT_FUSION_ROUND_ROBIN:
        ordered_keys = _round_robin_evidence_unit_keys(search_results)
        order = {key: index for index, key in enumerate(ordered_keys)}
        return sorted(
            records.values(),
            key=lambda record: (
                order.get(str(record.get("key")), 10**9),
                int(record.get("first_seen") or 10**9),
                str(record.get("key") or ""),
            ),
        )
    return sorted(records.values(), key=_evidence_unit_rrf_sort_key)


def _candidate_from_fused_record(
    *,
    rank: int,
    record: dict[str, Any],
    fusion_method: str,
) -> dict[str, Any]:
    candidate = _evidence_unit_candidate(rank=rank, hit=_mapping(record.get("hit")))
    matches = _list_of_dicts(record.get("matches"))
    candidate["candidate_sources"] = [
        _evidence_unit_source_summary(match) for match in sorted(matches, key=_source_sort_key)
    ]
    candidate["candidate_source_types"] = [
        str(match.get("query_role") or "")
        for match in sorted(matches, key=_source_sort_key)
        if match.get("query_role")
    ]
    candidate["candidate_fusion"] = {
        "method": fusion_method,
        "score": record.get("fusion_score"),
        "match_count": len(matches),
        "best_rank": record.get("best_rank"),
        "best_query_role": (
            candidate["candidate_sources"][0].get("query_role")
            if candidate["candidate_sources"]
            else None
        ),
        "dedupe_key_ref": (
            "candidate:"
            f"{hashlib.sha256(str(record.get('key') or '').encode('utf-8')).hexdigest()[:12]}"
        ),
    }
    return candidate


def _evidence_unit_query_planning_context(
    *,
    query_plan: dict[str, Any],
    search_results: list[dict[str, Any]],
    fused_records: list[dict[str, Any]],
    returned_candidate_count: int,
) -> dict[str, Any]:
    raw_hit_count = sum(len(_list_of_dicts(result.get("hits"))) for result in search_results)
    return {
        "strategy": query_plan.get("strategy"),
        "query_count": len(search_results),
        "query_roles": [result.get("role") for result in search_results],
        "depth": query_plan.get("depth"),
        "candidate_limit": query_plan.get("candidate_limit"),
        "fusion_method": query_plan.get("fusion_method"),
        "rrf_rank_constant": (
            DEFAULT_EVIDENCE_UNIT_RRF_RANK_CONSTANT
            if query_plan.get("fusion_method") == EVIDENCE_UNIT_FUSION_RRF
            else None
        ),
        "raw_hit_count_before_dedupe": raw_hit_count,
        "unique_candidate_count_before_limit": len(fused_records),
        "returned_candidate_count": returned_candidate_count,
        "dedupe_keys": [
            "evidence_unit_id",
            "target_segment_id",
            "stable_id",
            "hash",
        ],
        "variants": [
            {
                "role": result.get("role"),
                "query_ref": result.get("query_ref"),
                "depth": result.get("depth"),
                "hit_count": len(_list_of_dicts(result.get("hits"))),
            }
            for result in search_results
        ],
        "public_note": (
            "Evidence-unit query planning exposes variant roles, depth, fusion, counts, "
            "and hashed query refs only; raw query text, transcript text, and local paths "
            "are omitted."
        ),
    }


def classify_evidence_unit_query_modality(query: str) -> str:
    terms = _coverage_terms(query)
    if not terms:
        return QUERY_TYPE_UNKNOWN
    visual_hits = len(terms & _VISUAL_QUERY_TERMS)
    speech_hits = len(terms & _SPEECH_QUERY_TERMS)
    if visual_hits and speech_hits:
        return QUERY_TYPE_MIXED
    if visual_hits:
        return QUERY_TYPE_VISUAL_HEAVY
    if speech_hits:
        return QUERY_TYPE_SPEECH_HEAVY
    return QUERY_TYPE_UNKNOWN


def modality_aware_rerank_candidates(
    candidates: list[dict[str, Any]],
    *,
    query: str,
) -> list[dict[str, Any]]:
    query_type = classify_evidence_unit_query_modality(query)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for fallback_rank, candidate in enumerate(candidates, start=1):
        original_rank = _optional_int(candidate.get("rank")) or fallback_rank
        breakdown = modality_aware_rerank_breakdown(
            candidate,
            query=query,
            query_type=query_type,
            original_rank=original_rank,
        )
        updated = dict(candidate)
        updated["modality_aware_rerank"] = {
            "strategy": MODALITY_AWARE_RERANK,
            "query_type": query_type,
            "score": breakdown["score"],
            "original_rank": original_rank,
            "components": breakdown["components"],
            "flags": breakdown["flags"],
            "query_term_overlap": breakdown["query_term_overlap"],
            "public_note": (
                "Modality-aware rerank uses sanitized query type, candidate quality flags, "
                "and query-term overlap counts only; raw query text is not stored here."
            ),
        }
        scored.append((float(breakdown["score"]), original_rank, updated))

    reranked: list[dict[str, Any]] = []
    for new_rank, (_, _, candidate) in enumerate(
        sorted(scored, key=lambda item: (-item[0], item[1])),
        start=1,
    ):
        updated = dict(candidate)
        rerank = dict(_mapping(updated.get("modality_aware_rerank")))
        rerank["reranked_rank"] = new_rank
        updated["modality_aware_rerank"] = rerank
        updated["rank"] = new_rank
        reranked.append(updated)
    return reranked


def modality_aware_rerank_breakdown(
    candidate: dict[str, Any],
    *,
    query: str,
    query_type: str | None = None,
    original_rank: int | None = None,
) -> dict[str, Any]:
    resolved_query_type = (
        query_type
        if query_type in MODALITY_QUERY_TYPES
        else classify_evidence_unit_query_modality(query)
    )
    rank = original_rank or _optional_int(candidate.get("rank")) or 1
    quality = _mapping(candidate.get("source_quality"))
    score = _optional_float(candidate.get("score"))
    if score is None:
        score = _optional_float(candidate.get("_rankingScore"))
    evidence_overlap = _query_text_overlap(query, _text_for_coverage(candidate.get("evidence_text")))
    semantic_overlap = _query_text_overlap(query, _text_for_coverage(candidate.get("semantic_text")))
    transcript_overlap = _query_text_overlap(
        query,
        _text_for_coverage(candidate.get("transcript_window_text")),
    )
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
    verified_link_count = _verified_link_count(candidate, quality)
    candidate_link_count = _candidate_link_count(candidate, quality)
    timestamp_fallback_count = _timestamp_fallback_count(candidate, quality)
    has_vlm_entity = bool(quality.get("has_vlm_entity"))
    uses_ocr_only = bool(quality.get("uses_ocr_only")) and not has_vlm_entity
    has_verified_link = bool(quality.get("has_verified_link")) or verified_link_count > 0
    has_timestamp_fallback = (
        bool(quality.get("has_timestamp_fallback_link")) or timestamp_fallback_count > 0
    )
    transcript_only = str(candidate.get("alignment_status") or "").casefold() == "transcript_only"
    text_weight, visual_weight = _query_type_weights(resolved_query_type)
    combined_overlap = max(
        float(evidence_overlap["ratio"]),
        float(semantic_overlap["ratio"]),
        float(transcript_overlap["ratio"]),
    )
    components = {
        "rank_preservation": round(0.12 / max(rank, 1), 6),
        "meili_score": round(0.18 * max(0.0, min(float(score or 0.0), 1.0)), 6),
        "evidence_text_query_overlap": round(text_weight * float(evidence_overlap["ratio"]), 6),
        "semantic_text_query_overlap": round(
            (text_weight + 0.05) * float(semantic_overlap["ratio"]),
            6,
        ),
        "transcript_query_overlap": round(
            (text_weight * 0.55) * float(transcript_overlap["ratio"]),
            6,
        ),
        "visual_state": round(visual_weight * min(visual_state_count, 3) * 0.18, 6),
        "vlm_entity": round(visual_weight * (0.45 if has_vlm_entity else 0.0), 6),
        "verified_link": round(visual_weight * (0.7 if has_verified_link else 0.0), 6),
        "candidate_support": round(visual_weight * min(candidate_link_count, 4) * 0.08, 6),
        "visual_entity_count": round(visual_weight * min(visual_entity_count, 6) * 0.04, 6),
        "ocr_only_penalty": -0.45 if uses_ocr_only else 0.0,
        "timestamp_fallback_penalty": -0.28 if has_timestamp_fallback else 0.0,
        "transcript_only_penalty": (
            -0.2 if resolved_query_type == QUERY_TYPE_VISUAL_HEAVY and transcript_only else 0.0
        ),
    }
    score_total = round(sum(float(value) for value in components.values()), 6)
    return {
        "score": score_total,
        "components": components,
        "flags": {
            "query_type": resolved_query_type,
            "has_visual_state": visual_state_count > 0,
            "has_vlm_entity": has_vlm_entity,
            "has_verified_link": has_verified_link,
            "has_candidate_support": candidate_link_count > 0,
            "uses_ocr_only": uses_ocr_only,
            "has_timestamp_fallback_link": has_timestamp_fallback,
            "transcript_only": transcript_only,
            "combined_query_overlap_bucket": _ratio_bucket(combined_overlap),
        },
        "query_term_overlap": {
            "query_term_count": evidence_overlap["query_term_count"],
            "evidence_text_match_count": evidence_overlap["match_count"],
            "evidence_text_match_ratio": evidence_overlap["ratio"],
            "evidence_text_match_bucket": evidence_overlap["bucket"],
            "semantic_text_match_count": semantic_overlap["match_count"],
            "semantic_text_match_ratio": semantic_overlap["ratio"],
            "semantic_text_match_bucket": semantic_overlap["bucket"],
            "transcript_match_count": transcript_overlap["match_count"],
            "transcript_match_ratio": transcript_overlap["ratio"],
            "transcript_match_bucket": transcript_overlap["bucket"],
        },
    }


def _raw_query_variant(query: str) -> str:
    return " ".join(str(query or "").split())


def _broad_query_variant(query: str) -> str:
    terms = [
        term
        for term in _ordered_query_terms(query)
        if term not in _QUERY_PLANNING_STOPWORDS
    ]
    return " ".join(terms[:12]) or _raw_query_variant(query)


def _concept_query_variant(query: str) -> str:
    terms = [
        term
        for term in _ordered_query_terms(query)
        if term not in _QUERY_PLANNING_STOPWORDS
        and term not in _VISUAL_QUERY_TERMS
        and term not in _SPEECH_QUERY_TERMS
    ]
    if not terms:
        terms = [
            term
            for term in _ordered_query_terms(query)
            if term not in _QUERY_PLANNING_STOPWORDS
        ]
    phrase_terms: list[str] = []
    for index in range(max(0, len(terms) - 1)):
        phrase_terms.append(f"{terms[index]} {terms[index + 1]}")
        if len(phrase_terms) >= 4:
            break
    return " ".join([*phrase_terms, *terms[:8]]).strip() or _broad_query_variant(query)


def _ordered_query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in re.findall(r"[A-Za-z0-9]+", str(query or "").casefold()):
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _normalize_query_variant(query: str) -> str:
    return " ".join(str(query or "").casefold().split())


def _query_ref(*, role: str, query: str) -> str:
    digest = hashlib.sha256(f"{role}\0{query}".encode("utf-8")).hexdigest()[:12]
    return f"query:{digest}"


def _candidate_depth(
    value: int | None,
    *,
    result_limit: int,
    default: int | None = None,
) -> int:
    if value is None:
        resolved = default if default is not None else DEFAULT_EVIDENCE_UNIT_CANDIDATE_DEPTH
    else:
        resolved = value
    depth = _positive_int(resolved, field_name="candidate_depth")
    return max(result_limit, depth)


def _evidence_unit_fusion_method(value: str | None) -> str:
    normalized = str(value or DEFAULT_EVIDENCE_UNIT_FUSION).strip().casefold().replace("-", "_")
    if normalized in {"rrf", "reciprocal_rank_fusion", "reciprocal_rank"}:
        return EVIDENCE_UNIT_FUSION_RRF
    if normalized in {"round_robin", "roundrobin"}:
        return EVIDENCE_UNIT_FUSION_ROUND_ROBIN
    raise ValueError(f"Unsupported evidence unit candidate fusion: {value}")


def _candidate_dedupe_key(hit: dict[str, Any]) -> str:
    return _candidate_dedupe_keys(hit)[0]


def _candidate_dedupe_keys(hit: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in (
        "evidence_unit_id",
        "target_segment_id",
        "stable_id",
        "stable_hash",
        "content_hash",
        "hash",
        "id",
    ):
        value = hit.get(field)
        if value not in (None, ""):
            keys.append(f"{field}:{value}")
    if keys:
        return keys
    digest = hashlib.sha256(
        repr(sorted((str(key), str(value)) for key, value in hit.items())).encode("utf-8")
    ).hexdigest()[:16]
    return [f"document_hash:{digest}"]


def _is_better_evidence_unit_hit(
    *,
    score: float | None,
    rank: int,
    query_index: int,
    record: dict[str, Any],
) -> bool:
    best_score = _optional_float(record.get("best_score"))
    if score is not None and best_score is not None and score != best_score:
        return score > best_score
    if score is not None and best_score is None:
        return True
    if score is None and best_score is not None:
        return False
    best_query_index = _optional_int(record.get("best_query_index")) or 10**9
    if query_index != best_query_index:
        return query_index < best_query_index
    best_rank = _optional_int(record.get("best_rank")) or 10**9
    return rank < best_rank


def _evidence_unit_rrf_score(matches: list[dict[str, Any]]) -> float:
    score = 0.0
    for match in matches:
        rank = _optional_int(match.get("rank")) or 10**9
        score += 1.0 / (DEFAULT_EVIDENCE_UNIT_RRF_RANK_CONSTANT + rank)
    return round(score, 8)


def _evidence_unit_rrf_sort_key(record: dict[str, Any]) -> tuple[float, float, int, int, int, str]:
    best_score = _optional_float(record.get("best_score"))
    return (
        -float(record.get("fusion_score") or 0.0),
        -(best_score if best_score is not None else float("-inf")),
        int(record.get("best_query_index") or 0),
        int(record.get("best_rank") or 10**9),
        int(record.get("first_seen") or 10**9),
        str(record.get("key") or ""),
    )


def _round_robin_evidence_unit_keys(search_results: list[dict[str, Any]]) -> list[str]:
    per_variant = [
        [_candidate_dedupe_key(hit) for hit in _list_of_dicts(result.get("hits"))]
        for result in search_results
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    max_len = max((len(keys) for keys in per_variant), default=0)
    for rank_index in range(max_len):
        for keys in per_variant:
            if rank_index >= len(keys):
                continue
            key = keys[rank_index]
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def _source_sort_key(match: dict[str, Any]) -> tuple[int, int]:
    return (
        int(match.get("query_index") or 0),
        int(match.get("rank") or 10**9),
    )


def _evidence_unit_source_summary(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_role": match.get("query_role"),
        "query_ref": match.get("query_ref"),
        "rank": match.get("rank"),
        "depth": match.get("depth"),
        "score": _optional_float(match.get("score")),
    }


def _combined_processing_time_ms(search_results: list[dict[str, Any]]) -> int | float | None:
    values = [
        _optional_float(result.get("processing_time_ms"))
        for result in search_results
        if _optional_float(result.get("processing_time_ms")) is not None
    ]
    if not values:
        return None
    total = sum(values)
    return int(total) if total.is_integer() else round(total, 4)


def _project_id_from_evidence_units(
    *,
    project_dir: Path,
    evidence_units: Path | None,
) -> str:
    try:
        artifact_path = evidence_unit_artifact_path(project_dir, evidence_units=evidence_units)
    except FileNotFoundError:
        if evidence_units is not None:
            raise
        return project_dir.name
    for document in iter_jsonl_documents(artifact_path):
        project_id = str(document.get("project_id") or "").strip()
        if project_id:
            return project_id
    return project_dir.name


def _evidence_unit_candidate(*, rank: int, hit: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "evidence_unit_id": str(hit.get("evidence_unit_id") or ""),
        "project_id": str(hit.get("project_id") or ""),
        "video_id": str(hit.get("video_id") or ""),
        "target_segment_id": str(hit.get("target_segment_id") or ""),
        "source_segment_ids": _string_list(hit.get("source_segment_ids")),
        "start_time": _optional_float(hit.get("start_time")),
        "end_time": _optional_float(hit.get("end_time")),
        "visual_state_ids": _string_list(hit.get("visual_state_ids")),
        "visual_entity_ids": _string_list(hit.get("visual_entity_ids")),
        "concept_ids": _string_list(hit.get("concept_ids")),
        "concept_labels": _string_list(hit.get("concept_labels")),
        "concept_aliases": _string_list(hit.get("concept_aliases")),
        "concept_relation_text": str(hit.get("concept_relation_text") or ""),
        "concept_search_text": str(hit.get("concept_search_text") or ""),
        "concepts": _list_of_dicts(hit.get("concepts")),
        "concept_relations": _list_of_dicts(hit.get("concept_relations")),
        "transcript_keywords": _string_list(hit.get("transcript_keywords")),
        "visual_state_text": str(hit.get("visual_state_text") or ""),
        "visual_entity_text": str(hit.get("visual_entity_text") or ""),
        "verified_entity_link_ids": _string_list(hit.get("verified_entity_link_ids")),
        "candidate_entity_link_ids": _string_list(hit.get("candidate_entity_link_ids")),
        "candidate_entity_link_statuses": _mapping(hit.get("candidate_entity_link_statuses")),
        "candidate_link_signal_summary": str(hit.get("candidate_link_signal_summary") or ""),
        "verified_link_signal_summary": str(hit.get("verified_link_signal_summary") or ""),
        "alignment_status": str(hit.get("alignment_status") or ""),
        "source_quality": _mapping(hit.get("source_quality")),
    }
    candidate["rank"] = rank
    candidate["score"] = _optional_float(hit.get("_rankingScore"))
    candidate["evidence_text"] = hit.get("evidence_text")
    candidate["semantic_text"] = hit.get("semantic_text")
    candidate["transcript_window_text"] = hit.get("transcript_window_text")
    return candidate


def _evidence_unit_rerank_mode(
    *,
    evidence_unit_rerank: str | None,
    modality_aware_rerank: bool,
) -> str | None:
    if modality_aware_rerank:
        return MODALITY_AWARE_RERANK
    normalized = str(evidence_unit_rerank or "").strip().casefold().replace("-", "_")
    if not normalized or normalized in {"none", "disabled", "false"}:
        return None
    if normalized in {"modality_aware", "modality"}:
        return MODALITY_AWARE_RERANK
    raise ValueError(f"Unsupported evidence_unit_rerank: {evidence_unit_rerank}")


def _evidence_unit_rerank_context(
    *,
    enabled: bool,
    mode: str | None,
    query: str,
    base_candidates: list[dict[str, Any]],
    reranked_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "strategy": None}
    base_top = base_candidates[0] if base_candidates else {}
    reranked_top = reranked_candidates[0] if reranked_candidates else {}
    top_rerank = _mapping(reranked_top.get("modality_aware_rerank"))
    return {
        "enabled": True,
        "strategy": mode,
        "query_type": classify_evidence_unit_query_modality(query),
        "candidate_count": len(reranked_candidates),
        "top_changed": _candidate_identity(base_top) != _candidate_identity(reranked_top),
        "base_top_ref": _candidate_ref(base_top),
        "reranked_top_ref": _candidate_ref(reranked_top),
        "reranked_top_score": top_rerank.get("score"),
        "reranked_top_original_rank": top_rerank.get("original_rank"),
        "reranked_top_rank": top_rerank.get("reranked_rank"),
        "component_names": sorted(_mapping(top_rerank.get("components"))),
        "public_note": (
            "Evidence-unit rerank diagnostics expose hashed refs, query type, score, "
            "feature names, counts, and flags only; raw query text and evidence text are omitted."
        ),
    }


def _candidate_identity(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("evidence_unit_id")
        or candidate.get("target_segment_id")
        or candidate.get("rank")
        or ""
    )


def _candidate_ref(candidate: dict[str, Any]) -> str | None:
    identity = _candidate_identity(candidate)
    if not identity:
        return None
    return f"evu:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"


def _coverage_terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if len(term) >= 2
    }


def _text_for_coverage(value: Any) -> str:
    if value in (None, ""):
        return ""
    return " ".join(str(value).split())


def _query_text_overlap(query: str, candidate_text: str) -> dict[str, Any]:
    query_terms = _coverage_terms(query)
    candidate_terms = _coverage_terms(candidate_text)
    match_count = len(query_terms & candidate_terms)
    ratio = round(match_count / len(query_terms), 6) if query_terms else 0.0
    return {
        "query_term_count": len(query_terms),
        "match_count": match_count,
        "ratio": ratio,
        "bucket": _ratio_bucket(ratio),
    }


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


def _query_type_weights(query_type: str) -> tuple[float, float]:
    if query_type == QUERY_TYPE_VISUAL_HEAVY:
        return 0.25, 1.0
    if query_type == QUERY_TYPE_SPEECH_HEAVY:
        return 0.85, 0.28
    if query_type == QUERY_TYPE_MIXED:
        return 0.6, 0.72
    return 0.55, 0.45


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
    return max(count, sum(1 for status in statuses.values() if str(status) == "verified"))


def _candidate_link_count(candidate: dict[str, Any], quality: dict[str, Any]) -> int:
    count = _quality_count(
        quality,
        "candidate_link_count",
        fallback=0,
    )
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


def _positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _escape_meili_filter_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
