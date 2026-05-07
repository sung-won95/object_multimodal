from __future__ import annotations

import copy
import math
import re
from typing import Any


STRATEGY_NAME = "deterministic_evidence_v1"
DEFAULT_WEIGHTS: dict[str, float] = {
    "original": 0.4,
    "query_overlap": 0.22,
    "timestamp_proximity": 0.14,
    "segment_length": 0.08,
    "frame_backed": 0.08,
    "entity_link": 0.08,
}


def rerank_bundles(
    *,
    bundles: list[dict[str, Any]],
    query: str,
    timestamp_hint: str | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_weights = _normalize_weights(weights or DEFAULT_WEIGHTS)
    metadata = rerank_metadata(
        enabled=True,
        query=query,
        timestamp_hint=timestamp_hint,
        weights=resolved_weights,
    )
    if not bundles:
        return [], metadata

    timestamp_ranges = _timestamp_ranges_from_metadata(metadata)
    query_terms = _tokenize(query)
    search_scores = [_optional_float(_candidate(bundle).get("score")) for bundle in bundles]
    normalized_search_scores = _normalized_search_scores(search_scores)

    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for position, bundle in enumerate(bundles, start=1):
        original_rank = _candidate_rank(bundle, fallback=position)
        score, breakdown = _score_bundle(
            bundle=bundle,
            query_terms=query_terms,
            timestamp_ranges=timestamp_ranges,
            normalized_search_score=normalized_search_scores[position - 1],
            weights=resolved_weights,
            original_rank=original_rank,
        )
        updated = copy.deepcopy(bundle)
        updated["rerank"] = {
            "strategy": STRATEGY_NAME,
            "rank": None,
            "score": round(score, 4),
            "original_rank": original_rank,
            "breakdown": breakdown,
            "explanation": _human_explanation(breakdown),
        }
        segment_id = str(_candidate(updated).get("segment_id", ""))
        scored.append((score, original_rank, segment_id, updated))

    ordered = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    reranked: list[dict[str, Any]] = []
    for rank, (_, _, _, bundle) in enumerate(ordered, start=1):
        bundle["rank"] = rank
        bundle["rerank"]["rank"] = rank
        reranked.append(bundle)
    metadata["order"] = [
        {
            "rank": bundle["rerank"]["rank"],
            "segment_id": _candidate(bundle).get("segment_id"),
            "original_rank": bundle["rerank"]["original_rank"],
            "score": bundle["rerank"]["score"],
        }
        for bundle in reranked
    ]
    return reranked, metadata


def rerank_metadata(
    *,
    enabled: bool,
    query: str = "",
    timestamp_hint: str | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    timestamp_text = " ".join(part for part in (query, timestamp_hint or "") if part.strip())
    timestamp_ranges = parse_timestamp_hints(timestamp_text)
    return {
        "enabled": enabled,
        "strategy": STRATEGY_NAME if enabled else None,
        "weights": _round_mapping(_normalize_weights(weights or DEFAULT_WEIGHTS)) if enabled else None,
        "query_term_count": len(_tokenize(query)) if enabled else None,
        "timestamp_hint_provided": bool(timestamp_hint and timestamp_hint.strip()),
        "timestamp_ranges": [
            {"start": round(start, 4), "end": round(end, 4)} for start, end in timestamp_ranges
        ],
    }


def parse_timestamp_hints(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    range_pattern = re.compile(
        r"(?<![\w:])(\d+(?:\.\d+)?)\s*(?:-|to|~|부터)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:s|sec(?:onds?)?|초)?\b",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(value):
        start = float(match.group(1))
        end = float(match.group(2))
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    if ranges:
        return ranges

    point_pattern = re.compile(
        r"(?<![\w:])(\d+(?:\.\d+)?)\s*(?:s|sec(?:onds?)?|초)\b",
        re.IGNORECASE,
    )
    for match in point_pattern.finditer(value):
        point = float(match.group(1))
        ranges.append((point, point))
    return ranges


def _score_bundle(
    *,
    bundle: dict[str, Any],
    query_terms: set[str],
    timestamp_ranges: list[tuple[float, float]],
    normalized_search_score: float | None,
    weights: dict[str, float],
    original_rank: int,
) -> tuple[float, dict[str, Any]]:
    signals = {
        "original": _original_signal(
            original_rank=original_rank,
            normalized_search_score=normalized_search_score,
            raw_search_score=_optional_float(_candidate(bundle).get("score")),
        ),
        "query_overlap": _query_overlap_signal(bundle=bundle, query_terms=query_terms),
        "timestamp_proximity": _timestamp_signal(
            bundle=bundle,
            timestamp_ranges=timestamp_ranges,
        ),
        "segment_length": _segment_length_signal(bundle),
        "frame_backed": _frame_backed_signal(bundle),
        "entity_link": _entity_link_signal(bundle),
    }

    weighted: dict[str, Any] = {}
    total = 0.0
    for name, signal in signals.items():
        weight = weights[name]
        contribution = signal["score"] * weight
        total += contribution
        weighted[name] = {
            **signal,
            "weight": round(weight, 4),
            "contribution": round(contribution, 4),
        }
    return total, {"total": round(total, 4), "signals": weighted}


def _original_signal(
    *,
    original_rank: int,
    normalized_search_score: float | None,
    raw_search_score: float | None,
) -> dict[str, Any]:
    rank_component = 1.0 / original_rank if original_rank > 0 else 0.0
    if normalized_search_score is None:
        score = rank_component
    else:
        score = (0.65 * rank_component) + (0.35 * normalized_search_score)
    return {
        "score": round(score, 4),
        "rank_component": round(rank_component, 4),
        "search_score_component": round(normalized_search_score, 4)
        if normalized_search_score is not None
        else None,
        "raw_search_score": raw_search_score,
        "detail": f"original rank #{original_rank}"
        + (f", normalized search score {normalized_search_score:.3f}" if normalized_search_score is not None else ""),
    }


def _query_overlap_signal(*, bundle: dict[str, Any], query_terms: set[str]) -> dict[str, Any]:
    if not query_terms:
        return {"score": 0.0, "matched_terms": [], "detail": "no query terms"}
    evidence_terms = _tokenize(" ".join(_bundle_text_parts(bundle)))
    matched_terms = sorted(query_terms & evidence_terms)
    score = len(matched_terms) / len(query_terms)
    return {
        "score": round(score, 4),
        "matched_terms": matched_terms,
        "matched_term_count": len(matched_terms),
        "query_term_count": len(query_terms),
        "detail": f"{len(matched_terms)}/{len(query_terms)} query terms matched",
    }


def _timestamp_signal(
    *,
    bundle: dict[str, Any],
    timestamp_ranges: list[tuple[float, float]],
) -> dict[str, Any]:
    if not timestamp_ranges:
        return {"score": 0.0, "distance_seconds": None, "detail": "no timestamp hint"}

    center = _candidate_center(_candidate(bundle))
    if center is None:
        return {"score": 0.0, "distance_seconds": None, "detail": "candidate has no timestamp"}

    distance = min(_distance_to_range(center, expected_range) for expected_range in timestamp_ranges)
    score = 1.0 / (1.0 + (distance / 10.0))
    return {
        "score": round(score, 4),
        "candidate_center": round(center, 4),
        "distance_seconds": round(distance, 4),
        "detail": "inside timestamp hint" if math.isclose(distance, 0.0) else f"{distance:.2f}s from timestamp hint",
    }


def _segment_length_signal(bundle: dict[str, Any]) -> dict[str, Any]:
    candidate = _candidate(bundle)
    start = _optional_float(candidate.get("start_time"))
    end = _optional_float(candidate.get("end_time"))
    duration = end - start if start is not None and end is not None else None
    text_terms = _tokenize(" ".join(_target_text_parts(bundle)) or str(candidate.get("transcript_excerpt", "")))
    duration_score = _range_quality(duration, ideal_min=2.0, ideal_max=45.0, hard_max=120.0)
    text_score = _range_quality(float(len(text_terms)), ideal_min=3.0, ideal_max=80.0, hard_max=180.0)
    score = (duration_score + text_score) / 2.0
    return {
        "score": round(score, 4),
        "penalty": round(1.0 - score, 4),
        "duration_seconds": round(duration, 4) if duration is not None else None,
        "term_count": len(text_terms),
        "detail": f"length penalty {1.0 - score:.3f}",
    }


def _frame_backed_signal(bundle: dict[str, Any]) -> dict[str, Any]:
    evidence_window = _evidence_window(bundle)
    target_segment = evidence_window.get("target_segment")
    target_frame_refs = (
        target_segment.get("frame_refs", []) if isinstance(target_segment, dict) else []
    )
    frame_refs = evidence_window.get("frame_refs") or []
    frame_count = len(frame_refs) if isinstance(frame_refs, list) else 0
    target_frame_count = len(target_frame_refs) if isinstance(target_frame_refs, list) else 0
    score = 1.0 if target_frame_count else 0.0
    return {
        "score": score,
        "target_frame_ref_count": target_frame_count,
        "window_frame_ref_count": frame_count,
        "detail": "target frame-backed evidence" if score else "no target frame-backed evidence",
    }


def _entity_link_signal(bundle: dict[str, Any]) -> dict[str, Any]:
    linked_entities = bundle.get("linked_entities") or []
    scores = [
        _optional_float(link.get("score"))
        for link in linked_entities
        if isinstance(link, dict) and _optional_float(link.get("score")) is not None
    ]
    if not scores:
        return {"score": 0.0, "link_count": 0, "max_link_score": None, "detail": "no entity links"}
    max_score = max(scores)
    score = min(1.0, max_score / 1.5)
    return {
        "score": round(score, 4),
        "link_count": len(scores),
        "max_link_score": round(max_score, 4),
        "detail": f"{len(scores)} entity link(s), max score {max_score:.3f}",
    }


def _human_explanation(breakdown: dict[str, Any]) -> str:
    signals = breakdown.get("signals", {})
    parts = []
    for name in DEFAULT_WEIGHTS:
        signal = signals.get(name, {})
        detail = signal.get("detail")
        contribution = signal.get("contribution")
        if detail is not None and contribution is not None:
            parts.append(f"{name}: {detail} (+{contribution:.4f})")
    return "; ".join(parts)


def _candidate(bundle: dict[str, Any]) -> dict[str, Any]:
    candidate = bundle.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def _evidence_window(bundle: dict[str, Any]) -> dict[str, Any]:
    evidence_window = bundle.get("evidence_window")
    return evidence_window if isinstance(evidence_window, dict) else {}


def _candidate_rank(bundle: dict[str, Any], *, fallback: int) -> int:
    value = _candidate(bundle).get("rank", bundle.get("rank"))
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return fallback
    return rank if rank > 0 else fallback


def _candidate_center(candidate: dict[str, Any]) -> float | None:
    center = _optional_float(candidate.get("timestamp_center"))
    if center is not None:
        return center
    start = _optional_float(candidate.get("start_time"))
    end = _optional_float(candidate.get("end_time"))
    if start is not None and end is not None:
        return (start + end) / 2.0
    return start if start is not None else end


def _distance_to_range(point: float, expected_range: tuple[float, float]) -> float:
    start, end = expected_range
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def _bundle_text_parts(bundle: dict[str, Any]) -> list[str]:
    parts = [str(_candidate(bundle).get("transcript_excerpt", ""))]
    parts.extend(_target_text_parts(bundle))
    linked_entities = bundle.get("linked_entities")
    if isinstance(linked_entities, list):
        for link in linked_entities:
            if not isinstance(link, dict):
                continue
            entity = link.get("entity")
            if isinstance(entity, dict):
                parts.append(str(entity.get("text", "")))
    return [part for part in parts if part]


def _target_text_parts(bundle: dict[str, Any]) -> list[str]:
    evidence_window = _evidence_window(bundle)
    target_segment = evidence_window.get("target_segment")
    if isinstance(target_segment, dict):
        return [str(target_segment.get("transcript_text", ""))]
    return []


def _range_quality(
    value: float | None,
    *,
    ideal_min: float,
    ideal_max: float,
    hard_max: float,
) -> float:
    if value is None:
        return 0.5
    if value <= 0:
        return 0.0
    if value < ideal_min:
        return max(0.0, value / ideal_min)
    if value <= ideal_max:
        return 1.0
    if value >= hard_max:
        return 0.0
    return max(0.0, 1.0 - ((value - ideal_max) / (hard_max - ideal_max)))


def _normalized_search_scores(scores: list[float | None]) -> list[float | None]:
    present = [score for score in scores if score is not None]
    if not present:
        return [None for _ in scores]

    low = min(present)
    high = max(present)
    if math.isclose(low, high):
        return [_bounded_score(score) if score is not None else None for score in scores]
    return [
        round((score - low) / (high - low), 6) if score is not None else None for score in scores
    ]


def _bounded_score(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    if value <= 0.0:
        return 0.0
    return value / (1.0 + value)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    normalized = {name: max(0.0, float(weights.get(name, 0.0))) for name in DEFAULT_WEIGHTS}
    total = sum(normalized.values())
    if total <= 0.0:
        return DEFAULT_WEIGHTS.copy()
    return {name: value / total for name, value in normalized.items()}


def _round_mapping(values: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in values.items()}


def _timestamp_ranges_from_metadata(metadata: dict[str, Any]) -> list[tuple[float, float]]:
    ranges = []
    for item in metadata.get("timestamp_ranges") or []:
        if not isinstance(item, dict):
            continue
        start = _optional_float(item.get("start"))
        end = _optional_float(item.get("end"))
        if start is not None and end is not None:
            ranges.append((start, end))
    return ranges


def _tokenize(value: str) -> set[str]:
    tokens = re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
    return {token for token in tokens if len(token) >= 2 or _contains_cjk(token)}


def _contains_cjk(value: str) -> bool:
    return any("\u3130" <= char <= "\u9fff" for char in value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
