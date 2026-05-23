from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from oarag.core.schemas import EduVidQARecord, SearchCandidate

DEFAULT_RECALL_KS = [1, 5, 10, 50]


@dataclass(frozen=True)
class QueryEval:
    sample_id: str
    hit_by_delta: dict[int, bool]
    best_abs_error: float | None
    top_candidate_sample_id: str | None
    recall_by_k: dict[int, bool]


def absolute_errors(record: EduVidQARecord, candidates: list[SearchCandidate]) -> list[float]:
    errors: list[float] = []
    if not record.timestamp_points:
        return errors
    for candidate in candidates:
        if candidate.timestamp_center is None:
            continue
        errors.extend(abs(candidate.timestamp_center - gold) for gold in record.timestamp_points)
    return errors


def evaluate_query(
    record: EduVidQARecord,
    candidates: list[SearchCandidate],
    deltas: list[int],
    recall_ks: list[int] | None = None,
) -> QueryEval:
    errors = absolute_errors(record, candidates)
    best = min(errors) if errors else None
    ks = recall_ks or DEFAULT_RECALL_KS
    return QueryEval(
        sample_id=record.sample_id,
        hit_by_delta={delta: best is not None and best <= delta for delta in deltas},
        best_abs_error=best,
        top_candidate_sample_id=candidates[0].sample_id if candidates else None,
        recall_by_k={k: _same_sample_in_top_k(record, candidates, k) for k in ks},
    )


def summarize(evals: list[QueryEval], deltas: list[int]) -> dict:
    total = len(evals)
    if total == 0:
        return {"count": 0}
    errors = [item.best_abs_error for item in evals if item.best_abs_error is not None]
    summary = {
        "count": total,
        "mean_abs_error": round(sum(errors) / len(errors), 4) if errors else None,
        "median_abs_error": _percentile(errors, 0.5),
        "p75_abs_error": _percentile(errors, 0.75),
        "p90_abs_error": _percentile(errors, 0.9),
    }
    for delta in deltas:
        hits = sum(1 for item in evals if item.hit_by_delta.get(delta, False))
        summary[f"hit_at_{delta}s"] = round(hits / total, 4)
    recall_ks = sorted({k for item in evals for k in item.recall_by_k})
    for k in recall_ks:
        hits = sum(1 for item in evals if item.recall_by_k.get(k, False))
        summary[f"top{k}_recall"] = round(hits / total, 4)
    return summary


def candidate_diagnostics(
    record: EduVidQARecord,
    candidates: list[SearchCandidate],
    *,
    top_k: int,
    include_private_fields: bool = False,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    candidate_errors = candidate_abs_errors(record, candidates)
    rows: list[dict[str, Any]] = []
    for candidate, abs_error in zip(candidates[:top_k], candidate_errors[:top_k]):
        row: dict[str, Any] = {
            "rank": candidate.rank,
            "segment_id": candidate.segment_id,
            "sample_id": candidate.sample_id,
            "video_id": candidate.video_id,
            "timestamp_center": candidate.timestamp_center,
            "abs_error": abs_error,
            "same_sample": _same_sample(record, candidate),
            "same_video": _same_video(record, candidate),
        }
        if include_private_fields:
            row["transcript_excerpt"] = candidate.transcript_excerpt
            if candidate.semantic_source_fields:
                row["semantic_source_fields"] = list(candidate.semantic_source_fields)
        rows.append(row)
    return rows


def candidate_abs_errors(
    record: EduVidQARecord,
    candidates: list[SearchCandidate],
) -> list[float | None]:
    errors: list[float | None] = []
    for candidate in candidates:
        if candidate.timestamp_center is None or not record.timestamp_points:
            errors.append(None)
            continue
        errors.append(
            min(abs(candidate.timestamp_center - gold) for gold in record.timestamp_points)
        )
    return errors


def _same_sample_in_top_k(
    record: EduVidQARecord,
    candidates: list[SearchCandidate],
    top_k: int,
) -> bool:
    if top_k <= 0:
        return False
    return any(_same_sample(record, candidate) for candidate in candidates[:top_k])


def _same_sample(record: EduVidQARecord, candidate: SearchCandidate) -> bool:
    return bool(record.sample_id) and candidate.sample_id == record.sample_id


def _same_video(record: EduVidQARecord, candidate: SearchCandidate) -> bool:
    return bool(record.video_name) and candidate.video_id == record.video_name


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 4)
