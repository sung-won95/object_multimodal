from __future__ import annotations

from dataclasses import dataclass

from oarag.core.schemas import EduVidQARecord, SearchCandidate


@dataclass(frozen=True)
class QueryEval:
    sample_id: str
    hit_by_delta: dict[int, bool]
    best_abs_error: float | None
    top_candidate_sample_id: str | None


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
) -> QueryEval:
    errors = absolute_errors(record, candidates)
    best = min(errors) if errors else None
    return QueryEval(
        sample_id=record.sample_id,
        hit_by_delta={delta: best is not None and best <= delta for delta in deltas},
        best_abs_error=best,
        top_candidate_sample_id=candidates[0].sample_id if candidates else None,
    )


def summarize(evals: list[QueryEval], deltas: list[int]) -> dict:
    total = len(evals)
    if total == 0:
        return {"count": 0}
    errors = [item.best_abs_error for item in evals if item.best_abs_error is not None]
    summary = {
        "count": total,
        "mean_abs_error": round(sum(errors) / len(errors), 4) if errors else None,
    }
    for delta in deltas:
        hits = sum(1 for item in evals if item.hit_by_delta.get(delta, False))
        summary[f"hit_at_{delta}s"] = round(hits / total, 4)
    return summary
