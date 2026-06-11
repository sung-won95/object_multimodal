from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.project_index import segment_artifact_path


STRICT_DETERMINISTIC_VERIFIER_SCHEMA_VERSION = "oarag-strict-deterministic-link-verifier-v1"
STRICT_DETERMINISTIC_REPORT_SCHEMA_VERSION = (
    "oarag-strict-deterministic-link-verifier-report-public-v1"
)
STRICT_DETERMINISTIC_LINK_SOURCE = "strict_deterministic_rule"
DEFAULT_STRICT_LEXICAL_OVERLAP_THRESHOLD = 0.5
DEFAULT_SWEEP_THRESHOLDS = (0.25, 0.4, 0.5, 0.6, 0.75)
VISUAL_TEXT_FIELDS = ("visible_text", "detected_text", "text")
TERM_STOPWORDS = {
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
    "with",
}
TIME_OVERLAP_ONLY_EVIDENCE = {"time_overlap", "timestamp_fallback"}


def verify_entity_links_strict_deterministic(
    *,
    project_dir: Path,
    entity_links_path: Path | None = None,
    segments_path: Path | None = None,
    visual_entities_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
    lexical_overlap_threshold: float = DEFAULT_STRICT_LEXICAL_OVERLAP_THRESHOLD,
    sweep_thresholds: Iterable[float] | None = DEFAULT_SWEEP_THRESHOLDS,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_segments_path = segment_artifact_path(resolved_project_dir, segments=segments_path)
    resolved_visual_entities_path = _resolve_existing_project_path(
        project_dir=resolved_project_dir,
        path=visual_entities_path,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
        artifact_name="visual entities",
    )
    resolved_entity_links_path = _resolve_existing_project_path(
        project_dir=resolved_project_dir,
        path=entity_links_path,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
        artifact_name="entity links",
    )
    resolved_output_path = _resolve_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / "manifests" / "entity_links.verified.jsonl",
    )
    resolved_report_path = (
        _resolve_output_path(
            project_dir=resolved_project_dir,
            path=report_path,
            default=resolved_project_dir
            / "reports"
            / "strict_deterministic_entity_link_verifier.json",
        )
        if report_path is not None
        else None
    )

    segment_rows = _read_jsonl(resolved_segments_path)
    entity_rows = _read_jsonl(resolved_visual_entities_path)
    link_rows = _read_jsonl(resolved_entity_links_path)
    segments_by_id = {_text(row.get("segment_id")): row for row in segment_rows}
    entities_by_id = {_text(row.get("entity_id")): row for row in entity_rows}

    threshold = _threshold(lexical_overlap_threshold)
    decisions = [
        _link_decision(
            link,
            segment=segments_by_id.get(_text(link.get("segment_id")), {}),
            entity=entities_by_id.get(_text(link.get("entity_id")), {}),
            lexical_overlap_threshold=threshold,
        )
        for link in link_rows
    ]
    output_rows = [
        _promote_link(row, decision) if decision["promoted"] else dict(row)
        for row, decision in zip(link_rows, decisions)
    ]
    write_jsonl(resolved_output_path, output_rows)

    report = _public_report(
        links=output_rows,
        decisions=decisions,
        threshold=threshold,
        sweep_thresholds=[_threshold(item) for item in (sweep_thresholds or [])],
        segments_by_id=segments_by_id,
    )
    if resolved_report_path is not None:
        write_json(resolved_report_path, report)

    return {
        "schema_version": STRICT_DETERMINISTIC_VERIFIER_SCHEMA_VERSION,
        "counts": report["counts"],
        "threshold": threshold,
        "paths": {
            "entity_links": str(resolved_entity_links_path),
            "segments": str(resolved_segments_path),
            "visual_entities": str(resolved_visual_entities_path),
            "output": str(resolved_output_path),
            "report": str(resolved_report_path) if resolved_report_path is not None else None,
        },
        "public_report": report,
    }


def _link_decision(
    link: dict[str, Any],
    *,
    segment: dict[str, Any],
    entity: dict[str, Any],
    lexical_overlap_threshold: float,
) -> dict[str, Any]:
    base = {
        "link_id": _text(link.get("link_id")),
        "promoted": False,
        "reason": "not_evaluated",
        "score": 0.0,
        "matched_term_count": 0,
        "visual_term_count": 0,
        "transcript_term_count": 0,
        "visual_text_fields": [],
    }
    if _is_verified(link):
        return {**base, "reason": "already_verified"}
    if _is_time_overlap_only_link(link):
        return {**base, "reason": "time_overlap_only_excluded"}
    transcript_terms = _terms(_text(segment.get("transcript_text")))
    visual_field_terms = _visual_text_terms_by_field(entity)
    visual_terms = set().union(*visual_field_terms.values()) if visual_field_terms else set()
    matched_terms = transcript_terms & visual_terms
    score = _lexical_overlap_score(matched_terms=matched_terms, visual_terms=visual_terms)
    evaluated = {
        **base,
        "score": score,
        "threshold": lexical_overlap_threshold,
        "matched_term_count": len(matched_terms),
        "visual_term_count": len(visual_terms),
        "transcript_term_count": len(transcript_terms),
        "visual_text_fields": sorted(
            field for field, terms in visual_field_terms.items() if terms & matched_terms
        ),
    }
    if not segment:
        return {**evaluated, "reason": "missing_segment"}
    if not entity:
        return {**evaluated, "reason": "missing_visual_entity"}
    if not matched_terms:
        return {**evaluated, "reason": "no_exact_transcript_visual_text_match"}
    if score <= lexical_overlap_threshold:
        return {**evaluated, "reason": "below_lexical_overlap_threshold"}
    return {**evaluated, "promoted": True, "reason": "exact_text_match_above_threshold"}


def _promote_link(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    promoted = dict(row)
    reason_metadata = dict(_mapping(promoted.get("reason_metadata")))
    strict_reason = {
        "source": STRICT_DETERMINISTIC_LINK_SOURCE,
        "verification_source": STRICT_DETERMINISTIC_LINK_SOURCE,
        "summary": "strict_deterministic_exact_text_overlap",
        "rule": "exact_transcript_visual_text_match_with_lexical_overlap_threshold",
        "lexical_overlap_score": decision["score"],
        "lexical_overlap_threshold": decision.get("threshold"),
        "matched_term_count": decision["matched_term_count"],
        "visual_term_count": decision["visual_term_count"],
        "transcript_term_count": decision["transcript_term_count"],
        "visual_text_fields": decision["visual_text_fields"],
        "public_safe": True,
    }
    strict_reason = {key: value for key, value in strict_reason.items() if value is not None}
    reason_metadata["strict_deterministic_verifier"] = strict_reason
    reason_metadata["verification_source"] = STRICT_DETERMINISTIC_LINK_SOURCE
    promoted["reason_metadata"] = reason_metadata
    promoted["alignment_status"] = "verified"
    promoted["verification_status"] = "verified"
    promoted["verified_link_source"] = STRICT_DETERMINISTIC_LINK_SOURCE
    promoted["verification_source"] = STRICT_DETERMINISTIC_LINK_SOURCE
    return promoted


def _public_report(
    *,
    links: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    threshold: float,
    sweep_thresholds: list[float],
    segments_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    promoted = [decision for decision in decisions if decision["promoted"]]
    reason_counts = Counter(str(decision["reason"]) for decision in decisions)
    lecture_counts: dict[str, int] = defaultdict(int)
    for link, decision in zip(links, decisions):
        if not decision["promoted"]:
            continue
        lecture_ref = _public_lecture_ref(link, segments_by_id)
        lecture_counts[lecture_ref] += 1
    sweep = [
        _sweep_row(
            threshold=item,
            decisions=decisions,
            links=links,
            segments_by_id=segments_by_id,
        )
        for item in sorted(set(sweep_thresholds + [threshold]))
    ]
    return {
        "schema_version": STRICT_DETERMINISTIC_REPORT_SCHEMA_VERSION,
        "privacy": {
            "raw_transcript": "redacted",
            "raw_visual_text": "redacted",
            "raw_ids": "omitted",
            "local_paths": "omitted",
        },
        "rule": {
            "source": STRICT_DETERMINISTIC_LINK_SOURCE,
            "exact_match_required": True,
            "lexical_overlap_threshold": threshold,
            "time_overlap_only_promoted": False,
            "visual_text_sources": list(VISUAL_TEXT_FIELDS),
        },
        "counts": {
            "links_total": len(links),
            "promoted_links": len(promoted),
            "already_verified_links": reason_counts["already_verified"],
            "time_overlap_only_excluded": reason_counts["time_overlap_only_excluded"],
            "below_threshold": reason_counts["below_lexical_overlap_threshold"],
            "no_exact_match": reason_counts["no_exact_transcript_visual_text_match"],
        },
        "promotion_distribution": {
            "by_public_lecture_ref": dict(sorted(lecture_counts.items())),
        },
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "threshold_sweep": sweep,
        "public_note": (
            "This report contains aggregate counts only. Raw transcripts, visual text, "
            "local paths, and raw link/entity IDs are intentionally omitted."
        ),
    }


def _sweep_row(
    *,
    threshold: float,
    decisions: list[dict[str, Any]],
    links: list[dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lecture_counts: dict[str, int] = defaultdict(int)
    promoted_count = 0
    for link, decision in zip(links, decisions):
        if decision["reason"] in {"already_verified", "time_overlap_only_excluded"}:
            continue
        if decision["matched_term_count"] <= 0:
            continue
        if float(decision["score"]) <= threshold:
            continue
        promoted_count += 1
        lecture_counts[_public_lecture_ref(link, segments_by_id)] += 1
    return {
        "threshold": threshold,
        "promoted_links": promoted_count,
        "by_public_lecture_ref": dict(sorted(lecture_counts.items())),
    }


def _public_lecture_ref(
    link: dict[str, Any],
    segments_by_id: dict[str, dict[str, Any]],
) -> str:
    segment = segments_by_id.get(_text(link.get("segment_id")), {})
    video_id = _text(segment.get("video_id") or link.get("video_id"))
    project_id = _text(segment.get("project_id") or link.get("project_id"))
    if video_id:
        return f"video:{_short_hash(video_id)}"
    if project_id:
        return f"project:{_short_hash(project_id)}"
    return "unknown"


def _is_verified(link: dict[str, Any]) -> bool:
    status = _text(
        link.get("alignment_status")
        or link.get("verification_status")
        or link.get("status")
    ).casefold()
    return status == "verified" or link.get("verified") is True


def _is_time_overlap_only_link(link: dict[str, Any]) -> bool:
    evidence = {_text(item).casefold() for item in _list(link.get("evidence"))}
    link_type = _text(link.get("link_type")).casefold()
    link_type_parts = {part for part in link_type.split("+") if part}
    score_breakdown_keys = {
        _text(key).casefold()
        for key in _mapping(link.get("score_breakdown"))
        if _text(key)
    }
    reason_summary = _text(_mapping(link.get("reason_metadata")).get("summary")).casefold()
    if "timestamp_fallback" in evidence or reason_summary == "timestamp_fallback_only":
        return True
    strong_evidence = (
        evidence
        | link_type_parts
        | score_breakdown_keys
    ) - TIME_OVERLAP_ONLY_EVIDENCE
    return not strong_evidence


def _visual_text_terms_by_field(entity: dict[str, Any]) -> dict[str, set[str]]:
    terms_by_field: dict[str, set[str]] = {}
    for field in VISUAL_TEXT_FIELDS:
        terms = _terms(" ".join(_text_values(entity.get(field))))
        if terms:
            terms_by_field[field] = terms
    return terms_by_field


def _lexical_overlap_score(*, matched_terms: set[str], visual_terms: set[str]) -> float:
    if not visual_terms:
        return 0.0
    return round(len(matched_terms) / len(visual_terms), 6)


def _terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9a-zA-Z가-힣]+", value.casefold())
        if len(token) >= 2 and not token.isdigit() and token not in TERM_STOPWORDS
    }


def _resolve_existing_project_path(
    *,
    project_dir: Path,
    path: Path | None,
    default: Path,
    artifact_name: str,
) -> Path:
    resolved = _resolve_output_path(project_dir=project_dir, path=path, default=default)
    if not resolved.exists():
        raise FileNotFoundError(f"{artifact_name} not found: {resolved}")
    return resolved


def _resolve_output_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    candidate = path or default
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return candidate.expanduser().resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {path}:{line_number}")
            rows.append(payload)
    return rows


def _threshold(value: Any) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("lexical overlap threshold must be between 0.0 and 1.0")
    return round(parsed, 6)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[str] = []
        for child in value.values():
            values.extend(_text_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_text_values(child))
        return values
    text = _text(value)
    return [text] if text else []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _short_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
