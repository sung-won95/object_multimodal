from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "remaining-miss-audit-public-v1"
DEFAULT_DEPTHS = (5, 10, 30, 50, 100)
TARGET_VARIANT_ID = "evidence_unit_candidate"
CAUSE_CODES = (
    "candidate_depth_issue",
    "query_cleaning_issue",
    "missing_search_field",
    "verified_coverage_issue",
    "evaluation_match_issue",
)
QUALITY_ORDER = {
    "none": 0,
    "transcript_only": 1,
    "timestamp_fallback": 2,
    "visual_candidate": 3,
    "vlm_candidate": 4,
    "verified_visual": 5,
    "unknown": -1,
}
SEARCH_COVERAGE_ORDER = {
    "empty": 0,
    "minimal": 1,
    "partial": 2,
    "strong": 3,
    "full": 4,
    "unknown": -1,
}
PUBLIC_NOTE = (
    "This audit is public-safe: raw query text, raw answers, transcript excerpts, "
    "raw candidate ids, evidence text, and local absolute paths are excluded. Gold labels "
    "are consumed only through benchmark post-processing diagnostics, not as search input."
)


def build_remaining_miss_audit(
    *,
    query_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]] | None = None,
    run_label: str = "remaining_miss_audit",
    input_status: str = "loaded",
    unrun_reason: str | None = None,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
) -> dict[str, Any]:
    depth_lookup = _depth_lookup(depth_rows or [])
    rows_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        query_id = str(row.get("query_id") or "").strip()
        if query_id:
            rows_by_query[query_id].append(row)

    miss_rows = [
        row
        for row in query_rows
        if str(row.get("variant_id") or "") == TARGET_VARIANT_ID
        and _target_bucket(row) == "not_found"
        and _target_configured(row)
    ]

    audits = [
        _audit_miss_row(
            row=row,
            sibling_rows=rows_by_query.get(str(row.get("query_id") or ""), []),
            depth_row=depth_lookup.get(str(row.get("query_id") or "")),
            depths=depths,
        )
        for row in miss_rows
    ]
    metrics = _audit_metrics(
        audits=audits,
        query_rows=query_rows,
        input_status=input_status,
        unrun_reason=unrun_reason,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": run_label,
        "scope": {
            "lecture_slice": "1-4",
            "target_variant_id": TARGET_VARIANT_ID,
            "miss_filter": "target_rank_bucket == not_found",
            "meili_depths": list(depths),
        },
        "inputs": {
            "query_results": {"status": input_status, "row_count": len(query_rows)},
            "depth_diagnostics": {
                "status": "loaded" if depth_rows else "not_provided",
                "row_count": len(depth_rows or []),
            },
            "unrun_reason": unrun_reason,
        },
        "privacy": _privacy_payload(),
        "metrics": metrics,
        "misses": audits,
        "public_note": PUBLIC_NOTE,
    }


def write_remaining_miss_audit_outputs(
    *,
    output_dir: Path,
    audit: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.json"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(audit["metrics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(audit_summary_markdown(audit), encoding="utf-8")
    return {"audit": audit_path, "metrics": metrics_path, "summary": summary_path}


def audit_summary_markdown(audit: dict[str, Any]) -> str:
    metrics = _mapping(audit.get("metrics"))
    inputs = _mapping(audit.get("inputs"))
    scope = _mapping(audit.get("scope"))
    lines = [
        "# Issue 256 Remaining Miss Audit",
        "",
        "## Scope",
        "",
        f"- lecture slice: `{scope.get('lecture_slice')}`",
        f"- target variant: `{scope.get('target_variant_id')}`",
        f"- miss filter: `{scope.get('miss_filter')}`",
        f"- Meili depths: `{json.dumps(scope.get('meili_depths') or [])}`",
        "",
        "## Public-Safe Status",
        "",
        f"- query results input: `{_mapping(inputs.get('query_results')).get('status')}`",
        f"- depth diagnostics input: `{_mapping(inputs.get('depth_diagnostics')).get('status')}`",
    ]
    if inputs.get("unrun_reason"):
        lines.append(f"- unrun reason: `{inputs['unrun_reason']}`")
    lines.extend(
        [
            "- raw query text: `excluded`",
            "- raw answers/transcripts/evidence text: `excluded`",
            "- raw candidate ids and local paths: `excluded`",
            "",
            "## Aggregate",
            "",
            f"- candidate rows audited: `{metrics.get('candidate_variant_row_count')}`",
            f"- remaining miss seed count: `{metrics.get('remaining_miss_seed_count')}`",
            f"- cause counts: `{json.dumps(metrics.get('cause_counts') or {}, sort_keys=True)}`",
            f"- Meili depth found counts: `{json.dumps(metrics.get('meili_depth_found_counts') or {}, sort_keys=True)}`",
            f"- target search field coverage buckets: `{json.dumps(metrics.get('target_search_field_coverage_bucket_counts') or {}, sort_keys=True)}`",
            f"- quality bucket deltas: `{json.dumps(metrics.get('quality_bucket_delta_counts') or {}, sort_keys=True)}`",
            "",
            "## Miss Rows",
            "",
        ]
    )
    misses = _list_of_dicts(audit.get("misses"))
    if not misses:
        lines.append("- No `evidence_unit_candidate` `not_found` seed rows were available.")
    for miss in misses:
        lines.extend(
            [
                f"- seed: `{miss.get('seed_ref')}`",
                f"  - causes: `{json.dumps(miss.get('cause_codes') or [])}`",
                f"  - top/target quality: `{_mapping(miss.get('quality_bucket_delta')).get('top_candidate_quality_bucket')}` -> `{_mapping(miss.get('quality_bucket_delta')).get('target_candidate_quality_bucket')}`",
                f"  - target search coverage: `{miss.get('target_search_field_coverage_bucket')}`",
                f"  - Meili depth found: `{json.dumps(miss.get('meili_depth_found') or {}, sort_keys=True)}`",
            ]
        )
    lines.extend(["", f"Public note: {audit.get('public_note')}", ""])
    return "\n".join(lines)


def synthetic_remaining_miss_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_rows = [
        {
            "suite_id": "synthetic_lecture_slice",
            "query_id": "synthetic_seed_candidate_depth",
            "query_label": "synthetic public seed A",
            "variant_id": TARGET_VARIANT_ID,
            "target_rank_bucket": "not_found",
            "target_rank_diagnostics": {
                "target_configured": True,
                "target_rank_bucket": "not_found",
                "target_match_bucket": "not_found",
                "match_found": {"exact": False, "window": False, "time_overlap": False},
                "match_rank_buckets": {
                    "exact": "not_found",
                    "window": "not_found",
                    "time_overlap": "not_found",
                },
            },
            "object_evidence_coverage": {
                "has_visual_state": True,
                "has_visual_entity": True,
                "has_vlm_entity": False,
                "uses_ocr_only": True,
                "has_candidate_link": True,
                "has_verified_link": False,
                "has_timestamp_fallback_link": False,
            },
            "top_candidate": {
                "rank": 1,
                "timestamp_available": True,
                "object_evidence_coverage": {
                    "has_visual_state": True,
                    "has_visual_entity": True,
                    "has_vlm_entity": False,
                    "has_verified_link": False,
                    "has_candidate_link": True,
                },
            },
        },
        {
            "suite_id": "synthetic_lecture_slice",
            "query_id": "synthetic_seed_candidate_depth",
            "query_label": "synthetic public seed A",
            "variant_id": "domain_lexicon",
            "target_rank_bucket": "top10",
            "target_rank_diagnostics": {
                "target_configured": True,
                "target_rank_bucket": "top10",
                "target_match_bucket": "time_overlap",
                "match_found": {"exact": False, "window": False, "time_overlap": True},
                "match_rank_buckets": {
                    "exact": "not_found",
                    "window": "not_found",
                    "time_overlap": "top10",
                },
            },
        },
        {
            "suite_id": "synthetic_lecture_slice",
            "query_id": "synthetic_seed_missing_fields",
            "query_label": "synthetic public seed B",
            "variant_id": TARGET_VARIANT_ID,
            "target_rank_bucket": "not_found",
            "target_rank_diagnostics": {
                "target_configured": True,
                "target_rank_bucket": "not_found",
                "target_match_bucket": "not_found",
                "match_found": {"exact": False, "window": False, "time_overlap": False},
                "match_rank_buckets": {
                    "exact": "not_found",
                    "window": "not_found",
                    "time_overlap": "not_found",
                },
            },
            "object_evidence_coverage": {
                "has_visual_state": False,
                "has_visual_entity": False,
                "has_vlm_entity": False,
                "uses_ocr_only": False,
                "has_candidate_link": False,
                "has_verified_link": False,
                "has_timestamp_fallback_link": False,
            },
            "top_candidate": {"rank": 1, "timestamp_available": True},
        },
    ]
    depth_rows = [
        {
            "query_id": "synthetic_seed_candidate_depth",
            "meili_depth_found": {"5": False, "10": False, "30": True, "50": True, "100": True},
            "target_candidate_quality_bucket": "verified_visual",
            "target_search_field_coverage": {
                "evidence_text": True,
                "semantic_text": True,
                "transcript_keywords": True,
                "concept_search_text": True,
                "visual_state_text": True,
                "visual_entity_text": True,
            },
        },
        {
            "query_id": "synthetic_seed_missing_fields",
            "meili_depth_found": {
                "5": False,
                "10": False,
                "30": False,
                "50": False,
                "100": False,
            },
            "target_candidate_quality_bucket": "transcript_only",
            "target_search_field_coverage": {
                "evidence_text": True,
                "semantic_text": False,
                "transcript_keywords": False,
                "concept_search_text": False,
                "visual_state_text": False,
                "visual_entity_text": False,
            },
        },
    ]
    return query_rows, depth_rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(payload)
    return rows


def _audit_miss_row(
    *,
    row: dict[str, Any],
    sibling_rows: list[dict[str, Any]],
    depth_row: dict[str, Any] | None,
    depths: tuple[int, ...],
) -> dict[str, Any]:
    depth_row = depth_row or {}
    variant_buckets = {
        str(sibling.get("variant_id") or "unknown"): _variant_target_diagnostics(sibling)
        for sibling in sibling_rows
        if str(sibling.get("variant_id") or "").strip()
    }
    meili_depth_found = _meili_depth_found(row=row, depth_row=depth_row, depths=depths)
    top_quality_bucket = _quality_bucket_from_row(row)
    target_quality_bucket = str(
        depth_row.get("target_candidate_quality_bucket")
        or _mapping(_mapping(row.get("target_rank_diagnostics")).get("target_candidate")).get(
            "quality_bucket"
        )
        or "unknown"
    )
    if target_quality_bucket not in QUALITY_ORDER:
        target_quality_bucket = "unknown"
    target_search_bucket = _target_search_field_coverage_bucket(row=row, depth_row=depth_row)
    quality_delta = _quality_bucket_delta(
        top_bucket=top_quality_bucket,
        target_bucket=target_quality_bucket,
    )
    causes = _cause_codes(
        row=row,
        variant_buckets=variant_buckets,
        meili_depth_found=meili_depth_found,
        target_search_bucket=target_search_bucket,
        top_quality_bucket=top_quality_bucket,
        target_quality_bucket=target_quality_bucket,
        depth_row=depth_row,
    )
    return {
        "seed_ref": _seed_ref(row),
        "suite_ref": _short_hash(str(row.get("suite_id") or "")),
        "query_label_available": bool(str(row.get("query_label") or "").strip()),
        "target_variant_id": TARGET_VARIANT_ID,
        "candidate_target_rank_bucket": _target_bucket(row),
        "variant_target_diagnostics": variant_buckets,
        "exact_window_time_overlap": _match_flags(row),
        "quality_bucket_delta": quality_delta,
        "target_search_field_coverage_bucket": target_search_bucket,
        "meili_depth_found": meili_depth_found,
        "cause_codes": causes,
        "privacy": {
            "seed_id": "hashed",
            "query_text": "excluded",
            "raw_candidate_id": "excluded",
        },
    }


def _audit_metrics(
    *,
    audits: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    input_status: str,
    unrun_reason: str | None,
) -> dict[str, Any]:
    cause_counts: Counter[str] = Counter()
    depth_counts: dict[str, int] = {str(depth): 0 for depth in DEFAULT_DEPTHS}
    coverage_counts: Counter[str] = Counter()
    quality_delta_counts: Counter[str] = Counter()
    for audit in audits:
        cause_counts.update(str(code) for code in audit.get("cause_codes") or [])
        for depth, found in _mapping(audit.get("meili_depth_found")).items():
            if found is True:
                depth_counts[str(depth)] = depth_counts.get(str(depth), 0) + 1
        coverage_counts[str(audit.get("target_search_field_coverage_bucket") or "unknown")] += 1
        quality_delta_counts[
            str(_mapping(audit.get("quality_bucket_delta")).get("delta_bucket") or "unknown")
        ] += 1
    candidate_rows = [
        row for row in query_rows if str(row.get("variant_id") or "") == TARGET_VARIANT_ID
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "input_status": input_status,
        "unrun_reason": unrun_reason,
        "candidate_variant_row_count": len(candidate_rows),
        "remaining_miss_seed_count": len(audits),
        "cause_counts": {code: cause_counts.get(code, 0) for code in CAUSE_CODES},
        "meili_depth_found_counts": dict(sorted(depth_counts.items(), key=lambda item: int(item[0]))),
        "target_search_field_coverage_bucket_counts": dict(sorted(coverage_counts.items())),
        "quality_bucket_delta_counts": dict(sorted(quality_delta_counts.items())),
        "public_note": PUBLIC_NOTE,
    }


def _variant_target_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping(row.get("target_rank_diagnostics"))
    return {
        "target_rank_bucket": _target_bucket(row),
        "target_match_bucket": str(diagnostics.get("target_match_bucket") or "not_found"),
        "match_found": _match_flags(row),
        "match_rank_buckets": {
            key: str(_mapping(diagnostics.get("match_rank_buckets")).get(key) or "not_found")
            for key in ("exact", "window", "time_overlap")
        },
    }


def _cause_codes(
    *,
    row: dict[str, Any],
    variant_buckets: dict[str, Any],
    meili_depth_found: dict[str, bool | str],
    target_search_bucket: str,
    top_quality_bucket: str,
    target_quality_bucket: str,
    depth_row: dict[str, Any],
) -> list[str]:
    causes: set[str] = set()
    found_depths = [
        int(depth)
        for depth, found in meili_depth_found.items()
        if found is True and str(depth).isdigit()
    ]
    known_depths = [found for found in meili_depth_found.values() if found != "unknown"]
    if found_depths or (known_depths and not any(found is True for found in known_depths)):
        causes.add("candidate_depth_issue")
    if any(
        variant_id != TARGET_VARIANT_ID
        and _mapping(diagnostics).get("target_rank_bucket") not in {"not_found", "not_configured"}
        for variant_id, diagnostics in variant_buckets.items()
    ):
        causes.add("query_cleaning_issue")
    if target_search_bucket in {"empty", "minimal"}:
        causes.add("missing_search_field")
    if target_search_bucket == "unknown" and not depth_row:
        causes.add("missing_search_field")
    if target_quality_bucket in {"transcript_only", "timestamp_fallback", "visual_candidate"}:
        causes.add("verified_coverage_issue")
    top_order = QUALITY_ORDER.get(top_quality_bucket, -1)
    target_order = QUALITY_ORDER.get(target_quality_bucket, -1)
    if target_order >= 0 and top_order >= 0 and target_order < top_order:
        causes.add("verified_coverage_issue")
    diagnostics = _mapping(row.get("target_rank_diagnostics"))
    if diagnostics.get("target_configured") is not True or (
        not any(_match_flags(row).values()) and not depth_row and not known_depths
    ):
        causes.add("evaluation_match_issue")
    if not causes:
        causes.add("evaluation_match_issue")
    return [code for code in CAUSE_CODES if code in causes]


def _meili_depth_found(
    *,
    row: dict[str, Any],
    depth_row: dict[str, Any],
    depths: tuple[int, ...],
) -> dict[str, bool | str]:
    depth_payload = (
        _mapping(depth_row.get("meili_depth_found"))
        or _mapping(depth_row.get("depth_found"))
        or _mapping(_mapping(row.get("remaining_miss_audit")).get("meili_depth_found"))
    )
    values: dict[str, bool | str] = {}
    target_rank = _optional_int(depth_row.get("target_rank"))
    for depth in depths:
        raw = depth_payload.get(str(depth), depth_payload.get(depth))
        if isinstance(raw, bool):
            values[str(depth)] = raw
        elif raw in (0, 1):
            values[str(depth)] = bool(raw)
        elif target_rank is not None:
            values[str(depth)] = target_rank <= depth
        else:
            values[str(depth)] = "unknown"
    return values


def _target_search_field_coverage_bucket(
    *,
    row: dict[str, Any],
    depth_row: dict[str, Any],
) -> str:
    direct = str(
        depth_row.get("target_search_field_coverage_bucket")
        or _mapping(_mapping(row.get("target_rank_diagnostics")).get("target_candidate")).get(
            "search_field_coverage_bucket"
        )
        or ""
    ).strip()
    if direct in SEARCH_COVERAGE_ORDER:
        return direct
    coverage = (
        _mapping(depth_row.get("target_search_field_coverage"))
        or _mapping(_mapping(row.get("target_rank_diagnostics")).get("target_search_field_coverage"))
    )
    if not coverage:
        return "unknown"
    count = sum(1 for value in coverage.values() if value is True)
    if count <= 0:
        return "empty"
    if count == 1:
        return "minimal"
    if count <= 3:
        return "partial"
    if count <= 5:
        return "strong"
    return "full"


def _quality_bucket_from_row(row: dict[str, Any]) -> str:
    top_candidate = _mapping(row.get("top_candidate"))
    coverage = _mapping(top_candidate.get("object_evidence_coverage")) or _mapping(
        row.get("object_evidence_coverage")
    )
    if not coverage and not top_candidate:
        return "none"
    if coverage.get("has_verified_link") is True:
        return "verified_visual"
    if coverage.get("has_vlm_entity") is True:
        return "vlm_candidate"
    if (
        coverage.get("has_visual_entity") is True
        or coverage.get("has_visual_state") is True
        or coverage.get("has_candidate_link") is True
    ):
        return "visual_candidate"
    if coverage.get("has_timestamp_fallback_link") is True:
        return "timestamp_fallback"
    if top_candidate.get("timestamp_available") is True or coverage:
        return "transcript_only"
    return "none"


def _quality_bucket_delta(*, top_bucket: str, target_bucket: str) -> dict[str, Any]:
    top_order = QUALITY_ORDER.get(top_bucket, -1)
    target_order = QUALITY_ORDER.get(target_bucket, -1)
    delta = None if top_order < 0 or target_order < 0 else target_order - top_order
    if delta is None:
        delta_bucket = "unknown"
    elif delta > 0:
        delta_bucket = "target_stronger"
    elif delta < 0:
        delta_bucket = "target_weaker"
    else:
        delta_bucket = "same"
    return {
        "top_candidate_quality_bucket": top_bucket,
        "target_candidate_quality_bucket": target_bucket,
        "delta": delta,
        "delta_bucket": delta_bucket,
    }


def _depth_lookup(depth_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in depth_rows:
        query_id = str(row.get("query_id") or "").strip()
        variant_id = str(row.get("variant_id") or TARGET_VARIANT_ID)
        if query_id and variant_id == TARGET_VARIANT_ID:
            lookup[query_id] = row
    return lookup


def _target_bucket(row: dict[str, Any]) -> str:
    diagnostics = _mapping(row.get("target_rank_diagnostics"))
    return str(
        diagnostics.get("target_rank_bucket")
        or row.get("target_rank_bucket")
        or "not_found"
    )


def _target_configured(row: dict[str, Any]) -> bool:
    diagnostics = _mapping(row.get("target_rank_diagnostics"))
    if diagnostics:
        return diagnostics.get("target_configured") is True
    return bool(row.get("expected_time_available") or row.get("expected_segment_available"))


def _match_flags(row: dict[str, Any]) -> dict[str, bool]:
    flags = _mapping(_mapping(row.get("target_rank_diagnostics")).get("match_found"))
    return {key: flags.get(key) is True for key in ("exact", "window", "time_overlap")}


def _seed_ref(row: dict[str, Any]) -> str:
    seed = f"{row.get('suite_id') or ''}:{row.get('query_id') or ''}"
    return f"seed:{_short_hash(seed)}"


def _short_hash(value: str) -> str:
    if not value:
        value = "unknown"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _privacy_payload() -> dict[str, Any]:
    return {
        "public_outputs_are_sanitized": True,
        "raw_query_text": "excluded",
        "answer_text": "excluded",
        "transcript_excerpt": "excluded",
        "candidate_evidence_text": "excluded",
        "raw_candidate_ids": "excluded",
        "local_absolute_paths": "excluded",
        "gold_labels_used_as_search_input": False,
        "gold_labels_used_only_for_posthoc_target_diagnostics": True,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value or [] if isinstance(item, dict)]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
