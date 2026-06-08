from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oarag.core.config import default_paths
from oarag.core.io import write_json, write_jsonl
from oarag.integrations.meili import EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE
from oarag.retrieval.evidence_unit_index import query_project_evidence_units
from oarag.retrieval.evidence_units import build_project_evidence_units
from oarag.retrieval.project_index import index_project_evidence_units, iter_jsonl_documents
from oarag.vision.vlm_evidence_validator import validate_vlm_object_evidence


PUBLIC_SCHEMA_VERSION = "evidence-unit-retrieval-smoke-public-v1"
DEFAULT_OUTPUT_ROOT = Path("reports") / "paper" / "evidence_units_retrieval_smoke"
DEFAULT_INDEX_PREFIX = "evidence_units_smoke"
DEFAULT_TARGET_RANK_LIMIT = 50


class EvidenceUnitSmokeClient(Protocol):
    def health(self) -> dict[str, Any]: ...

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        filter: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EvidenceUnitSmokeRun:
    run_id: str
    output_dir: Path
    metrics_path: Path
    query_results_path: Path
    summary_path: Path
    payload: dict[str, Any]


def run_evidence_unit_smoke(
    *,
    client: EvidenceUnitSmokeClient,
    manifest_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    dry_run: bool = False,
    quality_rerank: bool = False,
) -> EvidenceUnitSmokeRun:
    manifest = _read_json(manifest_path)
    base_dir = manifest_path.expanduser().resolve().parent
    run_id = str(manifest.get("run_id") or f"evidence_unit_smoke_{int(time.time())}")
    resolved_repo_root = (repo_root or default_paths().repo_root).expanduser().resolve()
    resolved_output_dir = _resolve_output_dir(
        manifest=manifest,
        output_dir=output_dir,
        base_dir=base_dir,
        run_id=run_id,
    )

    suites = manifest.get("suites") or []
    if not isinstance(suites, list):
        raise ValueError("evidence-unit-smoke manifest requires a list under 'suites'")

    health = (
        {"available": False, "status": "dry_run", "skip_reason": "dry_run_requested"}
        if dry_run
        else _meili_health(client)
    )
    suite_summaries: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for suite_index, suite in enumerate(suites, start=1):
        if not isinstance(suite, dict):
            raise ValueError(f"evidence-unit-smoke suite #{suite_index} must be a JSON object")
        suite_summary, rows = _run_suite(
            client=client,
            suite=suite,
            base_dir=base_dir,
            repo_root=resolved_repo_root,
            run_id=run_id,
            health=health,
            dry_run=dry_run,
            quality_rerank=quality_rerank,
        )
        suite_summaries.append(suite_summary)
        query_rows.extend(rows)

    payload = _summary_payload(
        run_id=run_id,
        manifest_path=manifest_path,
        suites=suite_summaries,
        query_rows=query_rows,
        health=health,
        dry_run=dry_run,
        quality_rerank=quality_rerank,
    )
    metrics_path = resolved_output_dir / "metrics.json"
    query_results_path = resolved_output_dir / "query_results.jsonl"
    summary_path = resolved_output_dir / "summary.md"
    write_json(metrics_path, payload)
    write_jsonl(query_results_path, query_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_markdown(payload), encoding="utf-8")

    return EvidenceUnitSmokeRun(
        run_id=run_id,
        output_dir=resolved_output_dir,
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        summary_path=summary_path,
        payload=payload,
    )


def _run_suite(
    *,
    client: EvidenceUnitSmokeClient,
    suite: dict[str, Any],
    base_dir: Path,
    repo_root: Path,
    run_id: str,
    health: dict[str, Any],
    dry_run: bool,
    quality_rerank: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_id = str(suite.get("suite_id") or suite.get("project_id") or "lecture_suite")
    project_dir = _project_dir_from_suite(suite=suite, base_dir=base_dir, repo_root=repo_root)
    evidence_units_path = _optional_path(suite.get("evidence_units"))
    visual_states_path = _optional_path(suite.get("visual_states"))
    visual_states_output_path = _optional_path(suite.get("visual_states_output"))
    index_uid = str(suite.get("index") or _default_index_uid(run_id, suite_id))
    segment_index_uid = _optional_str(suite.get("segment_index"))
    limit = _positive_int(suite.get("limit", 5), field_name="limit")
    target_rank_limit = _positive_int(
        suite.get("target_rank_limit", max(limit, DEFAULT_TARGET_RANK_LIMIT)),
        field_name="target_rank_limit",
    )
    target_rank_limit = max(limit, target_rank_limit)
    suite_quality_rerank = bool(suite.get("quality_rerank", quality_rerank))
    queries = _read_queries(base_dir=base_dir, suite=suite)

    build_summary = None
    if not dry_run:
        build_summary = build_project_evidence_units(
            project_dir=project_dir,
            output_path=evidence_units_path,
            segments=_optional_path(suite.get("segments")),
            frames_manifest=_optional_path(suite.get("frames_manifest")),
            visual_states=visual_states_path,
            visual_states_output=visual_states_output_path,
            visual_entities=_optional_path(suite.get("visual_entities")),
            entity_links=_optional_path(suite.get("entity_links")),
            manifest_path=_optional_path(suite.get("project_manifest")),
            window_seconds=_optional_float(suite.get("window_seconds")),
            neighbor_count=int(suite.get("neighbor_count", 1)),
            previous_neighbor_count=_optional_int(suite.get("previous_neighbor_count")),
            next_neighbor_count=_optional_int(suite.get("next_neighbor_count")),
            window_before_seconds=_optional_float(suite.get("window_before_seconds")),
            window_after_seconds=_optional_float(suite.get("window_after_seconds")),
            state_padding_seconds=float(suite.get("state_padding_seconds", 15.0)),
            visual_state_min_coverage_ratio=_optional_float(
                suite.get("visual_state_min_coverage_ratio")
                if suite.get("visual_state_min_coverage_ratio") is not None
                else suite.get("visual_state_min_unit_coverage_ratio")
            ),
            visual_state_min_total=_optional_int(suite.get("visual_state_min_total")),
            fail_on_visual_state_gate=bool(suite.get("fail_on_visual_state_gate", False)),
        )
        evidence_units_path = _path_from_build_summary(build_summary)

    artifact_summary = _artifact_summary(
        project_dir=project_dir,
        evidence_units=evidence_units_path,
        visual_states=visual_states_path or visual_states_output_path,
        build_summary=build_summary,
        dry_run=dry_run,
    )
    index_summary: dict[str, Any]
    query_rows: list[dict[str, Any]] = []
    if dry_run:
        index_summary = {"status": "dry_run", "skip_reason": "dry_run_requested"}
    elif not health["available"]:
        index_summary = {
            "status": "skipped",
            "skip_reason": health["skip_reason"],
            "indexed_documents": 0,
        }
        query_rows = [
            _skipped_query_row(
                run_id=run_id,
                suite_id=suite_id,
                query_row=query_row,
                index_uid=index_uid,
                segment_index_uid=segment_index_uid,
                skip_reason=health["skip_reason"],
            )
            for query_row in queries
        ]
    else:
        try:
            raw_index_summary = index_project_evidence_units(
                client,  # type: ignore[arg-type]
                index_uid=index_uid,
                project_dir=project_dir,
                batch_size=_positive_int(suite.get("batch_size", 500), field_name="batch_size"),
                reset=bool(suite.get("reset", True)),
                evidence_units=evidence_units_path,
                settings_profile=str(suite.get("settings_profile") or EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE),
            )
            index_summary = _public_index_summary(raw_index_summary)
            for query_row in queries:
                query_rows.append(
                    _query_row(
                        client=client,
                        run_id=run_id,
                        suite_id=suite_id,
                        project_dir=project_dir,
                        evidence_units=evidence_units_path,
                        index_uid=index_uid,
                        segment_index_uid=segment_index_uid,
                        query_row=query_row,
                        limit=limit,
                        target_rank_limit=target_rank_limit,
                        quality_rerank=suite_quality_rerank,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - smoke runner should record skip/failure context.
            reason = _exception_reason(exc)
            index_summary = {"status": "failed", "skip_reason": reason, "indexed_documents": 0}
            query_rows = [
                _skipped_query_row(
                    run_id=run_id,
                    suite_id=suite_id,
                    query_row=query_row,
                    index_uid=index_uid,
                    segment_index_uid=segment_index_uid,
                    skip_reason=reason,
                )
                for query_row in queries
            ]

    suite_summary = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "suite_id": suite_id,
        "project_ref": _short_hash(str(project_dir)),
        "query_count": len(queries),
        "build": artifact_summary,
        "meilisearch": {
            "available": health["available"],
            "status": health["status"],
        },
        "index": {
            "ref": _index_ref(index_uid),
            **index_summary,
        },
        "rag_input_inspection": _rag_input_inspection(query_rows, artifact_summary),
        "target_rank_diagnostics": _target_rank_inspection(query_rows),
        "rerank_diagnostics": _rerank_inspection(query_rows),
    }
    return suite_summary, query_rows


def _query_row(
    *,
    client: EvidenceUnitSmokeClient,
    run_id: str,
    suite_id: str,
    project_dir: Path,
    evidence_units: Path | None,
    index_uid: str,
    segment_index_uid: str | None,
    query_row: dict[str, Any],
    limit: int,
    target_rank_limit: int,
    quality_rerank: bool,
) -> dict[str, Any]:
    query_id = _query_id(query_row)
    query_text = _query_text(query_row)
    expected_segment_ids = set(_string_list(query_row.get("expected_segment_ids")))
    expected_segment_id = _optional_str(query_row.get("expected_segment_id"))
    if expected_segment_id:
        expected_segment_ids.add(expected_segment_id)

    response = query_project_evidence_units(
        client=client,  # type: ignore[arg-type]
        index_uid=index_uid,
        project_dir=project_dir,
        query=query_text,
        limit=target_rank_limit,
        evidence_units=evidence_units,
    )
    candidates = _list_of_dicts(response.get("candidates"))
    top_candidate = candidates[0] if candidates else {}
    expected_match = _candidate_matches_expected(top_candidate, expected_segment_ids)
    target_diagnostics = _target_diagnostics(
        candidates=candidates,
        expected_segment_ids=expected_segment_ids,
        search_depth=target_rank_limit,
        query_text=query_text,
    )
    reranked_candidates = _quality_rerank_candidates(candidates) if quality_rerank else []
    reranked_top_candidate = reranked_candidates[0] if reranked_candidates else {}
    reranked_expected_match = (
        _candidate_matches_expected(reranked_top_candidate, expected_segment_ids)
        if quality_rerank
        else None
    )
    reranked_target_diagnostics = (
        _target_diagnostics(
            candidates=reranked_candidates,
            expected_segment_ids=expected_segment_ids,
            search_depth=target_rank_limit,
            query_text=query_text,
        )
        if quality_rerank
        else {}
    )
    baseline = _segment_baseline(
        client=client,
        segment_index_uid=segment_index_uid,
        query_text=query_text,
        limit=limit,
        expected_segment_ids=expected_segment_ids,
    )
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "query_id": query_id,
        "query_label": _optional_public_label(query_row),
        "privacy": _privacy_policy(),
        "index_ref": _index_ref(index_uid),
        "segment_baseline_index_ref": _index_ref(segment_index_uid) if segment_index_uid else None,
        "status": "queried",
        "search_hit_count": len(candidates),
        "processing_time_ms": _optional_float(response.get("processing_time_ms")),
        "top_evidence_unit": _public_candidate(top_candidate, query_text=query_text),
        "top_hit_memo": _top_hit_memo(
            candidate=top_candidate,
            expected_match=expected_match,
            expected_configured=bool(expected_segment_ids),
            target_rank_bucket=str(target_diagnostics.get("target_rank_bucket") or ""),
        ),
        "target_diagnostics": target_diagnostics,
        "reranked_top_evidence_unit": (
            _public_candidate(reranked_top_candidate, query_text=query_text) if quality_rerank else None
        ),
        "reranked_top_hit_memo": (
            _top_hit_memo(
                candidate=reranked_top_candidate,
                expected_match=reranked_expected_match,
                expected_configured=bool(expected_segment_ids),
                target_rank_bucket=str(reranked_target_diagnostics.get("target_rank_bucket") or ""),
            )
            if quality_rerank
            else None
        ),
        "rerank_diagnostics": _rerank_diagnostics(
            enabled=quality_rerank,
            base_top_candidate=top_candidate,
            reranked_candidates=reranked_candidates,
            expected_segment_ids=expected_segment_ids,
            base_target_diagnostics=target_diagnostics,
            reranked_target_diagnostics=reranked_target_diagnostics,
        ),
        "segment_baseline": baseline,
        "rag_input_inspectable": _candidate_is_rag_inspectable(top_candidate),
    }


def _segment_baseline(
    *,
    client: EvidenceUnitSmokeClient,
    segment_index_uid: str | None,
    query_text: str,
    limit: int,
    expected_segment_ids: set[str],
) -> dict[str, Any]:
    if not segment_index_uid:
        return {"status": "not_configured"}
    try:
        response = client.search(segment_index_uid, query_text, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "skip_reason": _exception_reason(exc)}
    hits = _list_of_dicts(response.get("hits"))
    top_hit = hits[0] if hits else {}
    top_segment_id = _optional_str(top_hit.get("segment_id") or top_hit.get("sample_id"))
    expected_match = top_segment_id in expected_segment_ids if expected_segment_ids else None
    return {
        "status": "queried",
        "search_hit_count": len(hits),
        "top_candidate": {
            "ref": _id_ref(top_segment_id, prefix="seg"),
            "timestamp_available": _optional_float(
                top_hit.get("timestamp_center") or top_hit.get("start_time")
            )
            is not None,
            "score_available": _optional_float(top_hit.get("_rankingScore")) is not None,
        },
        "top_hit_memo": {
            "expected_target_configured": bool(expected_segment_ids),
            "top_expected_match": expected_match,
            "public_note": _baseline_note(expected_match),
        },
    }


def _skipped_query_row(
    *,
    run_id: str,
    suite_id: str,
    query_row: dict[str, Any],
    index_uid: str,
    segment_index_uid: str | None,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "query_id": _query_id(query_row),
        "query_label": _optional_public_label(query_row),
        "privacy": _privacy_policy(),
        "index_ref": _index_ref(index_uid),
        "segment_baseline_index_ref": _index_ref(segment_index_uid) if segment_index_uid else None,
        "status": "skipped",
        "skip_reason": skip_reason,
        "search_hit_count": 0,
        "top_evidence_unit": None,
        "top_hit_memo": {
            "expected_target_configured": bool(
                _optional_str(query_row.get("expected_segment_id"))
                or _string_list(query_row.get("expected_segment_ids"))
            ),
            "top_expected_match": None,
            "public_note": "index/query skipped; no top-hit judgment",
        },
        "target_diagnostics": {
            "target_configured": bool(
                _optional_str(query_row.get("expected_segment_id"))
                or _string_list(query_row.get("expected_segment_ids"))
            ),
            "target_found_in_top_k": None,
            "target_rank_bucket": "not_queried",
            "target_search_depth": 0,
        },
        "rerank_diagnostics": {
            "enabled": False,
            "status": "not_queried",
        },
        "segment_baseline": {"status": "skipped", "skip_reason": skip_reason},
        "rag_input_inspectable": False,
    }


def _artifact_summary(
    *,
    project_dir: Path,
    evidence_units: Path | None,
    visual_states: Path | None,
    build_summary: dict[str, Any] | None,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "dry_run",
            "counts": {},
            "alignment_status_counts": {},
            "source_quality_counts": {},
            "link_counts": {},
            "visual_state_coverage": {},
            "vlm_object_evidence_coverage": {
                "status": "dry_run",
                "skip_reason": "dry_run_requested",
            },
        }
    vlm_object_evidence_coverage = validate_vlm_object_evidence(
        project_dir=project_dir,
        evidence_units_path=evidence_units,
    )
    if build_summary is not None:
        counts = _mapping(build_summary.get("counts"))
        return {
            "status": "built",
            "counts": _public_build_counts(counts),
            "alignment_status_counts": _mapping(build_summary.get("alignment_status_counts")),
            "source_quality_counts": {
                "units_with_visual_state": int(counts.get("units_with_visual_state") or 0),
                "units_with_visual_entity": int(counts.get("units_with_visual_entity") or 0),
                "units_with_vlm_entity": int(counts.get("units_with_vlm_entity") or 0),
                "units_with_verified_link": int(counts.get("units_with_verified_link") or 0),
                "units_with_detected_text": int(counts.get("units_with_detected_text") or 0),
                "units_with_visual_description": int(counts.get("units_with_visual_description") or 0),
            },
            "link_counts": {
                "candidate_links": int(counts.get("candidate_links") or 0),
                "verified_links": int(counts.get("verified_links") or 0),
                "timestamp_fallback_links": int(counts.get("timestamp_fallback_links") or 0),
            },
            "visual_state_coverage": _public_visual_state_coverage(
                build_summary.get("visual_state_coverage")
            ),
            "verified_alignment_note": _verified_alignment_note(counts),
            "vlm_object_evidence_coverage": vlm_object_evidence_coverage,
        }

    rows = _evidence_unit_rows(project_dir=project_dir, evidence_units=evidence_units)
    counts = Counter()
    source_quality = Counter()
    for row in rows:
        counts[str(row.get("alignment_status") or "unknown")] += 1
        quality = _mapping(row.get("source_quality"))
        for key in ("has_visual_state", "has_visual_entity", "has_vlm_entity", "has_verified_link"):
            if quality.get(key) is True:
                source_quality[key] += 1
        for key in ("has_detected_text", "has_visual_description"):
            if quality.get(key) is True:
                source_quality[key] += 1
        source_quality["candidate_links"] += int(quality.get("candidate_link_count") or 0)
        source_quality["verified_links"] += int(quality.get("verified_link_count") or 0)
        source_quality["timestamp_fallback_links"] += int(quality.get("timestamp_fallback_link_count") or 0)
    visual_state_rows = _visual_state_rows(project_dir=project_dir, visual_states=visual_states)
    return {
        "status": "loaded_existing",
        "counts": {"evidence_units_total": len(rows)},
        "alignment_status_counts": dict(counts),
        "source_quality_counts": {
            "units_with_visual_state": source_quality["has_visual_state"],
            "units_with_visual_entity": source_quality["has_visual_entity"],
            "units_with_vlm_entity": source_quality["has_vlm_entity"],
            "units_with_verified_link": source_quality["has_verified_link"],
            "units_with_detected_text": source_quality["has_detected_text"],
            "units_with_visual_description": source_quality["has_visual_description"],
        },
        "link_counts": {
            "candidate_links": source_quality["candidate_links"],
            "verified_links": source_quality["verified_links"],
            "timestamp_fallback_links": source_quality["timestamp_fallback_links"],
        },
        "visual_state_coverage": _loaded_visual_state_coverage(
            evidence_units=rows,
            visual_states=visual_state_rows,
        ),
        "verified_alignment_note": _verified_alignment_note(source_quality),
        "vlm_object_evidence_coverage": vlm_object_evidence_coverage,
    }


def _public_candidate(candidate: dict[str, Any], *, query_text: str | None = None) -> dict[str, Any] | None:
    if not candidate:
        return None
    source_quality = _mapping(candidate.get("source_quality"))
    return {
        "rank": candidate.get("rank"),
        "ref": _id_ref(_optional_str(candidate.get("evidence_unit_id")), prefix="evu"),
        "target_ref": _id_ref(_optional_str(candidate.get("target_segment_id")), prefix="seg"),
        "source_segment_ref_count": len(_string_list(candidate.get("source_segment_ids"))),
        "timestamp_available": _optional_float(candidate.get("start_time")) is not None,
        "visual_state_count": len(_string_list(candidate.get("visual_state_ids"))),
        "visual_entity_count": len(_string_list(candidate.get("visual_entity_ids"))),
        "verified_entity_link_count": len(_string_list(candidate.get("verified_entity_link_ids"))),
        "candidate_entity_link_count": len(_string_list(candidate.get("candidate_entity_link_ids"))),
        "alignment_status": _alignment_status(candidate.get("alignment_status")),
        "source_quality": {
            "has_visual_state": bool(source_quality.get("has_visual_state")),
            "has_visual_entity": bool(source_quality.get("has_visual_entity")),
            "has_vlm_entity": bool(source_quality.get("has_vlm_entity")),
            "has_verified_link": bool(source_quality.get("has_verified_link")),
            "has_timestamp_fallback_link": bool(source_quality.get("has_timestamp_fallback_link")),
            "has_detected_text": bool(source_quality.get("has_detected_text")),
            "has_visual_description": bool(source_quality.get("has_visual_description")),
            "visual_state_detected_text_count": int(source_quality.get("visual_state_detected_text_count") or 0),
            "visual_entity_detected_text_count": int(source_quality.get("visual_entity_detected_text_count") or 0),
            "visual_description_count": int(source_quality.get("visual_description_count") or 0),
        },
        "rag_fields": {
            "evidence_text_available": bool(candidate.get("evidence_text")),
            "semantic_text_available": bool(candidate.get("semantic_text")),
        },
        "content_coverage": _candidate_content_coverage(candidate),
        "query_term_coverage": _query_term_coverage(
            query_text=query_text or "",
            candidate=candidate,
        ),
    }


def _target_diagnostics(
    *,
    candidates: list[dict[str, Any]],
    expected_segment_ids: set[str],
    search_depth: int,
    query_text: str,
) -> dict[str, Any]:
    if not expected_segment_ids:
        return {
            "target_configured": False,
            "target_found_in_top_k": None,
            "target_rank_bucket": "not_configured",
            "target_search_depth": search_depth,
        }
    target_candidate = None
    target_rank = None
    for rank, candidate in enumerate(candidates, start=1):
        if _candidate_matches_expected(candidate, expected_segment_ids):
            target_candidate = candidate
            target_rank = rank
            break
    top_candidate = candidates[0] if candidates else {}
    found = target_candidate is not None and target_rank is not None
    return {
        "target_configured": True,
        "target_found_in_top_k": found,
        "target_rank": target_rank,
        "target_rank_bucket": _target_rank_bucket(target_rank),
        "target_search_depth": search_depth,
        "target_evidence_unit_ref": (
            _id_ref(_optional_str(target_candidate.get("evidence_unit_id")), prefix="evu")
            if target_candidate
            else None
        ),
        "target_evidence_unit_quality": _candidate_quality_summary(target_candidate or {}),
        "target_content_coverage": _candidate_content_coverage(target_candidate or {}),
        "target_query_term_coverage": _query_term_coverage(
            query_text=query_text,
            candidate=target_candidate or {},
        ),
        "top_vs_target_quality_delta": _candidate_quality_delta(
            top_candidate=top_candidate,
            target_candidate=target_candidate or {},
        ),
        "top_vs_target_content_delta": _candidate_content_delta(
            top_candidate=top_candidate,
            target_candidate=target_candidate or {},
            query_text=query_text,
        ),
    }


def _target_rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found"
    if rank <= 1:
        return "top1"
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "top10"
    if rank <= 50:
        return "top50"
    return "not_found"


def _candidate_quality_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    source_quality = _mapping(candidate.get("source_quality"))
    return {
        "alignment_status": _alignment_status(candidate.get("alignment_status")),
        "visual_state_count": len(_string_list(candidate.get("visual_state_ids"))),
        "visual_entity_count": len(_string_list(candidate.get("visual_entity_ids"))),
        "verified_entity_link_count": len(_string_list(candidate.get("verified_entity_link_ids"))),
        "candidate_entity_link_count": len(_string_list(candidate.get("candidate_entity_link_ids"))),
        "has_visual_state": bool(source_quality.get("has_visual_state")),
        "has_visual_entity": bool(source_quality.get("has_visual_entity")),
        "has_vlm_entity": bool(source_quality.get("has_vlm_entity")),
        "has_verified_link": bool(source_quality.get("has_verified_link")),
        "has_timestamp_fallback_link": bool(source_quality.get("has_timestamp_fallback_link")),
        "rag_fields": {
            "evidence_text_available": bool(candidate.get("evidence_text")),
            "semantic_text_available": bool(candidate.get("semantic_text")),
        },
        "content_coverage": _candidate_content_coverage(candidate),
    }


def _candidate_content_coverage(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    evidence_text = _text_for_coverage(candidate.get("evidence_text"))
    semantic_text = _text_for_coverage(candidate.get("semantic_text"))
    transcript_text = _text_for_coverage(candidate.get("transcript_window_text"))
    source_quality = _mapping(candidate.get("source_quality"))
    return {
        "evidence_text_char_count": len(evidence_text),
        "semantic_text_char_count": len(semantic_text),
        "transcript_window_char_count": len(transcript_text),
        "evidence_text_bucket": _char_count_bucket(len(evidence_text)),
        "semantic_text_bucket": _char_count_bucket(len(semantic_text)),
        "transcript_window_bucket": _char_count_bucket(len(transcript_text)),
        "visual_state_count": len(_string_list(candidate.get("visual_state_ids"))),
        "visual_entity_count": len(_string_list(candidate.get("visual_entity_ids"))),
        "candidate_entity_link_count": len(_string_list(candidate.get("candidate_entity_link_ids"))),
        "verified_entity_link_count": len(_string_list(candidate.get("verified_entity_link_ids"))),
        "timestamp_fallback_link_count": int(source_quality.get("timestamp_fallback_link_count") or 0),
        "visual_state_detected_text_count": int(source_quality.get("visual_state_detected_text_count") or 0),
        "visual_entity_detected_text_count": int(source_quality.get("visual_entity_detected_text_count") or 0),
        "visual_description_count": int(source_quality.get("visual_description_count") or 0),
    }


def _query_term_coverage(*, query_text: str, candidate: dict[str, Any]) -> dict[str, Any]:
    query_terms = _coverage_terms(query_text)
    evidence_terms = set(_coverage_terms(_text_for_coverage(candidate.get("evidence_text"))))
    semantic_terms = set(_coverage_terms(_text_for_coverage(candidate.get("semantic_text"))))
    transcript_terms = set(_coverage_terms(_text_for_coverage(candidate.get("transcript_window_text"))))
    combined_terms = evidence_terms | semantic_terms | transcript_terms
    query_term_count = len(query_terms)
    evidence_matches = len(query_terms & evidence_terms)
    semantic_matches = len(query_terms & semantic_terms)
    transcript_matches = len(query_terms & transcript_terms)
    combined_matches = len(query_terms & combined_terms)
    return {
        "query_term_count": query_term_count,
        "evidence_text_match_count": evidence_matches,
        "semantic_text_match_count": semantic_matches,
        "transcript_window_match_count": transcript_matches,
        "combined_match_count": combined_matches,
        "combined_match_ratio": _ratio(combined_matches, query_term_count),
        "combined_match_bucket": _ratio_bucket(combined_matches, query_term_count),
    }


def _candidate_content_delta(
    *,
    top_candidate: dict[str, Any],
    target_candidate: dict[str, Any],
    query_text: str,
) -> dict[str, Any]:
    if not top_candidate or not target_candidate:
        return {"status": "not_available"}
    top_content = _candidate_content_coverage(top_candidate)
    target_content = _candidate_content_coverage(target_candidate)
    top_terms = _query_term_coverage(query_text=query_text, candidate=top_candidate)
    target_terms = _query_term_coverage(query_text=query_text, candidate=target_candidate)
    return {
        "status": "available",
        "same_evidence_unit": top_candidate.get("evidence_unit_id") == target_candidate.get("evidence_unit_id"),
        "evidence_text_char_count_delta": int(top_content.get("evidence_text_char_count") or 0)
        - int(target_content.get("evidence_text_char_count") or 0),
        "semantic_text_char_count_delta": int(top_content.get("semantic_text_char_count") or 0)
        - int(target_content.get("semantic_text_char_count") or 0),
        "transcript_window_char_count_delta": int(top_content.get("transcript_window_char_count") or 0)
        - int(target_content.get("transcript_window_char_count") or 0),
        "combined_query_term_match_count_delta": int(top_terms.get("combined_match_count") or 0)
        - int(target_terms.get("combined_match_count") or 0),
        "top_query_term_match_bucket": top_terms.get("combined_match_bucket"),
        "target_query_term_match_bucket": target_terms.get("combined_match_bucket"),
        "target_has_more_query_term_matches": int(top_terms.get("combined_match_count") or 0)
        < int(target_terms.get("combined_match_count") or 0),
        "target_has_longer_semantic_text": int(top_content.get("semantic_text_char_count") or 0)
        < int(target_content.get("semantic_text_char_count") or 0),
        "public_note": (
            "Query-term diagnostics expose only counts, ratios, and buckets; raw query terms "
            "and raw evidence text remain redacted."
        ),
    }


def _candidate_quality_delta(
    *,
    top_candidate: dict[str, Any],
    target_candidate: dict[str, Any],
) -> dict[str, Any]:
    if not top_candidate or not target_candidate:
        return {"status": "not_available"}
    top_quality = _candidate_quality_summary(top_candidate)
    target_quality = _candidate_quality_summary(target_candidate)
    visual_entity_delta = int(top_quality.get("visual_entity_count") or 0) - int(
        target_quality.get("visual_entity_count") or 0
    )
    visual_state_delta = int(top_quality.get("visual_state_count") or 0) - int(
        target_quality.get("visual_state_count") or 0
    )
    verified_link_delta = int(top_quality.get("verified_entity_link_count") or 0) - int(
        target_quality.get("verified_entity_link_count") or 0
    )
    candidate_link_delta = int(top_quality.get("candidate_entity_link_count") or 0) - int(
        target_quality.get("candidate_entity_link_count") or 0
    )
    return {
        "status": "available",
        "same_evidence_unit": top_candidate.get("evidence_unit_id") == target_candidate.get("evidence_unit_id"),
        "alignment_status_changed": top_quality.get("alignment_status") != target_quality.get("alignment_status"),
        "top_alignment_status": top_quality.get("alignment_status"),
        "target_alignment_status": target_quality.get("alignment_status"),
        "visual_entity_count_delta": visual_entity_delta,
        "visual_state_count_delta": visual_state_delta,
        "verified_entity_link_count_delta": verified_link_delta,
        "candidate_entity_link_count_delta": candidate_link_delta,
        "top_has_more_visual_entities": visual_entity_delta > 0,
        "target_has_more_visual_entities": visual_entity_delta < 0,
        "top_has_verified_link_only": bool(top_quality.get("has_verified_link"))
        and not bool(target_quality.get("has_verified_link")),
        "target_has_verified_link_only": bool(target_quality.get("has_verified_link"))
        and not bool(top_quality.get("has_verified_link")),
        "timestamp_fallback_flag_changed": bool(top_quality.get("has_timestamp_fallback_link"))
        != bool(target_quality.get("has_timestamp_fallback_link")),
        "verified_alignment_note": (
            "has_verified_link only reflects explicit verified links; timestamp fallback "
            "is never counted as verified object alignment"
        ),
    }


def _quality_rerank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for fallback_rank, candidate in enumerate(candidates, start=1):
        original_rank = _optional_int(candidate.get("rank")) or fallback_rank
        breakdown = _quality_rerank_breakdown(candidate, original_rank=original_rank)
        updated = dict(candidate)
        updated["quality_rerank"] = {
            "strategy": "deterministic_quality_v1",
            "score": breakdown["score"],
            "original_rank": original_rank,
            "components": breakdown["components"],
            "flags": breakdown["flags"],
            "explanation": (
                "Deterministic smoke-only quality score from rank/score, text availability, "
                "visual evidence counts, VLM/verified-link flags, and transcript/timestamp fallback flags."
            ),
        }
        scored.append((float(breakdown["score"]), original_rank, updated))

    reranked: list[dict[str, Any]] = []
    for new_rank, (_, _, candidate) in enumerate(
        sorted(scored, key=lambda item: (-item[0], item[1])),
        start=1,
    ):
        updated = dict(candidate)
        rerank = dict(_mapping(updated.get("quality_rerank")))
        rerank["reranked_rank"] = new_rank
        updated["quality_rerank"] = rerank
        updated["rank"] = new_rank
        reranked.append(updated)
    return reranked


def _quality_rerank_breakdown(candidate: dict[str, Any], *, original_rank: int) -> dict[str, Any]:
    source_quality = _mapping(candidate.get("source_quality"))
    alignment_status = _alignment_status(candidate.get("alignment_status"))
    visual_state_count = len(_string_list(candidate.get("visual_state_ids")))
    visual_entity_count = len(_string_list(candidate.get("visual_entity_ids")))
    candidate_link_count = len(_string_list(candidate.get("candidate_entity_link_ids")))
    verified_link_count = len(_string_list(candidate.get("verified_entity_link_ids")))
    evidence_text_available = bool(candidate.get("evidence_text"))
    semantic_text_available = bool(candidate.get("semantic_text"))
    meili_score = _optional_float(candidate.get("score"))
    if meili_score is None:
        meili_score = _optional_float(candidate.get("_rankingScore"))
    has_vlm_entity = bool(source_quality.get("has_vlm_entity"))
    has_verified_link = bool(source_quality.get("has_verified_link")) or verified_link_count > 0
    has_timestamp_fallback = bool(source_quality.get("has_timestamp_fallback_link"))
    transcript_only = alignment_status == "transcript_only"
    components = {
        "rank_preservation": round(1.0 / max(original_rank, 1), 6),
        "meili_score": round(max(0.0, min(float(meili_score or 0.0), 1.0)), 6),
        "evidence_text_available": 0.35 if evidence_text_available else 0.0,
        "semantic_text_available": 0.25 if semantic_text_available else 0.0,
        "visual_state_count": round(min(visual_state_count, 3) * 0.18, 6),
        "visual_entity_count": round(min(visual_entity_count, 10) * 0.06, 6),
        "candidate_entity_link_count": round(min(candidate_link_count, 5) * 0.04, 6),
        "vlm_entity_presence": 0.7 if has_vlm_entity else 0.0,
        "verified_link_presence": 1.0 if has_verified_link else 0.0,
        "transcript_only_penalty": -0.65 if transcript_only else 0.0,
        "timestamp_fallback_penalty": -0.15 if has_timestamp_fallback else 0.0,
    }
    score = round(sum(float(value) for value in components.values()), 6)
    return {
        "score": score,
        "components": components,
        "flags": {
            "has_vlm_entity": has_vlm_entity,
            "has_verified_link": has_verified_link,
            "has_timestamp_fallback_link": has_timestamp_fallback,
            "transcript_only": transcript_only,
            "evidence_text_available": evidence_text_available,
            "semantic_text_available": semantic_text_available,
        },
    }


def _rerank_diagnostics(
    *,
    enabled: bool,
    base_top_candidate: dict[str, Any],
    reranked_candidates: list[dict[str, Any]],
    expected_segment_ids: set[str],
    base_target_diagnostics: dict[str, Any],
    reranked_target_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
        }
    reranked_top = reranked_candidates[0] if reranked_candidates else {}
    rerank = _mapping(reranked_top.get("quality_rerank"))
    base_top_match = _candidate_matches_expected(base_top_candidate, expected_segment_ids)
    reranked_top_match = _candidate_matches_expected(reranked_top, expected_segment_ids)
    return {
        "enabled": True,
        "status": "computed",
        "strategy": "deterministic_quality_v1",
        "candidate_count": len(reranked_candidates),
        "base_top_expected_match": base_top_match,
        "reranked_top_expected_match": reranked_top_match,
        "top_changed": base_top_candidate.get("evidence_unit_id") != reranked_top.get("evidence_unit_id"),
        "base_target_rank": base_target_diagnostics.get("target_rank"),
        "base_target_rank_bucket": base_target_diagnostics.get("target_rank_bucket"),
        "reranked_target_rank": reranked_target_diagnostics.get("target_rank"),
        "reranked_target_rank_bucket": reranked_target_diagnostics.get("target_rank_bucket"),
        "reranked_top": {
            "ref": _id_ref(_optional_str(reranked_top.get("evidence_unit_id")), prefix="evu"),
            "original_rank": rerank.get("original_rank"),
            "reranked_rank": rerank.get("reranked_rank"),
            "score": rerank.get("score"),
            "score_components": _mapping(rerank.get("components")),
            "flags": _mapping(rerank.get("flags")),
        },
        "public_note": (
            "Quality rerank is smoke-only and never uses expected target labels. "
            "Expected segments are used only for diagnostics after reranking."
        ),
    }


def _top_hit_memo(
    *,
    candidate: dict[str, Any],
    expected_match: bool | None,
    expected_configured: bool,
    target_rank_bucket: str = "",
) -> dict[str, Any]:
    public_candidate = _public_candidate(candidate) or {}
    return {
        "expected_target_configured": expected_configured,
        "top_expected_match": expected_match,
        "target_rank_bucket": target_rank_bucket or "unknown",
        "public_note": _evidence_unit_note(
            expected_match=expected_match,
            alignment_status=str(public_candidate.get("alignment_status") or ""),
            has_verified_link=bool(
                _mapping(public_candidate.get("source_quality")).get("has_verified_link")
            ),
        ),
    }


def _summary_payload(
    *,
    run_id: str,
    manifest_path: Path,
    suites: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    health: dict[str, Any],
    dry_run: bool,
    quality_rerank: bool,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "unknown") for row in query_rows)
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest_ref": _short_hash(str(manifest_path.expanduser().resolve())),
        "privacy": _privacy_policy(),
        "dry_run": dry_run,
        "meilisearch": {
            "available": health["available"],
            "status": health["status"],
            "skip_reason": health.get("skip_reason"),
        },
        "suite_count": len(suites),
        "query_count": len(query_rows),
        "query_status_counts": dict(status_counts),
        "suites": suites,
        "rag_input_inspection": _rag_input_inspection(query_rows, {}),
        "target_rank_diagnostics": _target_rank_inspection(query_rows),
        "rerank_diagnostics": _rerank_inspection(query_rows),
        "quality_rerank_requested": quality_rerank,
        "object_alignment_note": (
            "timestamp-only overlap is reported as candidate/fallback evidence only; "
            "it is not counted as verified object alignment"
        ),
        "visual_state_note": (
            "visual_state intervals are candidate support from sampled-frame midpoint "
            "coverage or an explicit artifact; interval overlap is not verified object alignment"
        ),
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Unit Retrieval Smoke",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "This public-safe report redacts raw query text, transcripts, local paths, and raw IDs.",
        "Timestamp-only overlap is not counted as verified object alignment.",
        "Visual-state interval overlap is reported as candidate support, not verified object alignment.",
        "",
        "## Meilisearch",
        "",
        f"- available: `{payload['meilisearch']['available']}`",
        f"- status: `{payload['meilisearch']['status']}`",
    ]
    if payload["meilisearch"].get("skip_reason"):
        lines.append(f"- skip reason: `{payload['meilisearch']['skip_reason']}`")
    lines.extend(["", "## Suites", ""])
    for suite in payload.get("suites", []):
        build = suite.get("build", {})
        index = suite.get("index", {})
        vlm_coverage = _mapping(build.get("vlm_object_evidence_coverage"))
        entity_coverage = _mapping(vlm_coverage.get("visual_entity_coverage"))
        unit_coverage = _mapping(vlm_coverage.get("evidence_unit_coverage"))
        visual_coverage = _mapping(build.get("visual_state_coverage"))
        interval_summary = _mapping(visual_coverage.get("interval_duration_seconds"))
        coverage_gate = _mapping(visual_coverage.get("coverage_gate"))
        lines.extend(
            [
                f"### {suite.get('suite_id')}",
                "",
                f"- evidence units: `{_mapping(build.get('counts')).get('evidence_units_total', 0)}`",
                f"- alignment statuses: `{json.dumps(build.get('alignment_status_counts', {}), sort_keys=True)}`",
                f"- link counts: `{json.dumps(build.get('link_counts', {}), sort_keys=True)}`",
                f"- visual state source: `{visual_coverage.get('source') or 'unknown'}`",
                f"- visual state coverage: `{visual_coverage.get('evidence_units_with_visual_state', 0)}`/`{visual_coverage.get('evidence_units_total', 0)}`",
                f"- visual state duration buckets: `{json.dumps(interval_summary.get('buckets', {}), sort_keys=True)}`",
                f"- visual state gate: `{coverage_gate.get('status') or 'not_configured'}`",
                f"- index status: `{index.get('status')}`",
                f"- RAG input inspectable top hits: `{suite.get('rag_input_inspection', {}).get('inspectable_top_hit_count', 0)}`",
                f"- target rank buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('rank_bucket_counts', {}), sort_keys=True)}`",
                f"- found target query-term buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('found_target_query_term_bucket_counts', {}), sort_keys=True)}`",
                f"- top-vs-target coverage flags: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('quality_delta_flag_counts', {}), sort_keys=True)}`",
                f"- reranked target rank buckets: `{json.dumps(suite.get('rerank_diagnostics', {}).get('reranked_target_rank_bucket_counts', {}), sort_keys=True)}`",
                f"- reranked top-hit matches: `{suite.get('rerank_diagnostics', {}).get('reranked_top_match_count', 0)}`",
                f"- VLM evidence status: `{vlm_coverage.get('status')}`",
                f"- paper-quality VLM entities: `{entity_coverage.get('paper_quality_vlm_entity_count', 0)}`",
                f"- OCR-only entities: `{entity_coverage.get('ocr_only_entity_count', 0)}`",
                f"- units with VLM entity: `{unit_coverage.get('units_with_vlm_entity', 0)}`",
                f"- units with visual description: `{unit_coverage.get('units_with_visual_description', 0)}`",
                f"- units with detected text: `{unit_coverage.get('units_with_detected_text', 0)}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _rag_input_inspection(query_rows: list[dict[str, Any]], artifact_summary: dict[str, Any]) -> dict[str, Any]:
    queried = [row for row in query_rows if row.get("status") == "queried"]
    inspectable = [row for row in queried if row.get("rag_input_inspectable") is True]
    top_with_visual = [
        row
        for row in queried
        if _mapping(_mapping(row.get("top_evidence_unit")).get("source_quality")).get("has_visual_state")
    ]
    return {
        "query_count": len(query_rows),
        "queried_count": len(queried),
        "inspectable_top_hit_count": len(inspectable),
        "top_hits_with_visual_state_count": len(top_with_visual),
        "artifact_evidence_units_total": _mapping(artifact_summary.get("counts")).get("evidence_units_total"),
        "public_note": (
            "Inspectable means the top hit exposes evidence unit metadata plus evidence_text/"
            "semantic_text availability flags; text content remains redacted from this report."
        ),
    }


def _target_rank_inspection(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [_mapping(row.get("target_diagnostics")) for row in query_rows]
    configured = [diag for diag in diagnostics if diag.get("target_configured") is True]
    found = [diag for diag in configured if diag.get("target_found_in_top_k") is True]
    buckets = Counter(
        str(diag.get("target_rank_bucket") or "unknown")
        for diag in configured
    )
    target_quality = Counter()
    top_better_counts = Counter()
    for diag in found:
        quality = _mapping(diag.get("target_evidence_unit_quality"))
        if quality.get("has_visual_state") is True:
            target_quality["has_visual_state"] += 1
        if quality.get("has_visual_entity") is True:
            target_quality["has_visual_entity"] += 1
        if quality.get("has_vlm_entity") is True:
            target_quality["has_vlm_entity"] += 1
        if quality.get("has_verified_link") is True:
            target_quality["has_verified_link"] += 1
        delta = _mapping(diag.get("top_vs_target_quality_delta"))
        for key in (
            "top_has_more_visual_entities",
            "target_has_more_visual_entities",
            "top_has_verified_link_only",
            "target_has_verified_link_only",
        ):
            if delta.get(key) is True:
                top_better_counts[key] += 1
        content_delta = _mapping(diag.get("top_vs_target_content_delta"))
        for key in (
            "target_has_more_query_term_matches",
            "target_has_longer_semantic_text",
        ):
            if content_delta.get(key) is True:
                top_better_counts[key] += 1
    query_term_buckets = Counter(
        str(_mapping(diag.get("target_query_term_coverage")).get("combined_match_bucket") or "unknown")
        for diag in found
    )
    return {
        "target_configured_count": len(configured),
        "target_found_in_top_k_count": len(found),
        "rank_bucket_counts": dict(buckets),
        "found_target_quality_counts": dict(target_quality),
        "quality_delta_flag_counts": dict(top_better_counts),
        "found_target_query_term_bucket_counts": dict(query_term_buckets),
        "public_note": (
            "Target rank diagnostics use only configured expected segment IDs and "
            "hashed/count/flag evidence-unit metadata; raw query text and raw IDs remain redacted."
        ),
    }


def _rerank_inspection(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [_mapping(row.get("rerank_diagnostics")) for row in query_rows]
    enabled = [diag for diag in diagnostics if diag.get("enabled") is True]
    computed = [diag for diag in enabled if diag.get("status") == "computed"]
    base_buckets = Counter(
        str(diag.get("base_target_rank_bucket") or "unknown")
        for diag in computed
    )
    reranked_buckets = Counter(
        str(diag.get("reranked_target_rank_bucket") or "unknown")
        for diag in computed
    )
    component_presence = Counter()
    top_changed_count = 0
    base_top_match_count = 0
    reranked_top_match_count = 0
    for diag in computed:
        if diag.get("top_changed") is True:
            top_changed_count += 1
        if diag.get("base_top_expected_match") is True:
            base_top_match_count += 1
        if diag.get("reranked_top_expected_match") is True:
            reranked_top_match_count += 1
        components = _mapping(_mapping(diag.get("reranked_top")).get("score_components"))
        for key, value in components.items():
            if _optional_float(value):
                component_presence[str(key)] += 1
    return {
        "enabled_query_count": len(enabled),
        "computed_query_count": len(computed),
        "top_changed_count": top_changed_count,
        "base_top_match_count": base_top_match_count,
        "reranked_top_match_count": reranked_top_match_count,
        "base_target_rank_bucket_counts": dict(base_buckets),
        "reranked_target_rank_bucket_counts": dict(reranked_buckets),
        "reranked_top_score_component_presence_counts": dict(component_presence),
        "public_note": (
            "Rerank diagnostics compare base and deterministic quality-aware ordering using "
            "only candidate metadata. Expected targets are evaluation-only labels."
        ),
    }


def _meili_health(client: EvidenceUnitSmokeClient) -> dict[str, Any]:
    try:
        response = client.health()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "status": "unavailable",
            "skip_reason": _exception_reason(exc),
        }
    status = str(response.get("status") or "").lower()
    available = status == "available" or bool(response.get("available") is True)
    return {
        "available": available,
        "status": status or "unknown",
        "skip_reason": None if available else f"health_status:{status or 'unknown'}",
    }


def _public_index_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "indexed",
        "indexed_documents": int(summary.get("indexed_documents") or 0),
        "indexed_batches": int(summary.get("indexed_batches") or 0),
        "alignment_status_counts": _mapping(summary.get("alignment_status_counts")),
        "source_quality_counts": _mapping(summary.get("source_quality_counts")),
        "settings_profile": str(summary.get("settings_profile") or ""),
        "settings_hash": str(summary.get("settings_hash") or ""),
    }


def _public_build_counts(counts: dict[str, Any]) -> dict[str, int]:
    keys = [
        "segments_total",
        "frames_total",
        "visual_entities_total",
        "entity_links_total",
        "evidence_units_total",
    ]
    return {key: int(counts.get(key) or 0) for key in keys}


def _public_visual_state_coverage(value: Any) -> dict[str, Any]:
    coverage = _mapping(value)
    gate = _mapping(coverage.get("coverage_gate"))
    interval = _mapping(coverage.get("interval_duration_seconds"))
    return {
        "schema_version": coverage.get("schema_version"),
        "source": coverage.get("source"),
        "visual_states_total": int(coverage.get("visual_states_total") or 0),
        "evidence_units_total": int(coverage.get("evidence_units_total") or 0),
        "evidence_units_with_visual_state": int(
            coverage.get("evidence_units_with_visual_state") or 0
        ),
        "transcript_only_units": int(coverage.get("transcript_only_units") or 0),
        "unit_coverage_ratio": coverage.get("unit_coverage_ratio"),
        "interval_duration_seconds": {
            "count": int(interval.get("count") or 0),
            "min": interval.get("min"),
            "max": interval.get("max"),
            "mean": interval.get("mean"),
            "buckets": dict(_mapping(interval.get("buckets"))),
        },
        "coverage_gate": {
            "status": gate.get("status"),
            "checked": bool(gate.get("checked")),
            "thresholds": dict(_mapping(gate.get("thresholds"))),
            "failure_count": int(gate.get("failure_count") or 0),
        },
        "public_note": coverage.get("public_note"),
    }


def _loaded_visual_state_coverage(
    *,
    evidence_units: list[dict[str, Any]],
    visual_states: list[dict[str, Any]],
) -> dict[str, Any]:
    units_with_visual_state = sum(
        1
        for row in evidence_units
        if _mapping(row.get("source_quality")).get("has_visual_state") is True
        or bool(_string_list(row.get("visual_state_ids")))
    )
    total = len(evidence_units)
    return {
        "schema_version": "oarag-visual-state-coverage-v1",
        "source": "loaded_existing",
        "visual_states_total": len(visual_states),
        "evidence_units_total": total,
        "evidence_units_with_visual_state": units_with_visual_state,
        "transcript_only_units": sum(
            1 for row in evidence_units if row.get("alignment_status") == "transcript_only"
        ),
        "unit_coverage_ratio": round(units_with_visual_state / total, 6) if total else None,
        "interval_duration_seconds": _visual_state_interval_summary(visual_states),
        "coverage_gate": {
            "status": "not_configured",
            "checked": False,
            "thresholds": {},
            "failure_count": 0,
        },
        "public_note": (
            "Visual states are candidate interval support; timestamp-only overlap is not "
            "counted as verified object alignment."
        ),
    }


def _visual_state_rows(*, project_dir: Path, visual_states: Path | None) -> list[dict[str, Any]]:
    if visual_states is None:
        return []
    path = visual_states
    if not path.is_absolute():
        path = project_dir / path
    if not path.exists():
        return []
    return list(iter_jsonl_documents(path))


def _visual_state_interval_summary(visual_states: list[dict[str, Any]]) -> dict[str, Any]:
    durations = []
    for state in visual_states:
        start_time = _optional_float(_mapping(state).get("valid_start_time"))
        end_time = _optional_float(_mapping(state).get("valid_end_time"))
        if start_time is None or end_time is None:
            continue
        durations.append(abs(end_time - start_time))
    buckets = {"0-5s": 0, "5-15s": 0, "15-30s": 0, "30-60s": 0, "60s+": 0}
    for duration in durations:
        if duration < 5.0:
            buckets["0-5s"] += 1
        elif duration < 15.0:
            buckets["5-15s"] += 1
        elif duration < 30.0:
            buckets["15-30s"] += 1
        elif duration < 60.0:
            buckets["30-60s"] += 1
        else:
            buckets["60s+"] += 1
    if not durations:
        return {"count": 0, "min": None, "max": None, "mean": None, "buckets": buckets}
    return {
        "count": len(durations),
        "min": round(min(durations), 3),
        "max": round(max(durations), 3),
        "mean": round(sum(durations) / len(durations), 3),
        "buckets": buckets,
    }


def _evidence_unit_rows(*, project_dir: Path, evidence_units: Path | None) -> list[dict[str, Any]]:
    path = evidence_units
    if path is None:
        path = Path("segments") / "evidence_units.jsonl"
    if not path.is_absolute():
        path = project_dir / path
    return list(iter_jsonl_documents(path))


def _path_from_build_summary(summary: dict[str, Any]) -> Path | None:
    paths = _mapping(summary.get("paths"))
    value = paths.get("evidence_units")
    return Path(str(value)) if value else None


def _candidate_matches_expected(candidate: dict[str, Any], expected_segment_ids: set[str]) -> bool | None:
    if not expected_segment_ids:
        return None
    target = _optional_str(candidate.get("target_segment_id"))
    source_ids = set(_string_list(candidate.get("source_segment_ids")))
    return bool((target and target in expected_segment_ids) or (source_ids & expected_segment_ids))


def _candidate_is_rag_inspectable(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate
        and candidate.get("evidence_unit_id")
        and (candidate.get("evidence_text") or candidate.get("semantic_text"))
        and candidate.get("alignment_status")
        and isinstance(candidate.get("source_quality"), dict)
    )


def _verified_alignment_note(counts: dict[str, Any]) -> str:
    verified = int(counts.get("verified_links") or counts.get("has_verified_link") or 0)
    fallback = int(counts.get("timestamp_fallback_links") or 0)
    if verified == 0 and fallback > 0:
        return "timestamp fallback links present, but verified object alignment count is zero"
    return "verified links are counted only from explicit verified link fields"


def _text_for_coverage(value: Any) -> str:
    if value in (None, ""):
        return ""
    return " ".join(str(value).split())


def _coverage_terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if len(term) >= 2
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _ratio_bucket(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "no_query_terms"
    ratio = numerator / denominator
    if ratio <= 0:
        return "none"
    if ratio < 0.25:
        return "low"
    if ratio < 0.5:
        return "medium"
    if ratio < 0.75:
        return "high"
    return "very_high"


def _char_count_bucket(count: int) -> str:
    if count <= 0:
        return "empty"
    if count < 120:
        return "short"
    if count < 600:
        return "medium"
    return "long"


def _evidence_unit_note(
    *,
    expected_match: bool | None,
    alignment_status: str,
    has_verified_link: bool,
) -> str:
    match_word = "unknown"
    if expected_match is True:
        match_word = "top hit matches configured target"
    elif expected_match is False:
        match_word = "top hit misses configured target"
    if alignment_status == "verified" and has_verified_link:
        return f"{match_word}; top hit has explicit verified link"
    if alignment_status == "candidate":
        return f"{match_word}; top hit is candidate-level, not verified object alignment"
    if alignment_status == "transcript_only":
        return f"{match_word}; top hit is transcript-only"
    return f"{match_word}; alignment status {alignment_status or 'unknown'}"


def _baseline_note(expected_match: bool | None) -> str:
    if expected_match is True:
        return "segment baseline top hit matches configured target"
    if expected_match is False:
        return "segment baseline top hit misses configured target"
    return "segment baseline top-hit target match unknown"


def _project_dir_from_suite(*, suite: dict[str, Any], base_dir: Path, repo_root: Path) -> Path:
    if suite.get("project_dir"):
        path = Path(str(suite["project_dir"])).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()
    project_id = _optional_str(suite.get("project_id"))
    if not project_id:
        raise ValueError("suite requires either project_dir or project_id")
    return (repo_root / "artifacts" / "projects" / project_id).resolve()


def _read_queries(*, base_dir: Path, suite: dict[str, Any]) -> list[dict[str, Any]]:
    if suite.get("queries_file"):
        path = Path(str(suite["queries_file"])).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data = _read_json(path)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        rows = data.get("queries") if isinstance(data, dict) else None
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        raise ValueError(f"queries_file must contain a list or object with queries: {path}")
    queries = suite.get("queries") or []
    if not isinstance(queries, list):
        raise ValueError("suite queries must be a list")
    return [query for query in queries if isinstance(query, dict)]


def _resolve_output_dir(
    *,
    manifest: dict[str, Any],
    output_dir: Path | None,
    base_dir: Path,
    run_id: str,
) -> Path:
    configured = output_dir or _optional_path(manifest.get("output_dir"))
    if configured is None:
        configured = DEFAULT_OUTPUT_ROOT / run_id
    if not configured.is_absolute():
        configured = base_dir / configured
    return configured.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _default_index_uid(run_id: str, suite_id: str) -> str:
    return f"{DEFAULT_INDEX_PREFIX}_{_slug(run_id)}_{_slug(suite_id)}"


def _privacy_policy() -> dict[str, Any]:
    return {
        "raw_query_text": "redacted",
        "transcript_excerpt": "redacted",
        "semantic_text": "redacted",
        "local_paths": "redacted",
        "raw_ids": "hashed_refs",
        "raw_response_in_public_output": False,
    }


def _query_text(query_row: dict[str, Any]) -> str:
    value = _optional_str(query_row.get("query_text") or query_row.get("query"))
    if not value:
        raise ValueError("query row requires query_text or query")
    return value


def _query_id(query_row: dict[str, Any]) -> str:
    return _optional_str(query_row.get("query_id") or query_row.get("id")) or "query"


def _optional_public_label(query_row: dict[str, Any]) -> str | None:
    return _optional_str(query_row.get("query_label") or query_row.get("label"))


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be > 0") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _alignment_status(value: Any) -> str:
    status = str(value or "").strip()
    if status == "verified":
        return "verified"
    if status == "transcript_only":
        return "transcript_only"
    return "candidate" if status == "candidate" else status


def _id_ref(value: str | None, *, prefix: str) -> str | None:
    if not value:
        return None
    return f"{prefix}:{_short_hash(value)}"


def _index_ref(value: str | None) -> str | None:
    if not value:
        return None
    return f"index:{_short_hash(value)}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    return "_".join(part for part in "".join(chars).split("_") if part)[:48] or "run"


def _exception_reason(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if not text:
        text = exc.__class__.__name__
    return text[:200]
