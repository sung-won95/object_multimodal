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
from oarag.retrieval.evidence_unit_index import (
    MODALITY_AWARE_RERANK,
    query_project_evidence_units,
)
from oarag.retrieval.evidence_units import (
    CANDIDATE_LINK_SIGNAL_KEYS,
    CONCEPT_FIELD_COVERAGE_SCHEMA_VERSION,
    CONCEPT_FIELD_PUBLIC_NOTE,
    LINK_DIAGNOSTICS_PUBLIC_NOTE,
    LINK_DIAGNOSTICS_SCHEMA_VERSION,
    OCR_ENGINE_DIAGNOSTICS_PUBLIC_NOTE,
    SEARCH_FIELD_COVERAGE_SCHEMA_VERSION,
    SEARCH_FIELD_PUBLIC_NOTE,
    VERIFIED_LINK_SOURCE_KEYS,
    build_project_evidence_units,
)
from oarag.retrieval.project_index import index_project_evidence_units, iter_jsonl_documents
from oarag.vision.vlm_evidence_validator import validate_vlm_object_evidence


PUBLIC_SCHEMA_VERSION = "evidence-unit-retrieval-smoke-public-v1"
PUBLIC_SAFE_SLICE_SCHEMA_VERSION = "evidence-unit-public-safe-slice-v1"
DEFAULT_OUTPUT_ROOT = Path("reports") / "paper" / "evidence_units_retrieval_smoke"
DEFAULT_INDEX_PREFIX = "evidence_units_smoke"
DEFAULT_TARGET_RANK_LIMIT = 50
TARGET_FOUND_BUCKETS = ("top1", "top5", "top10", "top50", "top100", "not_found")
NOT_FOUND_REASON_CODES = (
    "candidate_recall_failure",
    "text_coverage_failure",
    "modality_evidence_missing",
    "index_settings_issue",
)


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
    modality_aware_rerank: bool = False,
) -> EvidenceUnitSmokeRun:
    raw_manifest = _read_json(manifest_path)
    base_dir = manifest_path.expanduser().resolve().parent
    manifest = _normalize_manifest(raw_manifest, base_dir=base_dir)
    slice_summary = _public_safe_slice_summary(manifest)
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
            modality_aware_rerank=modality_aware_rerank,
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
        modality_aware_rerank=modality_aware_rerank,
        slice_summary=slice_summary,
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
    modality_aware_rerank: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_id = str(suite.get("suite_id") or suite.get("project_id") or "lecture_suite")
    project_dir = _project_dir_from_suite(suite=suite, base_dir=base_dir, repo_root=repo_root)
    evidence_units_path = _optional_path(suite.get("evidence_units"))
    visual_states_path = _optional_path(suite.get("visual_states"))
    visual_states_output_path = _optional_path(suite.get("visual_states_output"))
    concept_graph_path = _optional_path(suite.get("concept_graph"))
    index_uid = str(suite.get("index") or _default_index_uid(run_id, suite_id))
    segment_index_uid = _optional_str(suite.get("segment_index"))
    limit = _positive_int(suite.get("limit", 5), field_name="limit")
    target_rank_limit = _positive_int(
        suite.get("target_rank_limit", max(limit, DEFAULT_TARGET_RANK_LIMIT)),
        field_name="target_rank_limit",
    )
    target_rank_limit = max(limit, target_rank_limit)
    suite_quality_rerank = bool(suite.get("quality_rerank", quality_rerank))
    suite_modality_aware_rerank = _suite_modality_aware_rerank(
        suite,
        default=modality_aware_rerank,
    )
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
            concept_graph=concept_graph_path,
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
        query_rows = [
            _dry_run_query_row(
                run_id=run_id,
                suite_id=suite_id,
                query_row=query_row,
                index_uid=index_uid,
                segment_index_uid=segment_index_uid,
                target_rank_limit=target_rank_limit,
            )
            for query_row in queries
        ]
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
                        modality_aware_rerank=suite_modality_aware_rerank,
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
        "recall_coverage_diagnostics": _recall_coverage_diagnostics(query_rows, artifact_summary),
        "rerank_diagnostics": _rerank_inspection(query_rows),
        "modality_aware_rerank_diagnostics": _modality_aware_rerank_inspection(query_rows),
        "evaluation_slice": _suite_public_safe_slice_summary(
            suite_id=suite_id,
            query_rows=query_rows,
        ),
    }
    return suite_summary, query_rows


def _suite_modality_aware_rerank(suite: dict[str, Any], *, default: bool) -> bool:
    if "modality_aware_rerank" in suite:
        return bool(suite.get("modality_aware_rerank"))
    configured = _optional_str(suite.get("evidence_unit_rerank"))
    if configured is not None:
        return configured.casefold().replace("-", "_") in {
            "modality_aware",
            "modality",
        }
    return default


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
    modality_aware_rerank: bool,
) -> dict[str, Any]:
    query_id = _query_id(query_row)
    query_text = _query_text(query_row)
    expected_segment_ids = _expected_segment_ids(query_row)
    expected_evidence_unit_ids = _expected_evidence_unit_ids(query_row)
    target_artifact = _expected_target_artifact(
        project_dir=project_dir,
        evidence_units=evidence_units,
        expected_segment_ids=expected_segment_ids,
        expected_evidence_unit_ids=expected_evidence_unit_ids,
    )

    response = query_project_evidence_units(
        client=client,  # type: ignore[arg-type]
        index_uid=index_uid,
        project_dir=project_dir,
        query=query_text,
        limit=target_rank_limit,
        evidence_units=evidence_units,
        evidence_unit_rerank=MODALITY_AWARE_RERANK if modality_aware_rerank else None,
    )
    candidates = _list_of_dicts(response.get("candidates"))
    base_candidates = _list_of_dicts(response.get("base_candidates")) or candidates
    top_candidate = candidates[0] if candidates else {}
    expected_match = _candidate_matches_expected(
        top_candidate,
        expected_segment_ids,
        expected_evidence_unit_ids,
    )
    target_diagnostics = _target_diagnostics(
        candidates=candidates,
        expected_segment_ids=expected_segment_ids,
        expected_evidence_unit_ids=expected_evidence_unit_ids,
        search_depth=target_rank_limit,
        query_text=query_text,
        query_row=query_row,
        target_artifact=target_artifact,
    )
    base_target_diagnostics = _target_diagnostics(
        candidates=base_candidates,
        expected_segment_ids=expected_segment_ids,
        expected_evidence_unit_ids=expected_evidence_unit_ids,
        search_depth=target_rank_limit,
        query_text=query_text,
        query_row=query_row,
        target_artifact=target_artifact,
    )
    reranked_candidates = _quality_rerank_candidates(candidates) if quality_rerank else []
    reranked_top_candidate = reranked_candidates[0] if reranked_candidates else {}
    reranked_expected_match = (
        _candidate_matches_expected(
            reranked_top_candidate,
            expected_segment_ids,
            expected_evidence_unit_ids,
        )
        if quality_rerank
        else None
    )
    reranked_target_diagnostics = (
        _target_diagnostics(
            candidates=reranked_candidates,
            expected_segment_ids=expected_segment_ids,
            expected_evidence_unit_ids=expected_evidence_unit_ids,
            search_depth=target_rank_limit,
            query_text=query_text,
            query_row=query_row,
            target_artifact=target_artifact,
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
        "evaluation_target": _public_evaluation_target(query_row),
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
        "base_target_diagnostics": base_target_diagnostics if modality_aware_rerank else None,
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
            expected_evidence_unit_ids=expected_evidence_unit_ids,
            base_target_diagnostics=target_diagnostics,
            reranked_target_diagnostics=reranked_target_diagnostics,
        ),
        "modality_aware_rerank": _public_modality_aware_rerank(
            response.get("retrieval_context"),
            base_target_diagnostics=base_target_diagnostics,
            reranked_target_diagnostics=target_diagnostics,
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
        "evaluation_target": _public_evaluation_target(query_row),
        "privacy": _privacy_policy(),
        "index_ref": _index_ref(index_uid),
        "segment_baseline_index_ref": _index_ref(segment_index_uid) if segment_index_uid else None,
        "status": "skipped",
        "skip_reason": skip_reason,
        "search_hit_count": 0,
        "top_evidence_unit": None,
        "top_hit_memo": {
            "expected_target_configured": bool(
                _expected_target_configured(query_row)
            ),
            "top_expected_match": None,
            "public_note": "index/query skipped; no top-hit judgment",
        },
        "target_diagnostics": {
            "target_configured": bool(
                _expected_target_configured(query_row)
            ),
            "target_found_in_top_k": None,
            "target_rank_bucket": "not_queried",
            "target_found_bucket": "not_queried",
            "target_search_depth": 0,
            "target_found@10": None,
            "target_found@50": None,
            "target_found@100": None,
            "public_safe_reason_codes": ["not_queried"],
            "not_found_reason_codes": [],
        },
        "base_target_diagnostics": None,
        "rerank_diagnostics": {
            "enabled": False,
            "status": "not_queried",
        },
        "modality_aware_rerank": {
            "enabled": False,
            "status": "not_queried",
        },
        "segment_baseline": {"status": "skipped", "skip_reason": skip_reason},
        "rag_input_inspectable": False,
    }


def _dry_run_query_row(
    *,
    run_id: str,
    suite_id: str,
    query_row: dict[str, Any],
    index_uid: str,
    segment_index_uid: str | None,
    target_rank_limit: int,
) -> dict[str, Any]:
    target_configured = _expected_target_configured(query_row)
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "query_id": _query_id(query_row),
        "query_label": _optional_public_label(query_row),
        "evaluation_target": _public_evaluation_target(query_row),
        "privacy": _privacy_policy(),
        "index_ref": _index_ref(index_uid),
        "segment_baseline_index_ref": _index_ref(segment_index_uid) if segment_index_uid else None,
        "status": "dry_run",
        "skip_reason": "dry_run_requested",
        "search_hit_count": 0,
        "top_evidence_unit": None,
        "top_hit_memo": {
            "expected_target_configured": target_configured,
            "top_expected_match": None,
            "target_rank_bucket": "not_queried",
            "public_note": "dry run records target configuration without querying private content",
        },
        "target_diagnostics": {
            "target_configured": target_configured,
            "target_found_in_top_k": None,
            "target_rank": None,
            "target_rank_bucket": "not_queried" if target_configured else "not_configured",
            "target_found_bucket": "not_queried" if target_configured else "not_configured",
            "target_search_depth": target_rank_limit,
            "target_found@10": None,
            "target_found@50": None,
            "target_found@100": None,
            "public_safe_reason_codes": ["not_queried"],
            "not_found_reason_codes": [],
        },
        "base_target_diagnostics": None,
        "rerank_diagnostics": {
            "enabled": False,
            "status": "not_queried",
        },
        "modality_aware_rerank": {
            "enabled": False,
            "status": "not_queried",
        },
        "segment_baseline": {"status": "dry_run", "skip_reason": "dry_run_requested"},
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
            "link_diagnostics": _empty_public_link_diagnostics(),
            "ocr_engine_diagnostics": _empty_public_ocr_engine_diagnostics(),
            "visual_state_coverage": {},
            "concept_field_coverage": _empty_public_concept_field_coverage(),
            "search_field_coverage": _empty_public_search_field_coverage("dry_run"),
            "vlm_object_evidence_coverage": {
                "status": "dry_run",
                "skip_reason": "dry_run_requested",
            },
            "visual_vlm_coverage_summary": _visual_vlm_coverage_summary(
                build_summary=None,
                vlm_object_evidence_coverage={
                    "status": "dry_run",
                    "skip_reason": "dry_run_requested",
                },
            ),
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
                "units_with_vlm_visible_text": int(
                    counts.get("units_with_vlm_visible_text") or 0
                ),
                "units_with_ocr_engine_evidence": int(
                    counts.get("units_with_ocr_engine_evidence") or 0
                ),
                "units_with_candidate_link": int(counts.get("units_with_candidate_link") or 0),
                "units_with_verified_link": int(counts.get("units_with_verified_link") or 0),
                "units_with_timestamp_fallback_link": int(
                    counts.get("units_with_timestamp_fallback_link") or 0
                ),
                "units_with_candidate_visual_support": int(
                    counts.get("units_with_candidate_visual_support") or 0
                ),
                "units_with_verified_object_alignment": int(
                    counts.get("units_with_verified_object_alignment") or 0
                ),
                "units_with_detected_text": int(counts.get("units_with_detected_text") or 0),
                "units_with_visual_description": int(counts.get("units_with_visual_description") or 0),
                "units_with_concept": int(counts.get("units_with_concept") or 0),
                "units_with_concept_relation": int(counts.get("units_with_concept_relation") or 0),
                "units_with_concept_search_text": int(
                    counts.get("units_with_concept_search_text") or 0
                ),
            },
            "link_counts": {
                "candidate_links": int(counts.get("candidate_links") or 0),
                "verified_links": int(counts.get("verified_links") or 0),
                "timestamp_fallback_links": int(counts.get("timestamp_fallback_links") or 0),
                "ocr_engine_links": int(counts.get("ocr_engine_links") or 0),
                "concept_mentions": int(counts.get("concept_mentions") or 0),
                "concept_relation_mentions": int(counts.get("concept_relation_mentions") or 0),
            },
            "link_diagnostics": _public_link_diagnostics(
                build_summary.get("link_diagnostics")
            ),
            "ocr_engine_diagnostics": _public_ocr_engine_diagnostics(
                build_summary.get("ocr_engine_diagnostics")
            ),
            "concept_field_coverage": _public_concept_field_coverage(
                build_summary.get("concept_field_coverage")
            ),
            "search_field_coverage": _public_search_field_coverage(
                build_summary.get("search_field_coverage")
            ),
            "visual_state_coverage": _public_visual_state_coverage(
                build_summary.get("visual_state_coverage")
            ),
            "verified_alignment_note": _verified_alignment_note(counts),
            "vlm_object_evidence_coverage": vlm_object_evidence_coverage,
            "visual_vlm_coverage_summary": _visual_vlm_coverage_summary(
                build_summary=build_summary,
                vlm_object_evidence_coverage=vlm_object_evidence_coverage,
            ),
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
        for key in ("has_vlm_visible_text", "has_ocr_engine_evidence"):
            if quality.get(key) is True:
                source_quality[key] += 1
        for key in ("has_detected_text", "has_visual_description"):
            if quality.get(key) is True:
                source_quality[key] += 1
        if int(quality.get("candidate_link_count") or 0) > 0:
            source_quality["units_with_candidate_link"] += 1
        if int(quality.get("timestamp_fallback_link_count") or 0) > 0:
            source_quality["units_with_timestamp_fallback_link"] += 1
        if _mapping(quality.get("candidate_visual_support")).get("has_candidate_visual_support") is True:
            source_quality["units_with_candidate_visual_support"] += 1
        if _mapping(quality.get("verified_object_alignment")).get("has_verified_object_alignment") is True:
            source_quality["units_with_verified_object_alignment"] += 1
        if _has_concept_fields(row, quality):
            source_quality["units_with_concept"] += 1
        if _has_concept_relation_fields(row, quality):
            source_quality["units_with_concept_relation"] += 1
        if _has_concept_search_text(row, quality):
            source_quality["units_with_concept_search_text"] += 1
        source_quality["candidate_links"] += int(quality.get("candidate_link_count") or 0)
        source_quality["verified_links"] += int(quality.get("verified_link_count") or 0)
        source_quality["timestamp_fallback_links"] += int(quality.get("timestamp_fallback_link_count") or 0)
        source_quality["ocr_engine_links"] += int(quality.get("ocr_engine_link_count") or 0)
        source_quality["ocr_engine_entity_mentions"] += int(
            quality.get("ocr_engine_entity_count") or 0
        )
        source_quality["vlm_visible_text_mentions"] += int(
            quality.get("vlm_visible_text_count") or 0
        )
        source_quality["concept_mentions"] += int(
            quality.get("concept_count") or len(_string_list(row.get("concept_ids")))
        )
        source_quality["concept_relation_mentions"] += int(
            quality.get("concept_relation_count")
            or len(_list_of_dicts(row.get("concept_relations")))
        )
    visual_state_rows = _visual_state_rows(project_dir=project_dir, visual_states=visual_states)
    link_diagnostics = _link_diagnostics_from_rows(rows)
    return {
        "status": "loaded_existing",
        "counts": {"evidence_units_total": len(rows)},
        "alignment_status_counts": dict(counts),
        "source_quality_counts": {
            "units_with_visual_state": source_quality["has_visual_state"],
            "units_with_visual_entity": source_quality["has_visual_entity"],
            "units_with_vlm_entity": source_quality["has_vlm_entity"],
            "units_with_vlm_visible_text": source_quality["has_vlm_visible_text"],
            "units_with_ocr_engine_evidence": source_quality["has_ocr_engine_evidence"],
            "units_with_candidate_link": source_quality["units_with_candidate_link"],
            "units_with_verified_link": source_quality["has_verified_link"],
            "units_with_timestamp_fallback_link": source_quality["units_with_timestamp_fallback_link"],
            "units_with_candidate_visual_support": source_quality[
                "units_with_candidate_visual_support"
            ],
            "units_with_verified_object_alignment": source_quality[
                "units_with_verified_object_alignment"
            ],
            "units_with_detected_text": source_quality["has_detected_text"],
            "units_with_visual_description": source_quality["has_visual_description"],
            "units_with_concept": source_quality["units_with_concept"],
            "units_with_concept_relation": source_quality["units_with_concept_relation"],
            "units_with_concept_search_text": source_quality["units_with_concept_search_text"],
        },
        "link_counts": {
            "candidate_links": source_quality["candidate_links"],
            "verified_links": source_quality["verified_links"],
            "timestamp_fallback_links": source_quality["timestamp_fallback_links"],
            "ocr_engine_links": source_quality["ocr_engine_links"],
            "concept_mentions": source_quality["concept_mentions"],
            "concept_relation_mentions": source_quality["concept_relation_mentions"],
        },
        "link_diagnostics": link_diagnostics,
        "ocr_engine_diagnostics": _loaded_ocr_engine_diagnostics(rows),
        "concept_field_coverage": _loaded_concept_field_coverage(rows),
        "search_field_coverage": _loaded_search_field_coverage(rows),
        "visual_state_coverage": _loaded_visual_state_coverage(
            evidence_units=rows,
            visual_states=visual_state_rows,
        ),
        "verified_alignment_note": _verified_alignment_note(source_quality),
        "vlm_object_evidence_coverage": vlm_object_evidence_coverage,
        "visual_vlm_coverage_summary": _visual_vlm_coverage_summary(
            build_summary={
                "counts": {
                    "evidence_units_total": len(rows),
                    "units_with_candidate_link": source_quality["units_with_candidate_link"],
                    "units_with_verified_link": source_quality["has_verified_link"],
                    "units_with_timestamp_fallback_link": source_quality[
                        "units_with_timestamp_fallback_link"
                    ],
                    "units_with_ocr_engine_evidence": source_quality[
                        "has_ocr_engine_evidence"
                    ],
                    "units_with_detected_text": source_quality["has_detected_text"],
                    "units_with_visual_description": source_quality["has_visual_description"],
                    "ocr_engine_links": source_quality["ocr_engine_links"],
                },
                "visual_state_coverage": _loaded_visual_state_coverage(
                    evidence_units=rows,
                    visual_states=visual_state_rows,
                ),
                "link_diagnostics": link_diagnostics,
                "ocr_engine_diagnostics": _loaded_ocr_engine_diagnostics(rows),
            },
            vlm_object_evidence_coverage=vlm_object_evidence_coverage,
        ),
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
            "has_concept": _has_concept_fields(candidate, source_quality),
            "has_concept_relation": _has_concept_relation_fields(candidate, source_quality),
            "has_verified_link": bool(source_quality.get("has_verified_link")),
            "has_timestamp_fallback_link": bool(source_quality.get("has_timestamp_fallback_link")),
            "has_detected_text": bool(source_quality.get("has_detected_text")),
            "has_visual_description": bool(source_quality.get("has_visual_description")),
            "has_concept_search_text": _has_concept_search_text(candidate, source_quality),
            "candidate_link_count": int(source_quality.get("candidate_link_count") or 0),
            "timestamp_fallback_link_count": int(source_quality.get("timestamp_fallback_link_count") or 0),
            "verified_link_count": int(source_quality.get("verified_link_count") or 0),
            "visual_state_detected_text_count": int(source_quality.get("visual_state_detected_text_count") or 0),
            "visual_entity_detected_text_count": int(source_quality.get("visual_entity_detected_text_count") or 0),
            "visual_description_count": int(source_quality.get("visual_description_count") or 0),
            "concept_count": _concept_count(candidate, source_quality),
            "concept_label_count": int(
                source_quality.get("concept_label_count")
                or len(_string_list(candidate.get("concept_labels")))
            ),
            "concept_alias_count": int(
                source_quality.get("concept_alias_count")
                or len(_string_list(candidate.get("concept_aliases")))
            ),
            "concept_relation_count": _concept_relation_count(candidate, source_quality),
            "timestamp_only_concept_relation_count": int(
                source_quality.get("timestamp_only_concept_relation_count") or 0
            ),
            "candidate_link_signal_counts": _public_count_map(
                source_quality.get("candidate_link_signal_counts"),
                CANDIDATE_LINK_SIGNAL_KEYS,
            ),
            "verified_link_source_counts": _public_count_map(
                source_quality.get("verified_link_source_counts"),
                VERIFIED_LINK_SOURCE_KEYS,
            ),
        },
        "candidate_visual_support": _public_candidate_visual_support(candidate),
        "verified_object_alignment": _public_verified_object_alignment(candidate),
        "rag_fields": {
            "evidence_text_available": bool(candidate.get("evidence_text")),
            "semantic_text_available": bool(candidate.get("semantic_text")),
            "concept_search_text_available": _has_concept_search_text(candidate, source_quality),
            "transcript_keywords_available": bool(_string_list(candidate.get("transcript_keywords"))),
            "visual_state_text_available": bool(candidate.get("visual_state_text")),
            "visual_entity_text_available": bool(candidate.get("visual_entity_text")),
            "link_signal_search_text_available": bool(
                candidate.get("candidate_link_signal_summary")
                or candidate.get("verified_link_signal_summary")
            ),
        },
        "concept_field_coverage": _public_candidate_concept_field_coverage(candidate),
        "content_coverage": _candidate_content_coverage(candidate),
        "query_term_coverage": _query_term_coverage(
            query_text=query_text or "",
            candidate=candidate,
        ),
        "modality_aware_rerank": _public_candidate_modality_rerank(candidate),
    }


def _public_candidate_visual_support(candidate: dict[str, Any]) -> dict[str, Any]:
    source_quality = _mapping(candidate.get("source_quality"))
    nested = _mapping(source_quality.get("candidate_visual_support"))
    visual_state_count = len(_string_list(candidate.get("visual_state_ids")))
    visual_entity_count = len(_string_list(candidate.get("visual_entity_ids")))
    candidate_link_count = int(source_quality.get("candidate_link_count") or 0)
    if not candidate_link_count:
        candidate_link_count = len(_string_list(candidate.get("candidate_entity_link_ids")))
    timestamp_fallback_link_count = int(source_quality.get("timestamp_fallback_link_count") or 0)
    has_support = bool(
        nested.get("has_candidate_visual_support")
        or source_quality.get("has_visual_state")
        or source_quality.get("has_visual_entity")
        or visual_state_count
        or visual_entity_count
        or candidate_link_count
        or timestamp_fallback_link_count
    )
    return {
        "has_candidate_visual_support": has_support,
        "visual_state_count": int(nested.get("visual_state_count") or visual_state_count),
        "visual_entity_count": int(nested.get("visual_entity_count") or visual_entity_count),
        "candidate_link_count": candidate_link_count,
        "timestamp_fallback_link_count": timestamp_fallback_link_count,
        "candidate_link_signal_counts": _public_count_map(
            source_quality.get("candidate_link_signal_counts")
            or nested.get("candidate_link_signal_counts"),
            CANDIDATE_LINK_SIGNAL_KEYS,
        ),
        "paper_claim_eligible": False,
    }


def _public_verified_object_alignment(candidate: dict[str, Any]) -> dict[str, Any]:
    source_quality = _mapping(candidate.get("source_quality"))
    nested = _mapping(source_quality.get("verified_object_alignment"))
    verified_link_count = int(source_quality.get("verified_link_count") or 0)
    if not verified_link_count:
        verified_link_count = len(_string_list(candidate.get("verified_entity_link_ids")))
    has_verified = bool(
        nested.get("has_verified_object_alignment")
        or source_quality.get("has_verified_link")
        or verified_link_count
    )
    return {
        "has_verified_object_alignment": has_verified,
        "verified_link_count": verified_link_count,
        "verified_link_source_counts": _public_count_map(
            source_quality.get("verified_link_source_counts")
            or nested.get("verified_link_source_counts"),
            VERIFIED_LINK_SOURCE_KEYS,
        ),
        "timestamp_fallback_counted_as_verified": False,
        "paper_claim_eligible": has_verified,
    }


def _target_diagnostics(
    *,
    candidates: list[dict[str, Any]],
    expected_segment_ids: set[str],
    expected_evidence_unit_ids: set[str],
    search_depth: int,
    query_text: str,
    query_row: dict[str, Any],
    target_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_evidence_unit_ids = expected_evidence_unit_ids or set()
    if not expected_segment_ids and not expected_evidence_unit_ids:
        return {
            "target_configured": False,
            "target_found_in_top_k": None,
            "target_rank_bucket": "not_configured",
            "target_found_bucket": "not_configured",
            "target_search_depth": search_depth,
            "target_found@10": None,
            "target_found@50": None,
            "target_found@100": None,
            "public_safe_reason_codes": ["target_not_configured"],
            "not_found_reason_codes": [],
        }
    target_candidate = None
    target_rank = None
    for rank, candidate in enumerate(candidates, start=1):
        if _candidate_matches_expected(
            candidate,
            expected_segment_ids,
            expected_evidence_unit_ids,
        ):
            target_candidate = candidate
            target_rank = rank
            break
    top_candidate = candidates[0] if candidates else {}
    found = target_candidate is not None and target_rank is not None
    public_target = target_candidate or target_artifact or {}
    reason_codes = _target_reason_codes(
        query_row=query_row,
        query_text=query_text,
        found=found,
        target_rank=target_rank,
        search_depth=search_depth,
        top_candidate=top_candidate,
        target_candidate=target_candidate or {},
        target_artifact=target_artifact or {},
    )
    return {
        "target_configured": True,
        "target_found_in_top_k": found,
        "target_rank": target_rank,
        "target_rank_bucket": _target_rank_bucket(target_rank),
        "target_found_bucket": _target_rank_bucket(target_rank),
        "target_search_depth": search_depth,
        "target_found@10": _target_found_at(target_rank, search_depth=search_depth, k=10),
        "target_found@50": _target_found_at(target_rank, search_depth=search_depth, k=50),
        "target_found@100": _target_found_at(target_rank, search_depth=search_depth, k=100),
        "target_evidence_unit_ref": _id_ref(
            _optional_str(public_target.get("evidence_unit_id")),
            prefix="evu",
        )
        if public_target
        else None,
        "target_quality_source": (
            "candidate" if target_candidate else "artifact" if target_artifact else "not_available"
        ),
        "target_evidence_unit_quality": _candidate_quality_summary(public_target),
        "target_content_coverage": _candidate_content_coverage(public_target),
        "target_query_term_coverage": _query_term_coverage(
            query_text=query_text,
            candidate=public_target,
        ),
        "top_vs_target_quality_delta": _candidate_quality_delta(
            top_candidate=top_candidate,
            target_candidate=public_target,
        ),
        "top_vs_target_content_delta": _candidate_content_delta(
            top_candidate=top_candidate,
            target_candidate=public_target,
            query_text=query_text,
        ),
        "public_safe_reason_codes": reason_codes,
        "not_found_reason_codes": (
            [code for code in reason_codes if code in NOT_FOUND_REASON_CODES]
            if not found
            else []
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
    if rank <= 100:
        return "top100"
    return "not_found"


def _target_found_at(rank: int | None, *, search_depth: int, k: int) -> bool | None:
    if rank is not None:
        return rank <= k
    if search_depth >= k:
        return False
    return None


def _expected_target_artifact(
    *,
    project_dir: Path,
    evidence_units: Path | None,
    expected_segment_ids: set[str],
    expected_evidence_unit_ids: set[str],
) -> dict[str, Any] | None:
    if not expected_segment_ids and not expected_evidence_unit_ids:
        return None
    try:
        rows = _evidence_unit_rows(project_dir=project_dir, evidence_units=evidence_units)
    except Exception:  # noqa: BLE001 - diagnostics should still run if local artifacts are unavailable.
        return None
    for row in rows:
        if _candidate_matches_expected(row, expected_segment_ids, expected_evidence_unit_ids):
            return row
    return None


def _target_reason_codes(
    *,
    query_row: dict[str, Any],
    query_text: str,
    found: bool,
    target_rank: int | None,
    search_depth: int,
    top_candidate: dict[str, Any],
    target_candidate: dict[str, Any],
    target_artifact: dict[str, Any],
) -> list[str]:
    target = target_candidate or target_artifact
    codes: list[str] = []
    if found:
        codes.append(f"target_found_{_target_rank_bucket(target_rank)}")
    elif target:
        codes.append("candidate_recall_failure")
    else:
        codes.append("index_settings_issue")

    if not target:
        return _dedupe_preserve_order(codes)

    quality = _candidate_quality_summary(target)
    content = _candidate_content_coverage(target)
    term_coverage = _query_term_coverage(query_text=query_text, candidate=target)
    evidence_bucket = str(content.get("evidence_text_bucket") or "unknown")
    semantic_bucket = str(content.get("semantic_text_bucket") or "unknown")
    query_bucket = str(term_coverage.get("combined_match_bucket") or "unknown")
    if evidence_bucket in {"empty", "short"}:
        codes.append(f"target_evidence_text_{evidence_bucket}")
    if semantic_bucket in {"empty", "short"}:
        codes.append(f"target_semantic_text_{semantic_bucket}")
    if query_bucket in {"none", "low", "no_query_terms"}:
        codes.append("text_coverage_failure")
        codes.append(f"target_query_term_coverage_{query_bucket}")

    concept_coverage = _mapping(quality.get("concept_field_coverage"))
    if not bool(quality.get("has_concept")):
        codes.append("target_concept_missing")
    if _concept_alias_count(query_row) > 0 and int(concept_coverage.get("concept_alias_count") or 0) <= 0:
        codes.append("target_concept_alias_missing")

    candidate_support = _mapping(quality.get("candidate_visual_support"))
    verified_alignment = _mapping(quality.get("verified_object_alignment"))
    has_visual_state = bool(quality.get("has_visual_state"))
    has_vlm = bool(quality.get("has_vlm_entity"))
    has_candidate_support = bool(candidate_support.get("has_candidate_visual_support"))
    has_verified_alignment = bool(verified_alignment.get("has_verified_object_alignment"))
    if not has_visual_state:
        codes.append("target_visual_state_missing")
    if not has_vlm:
        codes.append("target_vlm_missing")
    if not has_candidate_support:
        codes.append("target_candidate_link_missing")
    if not has_verified_alignment:
        codes.append("target_verified_link_missing")
    if _query_expects_visual_evidence(query_row) and not (
        has_visual_state or has_vlm or has_candidate_support or has_verified_alignment
    ):
        codes.append("modality_evidence_missing")

    quality_delta = _candidate_quality_delta(
        top_candidate=top_candidate,
        target_candidate=target,
    )
    if any(
        quality_delta.get(key) is True
        for key in ("top_has_more_visual_entities", "top_has_verified_link_only")
    ):
        codes.append("top_hit_quality_stronger_than_target")
    if any(
        quality_delta.get(key) is True
        for key in ("target_has_more_visual_entities", "target_has_verified_link_only")
    ):
        codes.append("target_quality_stronger_than_top_hit")
    content_delta = _candidate_content_delta(
        top_candidate=top_candidate,
        target_candidate=target,
        query_text=query_text,
    )
    if int(content_delta.get("combined_query_term_match_count_delta") or 0) > 0:
        codes.append("top_hit_text_coverage_stronger_than_target")
    if content_delta.get("target_has_more_query_term_matches") is True:
        codes.append("target_text_coverage_stronger_than_top_hit")
    if not found and search_depth < 100:
        codes.append("candidate_depth_below_100")
    return _dedupe_preserve_order(codes)


def _query_expects_visual_evidence(query_row: dict[str, Any]) -> bool:
    modality = (_optional_str(query_row.get("expected_modality")) or "").casefold()
    query_type = (_optional_str(query_row.get("query_type")) or "").casefold()
    return modality in {"both", "visual"} or query_type in {
        "formula_table_lookup",
        "multimodal_grounded",
        "visual_object_reference",
    }


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


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
        "has_concept": _has_concept_fields(candidate, source_quality),
        "has_concept_relation": _has_concept_relation_fields(candidate, source_quality),
        "has_verified_link": bool(source_quality.get("has_verified_link")),
        "has_timestamp_fallback_link": bool(source_quality.get("has_timestamp_fallback_link")),
        "candidate_visual_support": _public_candidate_visual_support(candidate),
        "verified_object_alignment": _public_verified_object_alignment(candidate),
        "concept_field_coverage": _public_candidate_concept_field_coverage(candidate),
        "rag_fields": {
            "evidence_text_available": bool(candidate.get("evidence_text")),
            "semantic_text_available": bool(candidate.get("semantic_text")),
            "concept_search_text_available": _has_concept_search_text(candidate, source_quality),
            "transcript_keywords_available": bool(_string_list(candidate.get("transcript_keywords"))),
            "visual_state_text_available": bool(candidate.get("visual_state_text")),
            "visual_entity_text_available": bool(candidate.get("visual_entity_text")),
            "link_signal_search_text_available": bool(
                candidate.get("candidate_link_signal_summary")
                or candidate.get("verified_link_signal_summary")
            ),
        },
        "content_coverage": _candidate_content_coverage(candidate),
    }


def _candidate_content_coverage(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    evidence_text = _text_for_coverage(candidate.get("evidence_text"))
    semantic_text = _text_for_coverage(candidate.get("semantic_text"))
    transcript_text = _text_for_coverage(candidate.get("transcript_window_text"))
    concept_relation_text = _text_for_coverage(candidate.get("concept_relation_text"))
    concept_search_text = _text_for_coverage(candidate.get("concept_search_text"))
    visual_state_text = _text_for_coverage(candidate.get("visual_state_text"))
    visual_entity_text = _text_for_coverage(candidate.get("visual_entity_text"))
    link_signal_text = _text_for_coverage(
        " ".join(
            [
                str(candidate.get("candidate_link_signal_summary") or ""),
                str(candidate.get("verified_link_signal_summary") or ""),
            ]
        )
    )
    source_quality = _mapping(candidate.get("source_quality"))
    return {
        "evidence_text_char_count": len(evidence_text),
        "semantic_text_char_count": len(semantic_text),
        "transcript_window_char_count": len(transcript_text),
        "transcript_keyword_count": len(_string_list(candidate.get("transcript_keywords"))),
        "concept_relation_text_char_count": len(concept_relation_text),
        "concept_search_text_char_count": len(concept_search_text),
        "visual_state_text_char_count": len(visual_state_text),
        "visual_entity_text_char_count": len(visual_entity_text),
        "link_signal_text_char_count": len(link_signal_text),
        "evidence_text_bucket": _char_count_bucket(len(evidence_text)),
        "semantic_text_bucket": _char_count_bucket(len(semantic_text)),
        "transcript_window_bucket": _char_count_bucket(len(transcript_text)),
        "concept_search_text_bucket": _char_count_bucket(len(concept_search_text)),
        "visual_state_text_bucket": _char_count_bucket(len(visual_state_text)),
        "visual_entity_text_bucket": _char_count_bucket(len(visual_entity_text)),
        "link_signal_text_bucket": _char_count_bucket(len(link_signal_text)),
        "concept_relation_text_bucket": _char_count_bucket(len(concept_relation_text)),
        "visual_state_count": len(_string_list(candidate.get("visual_state_ids"))),
        "visual_entity_count": len(_string_list(candidate.get("visual_entity_ids"))),
        "concept_count": _concept_count(candidate, source_quality),
        "concept_label_count": int(
            source_quality.get("concept_label_count")
            or len(_string_list(candidate.get("concept_labels")))
        ),
        "concept_alias_count": int(
            source_quality.get("concept_alias_count")
            or len(_string_list(candidate.get("concept_aliases")))
        ),
        "concept_relation_count": _concept_relation_count(candidate, source_quality),
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
    concept_terms = set(
        _coverage_terms(
            " ".join(
                [
                    *_string_list(candidate.get("concept_labels")),
                    *_string_list(candidate.get("concept_aliases")),
                    _text_for_coverage(candidate.get("concept_relation_text")),
                    _text_for_coverage(candidate.get("concept_search_text")),
                ]
            )
        )
    )
    transcript_keyword_terms = set(_coverage_terms(" ".join(_string_list(candidate.get("transcript_keywords")))))
    visual_terms = set(
        _coverage_terms(
            " ".join(
                [
                    _text_for_coverage(candidate.get("visual_state_text")),
                    _text_for_coverage(candidate.get("visual_entity_text")),
                ]
            )
        )
    )
    link_signal_terms = set(
        _coverage_terms(
            " ".join(
                [
                    _text_for_coverage(candidate.get("candidate_link_signal_summary")),
                    _text_for_coverage(candidate.get("verified_link_signal_summary")),
                ]
            )
        )
    )
    combined_terms = (
        evidence_terms
        | semantic_terms
        | transcript_terms
        | transcript_keyword_terms
        | concept_terms
        | visual_terms
        | link_signal_terms
    )
    query_term_count = len(query_terms)
    evidence_matches = len(query_terms & evidence_terms)
    semantic_matches = len(query_terms & semantic_terms)
    transcript_matches = len(query_terms & transcript_terms)
    transcript_keyword_matches = len(query_terms & transcript_keyword_terms)
    concept_matches = len(query_terms & concept_terms)
    visual_matches = len(query_terms & visual_terms)
    link_signal_matches = len(query_terms & link_signal_terms)
    combined_matches = len(query_terms & combined_terms)
    return {
        "query_term_count": query_term_count,
        "evidence_text_match_count": evidence_matches,
        "semantic_text_match_count": semantic_matches,
        "transcript_window_match_count": transcript_matches,
        "transcript_keyword_match_count": transcript_keyword_matches,
        "concept_search_text_match_count": concept_matches,
        "visual_search_text_match_count": visual_matches,
        "link_signal_text_match_count": link_signal_matches,
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


def _public_candidate_modality_rerank(candidate: dict[str, Any]) -> dict[str, Any] | None:
    rerank = _mapping(candidate.get("modality_aware_rerank"))
    if not rerank:
        return None
    return {
        "strategy": rerank.get("strategy"),
        "query_type": rerank.get("query_type"),
        "original_rank": rerank.get("original_rank"),
        "reranked_rank": rerank.get("reranked_rank"),
        "score": rerank.get("score"),
        "score_bucket": _score_bucket(_optional_float(rerank.get("score"))),
        "score_components": _mapping(rerank.get("components")),
        "flags": _mapping(rerank.get("flags")),
        "query_term_overlap": _mapping(rerank.get("query_term_overlap")),
    }


def _public_modality_aware_rerank(
    retrieval_context: Any,
    *,
    base_target_diagnostics: dict[str, Any],
    reranked_target_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    context = _mapping(retrieval_context)
    rerank = _mapping(context.get("evidence_unit_rerank"))
    if not rerank or rerank.get("enabled") is not True:
        return {"enabled": False, "status": "disabled"}
    return {
        "enabled": True,
        "status": "computed",
        "strategy": rerank.get("strategy"),
        "query_type": rerank.get("query_type"),
        "candidate_count": int(rerank.get("candidate_count") or 0),
        "top_changed": bool(rerank.get("top_changed")),
        "base_top_ref": rerank.get("base_top_ref"),
        "reranked_top_ref": rerank.get("reranked_top_ref"),
        "reranked_top_score": rerank.get("reranked_top_score"),
        "reranked_top_score_bucket": _score_bucket(
            _optional_float(rerank.get("reranked_top_score"))
        ),
        "reranked_top_original_rank": rerank.get("reranked_top_original_rank"),
        "reranked_top_rank": rerank.get("reranked_top_rank"),
        "component_names": [
            str(name)
            for name in rerank.get("component_names", [])
            if isinstance(name, str)
        ],
        "base_target_rank": base_target_diagnostics.get("target_rank"),
        "base_target_rank_bucket": base_target_diagnostics.get("target_rank_bucket"),
        "reranked_target_rank": reranked_target_diagnostics.get("target_rank"),
        "reranked_target_rank_bucket": reranked_target_diagnostics.get("target_rank_bucket"),
        "failure_mode": _modality_rerank_failure_mode(
            base_target_diagnostics=base_target_diagnostics,
            reranked_target_diagnostics=reranked_target_diagnostics,
        ),
        "public_note": (
            "Modality-aware rerank is an opt-in retrieval path. It reports feature "
            "names, counts, buckets, scores, and hashed refs only."
        ),
    }


def _modality_rerank_failure_mode(
    *,
    base_target_diagnostics: dict[str, Any],
    reranked_target_diagnostics: dict[str, Any],
) -> str:
    if base_target_diagnostics.get("target_configured") is not True:
        return "target_not_configured"
    if base_target_diagnostics.get("target_found_in_top_k") is not True:
        return "candidate_recall_failure"
    if reranked_target_diagnostics.get("target_found_in_top_k") is not True:
        return "rerank_regression"
    base_rank = _optional_int(base_target_diagnostics.get("target_rank"))
    reranked_rank = _optional_int(reranked_target_diagnostics.get("target_rank"))
    if base_rank is not None and reranked_rank is not None and reranked_rank < base_rank:
        return "ranking_improved"
    if reranked_rank == 1:
        return "ranking_success"
    return "evidence_quality_or_ranking_tie"


def _rerank_diagnostics(
    *,
    enabled: bool,
    base_top_candidate: dict[str, Any],
    reranked_candidates: list[dict[str, Any]],
    expected_segment_ids: set[str],
    expected_evidence_unit_ids: set[str],
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
    expected_evidence_unit_ids = expected_evidence_unit_ids or set()
    base_top_match = _candidate_matches_expected(
        base_top_candidate,
        expected_segment_ids,
        expected_evidence_unit_ids,
    )
    reranked_top_match = _candidate_matches_expected(
        reranked_top,
        expected_segment_ids,
        expected_evidence_unit_ids,
    )
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


def _public_safe_slice_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != PUBLIC_SAFE_SLICE_SCHEMA_VERSION:
        return {"enabled": False}
    queries = _manifest_query_rows(manifest)
    summary = _query_metadata_summary(queries)
    slice_info = _mapping(manifest.get("evaluation_slice") or manifest.get("slice"))
    return {
        "enabled": True,
        "schema_version": PUBLIC_SAFE_SLICE_SCHEMA_VERSION,
        "slice_id": slice_info.get("slice_id") or manifest.get("slice_id") or manifest.get("run_id"),
        "lecture_count": int(slice_info.get("lecture_count") or 0),
        "query_target_count": len(queries),
        "target_configured_count": summary["target_configured_count"],
        "expected_modality_counts": summary["expected_modality_counts"],
        "query_type_counts": summary["query_type_counts"],
        "concept_alias_coverage": summary["concept_alias_coverage"],
        "timestamp_hint_counts": summary["timestamp_hint_counts"],
        "privacy": "raw_query_text/transcripts/answers/evidence_text/local_paths_excluded",
        "public_note": (
            "This slice stores public-safe query IDs, concept-alias counts, target "
            "configuration, modality/query-type aggregates, and timestamp-hint buckets. "
            "Raw query text and raw evidence text are excluded."
        ),
    }


def _suite_public_safe_slice_summary(
    *,
    suite_id: str,
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = [_mapping(row.get("evaluation_target")) for row in query_rows]
    summary = _query_metadata_summary(targets)
    return {
        "suite_id": suite_id,
        "query_target_count": len(query_rows),
        "target_configured_count": summary["target_configured_count"],
        "expected_modality_counts": summary["expected_modality_counts"],
        "query_type_counts": summary["query_type_counts"],
        "concept_alias_coverage": summary["concept_alias_coverage"],
        "timestamp_hint_counts": summary["timestamp_hint_counts"],
    }


def _query_metadata_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modality_counts = Counter()
    query_type_counts = Counter()
    timestamp_hint_counts = Counter()
    target_configured_count = 0
    alias_total = 0
    queries_with_aliases = 0
    max_aliases = 0
    for row in rows:
        if _expected_target_configured(row):
            target_configured_count += 1
        modality = _optional_str(row.get("expected_modality"))
        if modality:
            modality_counts[modality] += 1
        query_type = _optional_str(row.get("query_type"))
        if query_type:
            query_type_counts[query_type] += 1
        hint_bucket = _optional_str(row.get("timestamp_hint_bucket"))
        if hint_bucket:
            timestamp_hint_counts[hint_bucket] += 1
        alias_count = _concept_alias_count(row)
        alias_total += alias_count
        max_aliases = max(max_aliases, alias_count)
        if alias_count:
            queries_with_aliases += 1
    return {
        "target_configured_count": target_configured_count,
        "expected_modality_counts": dict(modality_counts),
        "query_type_counts": dict(query_type_counts),
        "timestamp_hint_counts": dict(timestamp_hint_counts),
        "concept_alias_coverage": {
            "queries_with_aliases": queries_with_aliases,
            "queries_without_aliases": len(rows) - queries_with_aliases,
            "concept_alias_total": alias_total,
            "max_aliases_per_query": max_aliases,
        },
    }


def _public_evaluation_target(query_row: dict[str, Any]) -> dict[str, Any]:
    target_kind = _optional_str(query_row.get("target_kind"))
    if not target_kind:
        target_kind = "evidence_unit" if _expected_evidence_unit_ids(query_row) else "segment"
    return {
        "target_configured": _expected_target_configured(query_row),
        "target_kind": target_kind,
        "target_ref_count": len(_expected_segment_ids(query_row))
        + len(_expected_evidence_unit_ids(query_row)),
        "expected_modality": _optional_str(query_row.get("expected_modality")),
        "query_type": _optional_str(query_row.get("query_type")),
        "concept_alias_count": _concept_alias_count(query_row),
        "timestamp_hint_bucket": _optional_str(query_row.get("timestamp_hint_bucket")),
        "public_note": (
            "Target metadata is public-safe: raw query text, raw answers, transcript "
            "excerpts, local paths, and raw evidence text are not emitted."
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
    modality_aware_rerank: bool,
    slice_summary: dict[str, Any],
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
        "evaluation_slice": slice_summary,
        "suites": suites,
        "rag_input_inspection": _rag_input_inspection(query_rows, {}),
        "target_rank_diagnostics": _target_rank_inspection(query_rows),
        "recall_coverage_diagnostics": _recall_coverage_diagnostics(query_rows, {}),
        "rerank_diagnostics": _rerank_inspection(query_rows),
        "modality_aware_rerank_diagnostics": _modality_aware_rerank_inspection(query_rows),
        "quality_rerank_requested": quality_rerank,
        "modality_aware_rerank_requested": modality_aware_rerank,
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
    slice_summary = _mapping(payload.get("evaluation_slice"))
    if slice_summary.get("enabled") is True:
        alias_coverage = _mapping(slice_summary.get("concept_alias_coverage"))
        lines.extend(
            [
                "",
                "## Evaluation Slice",
                "",
                f"- slice id: `{slice_summary.get('slice_id')}`",
                f"- query targets: `{slice_summary.get('query_target_count', 0)}`",
                f"- targets configured: `{slice_summary.get('target_configured_count', 0)}`",
                f"- expected modalities: `{json.dumps(slice_summary.get('expected_modality_counts', {}), sort_keys=True)}`",
                f"- query types: `{json.dumps(slice_summary.get('query_type_counts', {}), sort_keys=True)}`",
                f"- queries with concept aliases: `{alias_coverage.get('queries_with_aliases', 0)}`",
                f"- concept alias total: `{alias_coverage.get('concept_alias_total', 0)}`",
                f"- privacy: `{slice_summary.get('privacy')}`",
            ]
        )
    lines.extend(["", "## Suites", ""])
    for suite in payload.get("suites", []):
        build = suite.get("build", {})
        index = suite.get("index", {})
        link_diagnostics = _mapping(build.get("link_diagnostics"))
        candidate_support = _mapping(link_diagnostics.get("candidate_visual_support"))
        verified_alignment = _mapping(link_diagnostics.get("verified_object_alignment"))
        ocr_engine_diagnostics = _mapping(build.get("ocr_engine_diagnostics"))
        vlm_coverage = _mapping(build.get("vlm_object_evidence_coverage"))
        entity_coverage = _mapping(vlm_coverage.get("visual_entity_coverage"))
        unit_coverage = _mapping(vlm_coverage.get("evidence_unit_coverage"))
        visual_vlm_summary = _mapping(build.get("visual_vlm_coverage_summary"))
        visual_entity_channels = _mapping(visual_vlm_summary.get("visual_entity_channels"))
        evidence_unit_channels = _mapping(visual_vlm_summary.get("evidence_unit_channels"))
        object_link_coverage = _mapping(visual_vlm_summary.get("object_link_coverage"))
        visual_coverage = _mapping(build.get("visual_state_coverage"))
        concept_coverage = _mapping(build.get("concept_field_coverage"))
        search_coverage = _mapping(build.get("search_field_coverage"))
        search_field_counts = _mapping(search_coverage.get("field_unit_counts"))
        interval_summary = _mapping(visual_coverage.get("interval_duration_seconds"))
        coverage_gate = _mapping(visual_coverage.get("coverage_gate"))
        recall_coverage = _mapping(suite.get("recall_coverage_diagnostics"))
        current_recall = _mapping(recall_coverage.get("current"))
        baseline_recall = _mapping(recall_coverage.get("baseline"))
        lines.extend(
            [
                f"### {suite.get('suite_id')}",
                "",
                f"- evidence units: `{_mapping(build.get('counts')).get('evidence_units_total', 0)}`",
                f"- alignment statuses: `{json.dumps(build.get('alignment_status_counts', {}), sort_keys=True)}`",
                f"- link counts: `{json.dumps(build.get('link_counts', {}), sort_keys=True)}`",
                f"- candidate visual support: `{json.dumps(candidate_support, sort_keys=True)}`",
                f"- verified object alignment: `{json.dumps(verified_alignment, sort_keys=True)}`",
                f"- OCR engine diagnostics: `{json.dumps(ocr_engine_diagnostics, sort_keys=True)}`",
                f"- visual state source: `{visual_coverage.get('source') or 'unknown'}`",
                f"- visual state coverage: `{visual_coverage.get('evidence_units_with_visual_state', 0)}`/`{visual_coverage.get('evidence_units_total', 0)}`",
                f"- concept field coverage: `{concept_coverage.get('evidence_units_with_concept_search_text', 0)}`/`{concept_coverage.get('evidence_units_total', 0)}`",
                f"- search field coverage: `{json.dumps(search_field_counts, sort_keys=True)}`",
                f"- visual state duration buckets: `{json.dumps(interval_summary.get('buckets', {}), sort_keys=True)}`",
                f"- visual state gate: `{coverage_gate.get('status') or 'not_configured'}`",
                f"- index status: `{index.get('status')}`",
                f"- RAG input inspectable top hits: `{suite.get('rag_input_inspection', {}).get('inspectable_top_hit_count', 0)}`",
                f"- target rank buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('rank_bucket_counts', {}), sort_keys=True)}`",
                f"- target found buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('target_found_bucket_counts', {}), sort_keys=True)}`",
                f"- target_found@50: `{suite.get('target_rank_diagnostics', {}).get('target_found@50')}`",
                f"- target_found@10: `{suite.get('target_rank_diagnostics', {}).get('target_found@10')}`",
                f"- target_found@100: `{suite.get('target_rank_diagnostics', {}).get('target_found@100')}`",
                f"- recall/coverage current @10/@50: `{current_recall.get('target_found@10')}` / `{current_recall.get('target_found@50')}`",
                f"- recall/coverage baseline kind: `{baseline_recall.get('kind')}`",
                f"- not_found reason codes: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('not_found_reason_code_counts', {}), sort_keys=True)}`",
                f"- target evidence-text buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('target_evidence_text_bucket_counts', {}), sort_keys=True)}`",
                f"- target semantic-text buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('target_semantic_text_bucket_counts', {}), sort_keys=True)}`",
                f"- target feature coverage: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('target_feature_coverage_counts', {}), sort_keys=True)}`",
                f"- found target query-term buckets: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('found_target_query_term_bucket_counts', {}), sort_keys=True)}`",
                f"- found target candidate link signals: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('found_target_candidate_link_signal_counts', {}), sort_keys=True)}`",
                f"- found target verified link sources: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('found_target_verified_link_source_counts', {}), sort_keys=True)}`",
                f"- top-vs-target coverage flags: `{json.dumps(suite.get('target_rank_diagnostics', {}).get('quality_delta_flag_counts', {}), sort_keys=True)}`",
                f"- reranked target rank buckets: `{json.dumps(suite.get('rerank_diagnostics', {}).get('reranked_target_rank_bucket_counts', {}), sort_keys=True)}`",
                f"- reranked top-hit matches: `{suite.get('rerank_diagnostics', {}).get('reranked_top_match_count', 0)}`",
                f"- modality-aware query types: `{json.dumps(suite.get('modality_aware_rerank_diagnostics', {}).get('query_type_counts', {}), sort_keys=True)}`",
                f"- modality-aware failure modes: `{json.dumps(suite.get('modality_aware_rerank_diagnostics', {}).get('failure_mode_counts', {}), sort_keys=True)}`",
                f"- modality-aware top changes: `{suite.get('modality_aware_rerank_diagnostics', {}).get('top_changed_count', 0)}`",
                f"- VLM evidence status: `{vlm_coverage.get('status')}`",
                f"- paper-quality VLM entities: `{entity_coverage.get('paper_quality_vlm_entity_count', 0)}`",
                f"- OCR-only entities: `{entity_coverage.get('ocr_only_entity_count', 0)}`",
                f"- VLM object-description entities: `{visual_entity_channels.get('vlm_object_description_entity_count', 0)}`",
                f"- detected-text entities: `{visual_entity_channels.get('detected_text_entity_count', 0)}`",
                f"- units with VLM entity: `{unit_coverage.get('units_with_vlm_entity', 0)}`",
                f"- units with visual description: `{unit_coverage.get('units_with_visual_description', 0)}`",
                f"- units with detected text: `{unit_coverage.get('units_with_detected_text', 0)}`",
                f"- visual/VLM unit ratios: `{json.dumps({key: evidence_unit_channels.get(key) for key in ('vlm_entity_ratio', 'visual_description_ratio', 'detected_text_ratio')}, sort_keys=True)}`",
                f"- object link unit ratios: `{json.dumps({key: object_link_coverage.get(key) for key in ('candidate_link_unit_ratio', 'verified_link_unit_ratio', 'timestamp_fallback_unit_ratio')}, sort_keys=True)}`",
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
    top_with_concepts = [
        row
        for row in queried
        if _mapping(_mapping(row.get("top_evidence_unit")).get("source_quality")).get("has_concept")
    ]
    return {
        "query_count": len(query_rows),
        "queried_count": len(queried),
        "inspectable_top_hit_count": len(inspectable),
        "top_hits_with_visual_state_count": len(top_with_visual),
        "top_hits_with_concept_count": len(top_with_concepts),
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
    target_found_buckets = Counter({bucket: 0 for bucket in TARGET_FOUND_BUCKETS})
    target_found_buckets.update(
        str(diag.get("target_found_bucket") or diag.get("target_rank_bucket") or "unknown")
        for diag in configured
    )
    target_quality = Counter()
    target_candidate_signal_counts = Counter()
    target_verified_source_counts = Counter()
    top_better_counts = Counter()
    reason_code_counts = Counter()
    not_found_reason_code_counts = Counter()
    evidence_text_buckets = Counter()
    semantic_text_buckets = Counter()
    query_term_buckets_all = Counter()
    target_feature_coverage = Counter()
    target_found_at_10 = 0
    target_found_at_50 = 0
    target_found_at_100 = 0
    target_found_at_10_available = 0
    target_found_at_50_available = 0
    target_found_at_100_available = 0
    for diag in configured:
        if diag.get("target_found@10") is not None:
            target_found_at_10_available += 1
            if diag.get("target_found@10") is True:
                target_found_at_10 += 1
        if diag.get("target_found@50") is not None:
            target_found_at_50_available += 1
            if diag.get("target_found@50") is True:
                target_found_at_50 += 1
        if diag.get("target_found@100") is not None:
            target_found_at_100_available += 1
            if diag.get("target_found@100") is True:
                target_found_at_100 += 1
        for code in _string_list(diag.get("public_safe_reason_codes")):
            reason_code_counts[code] += 1
        for code in _string_list(diag.get("not_found_reason_codes")):
            not_found_reason_code_counts[code] += 1
        coverage = _mapping(diag.get("target_content_coverage"))
        if coverage:
            evidence_text_buckets[str(coverage.get("evidence_text_bucket") or "unknown")] += 1
            semantic_text_buckets[str(coverage.get("semantic_text_bucket") or "unknown")] += 1
        terms = _mapping(diag.get("target_query_term_coverage"))
        if terms:
            query_term_buckets_all[str(terms.get("combined_match_bucket") or "unknown")] += 1
        quality = _mapping(diag.get("target_evidence_unit_quality"))
        if quality:
            for key in (
                "has_visual_state",
                "has_visual_entity",
                "has_vlm_entity",
                "has_concept",
                "has_concept_relation",
                "has_verified_link",
                "has_timestamp_fallback_link",
            ):
                if quality.get(key) is True:
                    target_feature_coverage[key] += 1
            if _mapping(quality.get("candidate_visual_support")).get(
                "has_candidate_visual_support"
            ) is True:
                target_feature_coverage["has_candidate_visual_support"] += 1
            if _mapping(quality.get("verified_object_alignment")).get(
                "has_verified_object_alignment"
            ) is True:
                target_feature_coverage["has_verified_object_alignment"] += 1
            if _mapping(quality.get("rag_fields")).get("concept_search_text_available") is True:
                target_feature_coverage["has_concept_search_text"] += 1
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
        candidate_support = _mapping(quality.get("candidate_visual_support"))
        verified_alignment = _mapping(quality.get("verified_object_alignment"))
        if candidate_support.get("has_candidate_visual_support") is True:
            target_quality["has_candidate_visual_support"] += 1
        if verified_alignment.get("has_verified_object_alignment") is True:
            target_quality["has_verified_object_alignment"] += 1
        for key, value in _public_count_map(
            candidate_support.get("candidate_link_signal_counts"),
            CANDIDATE_LINK_SIGNAL_KEYS,
        ).items():
            target_candidate_signal_counts[key] += value
        for key, value in _public_count_map(
            verified_alignment.get("verified_link_source_counts"),
            VERIFIED_LINK_SOURCE_KEYS,
        ).items():
            target_verified_source_counts[key] += value
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
        "target_found_at_10_count": target_found_at_10,
        "target_found_at_50_count": target_found_at_50,
        "target_found_at_100_count": target_found_at_100,
        "target_found@10": _ratio_or_none(target_found_at_10, target_found_at_10_available),
        "target_found@50": _ratio_or_none(target_found_at_50, target_found_at_50_available),
        "target_found@100": _ratio_or_none(target_found_at_100, target_found_at_100_available),
        "rank_bucket_counts": dict(buckets),
        "target_found_bucket_counts": {
            bucket: int(target_found_buckets.get(bucket) or 0)
            for bucket in TARGET_FOUND_BUCKETS
        },
        "not_found_reason_code_counts": {
            code: int(not_found_reason_code_counts.get(code) or 0)
            for code in NOT_FOUND_REASON_CODES
        },
        "public_safe_reason_code_counts": dict(reason_code_counts),
        "target_evidence_text_bucket_counts": dict(evidence_text_buckets),
        "target_semantic_text_bucket_counts": dict(semantic_text_buckets),
        "target_query_term_bucket_counts": dict(query_term_buckets_all),
        "target_feature_coverage_counts": dict(target_feature_coverage),
        "found_target_quality_counts": dict(target_quality),
        "found_target_candidate_link_signal_counts": dict(target_candidate_signal_counts),
        "found_target_verified_link_source_counts": dict(target_verified_source_counts),
        "quality_delta_flag_counts": dict(top_better_counts),
        "found_target_query_term_bucket_counts": dict(query_term_buckets),
        "public_note": (
            "Target rank diagnostics use only configured expected segment IDs and "
            "hashed/count/flag evidence-unit metadata; raw query text and raw IDs remain redacted."
        ),
    }


def _recall_coverage_diagnostics(
    query_rows: list[dict[str, Any]],
    artifact_summary: dict[str, Any],
) -> dict[str, Any]:
    current = _target_rank_inspection(query_rows)
    base_rows = []
    for row in query_rows:
        base = row.get("base_target_diagnostics")
        if isinstance(base, dict):
            cloned = dict(row)
            cloned["target_diagnostics"] = base
            base_rows.append(cloned)
    baseline = _target_rank_inspection(base_rows) if base_rows else None
    return {
        "schema_version": "evidence-unit-recall-coverage-comparison-public-v1",
        "current": {
            "target_configured_count": current.get("target_configured_count"),
            "target_found@10": current.get("target_found@10"),
            "target_found@50": current.get("target_found@50"),
            "target_found@100": current.get("target_found@100"),
            "target_found_bucket_counts": current.get("target_found_bucket_counts", {}),
        },
        "baseline": (
            {
                "kind": "base_evidence_unit_order",
                "target_configured_count": baseline.get("target_configured_count"),
                "target_found@10": baseline.get("target_found@10"),
                "target_found@50": baseline.get("target_found@50"),
                "target_found@100": baseline.get("target_found@100"),
                "target_found_bucket_counts": baseline.get("target_found_bucket_counts", {}),
            }
            if baseline is not None
            else {
                "kind": "not_configured",
                "available": False,
            }
        ),
        "search_field_coverage": _public_search_field_coverage(
            artifact_summary.get("search_field_coverage")
        ),
        "public_note": (
            "Use this block to compare the same public-safe evaluation slice across "
            "baseline/current runs: recall metrics are @10/@50/@100, while coverage "
            "contains field availability counts only."
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


def _modality_aware_rerank_inspection(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [_mapping(row.get("modality_aware_rerank")) for row in query_rows]
    enabled = [diag for diag in diagnostics if diag.get("enabled") is True]
    computed = [diag for diag in enabled if diag.get("status") == "computed"]
    query_types = Counter(str(diag.get("query_type") or "unknown") for diag in computed)
    failure_modes = Counter(str(diag.get("failure_mode") or "unknown") for diag in computed)
    component_presence = Counter()
    top_changed_count = 0
    base_buckets = Counter()
    reranked_buckets = Counter()
    for diag in computed:
        if diag.get("top_changed") is True:
            top_changed_count += 1
        base_buckets[str(diag.get("base_target_rank_bucket") or "unknown")] += 1
        reranked_buckets[str(diag.get("reranked_target_rank_bucket") or "unknown")] += 1
        for name in diag.get("component_names", []):
            if isinstance(name, str):
                component_presence[name] += 1
    return {
        "enabled_query_count": len(enabled),
        "computed_query_count": len(computed),
        "top_changed_count": top_changed_count,
        "query_type_counts": dict(query_types),
        "failure_mode_counts": dict(failure_modes),
        "base_target_rank_bucket_counts": dict(base_buckets),
        "reranked_target_rank_bucket_counts": dict(reranked_buckets),
        "score_component_presence_counts": dict(component_presence),
        "public_note": (
            "Failure modes separate candidate recall failure, rerank regression, "
            "ranking improvement/success, and remaining evidence-quality or ranking-tie cases."
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
        "concept_graph_records_total",
        "concept_nodes_total",
        "concept_relation_edges_total",
        "evidence_units_total",
    ]
    return {key: int(counts.get(key) or 0) for key in keys}


def _public_link_diagnostics(value: Any) -> dict[str, Any]:
    diagnostics = _mapping(value)
    candidate_support = _mapping(diagnostics.get("candidate_visual_support"))
    verified_alignment = _mapping(diagnostics.get("verified_object_alignment"))
    excluded_ocr = _mapping(diagnostics.get("excluded_ocr_engine_evidence"))
    return {
        "schema_version": diagnostics.get("schema_version") or LINK_DIAGNOSTICS_SCHEMA_VERSION,
        "evidence_units_total": int(diagnostics.get("evidence_units_total") or 0),
        "candidate_visual_support": {
            "units_with_candidate_visual_support": int(
                candidate_support.get("units_with_candidate_visual_support") or 0
            ),
            "units_with_candidate_link": int(candidate_support.get("units_with_candidate_link") or 0),
            "units_with_timestamp_fallback_link": int(
                candidate_support.get("units_with_timestamp_fallback_link") or 0
            ),
            "candidate_links": int(candidate_support.get("candidate_links") or 0),
            "timestamp_fallback_links": int(candidate_support.get("timestamp_fallback_links") or 0),
            "candidate_link_signal_counts": _public_count_map(
                candidate_support.get("candidate_link_signal_counts")
                or diagnostics.get("candidate_link_signal_counts"),
                CANDIDATE_LINK_SIGNAL_KEYS,
            ),
            "paper_claim_eligible": False,
        },
        "verified_object_alignment": {
            "units_with_verified_object_alignment": int(
                verified_alignment.get("units_with_verified_object_alignment") or 0
            ),
            "verified_links": int(verified_alignment.get("verified_links") or 0),
            "verified_link_source_counts": _public_count_map(
                verified_alignment.get("verified_link_source_counts")
                or diagnostics.get("verified_link_source_counts"),
                VERIFIED_LINK_SOURCE_KEYS,
            ),
            "timestamp_fallback_counted_as_verified": False,
            "paper_claim_eligible_units": int(
                verified_alignment.get("paper_claim_eligible_units") or 0
            ),
        },
        "excluded_ocr_engine_evidence": {
            "units_with_ocr_engine_evidence": int(
                excluded_ocr.get("units_with_ocr_engine_evidence") or 0
            ),
            "ocr_engine_entity_mentions": int(
                excluded_ocr.get("ocr_engine_entity_mentions") or 0
            ),
            "ocr_engine_links": int(excluded_ocr.get("ocr_engine_links") or 0),
            "counted_as_candidate_visual_support": False,
            "counted_as_verified_object_alignment": False,
        },
        "candidate_link_signal_counts": _public_count_map(
            diagnostics.get("candidate_link_signal_counts")
            or candidate_support.get("candidate_link_signal_counts"),
            CANDIDATE_LINK_SIGNAL_KEYS,
        ),
        "verified_link_source_counts": _public_count_map(
            diagnostics.get("verified_link_source_counts")
            or verified_alignment.get("verified_link_source_counts"),
            VERIFIED_LINK_SOURCE_KEYS,
        ),
        "public_note": diagnostics.get("public_note") or LINK_DIAGNOSTICS_PUBLIC_NOTE,
    }


def _public_ocr_engine_diagnostics(value: Any) -> dict[str, Any]:
    diagnostics = _mapping(value)
    return {
        "source": diagnostics.get("source") or "visual_entities_artifact",
        "role": diagnostics.get("role") or "baseline_fallback_diagnostic_only",
        "main_path_excludes_ocr_engine_entities": bool(
            diagnostics.get("main_path_excludes_ocr_engine_entities")
        ),
        "input_ocr_engine_entities_total": int(
            diagnostics.get("input_ocr_engine_entities_total") or 0
        ),
        "input_ocr_engine_links_total": int(
            diagnostics.get("input_ocr_engine_links_total") or 0
        ),
        "evidence_units_with_ocr_engine_evidence": int(
            diagnostics.get("evidence_units_with_ocr_engine_evidence") or 0
        ),
        "ocr_engine_entity_mentions": int(
            diagnostics.get("ocr_engine_entity_mentions") or 0
        ),
        "ocr_engine_link_mentions": int(diagnostics.get("ocr_engine_link_mentions") or 0),
        "counted_as_candidate_visual_support": False,
        "counted_as_verified_object_alignment": False,
        "public_note": diagnostics.get("public_note") or OCR_ENGINE_DIAGNOSTICS_PUBLIC_NOTE,
    }


def _empty_public_ocr_engine_diagnostics() -> dict[str, Any]:
    return _public_ocr_engine_diagnostics({})


def _loaded_ocr_engine_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    units_with_ocr = 0
    entity_mentions = 0
    link_mentions = 0
    for row in rows:
        quality = _mapping(row.get("source_quality"))
        entity_count = int(
            quality.get("ocr_engine_entity_count")
            or quality.get("excluded_ocr_engine_entity_count")
            or 0
        )
        link_count = int(
            quality.get("ocr_engine_link_count")
            or quality.get("excluded_ocr_engine_link_count")
            or 0
        )
        if quality.get("has_ocr_engine_evidence") is True or entity_count or link_count:
            units_with_ocr += 1
        entity_mentions += entity_count
        link_mentions += link_count
    return _public_ocr_engine_diagnostics(
        {
            "source": "evidence_units_artifact",
            "role": "baseline_fallback_diagnostic_only",
            "main_path_excludes_ocr_engine_entities": True,
            "evidence_units_with_ocr_engine_evidence": units_with_ocr,
            "ocr_engine_entity_mentions": entity_mentions,
            "ocr_engine_link_mentions": link_mentions,
        }
    )


def _public_concept_field_coverage(value: Any) -> dict[str, Any]:
    coverage = _mapping(value)
    return {
        "schema_version": coverage.get("schema_version") or CONCEPT_FIELD_COVERAGE_SCHEMA_VERSION,
        "source": coverage.get("source") or "unknown",
        "concept_graph_loaded": bool(coverage.get("concept_graph_loaded")),
        "concept_graph_records_total": int(coverage.get("concept_graph_records_total") or 0),
        "concept_nodes_total": int(coverage.get("concept_nodes_total") or 0),
        "concept_relation_edges_total": int(coverage.get("concept_relation_edges_total") or 0),
        "evidence_units_total": int(coverage.get("evidence_units_total") or 0),
        "evidence_units_with_concepts": int(coverage.get("evidence_units_with_concepts") or 0),
        "evidence_units_with_concept_relations": int(
            coverage.get("evidence_units_with_concept_relations") or 0
        ),
        "evidence_units_with_concept_search_text": int(
            coverage.get("evidence_units_with_concept_search_text") or 0
        ),
        "unit_concept_coverage_ratio": coverage.get("unit_concept_coverage_ratio"),
        "unit_concept_relation_coverage_ratio": coverage.get(
            "unit_concept_relation_coverage_ratio"
        ),
        "concept_mentions": int(coverage.get("concept_mentions") or 0),
        "concept_relation_mentions": int(coverage.get("concept_relation_mentions") or 0),
        "timestamp_only_concept_relation_mentions": int(
            coverage.get("timestamp_only_concept_relation_mentions") or 0
        ),
        "timestamp_only_counted_as_verified_object_alignment": False,
        "public_note": coverage.get("public_note") or CONCEPT_FIELD_PUBLIC_NOTE,
    }


def _empty_public_concept_field_coverage() -> dict[str, Any]:
    return _public_concept_field_coverage(
        {
            "schema_version": CONCEPT_FIELD_COVERAGE_SCHEMA_VERSION,
            "source": "dry_run",
            "public_note": CONCEPT_FIELD_PUBLIC_NOTE,
        }
    )


def _public_search_field_coverage(value: Any) -> dict[str, Any]:
    coverage = _mapping(value)
    field_counts = _mapping(coverage.get("field_unit_counts"))
    field_ratios = _mapping(coverage.get("field_unit_ratios"))
    fields = (
        "evidence_text",
        "semantic_text",
        "transcript_keywords",
        "concept_search_text",
        "visual_state_text",
        "visual_entity_text",
        "candidate_link_signal_summary",
        "verified_link_signal_summary",
    )
    return {
        "schema_version": coverage.get("schema_version") or SEARCH_FIELD_COVERAGE_SCHEMA_VERSION,
        "source": coverage.get("source") or "build_summary",
        "evidence_units_total": int(coverage.get("evidence_units_total") or 0),
        "field_unit_counts": {
            field: int(field_counts.get(field) or 0) for field in fields
        },
        "field_unit_ratios": {
            field: field_ratios.get(field) for field in fields
        },
        "units_with_link_signal_search_text": int(
            coverage.get("units_with_link_signal_search_text") or 0
        ),
        "unit_link_signal_search_text_ratio": coverage.get(
            "unit_link_signal_search_text_ratio"
        ),
        "public_note": coverage.get("public_note") or SEARCH_FIELD_PUBLIC_NOTE,
    }


def _empty_public_search_field_coverage(source: str) -> dict[str, Any]:
    return _public_search_field_coverage(
        {
            "schema_version": SEARCH_FIELD_COVERAGE_SCHEMA_VERSION,
            "source": source,
            "public_note": SEARCH_FIELD_PUBLIC_NOTE,
        }
    )


def _loaded_search_field_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "evidence_text",
        "semantic_text",
        "transcript_keywords",
        "concept_search_text",
        "visual_state_text",
        "visual_entity_text",
        "candidate_link_signal_summary",
        "verified_link_signal_summary",
    )
    counts = {field: 0 for field in fields}
    link_signal_count = 0
    for row in rows:
        if _text_for_coverage(row.get("evidence_text")):
            counts["evidence_text"] += 1
        if _text_for_coverage(row.get("semantic_text")):
            counts["semantic_text"] += 1
        if _string_list(row.get("transcript_keywords")):
            counts["transcript_keywords"] += 1
        if _has_concept_search_text(row, _mapping(row.get("source_quality"))):
            counts["concept_search_text"] += 1
        if _text_for_coverage(row.get("visual_state_text")):
            counts["visual_state_text"] += 1
        if _text_for_coverage(row.get("visual_entity_text")):
            counts["visual_entity_text"] += 1
        if _text_for_coverage(row.get("candidate_link_signal_summary")):
            counts["candidate_link_signal_summary"] += 1
        if _text_for_coverage(row.get("verified_link_signal_summary")):
            counts["verified_link_signal_summary"] += 1
        if _text_for_coverage(row.get("candidate_link_signal_summary")) or _text_for_coverage(
            row.get("verified_link_signal_summary")
        ):
            link_signal_count += 1
    total = len(rows)
    return _public_search_field_coverage(
        {
            "schema_version": SEARCH_FIELD_COVERAGE_SCHEMA_VERSION,
            "source": "loaded_existing",
            "evidence_units_total": total,
            "field_unit_counts": counts,
            "field_unit_ratios": {
                field: _ratio_or_none(count, total) for field, count in counts.items()
            },
            "units_with_link_signal_search_text": link_signal_count,
            "unit_link_signal_search_text_ratio": _ratio_or_none(link_signal_count, total),
            "public_note": SEARCH_FIELD_PUBLIC_NOTE,
        }
    )


def _loaded_concept_field_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    units_with_concepts = 0
    units_with_relations = 0
    units_with_search_text = 0
    concept_mentions = 0
    relation_mentions = 0
    timestamp_only_relation_mentions = 0
    for row in rows:
        quality = _mapping(row.get("source_quality"))
        concept_count = _concept_count(row, quality)
        relation_count = _concept_relation_count(row, quality)
        timestamp_only_count = int(quality.get("timestamp_only_concept_relation_count") or 0)
        if concept_count:
            units_with_concepts += 1
        if relation_count:
            units_with_relations += 1
        if _has_concept_search_text(row, quality):
            units_with_search_text += 1
        concept_mentions += concept_count
        relation_mentions += relation_count
        timestamp_only_relation_mentions += timestamp_only_count
    total = len(rows)
    return _public_concept_field_coverage(
        {
            "schema_version": CONCEPT_FIELD_COVERAGE_SCHEMA_VERSION,
            "source": "loaded_existing",
            "evidence_units_total": total,
            "evidence_units_with_concepts": units_with_concepts,
            "evidence_units_with_concept_relations": units_with_relations,
            "evidence_units_with_concept_search_text": units_with_search_text,
            "unit_concept_coverage_ratio": _ratio_or_none(units_with_concepts, total),
            "unit_concept_relation_coverage_ratio": _ratio_or_none(
                units_with_relations,
                total,
            ),
            "concept_mentions": concept_mentions,
            "concept_relation_mentions": relation_mentions,
            "timestamp_only_concept_relation_mentions": timestamp_only_relation_mentions,
            "public_note": CONCEPT_FIELD_PUBLIC_NOTE,
        }
    )


def _public_candidate_concept_field_coverage(candidate: dict[str, Any]) -> dict[str, Any]:
    source_quality = _mapping(candidate.get("source_quality"))
    return {
        "has_concept_search_text": _has_concept_search_text(candidate, source_quality),
        "concept_count": _concept_count(candidate, source_quality),
        "concept_label_count": int(
            source_quality.get("concept_label_count")
            or len(_string_list(candidate.get("concept_labels")))
        ),
        "concept_alias_count": int(
            source_quality.get("concept_alias_count")
            or len(_string_list(candidate.get("concept_aliases")))
        ),
        "concept_relation_count": _concept_relation_count(candidate, source_quality),
        "timestamp_only_concept_relation_count": int(
            source_quality.get("timestamp_only_concept_relation_count") or 0
        ),
        "timestamp_only_counted_as_verified_object_alignment": False,
    }


def _empty_public_link_diagnostics() -> dict[str, Any]:
    return _public_link_diagnostics(
        {
            "schema_version": LINK_DIAGNOSTICS_SCHEMA_VERSION,
            "candidate_visual_support": {},
            "verified_object_alignment": {},
        }
    )


def _link_diagnostics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_signal_counts = Counter()
    verified_source_counts = Counter()
    units_with_candidate_visual_support = 0
    units_with_candidate_link = 0
    units_with_timestamp_fallback_link = 0
    units_with_verified_object_alignment = 0
    candidate_links = 0
    timestamp_fallback_links = 0
    verified_links = 0
    for row in rows:
        source_quality = _mapping(row.get("source_quality"))
        candidate_count = int(source_quality.get("candidate_link_count") or 0)
        timestamp_count = int(source_quality.get("timestamp_fallback_link_count") or 0)
        verified_count = int(source_quality.get("verified_link_count") or 0)
        if not candidate_count:
            statuses = _mapping(row.get("candidate_entity_link_statuses"))
            candidate_count = sum(1 for status in statuses.values() if str(status) == "candidate")
        if not timestamp_count:
            statuses = _mapping(row.get("candidate_entity_link_statuses"))
            timestamp_count = sum(1 for status in statuses.values() if str(status) == "timestamp_fallback")
        if not verified_count:
            verified_count = len(_string_list(row.get("verified_entity_link_ids")))
        candidate_links += candidate_count
        timestamp_fallback_links += timestamp_count
        verified_links += verified_count
        if candidate_count:
            units_with_candidate_link += 1
        if timestamp_count:
            units_with_timestamp_fallback_link += 1

        support = _mapping(source_quality.get("candidate_visual_support"))
        if (
            support.get("has_candidate_visual_support") is True
            or source_quality.get("has_visual_state") is True
            or source_quality.get("has_visual_entity") is True
            or candidate_count
            or timestamp_count
        ):
            units_with_candidate_visual_support += 1
        verified = _mapping(source_quality.get("verified_object_alignment"))
        if verified.get("has_verified_object_alignment") is True or verified_count:
            units_with_verified_object_alignment += 1
        for key, value in _public_count_map(
            source_quality.get("candidate_link_signal_counts"),
            CANDIDATE_LINK_SIGNAL_KEYS,
        ).items():
            candidate_signal_counts[key] += value
        for key, value in _public_count_map(
            source_quality.get("verified_link_source_counts"),
            VERIFIED_LINK_SOURCE_KEYS,
        ).items():
            verified_source_counts[key] += value
    return _public_link_diagnostics(
        {
            "schema_version": LINK_DIAGNOSTICS_SCHEMA_VERSION,
            "evidence_units_total": len(rows),
            "candidate_visual_support": {
                "units_with_candidate_visual_support": units_with_candidate_visual_support,
                "units_with_candidate_link": units_with_candidate_link,
                "units_with_timestamp_fallback_link": units_with_timestamp_fallback_link,
                "candidate_links": candidate_links,
                "timestamp_fallback_links": timestamp_fallback_links,
                "candidate_link_signal_counts": dict(candidate_signal_counts),
            },
            "verified_object_alignment": {
                "units_with_verified_object_alignment": units_with_verified_object_alignment,
                "verified_links": verified_links,
                "verified_link_source_counts": dict(verified_source_counts),
                "paper_claim_eligible_units": units_with_verified_object_alignment,
            },
        }
    )


def _public_count_map(value: Any, keys: tuple[str, ...]) -> dict[str, int]:
    mapping = _mapping(value)
    return {key: int(mapping.get(key) or 0) for key in keys}


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


def _visual_vlm_coverage_summary(
    *,
    build_summary: dict[str, Any] | None,
    vlm_object_evidence_coverage: dict[str, Any],
) -> dict[str, Any]:
    build = _mapping(build_summary)
    counts = _mapping(build.get("counts"))
    visual_state = _public_visual_state_coverage(build.get("visual_state_coverage"))
    link_diagnostics = _mapping(build.get("link_diagnostics"))
    candidate_support = _mapping(link_diagnostics.get("candidate_visual_support"))
    verified_alignment = _mapping(link_diagnostics.get("verified_object_alignment"))
    ocr_engine_diagnostics = _public_ocr_engine_diagnostics(
        build.get("ocr_engine_diagnostics")
    )
    entity_coverage = _mapping(vlm_object_evidence_coverage.get("visual_entity_coverage"))
    unit_coverage = _mapping(vlm_object_evidence_coverage.get("evidence_unit_coverage"))
    entity_ratios = _mapping(entity_coverage.get("ratios"))
    unit_ratios = _mapping(unit_coverage.get("ratios"))
    evidence_units_total = int(
        counts.get("evidence_units_total")
        or unit_coverage.get("evidence_units_total")
        or visual_state.get("evidence_units_total")
        or 0
    )
    visual_entities_total = int(entity_coverage.get("visual_entities_total") or 0)
    candidate_link_units = int(
        counts.get("units_with_candidate_link")
        or unit_coverage.get("units_with_candidate_link")
        or candidate_support.get("units_with_candidate_link")
        or 0
    )
    verified_link_units = int(
        counts.get("units_with_verified_link")
        or unit_coverage.get("units_with_verified_link")
        or verified_alignment.get("units_with_verified_object_alignment")
        or 0
    )
    timestamp_fallback_units = int(
        counts.get("units_with_timestamp_fallback_link")
        or unit_coverage.get("units_with_timestamp_fallback_link")
        or candidate_support.get("units_with_timestamp_fallback_link")
        or 0
    )
    return {
        "status": vlm_object_evidence_coverage.get("status") or build.get("status"),
        "visual_state_interval": {
            "source": visual_state.get("source"),
            "visual_states_total": visual_state["visual_states_total"],
            "evidence_units_with_visual_state": visual_state[
                "evidence_units_with_visual_state"
            ],
            "evidence_units_total": evidence_units_total,
            "unit_coverage_ratio": visual_state.get("unit_coverage_ratio"),
            "coverage_gate_status": _mapping(visual_state.get("coverage_gate")).get("status"),
        },
        "visual_entity_channels": {
            "visual_entities_total": visual_entities_total,
            "ocr_only_entity_count": int(entity_coverage.get("ocr_only_entity_count") or 0),
            "ocr_only_ratio": entity_ratios.get("ocr_only")
            if "ocr_only" in entity_ratios
            else _ratio_or_none(
                int(entity_coverage.get("ocr_only_entity_count") or 0),
                visual_entities_total,
            ),
            "vlm_source_entity_count": int(entity_coverage.get("vlm_source_entity_count") or 0),
            "paper_quality_vlm_entity_count": int(
                entity_coverage.get("paper_quality_vlm_entity_count") or 0
            ),
            "vlm_object_description_entity_count": int(
                entity_coverage.get("vlm_object_description_entity_count") or 0
            ),
            "vlm_object_description_ratio": entity_ratios.get("vlm_object_description")
            if "vlm_object_description" in entity_ratios
            else _ratio_or_none(
                int(entity_coverage.get("vlm_object_description_entity_count") or 0),
                visual_entities_total,
            ),
            "detected_text_entity_count": int(entity_coverage.get("detected_text_entity_count") or 0),
            "detected_text_ratio": entity_ratios.get("detected_text")
            if "detected_text" in entity_ratios
            else _ratio_or_none(
                int(entity_coverage.get("detected_text_entity_count") or 0),
                visual_entities_total,
            ),
        },
        "evidence_unit_channels": {
            "evidence_units_total": evidence_units_total,
            "units_with_vlm_entity": int(unit_coverage.get("units_with_vlm_entity") or 0),
            "units_with_visual_description": int(
                unit_coverage.get("units_with_visual_description")
                or counts.get("units_with_visual_description")
                or 0
            ),
            "units_with_detected_text": int(
                unit_coverage.get("units_with_detected_text")
                or counts.get("units_with_detected_text")
                or 0
            ),
            "units_using_ocr_only": int(unit_coverage.get("units_using_ocr_only") or 0),
            "units_with_ocr_engine_evidence": int(
                counts.get("units_with_ocr_engine_evidence")
                or ocr_engine_diagnostics.get("evidence_units_with_ocr_engine_evidence")
                or 0
            ),
            "units_with_vlm_visible_text": int(
                counts.get("units_with_vlm_visible_text") or 0
            ),
            "vlm_entity_ratio": unit_ratios.get("units_with_vlm_entity")
            if "units_with_vlm_entity" in unit_ratios
            else _ratio_or_none(
                int(unit_coverage.get("units_with_vlm_entity") or 0),
                evidence_units_total,
            ),
            "visual_description_ratio": unit_ratios.get("units_with_visual_description")
            if "units_with_visual_description" in unit_ratios
            else _ratio_or_none(
                int(
                    unit_coverage.get("units_with_visual_description")
                    or counts.get("units_with_visual_description")
                    or 0
                ),
                evidence_units_total,
            ),
            "detected_text_ratio": unit_ratios.get("units_with_detected_text")
            if "units_with_detected_text" in unit_ratios
            else _ratio_or_none(
                int(
                    unit_coverage.get("units_with_detected_text")
                    or counts.get("units_with_detected_text")
                    or 0
                ),
                evidence_units_total,
            ),
        },
        "object_link_coverage": {
            "units_with_candidate_link": candidate_link_units,
            "candidate_links": int(
                counts.get("candidate_links")
                or candidate_support.get("candidate_links")
                or 0
            ),
            "units_with_verified_link": verified_link_units,
            "verified_links": int(
                counts.get("verified_links")
                or verified_alignment.get("verified_links")
                or 0
            ),
            "units_with_timestamp_fallback_link": timestamp_fallback_units,
            "timestamp_fallback_links": int(
                counts.get("timestamp_fallback_links")
                or candidate_support.get("timestamp_fallback_links")
                or 0
            ),
            "candidate_link_unit_ratio": _ratio_or_none(
                candidate_link_units,
                evidence_units_total,
            ),
            "verified_link_unit_ratio": _ratio_or_none(
                verified_link_units,
                evidence_units_total,
            ),
            "timestamp_fallback_unit_ratio": _ratio_or_none(
                timestamp_fallback_units,
                evidence_units_total,
            ),
            "timestamp_fallback_counted_as_verified": False,
        },
        "ocr_engine_diagnostics": {
            "role": ocr_engine_diagnostics["role"],
            "main_path_excludes_ocr_engine_entities": ocr_engine_diagnostics[
                "main_path_excludes_ocr_engine_entities"
            ],
            "input_ocr_engine_entities_total": ocr_engine_diagnostics[
                "input_ocr_engine_entities_total"
            ],
            "input_ocr_engine_links_total": ocr_engine_diagnostics[
                "input_ocr_engine_links_total"
            ],
            "evidence_units_with_ocr_engine_evidence": ocr_engine_diagnostics[
                "evidence_units_with_ocr_engine_evidence"
            ],
            "ocr_engine_entity_mentions": ocr_engine_diagnostics[
                "ocr_engine_entity_mentions"
            ],
            "ocr_engine_link_mentions": ocr_engine_diagnostics[
                "ocr_engine_link_mentions"
            ],
            "counted_as_candidate_visual_support": False,
            "counted_as_verified_object_alignment": False,
        },
        "public_note": (
            "OCR engine diagnostics, VLM object-description, visible/detected text, "
            "candidate links, verified links, and timestamp fallback links are reported "
            "as separate public-safe aggregates. OCR engine evidence is excluded from "
            "main candidate/verified visual support; timestamp-only fallback remains "
            "candidate/fallback support and is never counted as verified object alignment."
        ),
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


def _candidate_matches_expected(
    candidate: dict[str, Any],
    expected_segment_ids: set[str],
    expected_evidence_unit_ids: set[str] | None = None,
) -> bool | None:
    expected_evidence_unit_ids = expected_evidence_unit_ids or set()
    if not expected_segment_ids and not expected_evidence_unit_ids:
        return None
    evidence_unit_id = _optional_str(candidate.get("evidence_unit_id"))
    if evidence_unit_id and evidence_unit_id in expected_evidence_unit_ids:
        return True
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


def _has_concept_fields(candidate: dict[str, Any], source_quality: dict[str, Any]) -> bool:
    return bool(
        source_quality.get("has_concept")
        or _concept_count(candidate, source_quality) > 0
        or _string_list(candidate.get("concept_labels"))
        or _string_list(candidate.get("concept_aliases"))
        or _list_of_dicts(candidate.get("concepts"))
    )


def _has_concept_relation_fields(candidate: dict[str, Any], source_quality: dict[str, Any]) -> bool:
    return bool(
        source_quality.get("has_concept_relation")
        or _concept_relation_count(candidate, source_quality) > 0
        or _text_for_coverage(candidate.get("concept_relation_text")).strip()
        or _list_of_dicts(candidate.get("concept_relations"))
    )


def _has_concept_search_text(candidate: dict[str, Any], source_quality: dict[str, Any]) -> bool:
    return bool(
        source_quality.get("has_concept_search_text")
        or _text_for_coverage(candidate.get("concept_search_text")).strip()
        or _string_list(candidate.get("concept_labels"))
        or _string_list(candidate.get("concept_aliases"))
        or _text_for_coverage(candidate.get("concept_relation_text")).strip()
    )


def _concept_count(candidate: dict[str, Any], source_quality: dict[str, Any]) -> int:
    return int(
        source_quality.get("concept_count")
        or len(_string_list(candidate.get("concept_ids")))
        or len(_list_of_dicts(candidate.get("concepts")))
    )


def _concept_relation_count(candidate: dict[str, Any], source_quality: dict[str, Any]) -> int:
    return int(
        source_quality.get("concept_relation_count")
        or len(_list_of_dicts(candidate.get("concept_relations")))
        or (1 if _text_for_coverage(candidate.get("concept_relation_text")).strip() else 0)
    )


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


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


def _normalize_manifest(manifest: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    if manifest.get("schema_version") == PUBLIC_SAFE_SLICE_SCHEMA_VERSION:
        _validate_public_safe_slice_manifest(manifest, base_dir=base_dir)
    return manifest


def _validate_public_safe_slice_manifest(manifest: dict[str, Any], *, base_dir: Path) -> None:
    del base_dir  # Reserved for future relative private-query overlays.
    queries = _manifest_query_rows(manifest)
    min_targets = _positive_int(manifest.get("min_query_targets", 20), field_name="min_query_targets")
    if len(queries) < min_targets:
        raise ValueError(
            f"public-safe evaluation slice requires at least {min_targets} query targets"
        )
    banned_keys = {
        "query",
        "query_text",
        "query_text_ko",
        "raw_query",
        "reference_answer",
        "raw_answer",
        "answer_text",
        "evidence_text",
        "raw_evidence_text",
        "transcript_excerpt",
        "transcript_text",
        "private_path",
        "local_path",
    }
    for suite_index, suite in enumerate(_manifest_suites(manifest), start=1):
        if not isinstance(suite, dict):
            raise ValueError(f"public-safe suite #{suite_index} must be a JSON object")
        if not (_optional_str(suite.get("project_id")) or _optional_str(suite.get("project_dir"))):
            raise ValueError(f"public-safe suite #{suite_index} requires project_id or project_dir")
        for query_index, query in enumerate(_suite_queries(suite), start=1):
            present_banned = sorted(key for key in banned_keys if key in query)
            if present_banned:
                raise ValueError(
                    "public-safe query target must not contain raw/private fields: "
                    + ", ".join(present_banned)
                )
            if not _optional_str(query.get("query_id") or query.get("id")):
                raise ValueError(
                    f"public-safe suite #{suite_index} query #{query_index} requires query_id"
                )
            if not _expected_target_configured(query):
                raise ValueError(
                    f"public-safe suite #{suite_index} query #{query_index} requires a target id"
                )
            if not _optional_str(query.get("expected_modality")):
                raise ValueError(
                    f"public-safe suite #{suite_index} query #{query_index} requires expected_modality"
                )
            if not _optional_str(query.get("query_type")):
                raise ValueError(
                    f"public-safe suite #{suite_index} query #{query_index} requires query_type"
                )
            if _concept_alias_count(query) <= 0:
                raise ValueError(
                    f"public-safe suite #{suite_index} query #{query_index} requires concept_aliases"
                )


def _manifest_suites(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    suites = manifest.get("suites") or []
    if not isinstance(suites, list):
        raise ValueError("evidence-unit-smoke manifest requires a list under 'suites'")
    return [suite for suite in suites if isinstance(suite, dict)]


def _manifest_query_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in _manifest_suites(manifest):
        rows.extend(_suite_queries(suite))
    return rows


def _suite_queries(suite: dict[str, Any]) -> list[dict[str, Any]]:
    queries = suite.get("queries") or []
    if not isinstance(queries, list):
        raise ValueError("suite queries must be a list")
    return [query for query in queries if isinstance(query, dict)]


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
        value = _synthetic_public_safe_query_text(query_row)
    if not value:
        raise ValueError("query row requires query_text/query or public-safe concept_aliases")
    return value


def _query_id(query_row: dict[str, Any]) -> str:
    return _optional_str(query_row.get("query_id") or query_row.get("id")) or "query"


def _optional_public_label(query_row: dict[str, Any]) -> str | None:
    return _optional_str(query_row.get("query_label") or query_row.get("label"))


def _synthetic_public_safe_query_text(query_row: dict[str, Any]) -> str | None:
    aliases = _query_concept_aliases(query_row)
    if not aliases:
        return None
    parts = [
        " ".join(aliases),
        _optional_str(query_row.get("expected_modality")) or "",
        _optional_str(query_row.get("query_type")) or "",
    ]
    return " ".join(part for part in parts if part).strip()


def _query_concept_aliases(query_row: dict[str, Any]) -> list[str]:
    aliases = query_row.get("concept_aliases")
    if aliases is None:
        aliases = query_row.get("concept_alias")
    return _string_list(aliases)


def _concept_alias_count(query_row: dict[str, Any]) -> int:
    if "concept_alias_count" in query_row:
        return int(query_row.get("concept_alias_count") or 0)
    return len(_query_concept_aliases(query_row))


def _expected_segment_ids(query_row: dict[str, Any]) -> set[str]:
    values = set(_string_list(query_row.get("expected_segment_ids")))
    value = _optional_str(
        query_row.get("expected_segment_id")
        or query_row.get("target_segment_id")
        or query_row.get("gold_segment_id")
    )
    if value:
        values.add(value)
    return values


def _expected_evidence_unit_ids(query_row: dict[str, Any]) -> set[str]:
    values = set(
        _string_list(
            query_row.get("expected_evidence_unit_ids")
            or query_row.get("target_evidence_unit_ids")
        )
    )
    value = _optional_str(
        query_row.get("expected_evidence_unit_id")
        or query_row.get("target_evidence_unit_id")
    )
    if value:
        values.add(value)
    return values


def _expected_target_configured(query_row: dict[str, Any]) -> bool:
    if "target_configured" in query_row:
        return bool(query_row.get("target_configured"))
    return bool(_expected_segment_ids(query_row) or _expected_evidence_unit_ids(query_row))


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
