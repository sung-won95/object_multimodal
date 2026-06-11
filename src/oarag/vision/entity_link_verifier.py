from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from oarag.core.io import write_json, write_jsonl
from oarag.retrieval.project_index import segment_artifact_path


VLM_ENTITY_LINK_VERIFIER_SCHEMA_VERSION = "oarag-vlm-entity-link-verifier-v1"
VLM_ENTITY_LINK_VERIFIER_REQUEST_SCHEMA_VERSION = (
    "vlm-entity-link-verifier-request-v1"
)
VLM_ENTITY_LINK_VERIFIER_REPORT_SCHEMA_VERSION = (
    "oarag-vlm-entity-link-verifier-report-public-v1"
)
VLM_ENTITY_LINK_VERIFIER_CACHE_SCHEMA_VERSION = (
    "oarag-vlm-entity-link-verifier-cache-public-v1"
)
VLM_ENTITY_LINK_VERIFIER_AUDIT_SCHEMA_VERSION = (
    "oarag-vlm-entity-link-verifier-human-audit-template-v1"
)
VLM_ENTITY_LINK_VERIFIER_SOURCE = "vlm_verifier"
DEFAULT_VERIFIER_PROMPT_TEMPLATE_VERSION = "vlm-entity-link-verifier-v1"
ALLOWED_DECISIONS = {"verified", "rejected", "uncertain"}


class VLMVerifierUnavailable(RuntimeError):
    pass


class VLMVerifierParseError(ValueError):
    pass


@dataclass(frozen=True)
class VLMVerifierConfig:
    backend: str
    model: str | None = None
    device: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: f"vlm_link_verify_{int(time.time())}")

    @property
    def model_version(self) -> str | None:
        value = self.options.get("model_version")
        return str(value) if value is not None else None

    @property
    def prompt_template_version(self) -> str:
        value = self.options.get("prompt_template_version")
        text = str(value).strip() if value is not None else ""
        return text or DEFAULT_VERIFIER_PROMPT_TEMPLATE_VERSION


def verify_entity_links_vlm(
    *,
    project_dir: Path,
    entity_links_path: Path | None = None,
    segments_path: Path | None = None,
    visual_entities_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    output_path: Path | None = None,
    cache_path: Path | None = None,
    report_path: Path | None = None,
    audit_template_path: Path | None = None,
    backend: str,
    model: str | None = None,
    device: str | None = None,
    options: Mapping[str, Any] | None = None,
    limit: int | None = None,
    skip_on_unavailable: bool = True,
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
    resolved_frames_manifest_path = _resolve_existing_project_path(
        project_dir=resolved_project_dir,
        path=frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
        artifact_name="frames manifest",
    )
    resolved_output_path = _resolve_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / "manifests" / "entity_links.vlm_verified.jsonl",
    )
    resolved_cache_path = (
        _resolve_output_path(
            project_dir=resolved_project_dir,
            path=cache_path,
            default=resolved_project_dir / "manifests" / "entity_links.vlm_verifier_cache.json",
        )
        if cache_path is not None
        else None
    )
    resolved_report_path = (
        _resolve_output_path(
            project_dir=resolved_project_dir,
            path=report_path,
            default=resolved_project_dir / "reports" / "vlm_entity_link_verifier.json",
        )
        if report_path is not None
        else None
    )
    resolved_audit_template_path = (
        _resolve_output_path(
            project_dir=resolved_project_dir,
            path=audit_template_path,
            default=(
                resolved_project_dir
                / "reports"
                / "vlm_entity_link_verifier_human_audit_template.jsonl"
            ),
        )
        if audit_template_path is not None
        else None
    )

    segment_rows = _read_jsonl(resolved_segments_path)
    visual_entity_rows = _read_jsonl(resolved_visual_entities_path)
    frame_rows = _read_jsonl(resolved_frames_manifest_path)
    link_rows = _read_jsonl(resolved_entity_links_path)
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be zero or greater")
        link_rows_to_evaluate = link_rows[:limit]
    else:
        link_rows_to_evaluate = link_rows

    segments_by_id = {_text(row.get("segment_id")): row for row in segment_rows}
    entities_by_id = {_text(row.get("entity_id")): row for row in visual_entity_rows}
    frames_by_id = {_text(row.get("frame_id")): row for row in frame_rows}
    cache = _load_cache(resolved_cache_path)
    config = VLMVerifierConfig(
        backend=_text(backend).casefold(),
        model=_optional_text(model),
        device=_optional_text(device),
        options=_resolve_verifier_options(resolved_project_dir, dict(options or {})),
    )

    started_at = time.perf_counter()
    decisions_by_link_id: dict[str, dict[str, Any]] = {}
    evaluated_link_ids = {_text(row.get("link_id")) for row in link_rows_to_evaluate}
    cached_count = 0
    fresh_count = 0
    skipped = False
    skip_reason: str | None = None

    try:
        uncached_links = [
            link
            for link in link_rows_to_evaluate
            if _text(link.get("link_id"))
            and _cached_decision(cache, _text(link.get("link_id"))) is None
        ]
        if uncached_links:
            _preflight_backend(config)
        for link in link_rows_to_evaluate:
            link_id = _text(link.get("link_id"))
            if not link_id:
                continue
            cached = _cached_decision(cache, link_id)
            if cached is not None:
                decisions_by_link_id[link_id] = cached
                cached_count += 1
                continue
            request = _request_payload(
                link=link,
                segment=segments_by_id.get(_text(link.get("segment_id")), {}),
                entity=entities_by_id.get(_text(link.get("entity_id")), {}),
                frame=frames_by_id.get(_text(link.get("frame_id")), {}),
                config=config,
            )
            decision = _run_backend_decision(request=request, config=config)
            decisions_by_link_id[link_id] = decision
            _store_cached_decision(cache, link_id, decision)
            fresh_count += 1
    except VLMVerifierUnavailable as exc:
        if not skip_on_unavailable:
            raise
        skipped = True
        skip_reason = str(exc)

    output_rows = [
        _apply_decision(row, decisions_by_link_id.get(_text(row.get("link_id"))))
        if _text(row.get("link_id")) in evaluated_link_ids
        else dict(row)
        for row in link_rows
    ]
    write_jsonl(resolved_output_path, output_rows)
    if resolved_cache_path is not None:
        _write_cache(resolved_cache_path, cache, config=config)

    elapsed_seconds = round(time.perf_counter() - started_at, 4)
    report = _public_report(
        links=output_rows,
        decisions=list(decisions_by_link_id.values()),
        total_links=len(link_rows),
        evaluated_links=len(link_rows_to_evaluate),
        cached_count=cached_count,
        fresh_count=fresh_count,
        skipped=skipped,
        skip_reason=skip_reason,
        config=config,
        elapsed_seconds=elapsed_seconds,
    )
    if resolved_report_path is not None:
        write_json(resolved_report_path, report)
    if resolved_audit_template_path is not None:
        _write_human_audit_template(
            resolved_audit_template_path,
            links=link_rows_to_evaluate,
            decisions=decisions_by_link_id,
            segments_by_id=segments_by_id,
        )

    return {
        "schema_version": VLM_ENTITY_LINK_VERIFIER_SCHEMA_VERSION,
        "status": "skipped" if skipped else "completed",
        "backend": config.backend,
        "source_model": config.model,
        "counts": report["counts"],
        "paths": {
            "project_dir": str(resolved_project_dir),
            "entity_links": str(resolved_entity_links_path),
            "segments": str(resolved_segments_path),
            "visual_entities": str(resolved_visual_entities_path),
            "frames_manifest": str(resolved_frames_manifest_path),
            "output": str(resolved_output_path),
            "cache": str(resolved_cache_path) if resolved_cache_path is not None else None,
            "report": str(resolved_report_path) if resolved_report_path is not None else None,
            "human_audit_template": (
                str(resolved_audit_template_path)
                if resolved_audit_template_path is not None
                else None
            ),
        },
        "public_report": report,
    }


def _request_payload(
    *,
    link: dict[str, Any],
    segment: dict[str, Any],
    entity: dict[str, Any],
    frame: dict[str, Any],
    config: VLMVerifierConfig,
) -> dict[str, Any]:
    frame_path = _optional_text(entity.get("frame_path")) or _optional_text(frame.get("frame_path"))
    return {
        "schema_version": VLM_ENTITY_LINK_VERIFIER_REQUEST_SCHEMA_VERSION,
        "prompt_template_version": config.prompt_template_version,
        "model": config.model,
        "link": {
            "link_id": _text(link.get("link_id")),
            "project_id": _text(link.get("project_id")),
            "segment_id": _text(link.get("segment_id")),
            "entity_id": _text(link.get("entity_id")),
            "frame_id": _text(link.get("frame_id")),
            "score": _optional_float(link.get("score")),
            "evidence": _list_of_text(link.get("evidence")),
        },
        "frame": {
            "frame_id": _text(link.get("frame_id") or entity.get("frame_id") or frame.get("frame_id")),
            "frame_path": frame_path,
            "timestamp": _optional_float(entity.get("timestamp"))
            if entity.get("timestamp") is not None
            else _optional_float(frame.get("timestamp")),
        },
        "segment": {
            "segment_id": _text(segment.get("segment_id")),
            "start_time": _optional_float(segment.get("start_time")),
            "end_time": _optional_float(segment.get("end_time")),
            "timestamp_center": _optional_float(segment.get("timestamp_center")),
            "transcript_text": _text(segment.get("transcript_text")),
            "mention_candidates": _list_of_text(segment.get("mention_candidates")),
            "semantic_text": _text(segment.get("semantic_text")),
        },
        "visual_entity": _visual_entity_request(entity),
        "response_contract": {
            "decision": sorted(ALLOWED_DECISIONS),
            "public_safe_only": True,
            "required_fields": ["decision", "reason_code"],
        },
    }


def _visual_entity_request(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": _text(entity.get("entity_id")),
        "frame_id": _text(entity.get("frame_id")),
        "entity_type": _text(entity.get("entity_type")),
        "text": _text(entity.get("text")),
        "visual_description": _text(entity.get("visual_description")),
        "detected_text": _text(entity.get("detected_text") or entity.get("visible_text")),
        "position": _mapping(entity.get("position")),
        "relations": _list_of_mappings(entity.get("relations")),
        "confidence": _optional_float(entity.get("confidence")),
        "source": _text(entity.get("source")),
        "source_model": _text(entity.get("source_model")),
    }


def _run_backend_decision(
    *,
    request: dict[str, Any],
    config: VLMVerifierConfig,
) -> dict[str, Any]:
    if config.backend == "jsonl":
        return _jsonl_decision(request=request, config=config)
    if config.backend == "command":
        return _command_decision(request=request, config=config)
    if config.backend in {"deterministic", "mock"}:
        return _configured_decision(request=request, config=config)
    raise VLMVerifierUnavailable(f"Unsupported VLM verifier backend: {config.backend}")


def _configured_decision(
    *,
    request: dict[str, Any],
    config: VLMVerifierConfig,
) -> dict[str, Any]:
    by_link = _mapping(config.options.get("decisions_by_link_id"))
    link_id = _text(_mapping(request.get("link")).get("link_id"))
    payload = _mapping(by_link.get(link_id)) if by_link else {}
    if not payload:
        payload = {
            "decision": _text(config.options.get("default_decision")) or "uncertain",
            "reason_code": _text(config.options.get("default_reason_code"))
            or "configured_default_decision",
            "confidence": _optional_float(config.options.get("confidence")),
        }
    return _normalize_decision(payload, request=request, config=config, cached=False)


def _jsonl_decision(
    *,
    request: dict[str, Any],
    config: VLMVerifierConfig,
) -> dict[str, Any]:
    path = _jsonl_backend_path(config.options)
    link_id = _text(_mapping(request.get("link")).get("link_id"))
    for row in _read_jsonl(path):
        if _text(row.get("link_id") or _mapping(row.get("link")).get("link_id")) == link_id:
            return _normalize_decision(row, request=request, config=config, cached=False)
    raise VLMVerifierParseError(f"JSONL verifier fixture has no decision for link_id={link_id}")


def _command_decision(
    *,
    request: dict[str, Any],
    config: VLMVerifierConfig,
) -> dict[str, Any]:
    raw_command = config.options.get("command")
    if raw_command is None:
        raise VLMVerifierUnavailable("VLM command verifier requires vlm_options.command")
    command = _command_parts(raw_command)
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
            timeout=_optional_float(config.options.get("timeout_seconds")) or None,
        )
    except FileNotFoundError as exc:
        raise VLMVerifierUnavailable(f"VLM command verifier not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VLMVerifierUnavailable("VLM command verifier timed out") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise VLMVerifierUnavailable(f"VLM command verifier failed: {message}")
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise VLMVerifierParseError("VLM command verifier returned invalid JSON") from exc
    return _normalize_decision(payload, request=request, config=config, cached=False)


def _preflight_backend(config: VLMVerifierConfig) -> None:
    if config.backend == "jsonl":
        path = _jsonl_backend_path(config.options)
        if not path.exists():
            raise VLMVerifierUnavailable(f"VLM verifier JSONL fixture not found: {path}")
        return
    if config.backend == "command":
        raw_command = config.options.get("preflight_command")
        if raw_command is None:
            return
        command = _command_parts(raw_command)
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=_optional_float(config.options.get("preflight_timeout_seconds")) or None,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "preflight failed"
            raise VLMVerifierUnavailable(f"VLM verifier preflight failed: {message}")
        return
    if config.backend in {"deterministic", "mock"}:
        return
    raise VLMVerifierUnavailable(f"Unsupported VLM verifier backend: {config.backend}")


def _normalize_decision(
    payload: Mapping[str, Any],
    *,
    request: dict[str, Any],
    config: VLMVerifierConfig,
    cached: bool,
) -> dict[str, Any]:
    parent = payload
    if isinstance(payload.get("decision"), Mapping):
        parent = payload["decision"]
    decision = _text(parent.get("decision") or parent.get("status")).casefold()
    if decision not in ALLOWED_DECISIONS:
        raise VLMVerifierParseError(
            f"VLM verifier decision must be one of {sorted(ALLOWED_DECISIONS)}"
        )
    reason_code = _public_reason_code(
        parent.get("reason_code")
        or parent.get("public_reason_code")
        or parent.get("reason")
        or f"vlm_decision_{decision}"
    )
    public_reason = _public_reason_text(parent.get("public_reason") or reason_code)
    confidence = _optional_float(parent.get("confidence"))
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise VLMVerifierParseError("VLM verifier confidence must be between 0 and 1")
    link = _mapping(request.get("link"))
    frame = _mapping(request.get("frame"))
    return {
        "schema_version": VLM_ENTITY_LINK_VERIFIER_SCHEMA_VERSION,
        "link_id": _text(link.get("link_id")),
        "decision": decision,
        "reason_code": reason_code,
        "public_reason": public_reason,
        "confidence": confidence,
        "backend": _text(parent.get("backend")) or config.backend,
        "source_model": _optional_text(parent.get("source_model")) or config.model,
        "model_version": _optional_text(parent.get("model_version")) or config.model_version,
        "prompt_template_version": config.prompt_template_version,
        "frame_ref": _short_hash(_text(frame.get("frame_id"))),
        "segment_ref": _short_hash(_text(_mapping(request.get("segment")).get("segment_id"))),
        "entity_ref": _short_hash(_text(_mapping(request.get("visual_entity")).get("entity_id"))),
        "cached": cached,
        "public_safe": True,
    }


def _apply_decision(
    row: dict[str, Any],
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    if decision is None:
        return dict(row)
    updated = dict(row)
    reason_metadata = dict(_mapping(updated.get("reason_metadata")))
    verifier_reason = {
        "source": VLM_ENTITY_LINK_VERIFIER_SOURCE,
        "verification_source": VLM_ENTITY_LINK_VERIFIER_SOURCE,
        "verifier": VLM_ENTITY_LINK_VERIFIER_SOURCE,
        "decision": decision["decision"],
        "reason_code": decision["reason_code"],
        "public_reason": decision["public_reason"],
        "confidence": decision.get("confidence"),
        "backend": decision.get("backend"),
        "source_model": decision.get("source_model"),
        "model_version": decision.get("model_version"),
        "prompt_template_version": decision.get("prompt_template_version"),
        "public_safe": True,
    }
    verifier_reason = {key: value for key, value in verifier_reason.items() if value is not None}
    reason_metadata[VLM_ENTITY_LINK_VERIFIER_SOURCE] = verifier_reason
    if decision["decision"] == "verified":
        reason_metadata["verification_source"] = VLM_ENTITY_LINK_VERIFIER_SOURCE
        reason_metadata["verifier"] = VLM_ENTITY_LINK_VERIFIER_SOURCE
        updated["alignment_status"] = "verified"
        updated["verification_status"] = "verified"
        updated["verified_link_source"] = VLM_ENTITY_LINK_VERIFIER_SOURCE
        updated["verification_source"] = VLM_ENTITY_LINK_VERIFIER_SOURCE
        updated["verifier"] = VLM_ENTITY_LINK_VERIFIER_SOURCE
    updated["reason_metadata"] = reason_metadata
    return updated


def _public_report(
    *,
    links: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    total_links: int,
    evaluated_links: int,
    cached_count: int,
    fresh_count: int,
    skipped: bool,
    skip_reason: str | None,
    config: VLMVerifierConfig,
    elapsed_seconds: float,
) -> dict[str, Any]:
    decision_counts = Counter(decision["decision"] for decision in decisions)
    verified_promotions = sum(
        1
        for link in links
        if _text(link.get("verification_source")) == VLM_ENTITY_LINK_VERIFIER_SOURCE
        and _text(link.get("verification_status")) == "verified"
    )
    return {
        "schema_version": VLM_ENTITY_LINK_VERIFIER_REPORT_SCHEMA_VERSION,
        "privacy": {
            "raw_transcript": "redacted",
            "raw_visual_text": "redacted",
            "frame_image_bytes": "omitted",
            "frame_paths": "omitted",
            "local_paths": "omitted",
        },
        "status": "skipped" if skipped else "completed",
        "skip": {
            "skipped": skipped,
            "reason": _public_reason_text(skip_reason) if skip_reason else None,
            "no_candidate_auto_promotion": True,
        },
        "backend": {
            "name": config.backend,
            "source_model_configured": bool(config.model),
            "model_version": config.model_version,
            "prompt_template_version": config.prompt_template_version,
        },
        "counts": {
            "links_total": total_links,
            "links_evaluated": 0 if skipped else evaluated_links,
            "links_cached": cached_count,
            "links_fresh": fresh_count,
            "verified": decision_counts["verified"],
            "rejected": decision_counts["rejected"],
            "uncertain": decision_counts["uncertain"],
            "promoted_links": verified_promotions,
        },
        "decision_reason_counts": dict(
            sorted(Counter(decision["reason_code"] for decision in decisions).items())
        ),
        "manual_audit": {
            "required_for_issue_completion": True,
            "minimum_random_sample_size": 50,
            "human_audit_completed": False,
            "agreement_rate": None,
            "note": (
                "Use the optional human audit template to record human_decision values "
                "after direct video review. This automated report does not claim #281 "
                "human-audit completion."
            ),
        },
        "elapsed_seconds": elapsed_seconds,
        "public_note": (
            "This report contains aggregate counts and public-safe reason codes only. "
            "Raw transcripts, visual text, frame image bytes, and local paths are omitted."
        ),
    }


def _write_human_audit_template(
    path: Path,
    *,
    links: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for link in links:
        link_id = _text(link.get("link_id"))
        decision = decisions.get(link_id)
        if decision is None:
            continue
        segment = segments_by_id.get(_text(link.get("segment_id")), {})
        rows.append(
            {
                "schema_version": VLM_ENTITY_LINK_VERIFIER_AUDIT_SCHEMA_VERSION,
                "link_id": link_id,
                "segment_ref": decision["segment_ref"],
                "entity_ref": decision["entity_ref"],
                "frame_ref": decision["frame_ref"],
                "public_lecture_ref": _public_lecture_ref(link, segment),
                "vlm_decision": decision["decision"],
                "vlm_reason_code": decision["reason_code"],
                "vlm_confidence": decision.get("confidence"),
                "human_decision": "",
                "human_notes": "",
            }
        )
    write_jsonl(path, rows)


def load_human_audit_agreement(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    completed = [
        row
        for row in rows
        if _text(row.get("human_decision")).casefold() in ALLOWED_DECISIONS
        and _text(row.get("vlm_decision")).casefold() in ALLOWED_DECISIONS
    ]
    matches = [
        row
        for row in completed
        if _text(row.get("human_decision")).casefold()
        == _text(row.get("vlm_decision")).casefold()
    ]
    return {
        "schema_version": "oarag-vlm-entity-link-verifier-human-audit-agreement-v1",
        "rows_total": len(rows),
        "completed_rows": len(completed),
        "matches": len(matches),
        "agreement_rate": round(len(matches) / len(completed), 6) if completed else None,
        "minimum_random_sample_size": 50,
        "meets_issue_281_sample_size": len(completed) >= 50,
        "public_safe": True,
    }


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "schema_version": VLM_ENTITY_LINK_VERIFIER_CACHE_SCHEMA_VERSION,
            "decisions": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"VLM verifier cache must be a JSON object: {path}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict):
        payload["decisions"] = {}
    return payload


def _cached_decision(cache: dict[str, Any], link_id: str) -> dict[str, Any] | None:
    decisions = _mapping(cache.get("decisions"))
    cached = decisions.get(link_id)
    if not isinstance(cached, dict):
        return None
    decision = dict(cached)
    decision["cached"] = True
    return decision


def _store_cached_decision(
    cache: dict[str, Any],
    link_id: str,
    decision: dict[str, Any],
) -> None:
    public_decision = {
        key: decision.get(key)
        for key in (
            "schema_version",
            "link_id",
            "decision",
            "reason_code",
            "public_reason",
            "confidence",
            "backend",
            "source_model",
            "model_version",
            "prompt_template_version",
            "frame_ref",
            "segment_ref",
            "entity_ref",
            "public_safe",
        )
    }
    public_decision = {
        key: value for key, value in public_decision.items() if value is not None
    }
    cache.setdefault("decisions", {})[link_id] = public_decision


def _write_cache(path: Path, cache: dict[str, Any], *, config: VLMVerifierConfig) -> None:
    payload = {
        "schema_version": VLM_ENTITY_LINK_VERIFIER_CACHE_SCHEMA_VERSION,
        "backend": config.backend,
        "prompt_template_version": config.prompt_template_version,
        "privacy": {
            "raw_transcript": "omitted",
            "frame_image_bytes": "omitted",
            "frame_paths": "omitted",
        },
        "decisions": _mapping(cache.get("decisions")),
    }
    write_json(path, payload)


def _jsonl_backend_path(options: Mapping[str, Any]) -> Path:
    for key in ("jsonl_path", "fixture_path", "input_path", "path", "input"):
        value = _optional_text(options.get(key))
        if value:
            return Path(value).expanduser().resolve()
    raise VLMVerifierUnavailable("VLM verifier JSONL backend requires jsonl_path")


def _resolve_verifier_options(project_dir: Path, options: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(options)
    for key in ("jsonl_path", "fixture_path", "input_path", "path", "input"):
        value = _optional_text(resolved.get(key))
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        resolved[key] = str(path.resolve())
    return resolved


def _command_parts(raw_command: Any) -> list[str]:
    if isinstance(raw_command, list):
        command = [str(item) for item in raw_command if str(item).strip()]
    elif isinstance(raw_command, str):
        command = shlex.split(raw_command)
    else:
        raise VLMVerifierUnavailable("VLM verifier command must be a string or list")
    if not command:
        raise VLMVerifierUnavailable("VLM verifier command must not be empty")
    return command


def _public_reason_code(value: Any) -> str:
    text = _text(value).casefold()
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "unspecified_public_reason"


def _public_reason_text(value: Any) -> str:
    text = " ".join(_text(value).split())
    if not text:
        return "unspecified_public_reason"
    return text[:180]


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


def _public_lecture_ref(link: dict[str, Any], segment: dict[str, Any]) -> str:
    video_id = _text(segment.get("video_id") or link.get("video_id"))
    project_id = _text(segment.get("project_id") or link.get("project_id"))
    if video_id:
        return f"video:{_short_hash(video_id)}"
    if project_id:
        return f"project:{_short_hash(project_id)}"
    return "unknown"


def _short_hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _list_of_text(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [_text(item) for item in items if _text(item)]
