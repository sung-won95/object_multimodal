from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oarag.core.io import write_json


PAPER_ARTIFACT_REGISTRY_SCHEMA_VERSION = "paper-artifact-registry-v1"
PAPER_ARTIFACT_REGISTRY_JSON = "paper_artifact_registry.json"


@dataclass(frozen=True)
class PaperArtifactRegistryRun:
    json_path: Path | None
    payload: dict[str, Any]


def build_paper_artifact_registry(
    *,
    experiment_manifest_path: Path,
    readiness_audit_path: Path,
    claim_matrix_path: Path,
    quality_gate_result_path: Path | None = None,
    robustness_path: Path | None = None,
    output_path: Path | None = None,
) -> PaperArtifactRegistryRun:
    """Build a private-safe registry of paper run artifacts and coarse statuses."""
    resolved_manifest_path = experiment_manifest_path.expanduser().resolve()
    resolved_readiness_path = readiness_audit_path.expanduser().resolve()
    resolved_claim_path = claim_matrix_path.expanduser().resolve()
    manifest = _read_json(resolved_manifest_path)
    readiness = _read_json(resolved_readiness_path)
    claims = _read_json(resolved_claim_path)

    resolved_gate_path = _quality_gate_path(
        explicit_path=quality_gate_result_path,
        experiment_manifest=manifest,
        manifest_dir=resolved_manifest_path.parent,
    )
    resolved_robustness_path = _robustness_path(
        explicit_path=robustness_path,
        experiment_manifest=manifest,
        manifest_dir=resolved_manifest_path.parent,
    )
    gate = _read_optional_json(resolved_gate_path)
    robustness = _read_optional_json(resolved_robustness_path)
    output = output_path.expanduser().resolve() if output_path is not None else None
    artifacts = _registry_artifacts(
        experiment_manifest_path=resolved_manifest_path,
        manifest=manifest,
        readiness_audit_path=resolved_readiness_path,
        claim_matrix_path=resolved_claim_path,
        quality_gate_result_path=resolved_gate_path,
        robustness_path=resolved_robustness_path,
    )
    status = {
        "gate": _gate_status(gate),
        "readiness": _readiness_status(readiness),
        "claims": _claims_status(claims),
        "robustness": _robustness_status(
            robustness,
            provided=resolved_robustness_path is not None,
        ),
    }

    payload = {
        "schema_version": PAPER_ARTIFACT_REGISTRY_SCHEMA_VERSION,
        "run_id": _first_text(manifest.get("run_id"), readiness.get("run_id"), claims.get("run_id")),
        "commit": {
            "sha": _commit_sha(manifest=manifest),
        },
        "artifacts": artifacts,
        "status": status,
        "status_links": _status_links(artifacts=artifacts, status=status),
        "privacy": {
            "payload": "run_id_commit_artifact_filenames_and_statuses_only",
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
            "private_eval_values": "excluded",
        },
    }
    if output is not None:
        write_json(output, payload)
    return PaperArtifactRegistryRun(json_path=output, payload=payload)


def _quality_gate_path(
    *,
    explicit_path: Path | None,
    experiment_manifest: Mapping[str, Any],
    manifest_dir: Path,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    artifacts = experiment_manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    artifact_ref = artifacts.get("quality_gate_result")
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        return None
    path = Path(artifact_ref)
    if path.is_absolute():
        return path
    return (manifest_dir / path).resolve()


def _robustness_path(
    *,
    explicit_path: Path | None,
    experiment_manifest: Mapping[str, Any],
    manifest_dir: Path,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    artifacts = experiment_manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    for key in ("robustness", "metric_intervals", "paper_metric_intervals"):
        artifact_ref = artifacts.get(key)
        if not isinstance(artifact_ref, str) or not artifact_ref.strip():
            continue
        path = Path(artifact_ref)
        if path.is_absolute():
            return path
        return (manifest_dir / path).resolve()
    return None


def _registry_artifacts(
    *,
    experiment_manifest_path: Path,
    manifest: Mapping[str, Any],
    readiness_audit_path: Path,
    claim_matrix_path: Path,
    quality_gate_result_path: Path | None,
    robustness_path: Path | None,
) -> dict[str, str | None]:
    artifacts = {
        "experiment_manifest": _artifact_name(experiment_manifest_path),
        "quality_gate_result": _artifact_name(quality_gate_result_path),
        "readiness_audit": _artifact_name(readiness_audit_path),
        "claim_matrix": _artifact_name(claim_matrix_path),
        "robustness": _artifact_name(robustness_path),
    }
    manifest_artifacts = manifest.get("artifacts")
    if isinstance(manifest_artifacts, Mapping):
        for key, value in manifest_artifacts.items():
            if key not in artifacts:
                artifacts[str(key)] = _artifact_ref_name(value)
    return artifacts


def _status_links(
    *,
    artifacts: Mapping[str, str | None],
    status: Mapping[str, str],
) -> dict[str, dict[str, str | None]]:
    return {
        "gate": {
            "artifact": artifacts.get("quality_gate_result"),
            "status": status.get("gate"),
        },
        "readiness": {
            "artifact": artifacts.get("readiness_audit"),
            "status": status.get("readiness"),
        },
        "claims": {
            "artifact": artifacts.get("claim_matrix"),
            "status": status.get("claims"),
        },
        "robustness": {
            "artifact": artifacts.get("robustness"),
            "status": status.get("robustness"),
        },
    }


def _commit_sha(*, manifest: Mapping[str, Any]) -> str | None:
    commit = manifest.get("commit") if isinstance(manifest.get("commit"), Mapping) else {}
    sha = commit.get("sha") if isinstance(commit, Mapping) else None
    if sha:
        return str(sha)
    return None


def _gate_status(payload: Mapping[str, Any]) -> str:
    if payload.get("passed") is True:
        return "passed"
    if payload.get("passed") is False:
        return "failed"
    if payload.get("skipped") is True:
        return "skipped"
    return "unknown"


def _readiness_status(payload: Mapping[str, Any]) -> str:
    if payload.get("ready") is True:
        return "ready"
    if payload.get("ready") is False:
        return "gaps_found"
    return "unknown"


def _claims_status(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    status = str(summary.get("overall_status") or "").strip()
    if status:
        return status
    statuses = {
        str(claim.get("status"))
        for claim in _list_of_dicts(payload.get("claims"))
        if claim.get("status")
    }
    if "blocked" in statuses:
        return "blocked"
    if "needs_evidence" in statuses:
        return "needs_evidence"
    if statuses == {"supported"}:
        return "supported"
    return "unknown"


def _robustness_status(payload: Mapping[str, Any], *, provided: bool) -> str:
    if not provided:
        return "missing"
    if not payload:
        return "unknown"
    if payload.get("schema_version") == "paper-metric-intervals-v1":
        return _metric_intervals_robustness_status(payload)
    for key in ("passed", "ok", "ready"):
        if payload.get(key) is True:
            return "passed"
        if payload.get(key) is False:
            return "failed"
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    for raw_status in (
        payload.get("status"),
        payload.get("result"),
        summary.get("status"),
        summary.get("robustness_status"),
        summary.get("overall_status"),
    ):
        status = _normalize_robustness_status(raw_status)
        if status != "unknown":
            return status
    return "unknown"


def _metric_intervals_robustness_status(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    for raw_status in (
        payload.get("status"),
        summary.get("status"),
        summary.get("robustness_status"),
    ):
        status = _normalize_robustness_status(raw_status)
        if status != "unknown":
            return status

    caveats = [item for item in payload.get("caveats", []) if item]
    paired_deltas = _list_of_dicts(payload.get("paired_deltas"))
    row_count = _optional_int(payload.get("row_count"))
    if row_count == 0 or caveats or not paired_deltas:
        return "needs_evidence"
    return "passed"


def _normalize_robustness_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"pass", "passed", "success", "succeeded", "ok", "ready"}:
        return "passed"
    if status in {"fail", "failed", "error", "blocked", "not_ready"}:
        return "failed"
    if status in {"needs_evidence", "caveated", "descriptive", "partial"}:
        return "needs_evidence"
    if status in {"skip", "skipped", "missing", "not_run"}:
        return "missing"
    return "unknown"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required JSON artifact not found: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in artifact: {path.name}")
    return payload


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _read_json(path)


def _artifact_name(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.name


def _artifact_ref_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().rstrip("/")
    if not text:
        return None
    return Path(text).name


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
