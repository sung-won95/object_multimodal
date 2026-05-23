from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from oarag.core.io import write_json
from oarag.evaluation.benchmark import MATRIX_SCHEMA_VERSION
from oarag.evaluation.quality_gate import DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS
from oarag.evaluation.readiness import ANSWER_CITATION_METRICS


PAPER_CLAIM_MATRIX_SCHEMA_VERSION = "paper-claim-evidence-matrix-v1"
PAPER_CLAIM_MATRIX_JSON = "claim_evidence_matrix.json"
PAPER_CLAIM_MATRIX_MARKDOWN = "claim_evidence_matrix.md"

SUPPORTED = "supported"
NEEDS_EVIDENCE = "needs_evidence"
BLOCKED = "blocked"
CLAIM_STATUSES = [SUPPORTED, NEEDS_EVIDENCE, BLOCKED]

PRIVACY_KEYS = [
    "raw_queries",
    "answer_text",
    "transcript_content",
    "candidate_evidence_text",
    "local_paths",
]
PRIVACY_ALIASES = {
    "raw_queries": ["raw_queries", "query_content", "raw_query_text", "query_text"],
    "answer_text": ["answer_text"],
    "transcript_content": ["transcript_content", "transcript_excerpts", "transcript_excerpt"],
    "candidate_evidence_text": ["candidate_evidence_text", "evidence_content"],
    "local_paths": ["local_paths"],
}
PRIVACY_SAFE_MARKERS = ("excluded", "redacted", "omitted", "hashed")


@dataclass(frozen=True)
class PaperClaimMatrixRun:
    output_dir: Path
    json_path: Path
    markdown_path: Path
    payload: dict[str, Any]


def build_paper_claims(
    *,
    metrics_path: Path,
    reproducibility_path: Path,
    quality_gate_result_path: Path,
    readiness_audit_path: Path,
    output_dir: Path | None = None,
    robustness_path: Path | None = None,
) -> PaperClaimMatrixRun:
    """Build a conservative, private-safe paper claim/evidence matrix."""
    resolved_metrics_path = metrics_path.expanduser().resolve()
    resolved_reproducibility_path = reproducibility_path.expanduser().resolve()
    resolved_gate_path = quality_gate_result_path.expanduser().resolve()
    resolved_readiness_path = readiness_audit_path.expanduser().resolve()
    resolved_robustness_path = robustness_path.expanduser().resolve() if robustness_path else None
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else resolved_metrics_path.parent / "paper_claims"
    )

    metrics = _read_json(resolved_metrics_path)
    reproducibility = _read_json(resolved_reproducibility_path)
    quality_gate_result = _read_json(resolved_gate_path)
    readiness_audit = _read_json(resolved_readiness_path)
    robustness = _read_optional_json(resolved_robustness_path)

    artifacts = {
        "metrics": _artifact_name(resolved_metrics_path),
        "reproducibility": _artifact_name(resolved_reproducibility_path),
        "quality_gate_result": _artifact_name(resolved_gate_path),
        "readiness_audit": _artifact_name(resolved_readiness_path),
        "robustness": _artifact_name(resolved_robustness_path),
    }
    context = _ClaimContext(
        metrics=metrics,
        reproducibility=reproducibility,
        quality_gate_result=quality_gate_result,
        readiness_audit=readiness_audit,
        robustness=robustness,
        artifacts=artifacts,
    )
    claims = _claims(context)
    status_counts = dict(Counter(claim["status"] for claim in claims))
    payload = {
        "schema_version": PAPER_CLAIM_MATRIX_SCHEMA_VERSION,
        "run_id": _first_text(
            metrics.get("run_id"),
            readiness_audit.get("run_id"),
            reproducibility.get("run_id"),
        ),
        "artifacts": artifacts,
        "summary": {
            "overall_status": _overall_status(claims),
            "status_counts": {status: status_counts.get(status, 0) for status in CLAIM_STATUSES},
            "quality_gate_status": _quality_gate_status(quality_gate_result),
            "readiness_status": _readiness_status(readiness_audit),
            "robustness_status": _robustness_status(robustness, provided=resolved_robustness_path is not None),
        },
        "claims": claims,
        "privacy": {
            "payload": "artifact_names_statuses_and_aggregate_schema_metadata_only",
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
            "private_eval_values": "excluded",
        },
    }

    json_path = resolved_output_dir / PAPER_CLAIM_MATRIX_JSON
    markdown_path = resolved_output_dir / PAPER_CLAIM_MATRIX_MARKDOWN
    write_json(json_path, payload)
    markdown_path.write_text(paper_claims_markdown(payload), encoding="utf-8")
    return PaperClaimMatrixRun(
        output_dir=resolved_output_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        payload=payload,
    )


def paper_claims_markdown(payload: Mapping[str, Any]) -> str:
    run_id = payload.get("run_id") or "unknown"
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    lines = [
        f"# Paper Claim/Evidence Matrix: {run_id}",
        "",
        "This matrix is conservative and private-safe. It records claim status, artifact names, "
        "and aggregate schema evidence only. Private eval values, raw queries, answer text, "
        "transcripts, candidate evidence text, and local filesystem paths are excluded.",
        "",
        "## Summary",
        "",
        f"- Overall status: `{summary.get('overall_status') or 'unknown'}`",
        f"- Quality gate: `{summary.get('quality_gate_status') or 'unknown'}`",
        f"- Readiness: `{summary.get('readiness_status') or 'unknown'}`",
        f"- Robustness: `{summary.get('robustness_status') or 'unknown'}`",
        "",
        "## Matrix",
        "",
        "| claim id | status | evidence artifacts | note |",
        "| --- | --- | --- | --- |",
    ]
    for claim in _list_of_dicts(payload.get("claims")):
        evidence_artifacts = ", ".join(
            sorted(
                {
                    str(item.get("artifact"))
                    for item in _list_of_dicts(claim.get("evidence"))
                    if item.get("artifact")
                }
            )
        )
        note = str(claim.get("summary") or "")
        missing = [str(item) for item in claim.get("missing_evidence", []) if item]
        blockers = [
            str(item.get("reason"))
            for item in _list_of_dicts(claim.get("blockers"))
            if item.get("reason")
        ]
        if missing:
            note = f"{note} Missing: {', '.join(missing)}".strip()
        if blockers:
            note = f"{note} Blocked by: {', '.join(blockers)}".strip()
        lines.append(
            "| {claim_id} | {status} | {artifacts} | {note} |".format(
                claim_id=_markdown_cell(str(claim.get("claim_id") or "")),
                status=_markdown_cell(str(claim.get("status") or "")),
                artifacts=_markdown_cell(evidence_artifacts or "-"),
                note=_markdown_cell(note),
            )
        )
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class _ClaimContext:
    metrics: Mapping[str, Any]
    reproducibility: Mapping[str, Any]
    quality_gate_result: Mapping[str, Any]
    readiness_audit: Mapping[str, Any]
    robustness: Mapping[str, Any]
    artifacts: Mapping[str, str | None]


def _claims(context: _ClaimContext) -> list[dict[str, Any]]:
    blockers = _global_blockers(
        readiness_audit=context.readiness_audit,
        quality_gate_result=context.quality_gate_result,
    )
    return [
        _claim_readiness(context=context, blockers=blockers),
        _claim_quality_gate(context=context, readiness_blockers=blockers),
        _claim_matrix_coverage(context=context, blockers=blockers),
        _claim_answer_citation_metrics(context=context, blockers=blockers),
        _claim_private_safe_bundle(context=context, blockers=blockers),
        _claim_robustness(context=context, blockers=blockers),
    ]


def _claim_readiness(*, context: _ClaimContext, blockers: list[dict[str, Any]]) -> dict[str, Any]:
    ready = context.readiness_audit.get("ready")
    if ready is True:
        status = _blocked_if_supported(status=SUPPORTED, blockers=blockers)
        summary = "Paper readiness audit has no reported gaps."
        local_blockers = blockers if status == BLOCKED else []
        missing: list[str] = []
    elif ready is False:
        status = BLOCKED
        summary = "Paper readiness audit reports gaps."
        local_blockers = [_readiness_blocker(context.readiness_audit)]
        missing = []
    else:
        status = NEEDS_EVIDENCE
        summary = "Paper readiness audit does not expose a ready flag."
        local_blockers = []
        missing = ["readiness_audit.ready"]
    return _claim(
        claim_id="readiness_audit_passed",
        claim="Paper readiness audit supports moving aggregate results into paper claims.",
        status=status,
        summary=summary,
        evidence=[
            {
                "artifact": context.artifacts["readiness_audit"],
                "fields": ["schema_version", "ready", "gaps[].code"],
            }
        ],
        missing_evidence=missing,
        blockers=local_blockers,
    )


def _claim_quality_gate(
    *,
    context: _ClaimContext,
    readiness_blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    gate = context.quality_gate_result
    if gate.get("passed") is True:
        status = SUPPORTED
        summary = "Retrieval quality gate passed."
        missing: list[str] = []
        blockers: list[dict[str, Any]] = []
    elif gate.get("skipped") is True or gate.get("passed") is None:
        status = NEEDS_EVIDENCE
        summary = "Retrieval quality gate evidence is missing or skipped."
        missing = ["quality_gate_result.passed"]
        blockers = []
    else:
        status = BLOCKED
        summary = "Retrieval quality gate did not pass."
        missing = []
        blockers = [_gate_blocker(gate)]
    if status == SUPPORTED and readiness_blockers:
        status = BLOCKED
        blockers = readiness_blockers
    return _claim(
        claim_id="quality_gate_passed",
        claim="Aggregate retrieval quality gate supports the reported experiment bundle.",
        status=status,
        summary=summary,
        evidence=[
            {
                "artifact": context.artifacts["quality_gate_result"],
                "fields": ["schema_version", "gate_id", "passed", "failure_count"],
            }
        ],
        missing_evidence=missing,
        blockers=blockers,
    )


def _claim_matrix_coverage(
    *,
    context: _ClaimContext,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    profile = _matrix_profile(context.metrics)
    missing = []
    if not profile["matrix_suite_present"]:
        missing.append("metrics.suites[retrieval_answer_matrix]")
    if profile["missing_variant_ids"]:
        missing.append("metrics.suites[].variants[].variant_id")
    status = SUPPORTED if not missing else NEEDS_EVIDENCE
    return _claim(
        claim_id="retrieval_answer_matrix_complete",
        claim="Retrieval-answer matrix includes the required comparison variants.",
        status=_blocked_if_supported(status=status, blockers=blockers),
        summary=(
            "Required retrieval-answer matrix variants are present."
            if status == SUPPORTED
            else "Required retrieval-answer matrix evidence is incomplete."
        ),
        evidence=[
            {
                "artifact": context.artifacts["metrics"],
                "fields": ["suites[].schema_version", "suites[].variants[].variant_id"],
                "schema_version": MATRIX_SCHEMA_VERSION,
                "required_variant_ids": DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS,
                "covered_variant_ids": profile["covered_variant_ids"],
            }
        ],
        missing_evidence=missing + profile["missing_variant_ids"],
        blockers=blockers if status == SUPPORTED else [],
    )


def _claim_answer_citation_metrics(
    *,
    context: _ClaimContext,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    profile = _answer_citation_profile(context.metrics)
    missing = []
    if not profile["matrix_suite_present"]:
        missing.append("metrics.suites[retrieval_answer_matrix]")
    missing.extend(profile["missing_metric_names"])
    status = SUPPORTED if not missing else NEEDS_EVIDENCE
    return _claim(
        claim_id="answer_citation_metrics_available",
        claim="Answer grounding and citation proxy metrics are present for matrix evidence.",
        status=_blocked_if_supported(status=status, blockers=blockers),
        summary=(
            "Answer/citation metric fields are present."
            if status == SUPPORTED
            else "Answer/citation metric fields are incomplete."
        ),
        evidence=[
            {
                "artifact": context.artifacts["metrics"],
                "fields": [f"suites[].variants[].{name}" for name in ANSWER_CITATION_METRICS],
                "metric_names": ANSWER_CITATION_METRICS,
            }
        ],
        missing_evidence=missing,
        blockers=blockers if status == SUPPORTED else [],
    )


def _claim_private_safe_bundle(
    *,
    context: _ClaimContext,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    payloads = [
        context.metrics,
        context.reproducibility,
        context.quality_gate_result,
        context.readiness_audit,
    ]
    missing = [
        key
        for key in PRIVACY_KEYS
        if not any(_privacy_key_covered(payload=payload, key=key) for payload in payloads)
    ]
    status = SUPPORTED if not missing else NEEDS_EVIDENCE
    return _claim(
        claim_id="private_safe_artifact_bundle",
        claim="Public-facing paper artifacts exclude raw private text, local paths, and eval values.",
        status=_blocked_if_supported(status=status, blockers=blockers),
        summary=(
            "Private-safe exclusion statements are present."
            if status == SUPPORTED
            else "Private-safe exclusion statements are incomplete."
        ),
        evidence=[
            {
                "artifact": context.artifacts["metrics"],
                "fields": ["suites[].privacy"],
            },
            {
                "artifact": context.artifacts["reproducibility"],
                "fields": ["privacy"],
            },
            {
                "artifact": context.artifacts["quality_gate_result"],
                "fields": ["privacy"],
            },
            {
                "artifact": context.artifacts["readiness_audit"],
                "fields": ["privacy"],
            },
        ],
        missing_evidence=missing,
        blockers=blockers if status == SUPPORTED else [],
    )


def _claim_robustness(
    *,
    context: _ClaimContext,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    status = _robustness_status(context.robustness, provided=context.artifacts["robustness"] is not None)
    if status == "passed":
        claim_status = _blocked_if_supported(status=SUPPORTED, blockers=blockers)
        missing: list[str] = []
        local_blockers = blockers if claim_status == BLOCKED else []
        summary = "Robustness artifact reports a passing status."
    elif status == "failed":
        claim_status = BLOCKED
        missing = []
        local_blockers = [{"reason": "robustness_not_passed"}]
        summary = "Robustness artifact reports a failing status."
    else:
        claim_status = NEEDS_EVIDENCE
        missing = ["robustness artifact"]
        local_blockers = []
        summary = "Robustness artifact is not yet available."
    return _claim(
        claim_id="robustness_artifact_recorded",
        claim="Robustness evidence is registered before final paper claims are treated as complete.",
        status=claim_status,
        summary=summary,
        evidence=[
            {
                "artifact": context.artifacts["robustness"],
                "fields": ["schema_version", "status", "passed", "ok"],
            }
        ],
        missing_evidence=missing,
        blockers=local_blockers,
    )


def _claim(
    *,
    claim_id: str,
    claim: str,
    status: str,
    summary: str,
    evidence: list[dict[str, Any]],
    missing_evidence: list[str],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "missing_evidence": [item for item in missing_evidence if item],
        "blockers": blockers,
    }


def _global_blockers(
    *,
    readiness_audit: Mapping[str, Any],
    quality_gate_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if readiness_audit.get("ready") is False:
        blockers.append(_readiness_blocker(readiness_audit))
    if quality_gate_result.get("passed") is False:
        blockers.append(_gate_blocker(quality_gate_result))
    return blockers


def _readiness_blocker(readiness_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reason": "paper_readiness_audit_not_ready",
        "gap_codes": sorted(
            {
                str(gap.get("code"))
                for gap in _list_of_dicts(readiness_audit.get("gaps"))
                if gap.get("code")
            }
        ),
    }


def _gate_blocker(quality_gate_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reason": "quality_gate_not_passed",
        "failure_codes": sorted(
            {
                str(failure.get("code"))
                for failure in _list_of_dicts(quality_gate_result.get("failures"))
                if failure.get("code")
            }
        ),
    }


def _blocked_if_supported(*, status: str, blockers: list[dict[str, Any]]) -> str:
    if status == SUPPORTED and blockers:
        return BLOCKED
    return status


def _matrix_profile(metrics: Mapping[str, Any]) -> dict[str, Any]:
    suites = _matrix_suites(metrics)
    covered = set[str]()
    for suite in suites:
        covered.update(_variant_ids(suite))
    missing = [
        variant_id
        for variant_id in DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS
        if variant_id not in covered
    ]
    return {
        "matrix_suite_present": bool(suites),
        "covered_variant_ids": sorted(covered),
        "missing_variant_ids": missing,
    }


def _answer_citation_profile(metrics: Mapping[str, Any]) -> dict[str, Any]:
    suites = _matrix_suites(metrics)
    missing = set[str]()
    for suite in suites:
        for metric_name in ANSWER_CITATION_METRICS:
            if suite.get(metric_name) is None:
                missing.add(metric_name)
        for variant in _variant_metric_items(suite):
            for metric_name in ANSWER_CITATION_METRICS:
                if variant.get(metric_name) is None:
                    missing.add(metric_name)
    return {
        "matrix_suite_present": bool(suites),
        "missing_metric_names": sorted(missing),
    }


def _matrix_suites(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        suite
        for suite in _list_of_dicts(metrics.get("suites"))
        if suite.get("suite_type") == "retrieval_answer_matrix"
        or suite.get("schema_version") == MATRIX_SCHEMA_VERSION
    ]


def _variant_ids(suite: Mapping[str, Any]) -> set[str]:
    return {
        str(variant.get("variant_id"))
        for variant in _variant_metric_items(suite)
        if variant.get("variant_id")
    }


def _variant_metric_items(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = _list_of_dicts(suite.get("variants"))
    if variants:
        return variants
    variant_metrics = suite.get("variant_metrics")
    if isinstance(variant_metrics, Mapping):
        return [dict(value) for value in variant_metrics.values() if isinstance(value, Mapping)]
    return []


def _quality_gate_status(payload: Mapping[str, Any]) -> str:
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


def _robustness_status(payload: Mapping[str, Any], *, provided: bool) -> str:
    if not provided:
        return "missing"
    if not payload:
        return "unknown"
    for key in ("passed", "ok", "ready"):
        if payload.get(key) is True:
            return "passed"
        if payload.get(key) is False:
            return "failed"
    status = str(payload.get("status") or payload.get("result") or "").strip().lower()
    if status in {"pass", "passed", "success", "succeeded", "ok", "ready"}:
        return "passed"
    if status in {"fail", "failed", "error", "blocked", "not_ready"}:
        return "failed"
    if status in {"skip", "skipped", "missing", "not_run"}:
        return "missing"
    return "unknown"


def _overall_status(claims: list[dict[str, Any]]) -> str:
    statuses = {str(claim.get("status")) for claim in claims}
    if BLOCKED in statuses:
        return BLOCKED
    if NEEDS_EVIDENCE in statuses:
        return NEEDS_EVIDENCE
    if statuses == {SUPPORTED}:
        return SUPPORTED
    return "unknown"


def _privacy_key_covered(*, payload: Any, key: str) -> bool:
    aliases = PRIVACY_ALIASES.get(key, [key])
    for item in _walk_dicts(payload):
        for alias in aliases:
            value = item.get(alias)
            if any(marker in str(value or "").lower() for marker in PRIVACY_SAFE_MARKERS):
                return True
            if isinstance(value, bool) and value is False and alias.startswith("raw_"):
                return True
    return False


def _walk_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required JSON artifact not found: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in artifact: {path.name}")
    return payload


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    return _read_json(path)


def _artifact_name(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.name


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


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
