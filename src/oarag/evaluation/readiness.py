from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from oarag.core.io import write_json
from oarag.evaluation.benchmark import MATRIX_SCHEMA_VERSION
from oarag.evaluation.quality_gate import DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS


PAPER_READINESS_AUDIT_SCHEMA_VERSION = "paper-readiness-audit-v1"
PAPER_READINESS_AUDIT_JSON = "paper_readiness_audit.json"
PAPER_READINESS_AUDIT_MARKDOWN = "paper_readiness_audit.md"
PAPER_READINESS_DISCLAIMER = (
    "This audit is a private-safe readiness checklist and gap report. "
    "It is not a paper acceptance guarantee."
)

ANSWER_CITATION_METRICS = [
    "grounded_answer_ratio",
    "citation_coverage_ratio",
    "answer_citation_precision",
    "answer_citation_recall",
    "expected_citation_hit_ratio",
    "mean_unsupported_claim_count",
    "unsupported_claim_ratio",
]
PRIVACY_EXCLUSION_KEYS = [
    "raw_queries",
    "answer_text",
    "transcript_content",
    "transcript_excerpts",
    "candidate_evidence_text",
    "local_paths",
]
PRIVACY_KEY_ALIASES = {
    "raw_queries": ["raw_queries", "query_content"],
    "answer_text": ["answer_text"],
    "transcript_content": ["transcript_content", "transcript_excerpts"],
    "transcript_excerpts": ["transcript_excerpts", "transcript_content"],
    "candidate_evidence_text": ["candidate_evidence_text", "evidence_content"],
    "local_paths": ["local_paths"],
}


@dataclass(frozen=True)
class PaperReadinessAuditRun:
    output_dir: Path | None
    json_path: Path | None
    markdown_path: Path | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class _BundlePaths:
    experiment_manifest: Path
    metrics: Path | None
    query_results: Path | None
    quality_gate_result: Path | None
    reproducibility: Path | None
    semantic_smoke: Path | None


def audit_paper_readiness(
    *,
    experiment_manifest_path: Path,
    output_dir: Path | None = None,
    metrics_path: Path | None = None,
    query_results_path: Path | None = None,
    quality_gate_result_path: Path | None = None,
    reproducibility_path: Path | None = None,
    semantic_smoke_path: Path | None = None,
    required_variant_ids: Iterable[str] | None = None,
) -> PaperReadinessAuditRun:
    """Generate a private-safe paper readiness checklist from aggregate artifacts."""
    resolved_manifest_path = experiment_manifest_path.expanduser().resolve()
    experiment_manifest = _read_json(resolved_manifest_path)
    bundle = _bundle_paths(
        experiment_manifest_path=resolved_manifest_path,
        experiment_manifest=experiment_manifest,
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        quality_gate_result_path=quality_gate_result_path,
        reproducibility_path=reproducibility_path,
        semantic_smoke_path=semantic_smoke_path,
    )
    metrics = _read_optional_json(bundle.metrics)
    quality_gate_result = _read_optional_json(bundle.quality_gate_result)
    reproducibility = _read_optional_json(bundle.reproducibility)
    semantic_smoke = _read_optional_json(bundle.semantic_smoke)
    semantic_evidence = _semantic_evidence(
        metrics=metrics,
        query_results_path=bundle.query_results,
        semantic_smoke=semantic_smoke,
    )
    required_variants = list(required_variant_ids or DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS)

    checks = [
        _check_matrix_variant_coverage(metrics=metrics, required_variant_ids=required_variants),
        _check_semantic_live_smoke(semantic_evidence=semantic_evidence),
        _check_answer_citation_metrics(metrics=metrics),
        _check_quality_gate(quality_gate_result=quality_gate_result),
        _check_reproducibility(reproducibility=reproducibility),
        _check_privacy_statement(
            experiment_manifest=experiment_manifest,
            quality_gate_result=quality_gate_result,
            reproducibility=reproducibility,
        ),
    ]
    gaps = _gaps_from_checks(checks)
    payload = {
        "schema_version": PAPER_READINESS_AUDIT_SCHEMA_VERSION,
        "run_id": _optional_text(experiment_manifest.get("run_id") or metrics.get("run_id")),
        "ready": not gaps,
        "gap_count": len(gaps),
        "disclaimer": PAPER_READINESS_DISCLAIMER,
        "artifacts": _artifact_summary(bundle=bundle, base_dir=resolved_manifest_path.parent),
        "checks": checks,
        "gaps": gaps,
        "privacy": {
            "payload": "artifact_schema_and_aggregate_metadata_only",
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
            "raw_private_text": "excluded",
        },
    }

    resolved_output_dir = output_dir.expanduser().resolve() if output_dir is not None else None
    json_path: Path | None = None
    markdown_path: Path | None = None
    if resolved_output_dir is not None:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        json_path = resolved_output_dir / PAPER_READINESS_AUDIT_JSON
        markdown_path = resolved_output_dir / PAPER_READINESS_AUDIT_MARKDOWN
        write_json(json_path, payload)
        markdown_path.write_text(paper_readiness_markdown(payload), encoding="utf-8")

    return PaperReadinessAuditRun(
        output_dir=resolved_output_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        payload=payload,
    )


def paper_readiness_markdown(payload: Mapping[str, Any]) -> str:
    run_id = payload.get("run_id") or "unknown"
    status = "READY" if payload.get("ready") else "GAPS FOUND"
    lines = [
        f"# Paper Readiness Audit: {run_id}",
        "",
        f"> {PAPER_READINESS_DISCLAIMER}",
        "",
        "## Overall",
        "",
        f"- Status: {status}",
        f"- Gap count: {payload.get('gap_count')}",
        "",
        "## Checklist",
        "",
        "| check | status | summary |",
        "| --- | --- | --- |",
    ]
    for check in _list_of_dicts(payload.get("checks")):
        lines.append(
            "| {title} | {status} | {summary} |".format(
                title=_markdown_cell(str(check.get("title") or check.get("id") or "")),
                status=_markdown_cell(str(check.get("status") or "")),
                summary=_markdown_cell(str(check.get("summary") or "")),
            )
        )
    lines.extend(["", "## Gap Report", ""])
    gaps = _list_of_dicts(payload.get("gaps"))
    if gaps:
        for gap in gaps:
            lines.append(
                "- `{code}` ({check_id}): {message}".format(
                    code=gap.get("code"),
                    check_id=gap.get("check_id"),
                    message=str(gap.get("message") or ""),
                )
            )
    else:
        lines.append("- No readiness gaps found in the provided artifact bundle.")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "This output contains artifact names, schema metadata, aggregate counts, "
            "check statuses, and gap codes only. Raw queries, answer text, transcripts, "
            "candidate evidence, local filesystem paths, and raw private text are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def _bundle_paths(
    *,
    experiment_manifest_path: Path,
    experiment_manifest: Mapping[str, Any],
    metrics_path: Path | None,
    query_results_path: Path | None,
    quality_gate_result_path: Path | None,
    reproducibility_path: Path | None,
    semantic_smoke_path: Path | None,
) -> _BundlePaths:
    base_dir = experiment_manifest_path.parent
    artifacts = experiment_manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    return _BundlePaths(
        experiment_manifest=experiment_manifest_path,
        metrics=_resolve_override_or_artifact(metrics_path, artifacts.get("metrics"), base_dir),
        query_results=_resolve_override_or_artifact(
            query_results_path,
            artifacts.get("query_results"),
            base_dir,
        ),
        quality_gate_result=_resolve_override_or_artifact(
            quality_gate_result_path,
            artifacts.get("quality_gate_result"),
            base_dir,
        ),
        reproducibility=_resolve_override_or_artifact(
            reproducibility_path,
            artifacts.get("reproducibility_json"),
            base_dir,
        ),
        semantic_smoke=_resolve_override_or_artifact(
            semantic_smoke_path,
            artifacts.get("semantic_smoke"),
            base_dir,
        ),
    )


def _resolve_override_or_artifact(
    override_path: Path | None,
    artifact_ref: Any,
    base_dir: Path,
) -> Path | None:
    if override_path is not None:
        return override_path.expanduser().resolve()
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        return None
    path = Path(artifact_ref)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _check_matrix_variant_coverage(
    *,
    metrics: Mapping[str, Any],
    required_variant_ids: list[str],
) -> dict[str, Any]:
    suites = _matrix_suites(metrics)
    gaps: list[dict[str, Any]] = []
    suite_coverage: list[dict[str, Any]] = []
    if not suites:
        gaps.append(
            _gap(
                "matrix_suite_missing",
                "No retrieval_answer_matrix suite was found in metrics.json.",
            )
        )

    for suite in suites:
        variant_ids = _variant_ids(suite)
        missing = [variant_id for variant_id in required_variant_ids if variant_id not in variant_ids]
        if missing:
            gaps.append(
                _gap(
                    "matrix_variant_missing",
                    "A retrieval_answer_matrix suite is missing required variants.",
                )
            )
        suite_coverage.append(
            {
                "suite_id": _optional_text(suite.get("suite_id")),
                "schema_version": _optional_text(suite.get("schema_version")),
                "required_variant_count": len(required_variant_ids),
                "covered_variant_count": len(
                    [variant_id for variant_id in required_variant_ids if variant_id in variant_ids]
                ),
                "missing_variant_ids": missing,
            }
        )

    return _check(
        check_id="matrix_variant_coverage",
        title="Matrix Variant Coverage",
        passed=not gaps,
        summary=(
            "Required retrieval-answer matrix variants are covered."
            if not gaps
            else "Required retrieval-answer matrix variants are missing."
        ),
        evidence={
            "required_variant_ids": required_variant_ids,
            "matrix_suite_count": len(suites),
            "suite_coverage": suite_coverage,
        },
        gaps=gaps,
    )


def _check_semantic_live_smoke(
    *,
    semantic_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    passed = bool(semantic_evidence.get("explicit_smoke_ok")) or (
        int(semantic_evidence.get("semantic_query_result_count") or 0) > 0
    )
    gaps = []
    if not passed:
        gaps.append(
            _gap(
                "semantic_live_smoke_missing",
                (
                    "No semantic live smoke evidence was found. Provide a semantic smoke "
                    "artifact or query results with semantic hybrid retrieval evidence."
                ),
            )
        )
    return _check(
        check_id="semantic_live_smoke_evidence",
        title="Semantic Live Smoke Evidence",
        passed=passed,
        summary=(
            "Semantic live smoke evidence is present."
            if passed
            else "Semantic live smoke evidence is missing."
        ),
        evidence=dict(semantic_evidence),
        gaps=gaps,
    )


def _check_answer_citation_metrics(*, metrics: Mapping[str, Any]) -> dict[str, Any]:
    suites = _matrix_suites(metrics)
    gaps: list[dict[str, Any]] = []
    metric_coverage: list[dict[str, Any]] = []
    if not suites:
        gaps.append(
            _gap(
                "answer_citation_suite_missing",
                "Answer and citation metrics require a retrieval_answer_matrix suite.",
            )
        )
    for suite in suites:
        suite_missing = _missing_metrics(suite, ANSWER_CITATION_METRICS)
        variant_coverage: list[dict[str, Any]] = []
        if suite_missing:
            gaps.append(
                _gap(
                    "answer_citation_metric_missing",
                    "A retrieval_answer_matrix suite is missing answer/citation metrics.",
                )
            )
        for variant in _variant_metric_items(suite):
            missing = _missing_metrics(variant, ANSWER_CITATION_METRICS)
            if missing:
                gaps.append(
                    _gap(
                        "answer_citation_variant_metric_missing",
                        "A matrix variant is missing answer/citation metrics.",
                    )
                )
            variant_coverage.append(
                {
                    "variant_id": _optional_text(variant.get("variant_id")),
                    "missing_metric_names": missing,
                }
            )
        metric_coverage.append(
            {
                "suite_id": _optional_text(suite.get("suite_id")),
                "required_metric_names": ANSWER_CITATION_METRICS,
                "missing_metric_names": suite_missing,
                "variant_coverage": variant_coverage,
            }
        )
    return _check(
        check_id="answer_citation_metrics",
        title="Answer/Citation Metrics",
        passed=not gaps,
        summary=(
            "Answer and citation aggregate metrics are present."
            if not gaps
            else "Answer and citation aggregate metrics are incomplete."
        ),
        evidence={"suite_metric_coverage": metric_coverage},
        gaps=gaps,
    )


def _check_quality_gate(*, quality_gate_result: Mapping[str, Any]) -> dict[str, Any]:
    passed = quality_gate_result.get("passed") is True
    gaps: list[dict[str, Any]] = []
    if quality_gate_result.get("skipped") is True:
        gaps.append(_gap("quality_gate_skipped", "Quality gate was skipped."))
    elif not passed:
        gaps.append(_gap("quality_gate_not_passed", "Quality gate did not pass."))
    failures = _list_of_dicts(quality_gate_result.get("failures"))
    return _check(
        check_id="quality_gate_pass",
        title="Quality Gate Pass",
        passed=passed,
        summary="Quality gate passed." if passed else "Quality gate did not pass.",
        evidence={
            "schema_version": _optional_text(quality_gate_result.get("schema_version")),
            "gate_id": _optional_text(quality_gate_result.get("gate_id")),
            "passed": quality_gate_result.get("passed"),
            "skipped": bool(quality_gate_result.get("skipped")),
            "failure_count": _optional_int(quality_gate_result.get("failure_count")),
            "failure_codes": sorted(
                {str(failure.get("code")) for failure in failures if failure.get("code")}
            ),
        },
        gaps=gaps,
    )


def _check_reproducibility(*, reproducibility: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    if not reproducibility.get("schema_version"):
        missing.append("schema_version")
    if not reproducibility.get("run_id"):
        missing.append("run_id")
    if not ((reproducibility.get("commit") or {}).get("sha")):
        missing.append("commit.sha")
    if not ((reproducibility.get("command") or {}).get("argv")):
        missing.append("command.argv")
    if not ((reproducibility.get("artifacts") or {}).get("metrics")):
        missing.append("artifacts.metrics")
    if not _list_of_dicts(reproducibility.get("dataset_descriptors")):
        missing.append("dataset_descriptors")
    benchmark = reproducibility.get("benchmark") if isinstance(reproducibility, Mapping) else {}
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    for key in ("suite_count", "query_count", "deltas"):
        if benchmark.get(key) in (None, [], ""):
            missing.append(f"benchmark.{key}")

    gaps = [
        _gap(
            "reproducibility_metadata_missing",
            "Reproducibility metadata is incomplete.",
        )
    ] if missing else []
    return _check(
        check_id="reproducibility_metadata",
        title="Reproducibility Metadata",
        passed=not missing,
        summary=(
            "Reproducibility metadata is present."
            if not missing
            else "Reproducibility metadata is incomplete."
        ),
        evidence={
            "schema_version": _optional_text(reproducibility.get("schema_version")),
            "commit_sha_present": bool((reproducibility.get("commit") or {}).get("sha")),
            "command_present": bool((reproducibility.get("command") or {}).get("argv")),
            "dataset_descriptor_count": len(
                _list_of_dicts(reproducibility.get("dataset_descriptors"))
            ),
            "missing_fields": missing,
        },
        gaps=gaps,
    )


def _check_privacy_statement(
    *,
    experiment_manifest: Mapping[str, Any],
    quality_gate_result: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = {
        "experiment_manifest": experiment_manifest.get("privacy"),
        "quality_gate_result": quality_gate_result.get("privacy"),
        "reproducibility": reproducibility.get("privacy"),
    }
    coverage: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    for payload_name, payload in payloads.items():
        payload = payload if isinstance(payload, Mapping) else {}
        if not payload:
            missing_artifacts.append(payload_name)
        covered_keys = [
            key
            for key in PRIVACY_EXCLUSION_KEYS
            if _privacy_key_covered(payloads=[payload], key=key)
        ]
        coverage.append({"artifact": payload_name, "covered_exclusion_keys": covered_keys})

    missing_keys = [
        key
        for key in PRIVACY_EXCLUSION_KEYS
        if not _privacy_key_covered(payloads=payloads.values(), key=key)
    ]
    gaps = [
        _gap(
            "privacy_statement_missing",
            "One or more artifacts are missing the required private-safe exclusion statement.",
        )
    ] if missing_artifacts or missing_keys else []
    return _check(
        check_id="privacy_statement",
        title="Privacy Statement",
        passed=not missing_artifacts and not missing_keys,
        summary=(
            "Private-safe exclusion statements are present."
            if not missing_artifacts and not missing_keys
            else "Private-safe exclusion statements are incomplete."
        ),
        evidence={
            "privacy_coverage": coverage,
            "missing_artifacts": missing_artifacts,
            "missing_exclusion_keys": missing_keys,
        },
        gaps=gaps,
    )


def _semantic_evidence(
    *,
    metrics: Mapping[str, Any],
    query_results_path: Path | None,
    semantic_smoke: Mapping[str, Any],
) -> dict[str, Any]:
    metric_variant_count = 0
    for suite in _matrix_suites(metrics):
        for variant in _variant_metric_items(suite):
            config = variant.get("config") if isinstance(variant.get("config"), Mapping) else {}
            if config.get("hybrid_retrieval") is True:
                metric_variant_count += 1

    query_counts = _semantic_query_result_counts(query_results_path)
    return {
        "explicit_smoke_ok": _semantic_smoke_ok(semantic_smoke),
        "hybrid_metric_variant_count": metric_variant_count,
        "hybrid_query_result_count": query_counts["hybrid_query_result_count"],
        "semantic_query_result_count": query_counts["semantic_query_result_count"],
        "query_results_available": query_counts["query_results_available"],
    }


def _semantic_query_result_counts(query_results_path: Path | None) -> dict[str, Any]:
    counts = {
        "query_results_available": False,
        "hybrid_query_result_count": 0,
        "semantic_query_result_count": 0,
    }
    if query_results_path is None or not query_results_path.exists():
        return counts
    counts["query_results_available"] = True
    with query_results_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            config = row.get("config") if isinstance(row.get("config"), Mapping) else {}
            variant_id = str(row.get("variant_id") or "")
            hybrid = config.get("hybrid_retrieval") is True or "hybrid" in variant_id
            if not hybrid:
                continue
            counts["hybrid_query_result_count"] += 1
            top_candidate = (
                row.get("top_candidate") if isinstance(row.get("top_candidate"), Mapping) else {}
            )
            if top_candidate.get("retrieval_mode") == "semantic":
                counts["semantic_query_result_count"] += 1
    return counts


def _semantic_smoke_ok(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    for item in _walk_dicts(payload):
        for key in ("hybrid_embedder_live_smoke", "semantic_live_smoke", "semantic_smoke"):
            value = item.get(key)
            if isinstance(value, Mapping) and (value.get("ok") is True or value.get("passed") is True):
                return True
    return False


def _matrix_suites(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    suites = _list_of_dicts(metrics.get("suites"))
    return [
        suite
        for suite in suites
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
        return [value for value in variant_metrics.values() if isinstance(value, dict)]
    return []


def _missing_metrics(payload: Mapping[str, Any], metric_names: Iterable[str]) -> list[str]:
    return [
        metric_name
        for metric_name in metric_names
        if metric_name not in payload or payload.get(metric_name) is None
    ]


def _gaps_from_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for check in checks:
        for gap in _list_of_dicts(check.get("gaps")):
            item = dict(gap)
            item["check_id"] = check.get("id")
            gaps.append(item)
    return gaps


def _artifact_summary(*, bundle: _BundlePaths, base_dir: Path) -> dict[str, Any]:
    return {
        "experiment_manifest": _artifact_ref(bundle.experiment_manifest, base_dir=base_dir),
        "metrics": _artifact_ref(bundle.metrics, base_dir=base_dir),
        "query_results": _artifact_ref(bundle.query_results, base_dir=base_dir),
        "quality_gate_result": _artifact_ref(bundle.quality_gate_result, base_dir=base_dir),
        "reproducibility": _artifact_ref(bundle.reproducibility, base_dir=base_dir),
        "semantic_smoke": _artifact_ref(bundle.semantic_smoke, base_dir=base_dir),
    }


def _artifact_ref(path: Path | None, *, base_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _check(
    *,
    check_id: str,
    title: str,
    passed: bool,
    summary: str,
    evidence: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "status": "pass" if passed else "fail",
        "summary": summary,
        "evidence": evidence,
        "gaps": gaps,
    }


def _gap(code: str, message: str) -> dict[str, Any]:
    return {"code": code, "message": message}


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


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _walk_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _privacy_excludes(value: Any) -> bool:
    return "excluded" in str(value or "").lower()


def _privacy_key_covered(*, payloads: Iterable[Any], key: str) -> bool:
    aliases = PRIVACY_KEY_ALIASES.get(key, [key])
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        if any(_privacy_excludes(payload.get(alias)) for alias in aliases):
            return True
    return False


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
