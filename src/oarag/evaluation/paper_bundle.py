from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from oarag.core.io import write_json
from oarag.evaluation.benchmark import SearchClient
from oarag.evaluation.claims import PaperClaimMatrixRun, build_paper_claims
from oarag.evaluation.experiment import (
    PaperExperimentError,
    PaperExperimentRun,
    run_paper_experiment,
)
from oarag.evaluation.metric_intervals import (
    DEFAULT_BOOTSTRAP_SAMPLE_COUNT,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    PaperMetricIntervalsRun,
    generate_metric_intervals,
)
from oarag.evaluation.paper_registry import (
    PAPER_ARTIFACT_REGISTRY_JSON,
    PaperArtifactRegistryRun,
    build_paper_artifact_registry,
)
from oarag.evaluation.readiness import PaperReadinessAuditRun, audit_paper_readiness


PAPER_BUNDLE_RESULT_SCHEMA_VERSION = "paper-bundle-result-v1"
PAPER_BUNDLE_RESULT_JSON = "paper_bundle_result.json"


class PaperBundleError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        result_path: Path | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"paper bundle failed during {stage}: {message}")
        self.stage = stage
        self.result_path = result_path
        self.payload = payload or {}


@dataclass(frozen=True)
class PaperBundleRun:
    run_id: str | None
    output_dir: Path
    experiment: PaperExperimentRun
    metric_intervals: PaperMetricIntervalsRun
    readiness: PaperReadinessAuditRun
    claims: PaperClaimMatrixRun
    registry: PaperArtifactRegistryRun
    result_path: Path
    payload: dict[str, Any]


def run_paper_bundle(
    *,
    client: SearchClient,
    manifest_path: Path,
    output_dir: Path,
    run_id: str | None = None,
    gate_config_path: Path | None = None,
    baseline_variant_id: str | None = None,
    metrics: Iterable[str] | None = None,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    sample_count: int = DEFAULT_BOOTSTRAP_SAMPLE_COUNT,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    repo_root: Path | None = None,
    diagnostic_top_k: int | None = None,
    fail_on_gate: bool = True,
    command: list[str] | None = None,
) -> PaperBundleRun:
    """Run the full private-safe paper artifact bundle pipeline."""
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = resolved_output_dir / PAPER_BUNDLE_RESULT_JSON
    artifacts: dict[str, Path | None] = {"paper_bundle_result": result_path}
    stages: list[dict[str, Any]] = []
    command = command or []

    def fail(stage: str, exc: Exception) -> None:
        artifacts.update(_collect_existing_artifacts(resolved_output_dir))
        payload = _bundle_result_payload(
            run_id=run_id,
            output_dir=resolved_output_dir,
            status="failed",
            stages=[
                *stages,
                {
                    "stage": stage,
                    "status": "failed",
                    "message": _sanitize_message(str(exc)),
                },
            ],
            artifacts=artifacts,
            failed_stage=stage,
            error_message=str(exc),
            registry_payload=None,
        )
        write_json(result_path, payload)
        raise PaperBundleError(
            stage,
            str(exc),
            result_path=result_path,
            payload=payload,
        ) from exc

    try:
        experiment = run_paper_experiment(
            client=client,
            manifest_path=manifest_path,
            output_dir=resolved_output_dir,
            run_id=run_id,
            gate_config_path=gate_config_path,
            repo_root=repo_root,
            diagnostic_top_k=diagnostic_top_k,
            fail_on_gate=fail_on_gate,
            command=command,
        )
    except PaperExperimentError as exc:
        stage = f"run_paper_experiment.{exc.stage}" if exc.stage else "run_paper_experiment"
        fail(stage, exc)
    except Exception as exc:
        fail("run_paper_experiment", exc)

    artifacts.update(
        {
            "metrics": experiment.benchmark.metrics_path,
            "metrics_summary": experiment.benchmark.metrics_csv_path,
            "query_results": experiment.benchmark.query_results_path,
            "semantic_smoke": experiment.benchmark.semantic_smoke_path,
            "summary": experiment.benchmark.summary_path,
            "paper_report_dir": experiment.report.output_dir,
            "paper_table_csv": experiment.report.paper_table_csv_path,
            "paper_table_markdown": experiment.report.paper_table_markdown_path,
            "reproducibility_json": experiment.report.reproducibility_json_path,
            "quality_gate_result": experiment.quality_gate_result_path,
            "experiment_manifest": experiment.experiment_manifest_path,
        }
    )
    stages.append(
        _stage_record(
            stage="run_paper_experiment",
            status="passed",
            output_dir=resolved_output_dir,
            artifacts=[
                experiment.benchmark.metrics_path,
                experiment.benchmark.query_results_path,
                experiment.benchmark.semantic_smoke_path,
                experiment.report.output_dir,
                experiment.quality_gate_result_path,
                experiment.experiment_manifest_path,
            ],
            details={
                "gate_status": "passed"
                if experiment.quality_gate_result.get("passed") is True
                else "skipped"
                if experiment.quality_gate_result.get("skipped") is True
                else "failed",
            },
        )
    )

    try:
        metric_intervals = generate_metric_intervals(
            query_results_path=experiment.benchmark.query_results_path,
            output_dir=resolved_output_dir / "paper_metric_intervals",
            metrics_path=experiment.benchmark.metrics_path,
            report_path=experiment.report.paper_table_markdown_path,
            baseline_variant_id=baseline_variant_id,
            metrics=metrics,
            seed=seed,
            sample_count=sample_count,
            confidence_level=confidence_level,
        )
    except Exception as exc:
        fail("generate_metric_intervals", exc)

    artifacts.update(
        {
            "metric_intervals_json": metric_intervals.json_path,
            "metric_intervals_csv": metric_intervals.csv_path,
            "metric_intervals_markdown": metric_intervals.markdown_path,
            "robustness": metric_intervals.json_path,
        }
    )
    stages.append(
        _stage_record(
            stage="generate_metric_intervals",
            status="passed",
            output_dir=resolved_output_dir,
            artifacts=[
                metric_intervals.json_path,
                metric_intervals.csv_path,
                metric_intervals.markdown_path,
            ],
            details={
                "robustness_status": metric_intervals.payload.get("status"),
                "caveat_count": (metric_intervals.payload.get("summary") or {}).get(
                    "caveat_count"
                ),
            },
        )
    )

    try:
        readiness = audit_paper_readiness(
            experiment_manifest_path=experiment.experiment_manifest_path,
            output_dir=resolved_output_dir / "paper_readiness",
        )
    except Exception as exc:
        fail("audit_paper_readiness", exc)

    artifacts["readiness_audit"] = readiness.json_path
    stages.append(
        _stage_record(
            stage="audit_paper_readiness",
            status="passed",
            output_dir=resolved_output_dir,
            artifacts=[readiness.json_path, readiness.markdown_path],
            details={
                "readiness_status": "ready" if readiness.payload.get("ready") else "gaps_found",
                "gap_count": readiness.payload.get("gap_count"),
            },
        )
    )

    try:
        claims = build_paper_claims(
            metrics_path=experiment.benchmark.metrics_path,
            reproducibility_path=experiment.report.reproducibility_json_path,
            quality_gate_result_path=experiment.quality_gate_result_path,
            readiness_audit_path=readiness.json_path or resolved_output_dir,
            robustness_path=metric_intervals.json_path,
            output_dir=resolved_output_dir / "paper_claims",
        )
    except Exception as exc:
        fail("build_paper_claims", exc)

    artifacts["claim_matrix"] = claims.json_path
    stages.append(
        _stage_record(
            stage="build_paper_claims",
            status="passed",
            output_dir=resolved_output_dir,
            artifacts=[claims.json_path, claims.markdown_path],
            details={
                "claims_status": (claims.payload.get("summary") or {}).get("overall_status"),
                "robustness_status": (claims.payload.get("summary") or {}).get(
                    "robustness_status"
                ),
            },
        )
    )

    try:
        registry = build_paper_artifact_registry(
            experiment_manifest_path=experiment.experiment_manifest_path,
            readiness_audit_path=readiness.json_path or resolved_output_dir,
            claim_matrix_path=claims.json_path,
            quality_gate_result_path=experiment.quality_gate_result_path,
            robustness_path=metric_intervals.json_path,
            output_path=resolved_output_dir / PAPER_ARTIFACT_REGISTRY_JSON,
        )
    except Exception as exc:
        fail("build_paper_artifact_registry", exc)

    artifacts["artifact_registry"] = registry.json_path
    stages.append(
        _stage_record(
            stage="build_paper_artifact_registry",
            status="passed",
            output_dir=resolved_output_dir,
            artifacts=[registry.json_path],
            details=dict(registry.payload.get("status") or {}),
        )
    )

    payload = _bundle_result_payload(
        run_id=experiment.run_id,
        output_dir=resolved_output_dir,
        status="passed",
        stages=stages,
        artifacts=artifacts,
        failed_stage=None,
        error_message=None,
        registry_payload=registry.payload,
    )
    write_json(result_path, payload)
    return PaperBundleRun(
        run_id=experiment.run_id,
        output_dir=resolved_output_dir,
        experiment=experiment,
        metric_intervals=metric_intervals,
        readiness=readiness,
        claims=claims,
        registry=registry,
        result_path=result_path,
        payload=payload,
    )


def _bundle_result_payload(
    *,
    run_id: str | None,
    output_dir: Path,
    status: str,
    stages: list[dict[str, Any]],
    artifacts: Mapping[str, Path | None],
    failed_stage: str | None,
    error_message: str | None,
    registry_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    registry_status = {}
    registry_links = {}
    if registry_payload:
        registry_status = dict(registry_payload.get("status") or {})
        registry_links = dict(registry_payload.get("status_links") or {})
    payload = {
        "schema_version": PAPER_BUNDLE_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "failed_stage": failed_stage,
        "stages": stages,
        "artifacts": _artifact_summary(artifacts=artifacts, output_dir=output_dir),
        "registry_status": registry_status,
        "status_links": registry_links,
        "privacy": {
            "payload": "artifact_names_stage_statuses_and_coarse_statuses_only",
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
            "private_eval_values": "excluded",
        },
    }
    if error_message:
        payload["error"] = {
            "stage": failed_stage,
            "message": _sanitize_message(error_message),
        }
    return payload


def _stage_record(
    *,
    stage: str,
    status: str,
    output_dir: Path,
    artifacts: Iterable[Path | None],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "artifacts": [
            _relative_artifact(path, output_dir) for path in artifacts if path is not None
        ],
    }
    if details:
        record["details"] = {str(key): value for key, value in details.items()}
    return record


def _artifact_summary(
    *,
    artifacts: Mapping[str, Path | None],
    output_dir: Path,
) -> dict[str, str | None]:
    return {
        str(key): _relative_artifact(path, output_dir) if path is not None else None
        for key, path in artifacts.items()
    }


def _collect_existing_artifacts(output_dir: Path) -> dict[str, Path | None]:
    candidates = {
        "metrics": output_dir / "metrics.json",
        "metrics_summary": output_dir / "metrics_summary.csv",
        "query_results": output_dir / "query_results.jsonl",
        "semantic_smoke": output_dir / "semantic_smoke.json",
        "summary": output_dir / "summary.md",
        "paper_report_dir": output_dir / "paper_report",
        "paper_table_csv": output_dir / "paper_report" / "paper_table.csv",
        "paper_table_markdown": output_dir / "paper_report" / "paper_table.md",
        "reproducibility_json": output_dir / "paper_report" / "reproducibility.json",
        "quality_gate_result": output_dir / "quality_gate_result.json",
        "experiment_manifest": output_dir / "experiment_manifest.json",
        "metric_intervals_json": output_dir / "paper_metric_intervals" / "paper_metric_intervals.json",
        "readiness_audit": output_dir / "paper_readiness" / "paper_readiness_audit.json",
        "claim_matrix": output_dir / "paper_claims" / "claim_evidence_matrix.json",
        "artifact_registry": output_dir / PAPER_ARTIFACT_REGISTRY_JSON,
    }
    return {key: path for key, path in candidates.items() if path.exists()}


def _relative_artifact(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _sanitize_message(message: str) -> str:
    sanitized: list[str] = []
    for token in str(message).split():
        stripped = token.strip("\"'`.,:;()[]{}")
        if "/" in stripped or "\\" in stripped or stripped.startswith("~"):
            replacement = Path(stripped).name or "<path>"
            sanitized.append(token.replace(stripped, replacement))
        else:
            sanitized.append(token)
    return " ".join(sanitized)
