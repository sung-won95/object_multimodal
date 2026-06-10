from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oarag.core.io import write_json
from oarag.evaluation.benchmark import BenchmarkRun, SearchClient, run_benchmark
from oarag.evaluation.quality_gate import check_retrieval_quality_gate
from oarag.evaluation.reporting import EvaluationReport, generate_evaluation_report


PAPER_EXPERIMENT_MANIFEST_SCHEMA_VERSION = "paper-experiment-manifest-v1"
PAPER_EXPERIMENT_GATE_SKIPPED_SCHEMA_VERSION = "paper-experiment-quality-gate-skipped-v1"


class PaperExperimentError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"paper experiment failed during {stage}: {message}")
        self.stage = stage


@dataclass(frozen=True)
class PaperExperimentRun:
    run_id: str
    output_dir: Path
    benchmark: BenchmarkRun
    report: EvaluationReport
    quality_gate_result_path: Path
    experiment_manifest_path: Path
    quality_gate_result: dict[str, Any]
    experiment_manifest: dict[str, Any]


def run_paper_experiment(
    *,
    client: SearchClient,
    manifest_path: Path,
    output_dir: Path,
    run_id: str | None = None,
    gate_config_path: Path | None = None,
    repo_root: Path | None = None,
    diagnostic_top_k: int | None = None,
    fail_on_gate: bool = True,
    command: list[str] | None = None,
) -> PaperExperimentRun:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    quality_gate_result_path = resolved_output_dir / "quality_gate_result.json"
    experiment_manifest_path = resolved_output_dir / "experiment_manifest.json"

    try:
        benchmark = run_benchmark(
            client=client,
            manifest_path=manifest_path,
            output_dir=resolved_output_dir,
            repo_root=repo_root,
            diagnostic_top_k=diagnostic_top_k,
            run_id=run_id,
        )
    except Exception as exc:
        raise PaperExperimentError("benchmark", str(exc)) from exc

    try:
        report = generate_evaluation_report(
            metrics_path=benchmark.metrics_path,
            output_dir=resolved_output_dir / "paper_report",
            repo_root=repo_root,
            command=command or [],
        )
    except Exception as exc:
        raise PaperExperimentError("report", str(exc)) from exc

    gate_stage_status = "skipped"
    try:
        if gate_config_path is None:
            quality_gate_result = _skipped_gate_result()
        else:
            quality_gate_result = check_retrieval_quality_gate(
                metrics_path=benchmark.metrics_path,
                config_path=gate_config_path,
                semantic_smoke_path=benchmark.semantic_smoke_path,
            )
            gate_stage_status = "passed" if quality_gate_result.get("passed") else "failed"
        write_json(quality_gate_result_path, quality_gate_result)
    except Exception as exc:
        raise PaperExperimentError("gate", str(exc)) from exc

    experiment_manifest = _experiment_manifest(
        benchmark=benchmark,
        report=report,
        quality_gate_result=quality_gate_result,
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        gate_config_path=gate_config_path,
        gate_stage_status=gate_stage_status,
        command=command or [],
    )
    write_json(experiment_manifest_path, experiment_manifest)

    if gate_config_path is not None and not quality_gate_result.get("passed") and fail_on_gate:
        raise PaperExperimentError("gate", "quality gate did not pass")

    return PaperExperimentRun(
        run_id=benchmark.run_id,
        output_dir=resolved_output_dir,
        benchmark=benchmark,
        report=report,
        quality_gate_result_path=quality_gate_result_path,
        experiment_manifest_path=experiment_manifest_path,
        quality_gate_result=quality_gate_result,
        experiment_manifest=experiment_manifest,
    )


def _experiment_manifest(
    *,
    benchmark: BenchmarkRun,
    report: EvaluationReport,
    quality_gate_result: dict[str, Any],
    output_dir: Path,
    manifest_path: Path,
    gate_config_path: Path | None,
    gate_stage_status: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": PAPER_EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "run_id": benchmark.run_id,
        "status": "passed" if gate_stage_status in {"passed", "skipped"} else "failed",
        "inputs": {
            "benchmark_manifest": manifest_path.name,
            "quality_gate_config": gate_config_path.name if gate_config_path is not None else None,
        },
        "command": {
            "argv": _sanitize_command(command),
        },
        "commit": {
            "sha": (report.reproducibility.get("commit") or {}).get("sha"),
        },
        "artifacts": {
            "metrics": _relative_artifact(benchmark.metrics_path, output_dir),
            "metrics_summary": _relative_artifact(benchmark.metrics_csv_path, output_dir),
            "query_results": _relative_artifact(benchmark.query_results_path, output_dir),
            "semantic_smoke": _relative_artifact(benchmark.semantic_smoke_path, output_dir),
            "summary": _relative_artifact(benchmark.summary_path, output_dir),
            "paper_report_dir": _relative_artifact(report.output_dir, output_dir),
            "paper_table_csv": _relative_artifact(report.paper_table_csv_path, output_dir),
            "paper_table_markdown": _relative_artifact(
                report.paper_table_markdown_path,
                output_dir,
            ),
            "reproducibility_json": _relative_artifact(
                report.reproducibility_json_path,
                output_dir,
            ),
            "reproducibility_markdown": _relative_artifact(
                report.reproducibility_markdown_path,
                output_dir,
            ),
            "quality_gate_result": "quality_gate_result.json",
            "experiment_manifest": "experiment_manifest.json",
        },
        "stages": [
            {
                "stage": "benchmark",
                "status": "passed",
                "artifacts": [
                    "metrics.json",
                    "metrics_summary.csv",
                    "query_results.jsonl",
                    "semantic_smoke.json",
                ],
            },
            {
                "stage": "report",
                "status": "passed",
                "artifacts": ["summary.md", "paper_report/"],
            },
            {
                "stage": "quality_gate",
                "status": gate_stage_status,
                "artifact": "quality_gate_result.json",
                "failure_count": quality_gate_result.get("failure_count"),
            },
        ],
        "privacy": {
            "payload": "artifact_schema_and_aggregate_metadata_only",
            "raw_queries": "excluded",
            "answer_text": "excluded",
            "transcript_content": "excluded",
            "candidate_evidence_text": "excluded",
            "local_paths": "excluded",
        },
    }


def _skipped_gate_result() -> dict[str, Any]:
    return {
        "schema_version": PAPER_EXPERIMENT_GATE_SKIPPED_SCHEMA_VERSION,
        "gate_id": None,
        "passed": None,
        "skipped": True,
        "failure_count": 0,
        "failures": [],
        "note": "Quality gate was not configured for this paper experiment run.",
        "privacy": {
            "payload": "aggregate_metrics_only",
            "query_content": "excluded",
            "transcript_content": "excluded",
            "evidence_content": "excluded",
        },
    }


def _relative_artifact(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _sanitize_command(command: list[str]) -> list[str]:
    sanitized: list[str] = []
    for token in command:
        text = str(token)
        if "/" in text or "\\" in text or text.startswith("~"):
            name = Path(text).name
            sanitized.append(name or "<path>")
        else:
            sanitized.append(text)
    return sanitized
