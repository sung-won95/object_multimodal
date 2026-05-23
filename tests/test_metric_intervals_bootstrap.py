from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.cli import build_parser
from oarag.evaluation.metric_intervals import generate_metric_intervals


def test_generate_metric_intervals_writes_private_safe_bootstrap_and_paired_delta(
    tmp_path: Path,
) -> None:
    query_results_path = tmp_path / "query_results.jsonl"
    _write_jsonl(
        query_results_path,
        [
            _row(
                variant_id="segment_lexical",
                query_id="private_q1",
                hit_at_10s=True,
                mrr=1.0,
                top1_abs_error=0.0,
                answer_type="grounded_answer",
                expected_citation_hit=True,
                unsupported_claim_count=0,
            ),
            _row(
                variant_id="segment_lexical",
                query_id="private_q2",
                hit_at_10s=False,
                mrr=0.0,
                top1_abs_error=20.0,
                answer_type="candidate_evidence_only",
                expected_citation_hit=False,
                unsupported_claim_count=2,
            ),
            _row(
                variant_id="window_hybrid",
                query_id="private_q1",
                hit_at_10s=True,
                mrr=1.0,
                top1_abs_error=0.0,
                answer_type="grounded_answer",
                expected_citation_hit=True,
                unsupported_claim_count=0,
            ),
            _row(
                variant_id="window_hybrid",
                query_id="private_q2",
                hit_at_10s=True,
                mrr=0.5,
                top1_abs_error=8.0,
                answer_type="grounded_answer",
                expected_citation_hit=True,
                unsupported_claim_count=1,
            ),
            _row(
                variant_id="window_hybrid",
                query_id="private_q3",
                hit_at_10s=True,
                mrr=1.0,
                top1_abs_error=2.0,
                answer_type="grounded_answer",
                expected_citation_hit=True,
                unsupported_claim_count=0,
            ),
        ],
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"run_id": "interval_fixture"}), encoding="utf-8")

    run = generate_metric_intervals(
        query_results_path=query_results_path,
        metrics_path=metrics_path,
        report_path=tmp_path / "paper_table.md",
        output_dir=tmp_path / "intervals",
        baseline_variant_id="segment_lexical",
        seed=7,
        sample_count=100,
    )

    assert run.json_path.exists()
    assert run.csv_path.exists()
    assert run.markdown_path.exists()
    assert run.payload["schema_version"] == "paper-metric-intervals-v1"
    assert run.payload["run_id"] == "interval_fixture"
    assert run.payload["status"] == "needs_evidence"
    assert run.payload["summary"]["robustness_status"] == "needs_evidence"
    assert run.payload["summary"]["paired_delta_count"] > 0
    assert run.payload["summary"]["caveat_count"] > 0
    assert run.payload["input"] == {
        "query_results_artifact": "query_results.jsonl",
        "metrics_artifact": "metrics.json",
        "report_artifact": "paper_table.md",
    }
    variants = {item["variant_id"]: item for item in run.payload["variants"]}
    assert variants["segment_lexical"]["metrics"]["hit_at_10s"]["mean"] == 0.5
    assert variants["window_hybrid"]["metrics"]["hit_at_10s"]["mean"] == 1.0
    assert variants["window_hybrid"]["metrics"]["grounded_answer"]["mean"] == 1.0

    deltas = {
        (item["variant_id"], item["metric"]): item for item in run.payload["paired_deltas"]
    }
    hit_delta = deltas[("window_hybrid", "hit_at_10s")]
    assert hit_delta["paired_query_count"] == 2
    assert hit_delta["mean_delta"] == 0.5
    assert hit_delta["mean_improvement"] == 0.5
    error_delta = deltas[("window_hybrid", "top1_abs_error")]
    assert error_delta["mean_delta"] == -6.0
    assert error_delta["mean_improvement"] == 6.0
    assert error_delta["improvement_ci_low"] <= error_delta["improvement_ci_high"]
    unsupported_delta = deltas[("window_hybrid", "unsupported_claim_count")]
    assert unsupported_delta["mean_delta"] == -0.5
    assert unsupported_delta["mean_improvement"] == 0.5
    assert "very_small_sample:paired_query_count=2" in unsupported_delta["caveats"]

    public_text = "\n".join(
        [
            run.json_path.read_text(encoding="utf-8"),
            run.csv_path.read_text(encoding="utf-8"),
            run.markdown_path.read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "PRIVATE RAW QUERY",
        "PRIVATE ANSWER TEXT",
        "PRIVATE TRANSCRIPT",
        "private_q1",
        "private_q2",
        "private_q3",
        str(tmp_path),
    ]:
        assert sensitive not in public_text
    assert "aggregate_statistics_only" in public_text
    assert "small_sample_intervals_should_not_be_used_as_confirmatory_evidence" in public_text


def test_paper_metric_intervals_cli_help_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["paper-metric-intervals", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--query-results" in help_text
    assert "--baseline-variant-id" in help_text
    assert "--sample-count" in help_text


def _row(
    *,
    variant_id: str,
    query_id: str,
    hit_at_10s: bool,
    mrr: float,
    top1_abs_error: float,
    answer_type: str,
    expected_citation_hit: bool,
    unsupported_claim_count: int,
) -> dict:
    return {
        "schema_version": "retrieval-answer-ablation-matrix-v1",
        "run_id": "row_run_id_is_overridden_by_metrics",
        "suite_id": "private_suite",
        "suite_type": "retrieval_answer_matrix",
        "domain": "private_domain",
        "variant_id": variant_id,
        "query_id": query_id,
        "query_text": "PRIVATE RAW QUERY must not leak",
        "answer_text": "PRIVATE ANSWER TEXT must not leak",
        "transcript_text": "PRIVATE TRANSCRIPT must not leak",
        "hit_by_delta": {"10": hit_at_10s},
        "mrr": mrr,
        "top1_abs_error": top1_abs_error,
        "answer": {"enabled": True, "answer_type": answer_type},
        "answer_grounding": {
            "expected_citation_hit": expected_citation_hit,
            "unsupported_claim_count": unsupported_claim_count,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
