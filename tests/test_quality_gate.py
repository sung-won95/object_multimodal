from __future__ import annotations

import json

from oarag.evaluation.quality_gate import evaluate_retrieval_quality_gate


def test_retrieval_quality_gate_reports_clear_public_fixture_failures() -> None:
    metrics = {
        "run_id": "public_fixture_regression",
        "suites": [
            {
                "suite_id": "public_matrix",
                "suite_type": "retrieval_answer_matrix",
                "query_count": 1,
                "variant_count": 1,
                "variant_metrics": {
                    "segment_lexical": {
                        "variant_id": "segment_lexical",
                        "hit_at_10s": 0.4,
                        "mrr_at_max_delta": 0.5,
                    }
                },
            }
        ],
    }
    config = {
        "gate_id": "test_public_gate",
        "suites": [
            {
                "suite_id": "public_matrix",
                "suite_type": "retrieval_answer_matrix",
                "schema_version": "retrieval-answer-ablation-matrix-v1",
                "required_variants": ["segment_lexical", "hybrid"],
                "thresholds": {
                    "hit_at_10s": 1.0,
                    "mrr_at_max_delta": 0.5,
                    "grounded_answer_ratio": 1.0,
                },
            }
        ],
    }

    result = evaluate_retrieval_quality_gate(metrics=metrics, config=config)

    assert result["passed"] is False
    codes = {failure["code"] for failure in result["failures"]}
    assert codes == {
        "suite_schema_missing",
        "metric_below_threshold",
        "metric_missing",
        "variant_missing",
    }
    messages = "\n".join(failure["message"] for failure in result["failures"])
    assert (
        "suite=public_matrix type=retrieval_answer_matrix missing schema_version "
        "(expected retrieval-answer-ablation-matrix-v1)"
    ) in messages
    assert (
        "suite=public_matrix variant=segment_lexical metric=hit_at_10s "
        "value=0.4 below public fixture guard threshold=1.0"
    ) in messages
    assert (
        "suite=public_matrix variant=segment_lexical missing metric=grounded_answer_ratio"
        in messages
    )
    assert "suite=public_matrix missing required variant=hybrid" in messages

    payload = json.dumps(result, ensure_ascii=False)
    assert "PUBLIC RAW QUERY" not in payload
    assert "transcript_text" not in payload
    assert "evidence_text" not in payload


def test_retrieval_quality_gate_reports_missing_metrics_schema() -> None:
    result = evaluate_retrieval_quality_gate(
        metrics={"run_id": "empty_metrics"},
        config={
            "gate_id": "test_public_gate",
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "thresholds": {"hit_at_10s": 1.0},
                }
            ],
        },
    )

    assert result["passed"] is False
    assert result["failures"][0]["code"] == "metrics_suites_missing"
    assert "metrics payload missing suites list" in result["failures"][0]["message"]
