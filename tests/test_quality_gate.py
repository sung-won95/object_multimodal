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
                "hit_at_10s": 0.4,
                "mrr_at_max_delta": 0.5,
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
                },
                "variant_thresholds": {
                    "segment_lexical": {
                        "grounded_answer_ratio": 1.0,
                    }
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
        "suite=public_matrix variant=* metric=hit_at_10s "
        "value=0.4 below regression threshold=1.0"
    ) in messages
    assert (
        "suite=public_matrix variant=segment_lexical metric=grounded_answer_ratio missing metric"
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


def test_retrieval_quality_gate_fails_when_required_variant_skipped() -> None:
    result = evaluate_retrieval_quality_gate(
        metrics={
            "run_id": "skip_regression",
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "schema_version": "retrieval-answer-ablation-matrix-v1",
                    "hit_at_10s": 1.0,
                    "mrr_at_max_delta": 1.0,
                    "variant_metrics": {
                        "hybrid": {
                            "variant_id": "hybrid",
                            "skipped_count": 1,
                            "hit_at_10s": 1.0,
                        }
                    },
                }
            ],
        },
        config={
            "gate_id": "test_gate",
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "thresholds": {"hit_at_10s": 0.9, "mrr_at_max_delta": 0.9},
                    "required_variants": ["hybrid"],
                }
            ],
        },
    )

    assert result["passed"] is False
    assert {failure["code"] for failure in result["failures"]} == {
        "required_variant_skipped"
    }


def test_retrieval_quality_gate_fails_variant_threshold_miss() -> None:
    result = evaluate_retrieval_quality_gate(
        metrics={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "schema_version": "retrieval-answer-ablation-matrix-v1",
                    "hit_at_10s": 1.0,
                    "mrr_at_max_delta": 1.0,
                    "variant_metrics": {
                        "hybrid": {
                            "variant_id": "hybrid",
                            "skipped_count": 0,
                            "hit_at_10s": 0.5,
                        }
                    },
                }
            ]
        },
        config={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "thresholds": {"hit_at_10s": 0.9, "mrr_at_max_delta": 0.9},
                    "required_variants": ["hybrid"],
                    "variant_thresholds": {
                        "hybrid": {"hit_at_10s": 0.9},
                    },
                }
            ]
        },
    )

    assert result["passed"] is False
    assert any(
        failure["code"] == "metric_below_threshold"
        and failure.get("variant_id") == "hybrid"
        for failure in result["failures"]
    )


def test_retrieval_quality_gate_rejects_zero_threshold_without_allowance() -> None:
    result = evaluate_retrieval_quality_gate(
        metrics={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "schema_version": "retrieval-answer-ablation-matrix-v1",
                    "hit_at_10s": 1.0,
                    "mrr_at_max_delta": 1.0,
                    "variant_metrics": {
                        "segment_lexical": {
                            "variant_id": "segment_lexical",
                            "skipped_count": 0,
                            "hit_at_10s": 0.0,
                        }
                    },
                }
            ]
        },
        config={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "thresholds": {"hit_at_10s": 0.9, "mrr_at_max_delta": 0.9},
                    "required_variants": ["segment_lexical"],
                    "variant_thresholds": {
                        "segment_lexical": {"hit_at_10s": 0},
                    },
                }
            ]
        },
    )

    assert result["passed"] is False
    assert any(
        failure["code"] == "zero_threshold_not_allowed"
        for failure in result["failures"]
    )


def test_retrieval_quality_gate_rejects_suite_zero_threshold_even_with_allowance() -> None:
    result = evaluate_retrieval_quality_gate(
        metrics={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "schema_version": "retrieval-answer-ablation-matrix-v1",
                    "hit_at_10s": 0.0,
                    "mrr_at_max_delta": 1.0,
                    "variant_metrics": {
                        "segment_lexical": {
                            "variant_id": "segment_lexical",
                            "skipped_count": 0,
                        }
                    },
                }
            ]
        },
        config={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "thresholds": {
                        "hit_at_10s": {
                            "threshold": 0,
                            "allow_zero_threshold": True,
                            "reason": "Suite aggregate zero is not an allowed gate floor.",
                        },
                        "mrr_at_max_delta": 0.9,
                    },
                    "required_variants": ["segment_lexical"],
                }
            ]
        },
    )

    assert result["passed"] is False
    assert any(
        failure["code"] == "zero_threshold_not_allowed"
        and failure.get("variant_id") is None
        for failure in result["failures"]
    )


def test_retrieval_quality_gate_zero_threshold_allowance_requires_reason() -> None:
    base_metrics = {
        "suites": [
            {
                "suite_id": "public_matrix",
                "suite_type": "retrieval_answer_matrix",
                "schema_version": "retrieval-answer-ablation-matrix-v1",
                "hit_at_10s": 1.0,
                "mrr_at_max_delta": 1.0,
                "variant_metrics": {
                    "segment_lexical": {
                        "variant_id": "segment_lexical",
                        "skipped_count": 0,
                        "hit_at_10s": 0.0,
                    }
                },
            }
        ]
    }
    base_config = {
        "suites": [
            {
                "suite_id": "public_matrix",
                "suite_type": "retrieval_answer_matrix",
                "thresholds": {"hit_at_10s": 0.9, "mrr_at_max_delta": 0.9},
                "required_variants": ["segment_lexical"],
                "variant_thresholds": {
                    "segment_lexical": {
                        "hit_at_10s": {
                            "threshold": 0,
                            "allow_zero_threshold": True,
                        }
                    },
                },
            }
        ]
    }

    missing_reason = evaluate_retrieval_quality_gate(
        metrics=base_metrics,
        config=base_config,
    )
    assert missing_reason["passed"] is False
    assert any(
        failure["code"] == "zero_threshold_reason_missing"
        for failure in missing_reason["failures"]
    )

    base_config["suites"][0]["variant_thresholds"]["segment_lexical"]["hit_at_10s"][
        "reason"
    ] = "Observed zero in the provider-backed baseline; keep zero floor until improved."
    allowed = evaluate_retrieval_quality_gate(metrics=base_metrics, config=base_config)
    assert allowed["passed"] is True


def test_retrieval_quality_gate_fails_unbacked_semantic_smoke_with_hybrid_metrics() -> None:
    result = evaluate_retrieval_quality_gate(
        metrics={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "schema_version": "retrieval-answer-ablation-matrix-v1",
                    "hit_at_10s": 1.0,
                    "mrr_at_max_delta": 1.0,
                    "variant_metrics": {
                        "hybrid": {
                            "variant_id": "hybrid",
                            "skipped_count": 0,
                            "hit_at_10s": 1.0,
                        }
                    },
                }
            ]
        },
        config={
            "suites": [
                {
                    "suite_id": "public_matrix",
                    "suite_type": "retrieval_answer_matrix",
                    "thresholds": {"hit_at_10s": 0.9, "mrr_at_max_delta": 0.9},
                    "required_variants": ["hybrid"],
                }
            ]
        },
        semantic_smoke={
            "semantic_live_smoke": {
                "ok": False,
                "passed": False,
                "provider_backed": False,
            },
            "hybrid_variant_ids": ["hybrid"],
        },
    )

    assert result["passed"] is False
    assert any(
        failure["code"] == "semantic_smoke_provider_unbacked_hybrid_metrics"
        for failure in result["failures"]
    )
