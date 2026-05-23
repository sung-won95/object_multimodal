from __future__ import annotations

import json
from pathlib import Path

from oarag.evaluation.reporting import generate_evaluation_report


def test_generate_evaluation_report_writes_paper_table_and_reproducibility(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": "report_fixture",
                "suite_count": 1,
                "query_count": 2,
                "deltas": [5, 10, 15],
                "anti_overfit": {"domain_count": 1, "domains": {}},
                "suites": [
                    {
                        "suite_id": "public_matrix",
                        "suite_type": "retrieval_answer_matrix",
                        "domain": "public_synthetic",
                        "query_count": 1,
                        "dataset_descriptor": {
                            "descriptor_id": "public_fixture",
                            "split": "test",
                            "version": "1",
                            "privacy": "aggregate_only",
                        },
                        "variants": [
                            {
                                "variant_id": "segment_lexical",
                                "query_count": 1,
                                "hit_at_10s": 1.0,
                                "mrr_at_max_delta": 1.0,
                                "top1_mean_abs_error": 0.0,
                                "frame_backed_ratio": 1.0,
                                "linked_entity_backed_ratio": 0.0,
                                "grounded_answer_ratio": 1.0,
                                "citation_coverage_ratio": 1.0,
                                "mean_processing_time_ms": 3.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = generate_evaluation_report(
        metrics_path=metrics_path,
        output_dir=tmp_path / "report",
        command=["oarag", "report-evaluation", "--metrics", str(metrics_path)],
        commit_sha="abc123",
    )

    assert report.paper_table_csv_path.exists()
    assert report.paper_table_markdown_path.exists()
    assert report.reproducibility_json_path.exists()
    assert report.reproducibility_markdown_path.exists()
    assert report.paper_table_rows[0]["condition"] == "segment_lexical"
    assert report.reproducibility["commit"]["sha"] == "abc123"

    paper_table = report.paper_table_markdown_path.read_text(encoding="utf-8")
    assert "segment_lexical" in paper_table
    assert "Raw queries" in paper_table

    reproducibility = report.reproducibility_json_path.read_text(encoding="utf-8")
    assert "abc123" in reproducibility
    assert "metrics.json" in reproducibility
    assert str(tmp_path) not in reproducibility
