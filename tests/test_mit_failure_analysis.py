from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def test_failure_analysis_sanitizes_query_results(tmp_path: Path) -> None:
    module = _load_script_module()
    metrics_path = _write_metrics(tmp_path)
    query_results_path = tmp_path / "query_results.jsonl"
    _write_jsonl(
        query_results_path,
        [
            {
                "suite_id": "public_suite",
                "suite_type": "local_project",
                "domain": "public_domain",
                "query_text": "SECRET query /Users/private/input",
                "hit_by_delta": {"10": False, "30": False},
                "top1_abs_error": 720.0,
                "frame_backed": False,
                "linked_entity_backed": False,
                "segment_search_hits": 2,
                "visual_entity_search_hits": 0,
                "rerank": {"enabled": False},
                "top_candidate": {
                    "transcript_excerpt": "SECRET transcript",
                    "frame_path": "/private/tmp/frame.png",
                    "segment_id": "private_segment",
                },
            }
        ],
    )

    payload = module.analyze_failures(
        metrics_path=metrics_path,
        query_results_path=query_results_path,
        output_json_path=tmp_path / "failure_analysis.json",
        output_markdown_path=tmp_path / "failure_analysis.md",
    )

    assert payload["inputs"]["query_results"]["status"] == "loaded"
    bucket_counts = {
        bucket["bucket"]: bucket["affected_query_count"]
        for bucket in payload["failure_buckets"]
    }
    assert bucket_counts["no_hit_at_30s"] == 1
    assert bucket_counts["candidate_label_audit_needed"] == 1
    public_text = "\n".join(
        [
            (tmp_path / "failure_analysis.json").read_text(encoding="utf-8"),
            (tmp_path / "failure_analysis.md").read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "/Users/",
        "/private/tmp",
        "query_text",
        "transcript_text",
        "transcript_excerpt",
        "frame_path",
        "SECRET",
        "private_segment",
    ]:
        assert sensitive not in public_text


def test_failure_analysis_runs_without_query_results(tmp_path: Path) -> None:
    module = _load_script_module()
    metrics_path = _write_metrics(tmp_path)

    payload = module.analyze_failures(
        metrics_path=metrics_path,
        output_json_path=tmp_path / "failure_analysis.json",
        output_markdown_path=tmp_path / "failure_analysis.md",
    )

    assert payload["inputs"]["query_results"]["status"] == "not_found"
    bucket_estimates = {
        bucket["bucket"]: bucket.get("affected_query_estimate")
        for bucket in payload["failure_buckets"]
    }
    assert bucket_estimates["no_hit_at_30s"] == 2
    assert (tmp_path / "failure_analysis.json").exists()
    assert (tmp_path / "failure_analysis.md").exists()


def _load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_mit_deep_learning_eval_failures.py"
    )
    spec = importlib.util.spec_from_file_location("mit_failure_analysis", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metrics(tmp_path: Path) -> Path:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": "public_fixture",
                "suite_count": 1,
                "query_count": 2,
                "deltas": [5, 10, 30],
                "anti_overfit": {
                    "domain_count": 1,
                    "domains": {
                        "public_domain": {
                            "suite_count": 1,
                            "query_count": 2,
                            "mean_hit_at_10s": 0.0,
                            "mean_mrr_at_max_delta": 0.0,
                        }
                    },
                },
                "suites": [
                    {
                        "suite_id": "public_suite",
                        "suite_type": "local_project",
                        "domain": "public_domain",
                        "query_count": 2,
                        "domain_lexicon": {"enabled": False, "source_path": "/Users/private/lexicon.json"},
                        "rerank": {"enabled": False},
                        "mean_abs_error": 500.0,
                        "top1_mean_abs_error": 700.0,
                        "mrr_at_max_delta": 0.0,
                        "frame_backed_ratio": 0.0,
                        "linked_entity_backed_ratio": 0.0,
                        "mean_processing_time_ms": 1.0,
                        "hit_at_5s": 0.0,
                        "hit_at_10s": 0.0,
                        "hit_at_15s": 0.0,
                        "hit_at_30s": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return metrics_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
