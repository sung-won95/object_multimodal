from __future__ import annotations

import json
from pathlib import Path

import pytest

from oarag.evaluation.experiment import PaperExperimentError, run_paper_experiment


class FakePaperExperimentClient:
    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        hybrid: dict | None = None,
    ) -> dict:
        mode = "semantic" if hybrid else "lexical"
        if index_uid == "public_windows":
            hits = [
                {
                    "window_id": "window_loss_public",
                    "target_segment_id": "seg_loss_public",
                    "sample_id": "seg_loss_public",
                    "video_id": "public_demo_video",
                    "start_time": 0.0,
                    "end_time": 34.0,
                    "timestamp_center": 12.0,
                    "target_start_time": 10.0,
                    "target_end_time": 14.0,
                    "target_timestamp_center": 12.0,
                    "source_segment_ids": [
                        "seg_intro_public",
                        "seg_loss_public",
                        "seg_wrap_public",
                    ],
                    "transcript_window_text": "PUBLIC SYNTHETIC window must not leak",
                    "_rankingScore": 0.89,
                }
            ]
            return {"hits": hits[:limit], "processingTimeMs": 4, "indexUid": index_uid}
        if index_uid == "public_visual_entities":
            hits = [
                {
                    "entity_id": "ent_loss_public",
                    "frame_id": "frame_loss_public",
                    "timestamp": 12.0,
                    "text": "PUBLIC VISUAL LABEL loss curve",
                    "_rankingScore": 0.9,
                }
            ]
            return {"hits": hits[:limit], "processingTimeMs": 2, "indexUid": index_uid}
        if index_uid == "public_segments":
            hits = [
                {
                    "segment_id": "seg_wrap_public",
                    "sample_id": "seg_wrap_public",
                    "video_id": "public_demo_video",
                    "start_time": 30.0,
                    "end_time": 34.0,
                    "timestamp_center": 32.0,
                    "transcript_text": "PUBLIC SYNTHETIC wrong transcript must not leak",
                    "_rankingScore": 0.95 if mode == "lexical" else 0.7,
                },
                {
                    "segment_id": "seg_loss_public",
                    "sample_id": "seg_loss_public",
                    "video_id": "public_demo_video",
                    "start_time": 10.0,
                    "end_time": 14.0,
                    "timestamp_center": 12.0,
                    "transcript_text": "PUBLIC SYNTHETIC correct transcript must not leak",
                    "_rankingScore": 0.55 if mode == "lexical" else 0.96,
                },
            ]
            if "validation loss" in query:
                hits = list(reversed(hits))
            return {"hits": hits[:limit], "processingTimeMs": 5, "indexUid": index_uid}
        return {"hits": [], "processingTimeMs": 1, "indexUid": index_uid}


def test_run_paper_experiment_public_fixture_writes_end_to_end_artifacts(
    tmp_path: Path,
) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    output_dir = tmp_path / "paper_experiment"

    run = run_paper_experiment(
        client=FakePaperExperimentClient(),
        manifest_path=fixture_dir / "benchmark_matrix_manifest.json",
        output_dir=output_dir,
        run_id="paper_e2e_fixture",
        gate_config_path=fixture_dir / "retrieval_quality_gate.json",
        repo_root=Path.cwd(),
        command=[
            "oarag",
            "run-paper-experiment",
            "--manifest",
            str(fixture_dir / "benchmark_matrix_manifest.json"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run.run_id == "paper_e2e_fixture"
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "query_results.jsonl").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "paper_report" / "paper_table.csv").exists()
    assert (output_dir / "paper_report" / "paper_table.md").exists()
    assert (output_dir / "paper_report" / "reproducibility.json").exists()
    assert (output_dir / "quality_gate_result.json").exists()
    assert (output_dir / "experiment_manifest.json").exists()
    assert run.quality_gate_result["passed"] is True

    experiment_manifest = json.loads(
        (output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert experiment_manifest["schema_version"] == "paper-experiment-manifest-v1"
    assert experiment_manifest["status"] == "passed"
    assert experiment_manifest["artifacts"]["metrics"] == "metrics.json"
    assert experiment_manifest["artifacts"]["paper_report_dir"] == "paper_report"
    assert experiment_manifest["stages"][2]["stage"] == "quality_gate"
    assert experiment_manifest["stages"][2]["status"] == "passed"

    public_text = "\n".join(
        [
            (output_dir / "metrics.json").read_text(encoding="utf-8"),
            (output_dir / "metrics_summary.csv").read_text(encoding="utf-8"),
            (output_dir / "query_results.jsonl").read_text(encoding="utf-8"),
            (output_dir / "summary.md").read_text(encoding="utf-8"),
            (output_dir / "paper_report" / "paper_table.md").read_text(encoding="utf-8"),
            (output_dir / "paper_report" / "reproducibility.json").read_text(
                encoding="utf-8"
            ),
            (output_dir / "quality_gate_result.json").read_text(encoding="utf-8"),
            (output_dir / "experiment_manifest.json").read_text(encoding="utf-8"),
        ]
    )
    for sensitive in [
        "PUBLIC RAW QUERY",
        "loss curve slope should stay private-safe",
        "PUBLIC SYNTHETIC",
        "PUBLIC VISUAL LABEL",
        "seg_loss_public",
        "window_loss_public",
        "ent_loss_public",
        "frame_loss_public",
        str(fixture_dir),
        str(tmp_path),
    ]:
        assert sensitive not in public_text


def test_run_paper_experiment_gate_failure_names_stage(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/public_retrieval_ablation_project").resolve()
    gate_config = tmp_path / "strict_gate.json"
    gate_config.write_text(
        json.dumps(
            {
                "gate_id": "strict_gate",
                "suites": [
                    {
                        "suite_id": "public_matrix",
                        "suite_type": "retrieval_answer_matrix",
                        "thresholds": {"expected_citation_hit_ratio": 1.1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PaperExperimentError) as exc_info:
        run_paper_experiment(
            client=FakePaperExperimentClient(),
            manifest_path=fixture_dir / "benchmark_matrix_manifest.json",
            output_dir=tmp_path / "failed_experiment",
            gate_config_path=gate_config,
            repo_root=Path.cwd(),
        )

    assert exc_info.value.stage == "gate"
    assert "paper experiment failed during gate" in str(exc_info.value)
    assert (tmp_path / "failed_experiment" / "quality_gate_result.json").exists()
    assert (tmp_path / "failed_experiment" / "experiment_manifest.json").exists()
