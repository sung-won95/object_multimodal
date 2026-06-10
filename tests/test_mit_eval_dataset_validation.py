from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType


def test_current_seed_dataset_fails_default_completion_gate() -> None:
    module = _load_validator_module()
    repo_root = Path(__file__).resolve().parents[1]

    payload = module.validate_dataset(
        queries_path=repo_root / "eval" / "mit_deep_learning_stt" / "queries.csv",
        readme_path=repo_root / "eval" / "mit_deep_learning_stt" / "README.md",
    )

    assert payload["ok"] is False
    failure_codes = {failure["code"] for failure in payload["failures"]}
    assert "min_queries_not_met" in failure_codes
    assert "min_human_audit_ratio_not_met" in failure_codes
    assert "all_annotators_codex_generated" in failure_codes


def test_synthetic_audited_240_row_dataset_passes(tmp_path: Path) -> None:
    module = _load_validator_module()
    queries_path = _write_queries(tmp_path)
    readme_path = _write_readme(tmp_path, audited=True)

    payload = module.validate_dataset(
        queries_path=queries_path,
        readme_path=readme_path,
        require_readme_audit_summary=True,
    )

    assert payload["ok"] is True
    assert payload["summary"]["row_count"] == 240
    assert payload["summary"]["unique_video_count"] == 24
    assert payload["summary"]["human_audit_ratio"] == 0.2


def test_all_codex_annotators_fail_even_with_enough_rows(tmp_path: Path) -> None:
    module = _load_validator_module()
    queries_path = _write_queries(tmp_path, human_rows=0)
    readme_path = _write_readme(tmp_path, audited=True)

    payload = module.validate_dataset(
        queries_path=queries_path,
        readme_path=readme_path,
        require_readme_audit_summary=True,
    )

    failure_codes = {failure["code"] for failure in payload["failures"]}
    assert payload["ok"] is False
    assert "min_human_audit_ratio_not_met" in failure_codes
    assert "all_annotators_codex_generated" in failure_codes


def test_invalid_timestamp_fails(tmp_path: Path) -> None:
    module = _load_validator_module()
    queries_path = _write_queries(tmp_path, invalid_timestamp=True)
    readme_path = _write_readme(tmp_path, audited=True)

    payload = module.validate_dataset(
        queries_path=queries_path,
        readme_path=readme_path,
        require_readme_audit_summary=True,
    )

    failure_codes = {failure["code"] for failure in payload["failures"]}
    assert payload["ok"] is False
    assert "invalid_timestamps" in failure_codes


def test_missing_manifest_expected_video_id_fails(tmp_path: Path) -> None:
    module = _load_validator_module()
    queries_path = _write_queries(tmp_path)
    readme_path = _write_readme(tmp_path, audited=True)
    expected_video_ids = {f"mit6_7960f24_lec{index:02d}_mp4" for index in range(1, 25)}
    expected_video_ids.add("mit6_7960f24_review_mp4")

    payload = module.validate_dataset(
        queries_path=queries_path,
        readme_path=readme_path,
        expected_video_ids=expected_video_ids,
        require_readme_audit_summary=True,
    )

    failure_codes = {failure["code"] for failure in payload["failures"]}
    assert payload["ok"] is False
    assert "expected_videos_missing" in failure_codes
    assert payload["summary"]["missing_expected_video_count"] == 1


def test_readme_audit_summary_requirement(tmp_path: Path) -> None:
    module = _load_validator_module()
    queries_path = _write_queries(tmp_path)
    missing_summary_readme = _write_readme(tmp_path, audited=False)
    audited_readme = _write_readme(tmp_path, audited=True, name="README_audited.md")

    missing_payload = module.validate_dataset(
        queries_path=queries_path,
        readme_path=missing_summary_readme,
        require_readme_audit_summary=True,
    )
    passing_payload = module.validate_dataset(
        queries_path=queries_path,
        readme_path=audited_readme,
        require_readme_audit_summary=True,
    )

    assert missing_payload["ok"] is False
    assert "readme_audit_summary_missing" in {
        failure["code"] for failure in missing_payload["failures"]
    }
    assert passing_payload["ok"] is True


def _load_validator_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "validate_mit_deep_learning_eval_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("mit_eval_dataset_validator", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_queries(
    tmp_path: Path,
    *,
    human_rows: int = 48,
    invalid_timestamp: bool = False,
) -> Path:
    path = tmp_path / "queries.csv"
    fieldnames = [
        "query_id",
        "split",
        "video_id",
        "lecture_title",
        "gold_start_time",
        "gold_end_time",
        "gold_timestamp_center",
        "gold_modality",
        "gold_frame_ids",
        "question_type",
        "annotator_id",
        "confidence",
    ]
    question_types = [
        "concept_explanation",
        "definition_lookup",
        "multimodal_grounded",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(240):
            video_number = index // 10 + 1
            start = float(index * 3)
            end = start + 2.0
            if invalid_timestamp and index == 17:
                end = start - 1.0
            writer.writerow(
                {
                    "query_id": f"fixture_q{index:03d}",
                    "split": "dev" if index % 2 == 0 else "test",
                    "video_id": f"mit6_7960f24_lec{video_number:02d}_mp4",
                    "lecture_title": f"Lecture {video_number:02d}",
                    "gold_start_time": start,
                    "gold_end_time": end,
                    "gold_timestamp_center": (start + end) / 2.0,
                    "gold_modality": "both" if index % 3 == 0 else "audio",
                    "gold_frame_ids": "frame_000001" if index % 3 == 0 else "",
                    "question_type": question_types[index % len(question_types)],
                    "annotator_id": (
                        f"human_auditor_{index % 4}"
                        if index < human_rows
                        else "codex_stt_seed_v1"
                    ),
                    "confidence": "high",
                }
            )
    return path


def _write_readme(tmp_path: Path, *, audited: bool, name: str = "README.md") -> Path:
    path = tmp_path / name
    text = "# Fixture\n\n"
    if audited:
        text += (
            "## Human Audit Results\n\n"
            "Human audit result: audited label rows are complete. "
            "Timestamp correction summary includes mean timestamp correction."
        )
    else:
        text += "Seed evaluation notes only."
    path.write_text(text, encoding="utf-8")
    return path
