from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def test_apply_human_audit_outputs_validator_passing_packet(tmp_path: Path) -> None:
    module = load_script("apply_mit_deep_learning_human_audit.py")
    validator = load_script("validate_mit_deep_learning_eval_dataset.py")
    candidates_path = write_candidates(tmp_path)
    audit_path = write_audit_sample(tmp_path, audited_count=48)
    output_dir = tmp_path / "audited"

    result = module.apply_human_audit(
        candidates_path=candidates_path,
        audit_sample_path=audit_path,
        output_dir=output_dir,
    )

    assert result["ok"] is True
    assert result["audited_count"] == 48
    assert result["candidate_count"] == 240
    assert result["audited_ratio"] == 0.2

    output_rows = read_csv(output_dir / "queries_audited_candidate.csv")
    assert sum(not row["annotator_id"].startswith("codex_") for row in output_rows) == 48
    assert output_rows[1]["query_text"] == "Audited question 1?"
    assert output_rows[1]["reference_answer"] == "Audited answer 1."
    assert output_rows[1]["gold_start_time"] == "10.25"
    assert output_rows[1]["gold_end_time"] == "15.25"
    assert output_rows[1]["gold_timestamp_center"] == "13.5"
    assert output_rows[1]["expected_time_hint"] == "10.25-15.25s"
    assert output_rows[1]["gold_modality"] == "both"
    assert output_rows[1]["question_type"] == "multimodal_grounded"
    assert output_rows[1]["confidence"] == "high"

    payload = validator.validate_dataset(
        queries_path=output_dir / "queries_audited_candidate.csv",
        readme_path=output_dir / "human_audit_results.md",
        require_readme_audit_summary=True,
    )
    assert payload["ok"] is True
    assert payload["summary"]["human_audit_ratio"] == 0.2


def test_all_codex_auditors_fail(tmp_path: Path) -> None:
    module = load_script("apply_mit_deep_learning_human_audit.py")
    candidates_path = write_candidates(tmp_path)
    audit_path = write_audit_sample(tmp_path, audited_count=48, auditor_id="codex_reviewer")

    result = module.apply_human_audit(
        candidates_path=candidates_path,
        audit_sample_path=audit_path,
        output_dir=tmp_path / "audited",
    )

    assert result["ok"] is False
    assert result["audited_count"] == 0
    failure_codes = {failure["code"] for failure in result["failures"]}
    assert "invalid_auditor_id" in failure_codes
    assert "min_audited_ratio_not_met" in failure_codes


def test_timestamp_correction_mean_uses_explicit_numeric_when_present(tmp_path: Path) -> None:
    module = load_script("apply_mit_deep_learning_human_audit.py")
    candidates_path = write_candidates(tmp_path)
    audit_path = write_audit_sample(
        tmp_path,
        audited_count=48,
        explicit_corrections={0: "4.0", 1: "-2.0"},
    )

    result = module.apply_human_audit(
        candidates_path=candidates_path,
        audit_sample_path=audit_path,
        output_dir=tmp_path / "audited",
    )

    # 46 rows use audited center delta of 1.5s; two explicit values use abs(4.0), abs(-2.0).
    assert result["mean_timestamp_correction_seconds"] == (46 * 1.5 + 4.0 + 2.0) / 48


def test_rejected_rows_are_excluded_from_audited_ratio(tmp_path: Path) -> None:
    module = load_script("apply_mit_deep_learning_human_audit.py")
    candidates_path = write_candidates(tmp_path)
    audit_path = write_audit_sample(tmp_path, audited_count=47, rejected_count=1)

    result = module.apply_human_audit(
        candidates_path=candidates_path,
        audit_sample_path=audit_path,
        output_dir=tmp_path / "audited",
    )

    assert result["ok"] is False
    assert result["audited_count"] == 47
    assert result["rejected_count"] == 1
    assert result["audited_ratio"] == 47 / 240
    summary = json.loads((tmp_path / "audited" / "human_audit_results.json").read_text())
    assert summary["rejected_count"] == 1
    assert "query_text" not in json.dumps(summary)
    assert "reference_answer" not in json.dumps(summary)


def load_script(filename: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_candidates(tmp_path: Path) -> Path:
    module = load_script("apply_mit_deep_learning_human_audit.py")
    path = tmp_path / "queries_candidate.csv"
    question_types = [
        "concept_explanation",
        "definition_lookup",
        "multimodal_grounded",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.FIELDNAMES)
        writer.writeheader()
        for index in range(240):
            video_number = index // 10 + 1
            start = float(index * 10)
            writer.writerow(
                {
                    "query_id": f"fixture_q{index:03d}",
                    "split": "dev" if index % 2 == 0 else "test",
                    "video_id": f"mit6_7960f24_lec{video_number:02d}_mp4",
                    "lecture_title": f"Lecture {video_number:02d}",
                    "project_id": f"project_{video_number:02d}",
                    "project_dir": f"artifacts/project_{video_number:02d}",
                    "query_text": f"Candidate question {index}?",
                    "query_text_ko": "",
                    "reference_answer": f"Candidate answer {index}.",
                    "expected_topic": "fixture topic",
                    "expected_time_hint": f"{start}-{start + 4.0}s",
                    "expected_visual_hint": "",
                    "gold_start_time": str(start),
                    "gold_end_time": str(start + 4.0),
                    "gold_timestamp_center": str(start + 2.0),
                    "gold_modality": "audio",
                    "gold_visual_entity": "",
                    "gold_entity_link_note": "",
                    "question_type": question_types[index % len(question_types)],
                    "gold_segment_id": f"seg_{index:03d}",
                    "gold_frame_ids": "",
                    "gold_frame_timestamps": "",
                    "gold_visual_entity_ids": "",
                    "gold_visual_entity_texts": "",
                    "linked_entity_count": "0",
                    "frame_count": "0",
                    "annotator_id": "codex_stt_candidate_v1",
                    "confidence": "medium",
                    "source": "MIT OpenCourseWare 6.7960 Deep Learning, Fall 2024",
                    "license": "CC BY-NC-SA 4.0",
                    "notes": "candidate label generated from fixture",
                }
            )
    return path


def write_audit_sample(
    tmp_path: Path,
    *,
    audited_count: int,
    rejected_count: int = 0,
    auditor_id: str = "person_01",
    explicit_corrections: dict[int, str] | None = None,
) -> Path:
    path = tmp_path / "human_audit_sample.csv"
    fieldnames = [
        "query_id",
        "audit_status",
        "auditor_id",
        "audited_query_text",
        "audited_reference_answer",
        "audited_start_time",
        "audited_end_time",
        "audited_timestamp_center",
        "audited_modality",
        "audited_question_type",
        "audited_confidence",
        "timestamp_correction_seconds",
        "audit_notes",
    ]
    explicit_corrections = explicit_corrections or {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(audited_count + rejected_count):
            original_center = index * 10 + 2.0
            status = "corrected" if index % 2 else "approved"
            if index >= audited_count:
                status = "rejected"
            writer.writerow(
                {
                    "query_id": f"fixture_q{index:03d}",
                    "audit_status": status,
                    "auditor_id": auditor_id,
                    "audited_query_text": f"Audited question {index}?" if index % 2 else "",
                    "audited_reference_answer": f"Audited answer {index}." if index % 2 else "",
                    "audited_start_time": str(index * 10 + 0.25) if index % 2 else "",
                    "audited_end_time": str(index * 10 + 5.25) if index % 2 else "",
                    "audited_timestamp_center": str(original_center + 1.5),
                    "audited_modality": "both" if index % 2 else "",
                    "audited_question_type": "multimodal_grounded" if index % 2 else "",
                    "audited_confidence": "high" if index % 2 else "",
                    "timestamp_correction_seconds": explicit_corrections.get(index, ""),
                    "audit_notes": "",
                }
            )
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
