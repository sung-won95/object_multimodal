from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType


def test_expansion_candidates_packet_and_validator_expected_failure(tmp_path: Path) -> None:
    candidate_module = load_script("prepare_mit_deep_learning_eval_expansion_candidates.py")
    validator = load_script("validate_mit_deep_learning_eval_dataset.py")
    manifest_path = write_fixture_repo(tmp_path)

    assert candidate_module.main(["--repo-root", str(tmp_path), "--manifest", str(manifest_path)]) == 0

    output_dir = tmp_path / "eval" / "mit_deep_learning_stt" / "candidates" / "expanded_seed_v1"
    queries_path = output_dir / "queries_candidate.csv"
    audit_path = output_dir / "human_audit_sample.csv"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"

    rows = read_csv(queries_path)
    audit_rows = read_csv(audit_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    counts_by_video = Counter(row["video_id"] for row in rows)
    assert len(rows) == 240
    assert len(counts_by_video) == 24
    assert min(counts_by_video.values()) == 10
    assert max(counts_by_video.values()) == 10
    assert {row["annotator_id"] for row in rows} == {"codex_stt_candidate_v1"}
    assert all("human audit required before use" in row["notes"] for row in rows)
    assert all(row["project_dir"].startswith("artifacts/") for row in rows)
    assert {"concept_explanation", "definition_lookup", "multimodal_grounded"}.issubset(
        {row["question_type"] for row in rows}
    )

    assert len(audit_rows) == 48
    assert {"audit_status", "auditor_id", "timestamp_correction_seconds"}.issubset(
        audit_rows[0]
    )
    assert summary["row_count"] == 240
    assert summary["video_count"] == 24
    assert summary["per_video_min"] == 10
    assert summary["per_video_max"] == 10
    assert summary["audit_sample_count"] == 48

    payload = validator.validate_dataset(
        queries_path=queries_path,
        readme_path=readme_path,
        expected_video_ids=validator.read_expected_video_ids(manifest_path),
        require_readme_audit_summary=True,
    )
    failure_codes = {failure["code"] for failure in payload["failures"]}
    assert payload["ok"] is False
    assert "min_queries_not_met" not in failure_codes
    assert "min_videos_not_met" not in failure_codes
    assert "min_per_video_not_met" not in failure_codes
    assert "required_question_types_missing" not in failure_codes
    assert "min_human_audit_ratio_not_met" in failure_codes
    assert "all_annotators_codex_generated" in failure_codes
    assert "readme_audit_summary_missing" in failure_codes

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (queries_path, audit_path, summary_path, readme_path)
    )
    assert "/Users/" not in public_text
    assert "/private/" not in public_text
    assert "_vectors" not in public_text
    assert "Bearer " not in public_text


def load_script(filename: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture_repo(repo_root: Path) -> Path:
    eval_dir = repo_root / "eval" / "mit_deep_learning_stt"
    eval_dir.mkdir(parents=True)
    manifest_path = eval_dir / "benchmark_matrix_manifest.json"
    suites = []
    seed_rows = []

    for video_number in range(1, 25):
        video_id = f"mit6_7960f24_lec{video_number:02d}_mp4"
        project_id = f"fixture_project_{video_number:02d}"
        project_dir = repo_root / "artifacts" / "paper_mit_deep_learning" / "projects" / project_id
        write_project(project_dir=project_dir, project_id=project_id, video_id=video_id)
        suites.append(
            {
                "suite_id": f"mitdl_lec{video_number:02d}",
                "project_dir": f"../../artifacts/paper_mit_deep_learning/projects/{project_id}",
                "video_id": video_id,
            }
        )
        seed_rows.append(
            {
                "query_id": f"seed_{video_number:02d}",
                "split": "dev",
                "video_id": video_id,
                "lecture_title": f"Lec {video_number:02d}. Fixture Lecture",
            }
        )

    manifest_path.write_text(
        json.dumps({"run_id": "fixture", "suites": suites}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (eval_dir / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "split", "video_id", "lecture_title"],
        )
        writer.writeheader()
        writer.writerows(seed_rows)
    return manifest_path


def write_project(*, project_dir: Path, project_id: str, video_id: str) -> None:
    segments_dir = project_dir / "segments"
    segments_dir.mkdir(parents=True)
    segments = []
    evidence_units = []
    for index in range(12):
        start = float(index * 10)
        segment_id = f"seg_{video_id}_{index:03d}"
        frame_id = f"frame_{index:06d}"
        entity_id = f"ent_{index:06d}"
        has_visual = index < 6
        segments.append(
            {
                "segment_id": segment_id,
                "project_id": project_id,
                "video_id": video_id,
                "start_time": start,
                "end_time": start + 4.0,
                "timestamp_center": start + 2.0,
                "transcript_text": f"Fixture transcript explains gradient attention concept {index}.",
                "frame_refs": [frame_id] if has_visual else [],
            }
        )
        evidence_units.append(
            {
                "evidence_unit_id": f"evu_{index:03d}",
                "project_id": project_id,
                "video_id": video_id,
                "target_segment_id": segment_id,
                "start_time": start,
                "end_time": start + 4.0,
                "visual_entity_ids": [entity_id] if has_visual else [],
                "evidence_text": f"Evidence explains optimization gradient representation topic {index}.",
            }
        )
    write_jsonl(segments_dir / "lecture_segments_aligned.jsonl", segments)
    write_jsonl(segments_dir / "evidence_units.jsonl", evidence_units)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
