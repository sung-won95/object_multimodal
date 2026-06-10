from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("eval/mit_deep_learning_stt/benchmark_matrix_manifest.json")
DEFAULT_OUTPUT_DIR = Path("eval/mit_deep_learning_stt/candidates/expanded_seed_v1")
ANNOTATOR_ID = "codex_stt_candidate_v1"
NOTE = "candidate label generated from STT/evidence artifacts; human audit required before use"
SOURCE = "MIT OpenCourseWare 6.7960 Deep Learning, Fall 2024"
LICENSE = "CC BY-NC-SA 4.0"

FIELDNAMES = [
    "query_id",
    "split",
    "video_id",
    "lecture_title",
    "project_id",
    "project_dir",
    "query_text",
    "query_text_ko",
    "reference_answer",
    "expected_topic",
    "expected_time_hint",
    "expected_visual_hint",
    "gold_start_time",
    "gold_end_time",
    "gold_timestamp_center",
    "gold_modality",
    "gold_visual_entity",
    "gold_entity_link_note",
    "question_type",
    "gold_segment_id",
    "gold_frame_ids",
    "gold_frame_timestamps",
    "gold_visual_entity_ids",
    "gold_visual_entity_texts",
    "linked_entity_count",
    "frame_count",
    "annotator_id",
    "confidence",
    "source",
    "license",
    "notes",
]

AUDIT_FIELDNAMES = FIELDNAMES + [
    "audit_status",
    "auditor_id",
    "audited_query_text",
    "audited_reference_answer",
    "audited_start_time",
    "audited_end_time",
    "audited_timestamp_center",
    "audited_modality",
    "audited_question_type",
    "timestamp_correction_seconds",
    "audit_notes",
]

QUESTION_PLAN = (
    ["multimodal_grounded"] * 4
    + ["concept_explanation"] * 3
    + ["definition_lookup"] * 3
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    manifest_path = resolve_path(args.manifest, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    rows = build_rows(
        manifest_path=manifest_path,
        repo_root=repo_root,
        title_by_video=read_seed_titles(repo_root / "eval/mit_deep_learning_stt/queries.csv"),
    )
    audit_rows = build_audit_sample(rows)
    write_outputs(output_dir, rows, audit_rows)
    print(json.dumps(build_summary(rows, audit_rows), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare automatic MIT Deep Learning eval expansion candidates."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (repo_root / path).resolve()


def build_rows(
    *,
    manifest_path: Path,
    repo_root: Path,
    title_by_video: dict[str, str],
) -> list[dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suites = manifest.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("manifest must contain suites")

    rows: list[dict[str, str]] = []
    for suite_index, suite in enumerate(suites):
        video_id = text(suite.get("video_id"))
        project_dir_raw = text(suite.get("project_dir"))
        if not video_id or not project_dir_raw:
            raise ValueError("each suite must contain video_id and project_dir")
        project_dir = resolve_project_dir(project_dir_raw, manifest_path)
        sources = load_sources(project_dir, video_id)
        rows.extend(
            build_video_rows(
                suite_index=suite_index,
                video_id=video_id,
                lecture_title=title_by_video.get(video_id, video_id),
                project_dir=project_dir,
                repo_root=repo_root,
                sources=sources,
            )
        )

    validate_rows(rows, expected_videos=len(suites))
    return rows


def resolve_project_dir(project_dir_raw: str, manifest_path: Path) -> Path:
    project_dir = Path(project_dir_raw).expanduser()
    if project_dir.is_absolute():
        return project_dir.resolve()
    return (manifest_path.parent / project_dir).resolve()


def load_sources(project_dir: Path, video_id: str) -> list[dict[str, Any]]:
    evidence_path = project_dir / "segments" / "evidence_units.jsonl"
    segments_path = project_dir / "segments" / "lecture_segments_aligned.jsonl"
    segments = read_jsonl(segments_path) if segments_path.exists() else []
    segment_by_id = {text(row.get("segment_id")): row for row in segments}

    if evidence_path.exists():
        raw_sources = read_jsonl(evidence_path)
        sources = [source_from_evidence(row, segment_by_id, video_id) for row in raw_sources]
    elif segments_path.exists():
        sources = [source_from_segment(row, video_id) for row in segments]
    else:
        raise FileNotFoundError(f"missing evidence_units.jsonl or lecture_segments_aligned.jsonl for {video_id}")

    sources = [source for source in sources if source["reference_answer"]]
    if len(sources) < 10:
        raise ValueError(f"{video_id} needs at least 10 usable sources, got {len(sources)}")
    return sorted(sources, key=lambda row: (float(row["start"]), row["source_id"]))


def source_from_evidence(
    row: dict[str, Any],
    segment_by_id: dict[str, dict[str, Any]],
    video_id: str,
) -> dict[str, Any]:
    segment_id = text(row.get("target_segment_id"))
    segment = segment_by_id.get(segment_id, {})
    visual_ids = list_text(row.get("visual_entity_ids"))
    frame_ids = list_text(segment.get("frame_refs"))
    start = number(row.get("start_time"), number(segment.get("start_time")))
    end = number(row.get("end_time"), number(segment.get("end_time"), start))
    body = first_text(
        row.get("evidence_text"),
        row.get("semantic_text"),
        row.get("transcript_window_text"),
        segment.get("transcript_text"),
    )
    return {
        "source_id": text(row.get("evidence_unit_id")) or segment_id,
        "project_id": text(row.get("project_id")) or text(segment.get("project_id")),
        "video_id": text(row.get("video_id")) or video_id,
        "segment_id": segment_id or text(row.get("evidence_unit_id")),
        "start": start,
        "end": end,
        "center": (start + end) / 2.0,
        "reference_answer": clean_public_text(body),
        "frame_ids": frame_ids,
        "visual_entity_ids": visual_ids,
        "has_visual": bool(frame_ids or visual_ids),
    }


def source_from_segment(row: dict[str, Any], video_id: str) -> dict[str, Any]:
    frame_ids = list_text(row.get("frame_refs"))
    start = number(row.get("start_time"))
    end = number(row.get("end_time"), start)
    return {
        "source_id": text(row.get("segment_id")),
        "project_id": text(row.get("project_id")),
        "video_id": text(row.get("video_id")) or video_id,
        "segment_id": text(row.get("segment_id")),
        "start": start,
        "end": end,
        "center": number(row.get("timestamp_center"), (start + end) / 2.0),
        "reference_answer": clean_public_text(first_text(row.get("transcript_text"), row.get("text"))),
        "frame_ids": frame_ids,
        "visual_entity_ids": list_text(row.get("visual_entity_ids")),
        "has_visual": bool(frame_ids or list_text(row.get("visual_entity_ids"))),
    }


def build_video_rows(
    *,
    suite_index: int,
    video_id: str,
    lecture_title: str,
    project_dir: Path,
    repo_root: Path,
    sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    selected = select_sources(sources)
    project_rel = os.path.relpath(project_dir, repo_root)
    rows: list[dict[str, str]] = []
    for row_index, question_type in enumerate(QUESTION_PLAN):
        source = selected[row_index]
        topic = topic_from_text(source["reference_answer"], video_id)
        is_multimodal = question_type == "multimodal_grounded"
        rows.append(
            {
                "query_id": f"mitdl_{slug(video_id)}_candidate_{row_index + 1:02d}",
                "split": "dev" if (suite_index + row_index) % 2 == 0 else "test",
                "video_id": video_id,
                "lecture_title": lecture_title,
                "project_id": source["project_id"] or project_dir.name,
                "project_dir": project_rel,
                "query_text": make_query(topic, question_type),
                "query_text_ko": make_korean_query(topic, question_type),
                "reference_answer": excerpt(source["reference_answer"]),
                "expected_topic": topic,
                "expected_time_hint": f"{seconds(source['start'])}-{seconds(source['end'])}s",
                "expected_visual_hint": visual_hint(source) if is_multimodal else "",
                "gold_start_time": seconds(source["start"]),
                "gold_end_time": seconds(source["end"]),
                "gold_timestamp_center": seconds(source["center"]),
                "gold_modality": "both" if is_multimodal else "audio",
                "gold_visual_entity": visual_hint(source) if is_multimodal else "",
                "gold_entity_link_note": (
                    f"Candidate visual grounding around {source['segment_id']}; human audit required"
                    if is_multimodal
                    else ""
                ),
                "question_type": question_type,
                "gold_segment_id": source["segment_id"],
                "gold_frame_ids": "|".join(source["frame_ids"]),
                "gold_frame_timestamps": "",
                "gold_visual_entity_ids": "|".join(source["visual_entity_ids"]),
                "gold_visual_entity_texts": "",
                "linked_entity_count": str(len(source["visual_entity_ids"])),
                "frame_count": str(len(source["frame_ids"])),
                "annotator_id": ANNOTATOR_ID,
                "confidence": "medium" if is_multimodal else "low",
                "source": SOURCE,
                "license": LICENSE,
                "notes": NOTE,
            }
        )
    return rows


def select_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visual_sources = [source for source in sources if source["has_visual"]]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for question_type in QUESTION_PLAN:
        pool = visual_sources if question_type == "multimodal_grounded" else sources
        source = next((item for item in pool if item["source_id"] not in used), None)
        if source is None:
            source = next(item for item in sources if item["source_id"] not in used)
        selected.append(source)
        used.add(source["source_id"])
    return selected


def build_audit_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row)
    sample = []
    for video_id in sorted(by_video):
        sample.extend([by_video[video_id][0], by_video[video_id][5]])
    audit_rows = []
    for row in sample:
        audit_row = dict(row)
        for field in AUDIT_FIELDNAMES:
            audit_row.setdefault(field, "")
        audit_rows.append(audit_row)
    return audit_rows


def write_outputs(output_dir: Path, rows: list[dict[str, str]], audit_rows: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "queries_candidate.csv", rows, FIELDNAMES)
    write_csv(output_dir / "human_audit_sample.csv", audit_rows, AUDIT_FIELDNAMES)
    (output_dir / "summary.json").write_text(
        json.dumps(build_summary(rows, audit_rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(readme_text(rows, audit_rows), encoding="utf-8")


def build_summary(rows: list[dict[str, str]], audit_rows: list[dict[str, str]]) -> dict[str, Any]:
    per_video = Counter(row["video_id"] for row in rows)
    return {
        "row_count": len(rows),
        "video_count": len(per_video),
        "per_video_min": min(per_video.values()) if per_video else 0,
        "per_video_max": max(per_video.values()) if per_video else 0,
        "question_type_distribution": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "annotator_distribution": dict(sorted(Counter(row["annotator_id"] for row in rows).items())),
        "split_distribution": dict(sorted(Counter(row["split"] for row in rows).items())),
        "audit_sample_count": len(audit_rows),
        "public_safe_note": "automatic candidates only; repo-relative project_dir; no local absolute paths, raw vectors, or auth tokens",
        "human_audit_required": True,
        "replaces_queries_csv": False,
    }


def readme_text(rows: list[dict[str, str]], audit_rows: list[dict[str, str]]) -> str:
    summary = build_summary(rows, audit_rows)
    return "\n".join(
        [
            "# MIT Deep Learning Expanded Seed Candidates",
            "",
            "This packet contains automatic pre-audit candidates for #270.",
            "It does not replace `eval/mit_deep_learning_stt/queries.csv`.",
            "Human audit is required before these rows are used for final evaluation or issue closure.",
            "",
            "## Generate",
            "",
            "```bash",
            "python scripts/prepare_mit_deep_learning_eval_expansion_candidates.py \\",
            "  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json \\",
            "  --output-dir eval/mit_deep_learning_stt/candidates/expanded_seed_v1",
            "```",
            "",
            "## Files",
            "",
            "- `queries_candidate.csv`: automatic 240-row candidate set",
            "- `human_audit_sample.csv`: 48-row audit sample with audit columns",
            "- `summary.json`: public-safe aggregate summary",
            "",
            "## Summary",
            "",
            f"- Rows: {summary['row_count']}",
            f"- Videos: {summary['video_count']}",
            f"- Per-video rows: min={summary['per_video_min']}, max={summary['per_video_max']}",
            f"- Question types: {summary['question_type_distribution']}",
            f"- Audit sample rows: {summary['audit_sample_count']}",
            "",
            "Validator expected behavior: this candidate packet should fail the final completion gate",
            "because all annotators are codex-generated and no Human Audit Results section exists yet.",
            "",
        ]
    )


def validate_rows(rows: list[dict[str, str]], expected_videos: int) -> None:
    per_video = Counter(row["video_id"] for row in rows)
    if len(rows) != expected_videos * 10:
        raise ValueError(f"expected {expected_videos * 10} rows, got {len(rows)}")
    if len(per_video) != expected_videos:
        raise ValueError(f"expected {expected_videos} videos, got {len(per_video)}")
    if min(per_video.values()) != 10 or max(per_video.values()) != 10:
        raise ValueError(f"expected 10 rows per video, got {dict(per_video)}")


def read_seed_titles(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            text(row.get("video_id")): text(row.get("lecture_title"))
            for row in csv.DictReader(handle)
            if text(row.get("video_id")) and text(row.get("lecture_title"))
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_query(topic: str, question_type: str) -> str:
    if question_type == "multimodal_grounded":
        return f"Which visual evidence is aligned with the discussion of {topic}?"
    if question_type == "definition_lookup":
        return f"How does the lecture define or introduce {topic}?"
    return f"What is the main idea about {topic} in this lecture segment?"


def make_korean_query(topic: str, question_type: str) -> str:
    if question_type == "multimodal_grounded":
        return f"{topic} 논의와 연결된 시각 evidence는 무엇인가?"
    if question_type == "definition_lookup":
        return f"강의에서 {topic}은(는) 어떻게 정의되거나 소개되는가?"
    return f"이 강의 segment에서 {topic}의 핵심 아이디어는 무엇인가?"


def topic_from_text(value: str, fallback: str) -> str:
    stopwords = {"about", "because", "lecture", "segment", "this", "that", "with", "from", "what", "which", "does", "into", "using"}
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", value)
        if token.lower() not in stopwords
    ]
    if not tokens:
        return fallback.replace("_", " ")
    return " ".join(token for token, _ in Counter(tokens).most_common(4))


def visual_hint(source: dict[str, Any]) -> str:
    parts = []
    if source["frame_ids"]:
        parts.append("frames " + "|".join(source["frame_ids"][:4]))
    if source["visual_entity_ids"]:
        parts.append("visual_entity_ids " + "|".join(source["visual_entity_ids"][:4]))
    return "; ".join(parts)


def first_text(*values: Any) -> str:
    for value in values:
        cleaned = text(value)
        if cleaned:
            return cleaned
    return ""


def list_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text(item) for item in value if text(item)]


def clean_public_text(value: str) -> str:
    cleaned = text(value)
    cleaned = re.sub(r"/Users/[^\s,;]+", "[local_path_redacted]", cleaned)
    cleaned = re.sub(r"/private/[^\s,;]+", "[local_path_redacted]", cleaned)
    cleaned = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+", r"\1[redacted]", cleaned)
    cleaned = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[redacted]", cleaned)
    return cleaned


def excerpt(value: str, limit: int = 360) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def seconds(value: Any) -> str:
    return f"{number(value):.1f}"


def slug(video_id: str) -> str:
    return video_id.replace("mit6_7960f24_", "").replace("_mp4", "")


if __name__ == "__main__":
    sys.exit(main())
