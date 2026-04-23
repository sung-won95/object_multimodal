from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl
from .project_index import segment_artifact_path
from .schemas import EntityLink, VisualEntity, mention_candidates, slugify


def link_entities(
    *,
    project_dir: Path,
    segments_path: Path | None = None,
    visual_entities_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_segments_path = segment_artifact_path(resolved_project_dir, segments=segments_path)
    resolved_visual_entities_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=visual_entities_path,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
    )
    resolved_output_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=output_path,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
    )
    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )

    segments = _read_jsonl(resolved_segments_path)
    entities = [VisualEntity.from_dict(row) for row in _read_jsonl(resolved_visual_entities_path)]
    project_id = _project_id_from_manifest_or_dir(resolved_manifest_path, resolved_project_dir, segments, entities)

    links: list[EntityLink] = []
    for segment in _sorted_segments(segments):
        for entity in _sorted_entities(entities):
            time_overlap = _has_time_overlap(segment, entity)
            if not time_overlap:
                continue

            lexical_match = _lexical_match(segment, entity)
            mention_match = _mention_match(segment, entity)
            evidence = ["time_overlap"]
            if lexical_match:
                evidence.append("lexical_match")
            if mention_match:
                evidence.append("mention_candidate")

            links.append(
                EntityLink(
                    link_id=f"link_{slugify(str(segment.get('segment_id', '')))}_{slugify(entity.entity_id)}",
                    project_id=project_id,
                    segment_id=str(segment.get("segment_id", "")),
                    entity_id=entity.entity_id,
                    frame_id=entity.frame_id,
                    link_type=_link_type(lexical_match=lexical_match, mention_match=mention_match),
                    score=_score(lexical_match=lexical_match, mention_match=mention_match),
                    evidence=evidence,
                    time_overlap=True,
                    lexical_match=lexical_match,
                    mention_candidate=mention_match,
                )
            )

    write_jsonl(resolved_output_path, [link.to_dict() for link in links])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        entity_links_path=resolved_output_path,
        segments_total=len(segments),
        visual_entities_total=len(entities),
        links=links,
    )

    return {
        "project_id": project_id,
        "paths": {
            "project_dir": str(resolved_project_dir),
            "segments": str(resolved_segments_path),
            "visual_entities": str(resolved_visual_entities_path),
            "entity_links": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "counts": {
            "segments_total": len(segments),
            "visual_entities_total": len(entities),
            "entity_links": len(links),
            "lexical_links": sum(1 for link in links if link.lexical_match),
            "mention_links": sum(1 for link in links if link.mention_candidate),
            "timestamp_only_links": sum(
                1 for link in links if not link.lexical_match and not link.mention_candidate
            ),
        },
    }


def _sorted_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        segments,
        key=lambda segment: (
            _optional_float(segment.get("timestamp_center")) is None,
            _optional_float(segment.get("timestamp_center")) or _optional_float(segment.get("start_time")) or 0.0,
            str(segment.get("segment_id", "")),
        ),
    )


def _sorted_entities(entities: list[VisualEntity]) -> list[VisualEntity]:
    return sorted(
        entities,
        key=lambda entity: (
            entity.timestamp is None,
            entity.timestamp or 0.0,
            entity.frame_id,
            entity.entity_id,
        ),
    )


def _has_time_overlap(segment: dict[str, Any], entity: VisualEntity) -> bool:
    frame_refs = {str(frame_ref) for frame_ref in segment.get("frame_refs") or []}
    if entity.frame_id and entity.frame_id in frame_refs:
        return True

    entity_time = entity.timestamp
    if entity_time is None:
        return False
    start = _optional_float(segment.get("start_time"))
    end = _optional_float(segment.get("end_time"))
    center = _optional_float(segment.get("timestamp_center"))
    if start is None and end is None:
        return center is not None and entity_time == center
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return False
    return min(start, end) <= entity_time <= max(start, end)


def _lexical_match(segment: dict[str, Any], entity: VisualEntity) -> list[str]:
    transcript_terms = _terms(str(segment.get("transcript_text", "")))
    entity_terms = _terms(entity.text)
    return sorted(term for term in transcript_terms & entity_terms if len(term) >= 3)


def _mention_match(segment: dict[str, Any], entity: VisualEntity) -> list[str]:
    entity_text = entity.text.casefold()
    candidates = segment.get("mention_candidates")
    if not isinstance(candidates, list):
        candidates = mention_candidates(str(segment.get("transcript_text", "")))
    return sorted(
        {
            str(candidate)
            for candidate in candidates
            if str(candidate).strip() and str(candidate).casefold() in entity_text
        }
    )


def _terms(text: str) -> set[str]:
    return {match.group(0) for match in re.finditer(r"[a-z0-9]+", text.casefold())}


def _link_type(*, lexical_match: list[str], mention_match: list[str]) -> str:
    if lexical_match and mention_match:
        return "time_overlap+lexical_match+mention_candidate"
    if lexical_match:
        return "time_overlap+lexical_match"
    if mention_match:
        return "time_overlap+mention_candidate"
    return "time_overlap"


def _score(*, lexical_match: list[str], mention_match: list[str]) -> float:
    score = 1.0
    if lexical_match:
        score += 0.2 + (0.05 * len(lexical_match))
    if mention_match:
        score += 0.1 + (0.05 * len(mention_match))
    return round(score, 3)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL not found: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            rows.append(payload)
    return rows


def _project_id_from_manifest_or_dir(
    manifest_path: Path,
    project_dir: Path,
    segments: list[dict[str, Any]],
    entities: list[VisualEntity],
) -> str:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("project_id") is not None:
            return str(loaded["project_id"])
    if segments and segments[0].get("project_id") is not None:
        return str(segments[0]["project_id"])
    if entities:
        return entities[0].project_id
    return project_dir.name


def _update_project_manifest(
    *,
    manifest_path: Path,
    entity_links_path: Path,
    segments_total: int,
    visual_entities_total: int,
    links: list[EntityLink],
) -> None:
    payload: dict[str, Any]
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["entity_links"] = str(entity_links_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["entity_links"] = len(links)

    payload["entity_linking"] = {
        "segments_total": segments_total,
        "visual_entities_total": visual_entities_total,
        "entity_links": len(links),
        "lexical_links": sum(1 for link in links if link.lexical_match),
        "mention_links": sum(1 for link in links if link.mention_candidate),
        "timestamp_only_links": sum(
            1 for link in links if not link.lexical_match and not link.mention_candidate
        ),
    }
    write_json(manifest_path, payload)


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
