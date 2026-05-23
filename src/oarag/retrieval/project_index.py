from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from oarag.core.config import default_paths
from oarag.integrations.meili import (
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    MeiliClient,
)
from oarag.integrations.meili import (
    lecture_segment_settings,
    lecture_segment_settings_snapshot,
    visual_entity_settings,
    visual_entity_settings_snapshot,
)
from oarag.core.schemas import VisualEntity, ensure_lecture_segment_semantic_contract


def project_dir_from_args(*, project_id: str | None, project_dir: Path | None) -> Path:
    if bool(project_id) == bool(project_dir):
        raise ValueError("Provide exactly one of --project-id or --project-dir")
    if project_dir is not None:
        resolved_dir = project_dir.expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = (Path.cwd() / resolved_dir).resolve()
        else:
            resolved_dir = resolved_dir.resolve()
    else:
        resolved_dir = (default_paths().artifacts_dir / "projects" / str(project_id)).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_dir}")
    return resolved_dir


def segment_artifact_path(project_dir: Path, segments: Path | None = None) -> Path:
    if segments is not None:
        candidate = segments.expanduser()
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Segment artifact not found: {candidate}")
        return candidate

    aligned = project_dir / "segments" / "lecture_segments_aligned.jsonl"
    fallback = project_dir / "segments" / "lecture_segments.jsonl"
    if aligned.exists():
        return aligned
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No segment artifact found at {aligned} or {fallback}")


def visual_entity_artifact_path(project_dir: Path, visual_entities: Path | None = None) -> Path:
    if visual_entities is not None:
        candidate = visual_entities.expanduser()
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Visual entity artifact not found: {candidate}")
        return candidate

    default = project_dir / "manifests" / "visual_entities.jsonl"
    if default.exists():
        return default
    raise FileNotFoundError(f"No visual entity artifact found at {default}")


def iter_jsonl_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def iter_batches(documents: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    batch: list[dict[str, Any]] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_project_segments(
    client: MeiliClient,
    *,
    index_uid: str,
    project_dir: Path,
    batch_size: int = 500,
    reset: bool = False,
    segments: Path | None = None,
    visual_entities: Path | None = None,
    settings_profile: str = LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
) -> dict[str, Any]:
    segments_path = segment_artifact_path(project_dir, segments=segments)
    visual_entities_path = _optional_visual_entity_artifact_path(
        project_dir=project_dir,
        visual_entities=visual_entities,
    )
    visual_entity_context = _visual_entity_context(visual_entities_path)
    settings = lecture_segment_settings(settings_profile)
    settings_snapshot = lecture_segment_settings_snapshot(settings, profile=settings_profile)

    if reset:
        client.wait_task(client.delete_index(index_uid), ignored_error_codes={"index_not_found"})
    client.wait_task(client.create_index(index_uid, primary_key="segment_id"))
    client.wait_task(client.update_settings(index_uid, settings))

    indexed_documents = 0
    indexed_batches = 0
    semantic_source_field_counts: dict[str, int] = {}
    embedded_visual_entity_count = 0
    documents = (
        _segment_index_document(document, visual_entity_context=visual_entity_context)
        for document in iter_jsonl_documents(segments_path)
    )
    for batch in iter_batches(documents, batch_size=batch_size):
        client.wait_task(client.add_documents(index_uid, batch))
        indexed_documents += len(batch)
        indexed_batches += 1
        for document in batch:
            embedded_visual_entity_count += len(_list_of_dicts(document.get("visual_entities")))
            for source_field in document.get("semantic_source_fields", []):
                semantic_source_field_counts[str(source_field)] = (
                    semantic_source_field_counts.get(str(source_field), 0) + 1
                )

    return {
        "index": index_uid,
        "project_dir": str(project_dir),
        "segments_path": str(segments_path),
        "visual_entities_path": str(visual_entities_path) if visual_entities_path else None,
        "batch_size": batch_size,
        "reset": reset,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
        "embedded_visual_entities": embedded_visual_entity_count,
        "semantic_source_field_counts": semantic_source_field_counts,
        "settings_profile": settings_snapshot["profile"],
        "settings_hash": settings_snapshot["hash"],
        "settings_snapshot": settings_snapshot,
    }


def index_project_visual_entities(
    client: MeiliClient,
    *,
    index_uid: str,
    project_dir: Path,
    batch_size: int = 500,
    reset: bool = False,
    visual_entities: Path | None = None,
    settings_profile: str = VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
) -> dict[str, Any]:
    visual_entities_path = visual_entity_artifact_path(project_dir, visual_entities=visual_entities)
    settings = visual_entity_settings(settings_profile)
    settings_snapshot = visual_entity_settings_snapshot(settings, profile=settings_profile)

    if reset:
        client.wait_task(client.delete_index(index_uid), ignored_error_codes={"index_not_found"})
    client.wait_task(client.create_index(index_uid, primary_key="entity_id"))
    client.wait_task(client.update_settings(index_uid, settings))

    indexed_documents = 0
    indexed_batches = 0
    semantic_source_field_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    documents = (
        _visual_entity_index_document(document)
        for document in iter_jsonl_documents(visual_entities_path)
    )
    for batch in iter_batches(documents, batch_size=batch_size):
        client.wait_task(client.add_documents(index_uid, batch))
        indexed_documents += len(batch)
        indexed_batches += 1
        for document in batch:
            for source_field in document.get("semantic_source_fields", []):
                semantic_source_field_counts[str(source_field)] = (
                    semantic_source_field_counts.get(str(source_field), 0) + 1
                )
            source = document.get("source")
            if source not in (None, ""):
                source_counts[str(source)] = source_counts.get(str(source), 0) + 1

    return {
        "index": index_uid,
        "project_dir": str(project_dir),
        "visual_entities_path": str(visual_entities_path),
        "batch_size": batch_size,
        "reset": reset,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
        "semantic_source_field_counts": semantic_source_field_counts,
        "source_counts": source_counts,
        "settings_profile": settings_snapshot["profile"],
        "settings_hash": settings_snapshot["hash"],
        "settings_snapshot": settings_snapshot,
    }


def _visual_entity_index_document(document: dict[str, Any]) -> dict[str, Any]:
    entity = VisualEntity.from_dict(document)
    if not entity.entity_id:
        raise ValueError("visual entity document is missing entity_id")
    if not entity.frame_id:
        raise ValueError(f"visual entity document is missing frame_id: {entity.entity_id}")

    indexed = entity.to_dict()
    for optional_field in ("video_id", "segment_id"):
        if document.get(optional_field) not in (None, ""):
            indexed[optional_field] = document[optional_field]
    indexed["semantic_source_fields"] = _visual_entity_semantic_source_fields(indexed)
    return indexed


def _segment_index_document(
    document: dict[str, Any],
    *,
    visual_entity_context: dict[str, Any],
) -> dict[str, Any]:
    indexed = dict(document)
    if "visual_entities" not in indexed:
        linked_entities = _linked_visual_entities(indexed, visual_entity_context=visual_entity_context)
        if linked_entities:
            indexed["visual_entities"] = linked_entities
    return ensure_lecture_segment_semantic_contract(indexed)


def _optional_visual_entity_artifact_path(
    *,
    project_dir: Path,
    visual_entities: Path | None,
) -> Path | None:
    if visual_entities is not None:
        return visual_entity_artifact_path(project_dir, visual_entities=visual_entities)
    default = project_dir / "manifests" / "visual_entities.jsonl"
    return default if default.exists() else None


def _visual_entity_context(path: Path | None) -> dict[str, Any]:
    by_segment_id: dict[str, list[dict[str, Any]]] = {}
    by_frame_id: dict[str, list[dict[str, Any]]] = {}
    if path is None:
        return {"by_segment_id": by_segment_id, "by_frame_id": by_frame_id}

    for row in iter_jsonl_documents(path):
        entity = _compact_visual_entity(row)
        segment_id = row.get("segment_id")
        if segment_id not in (None, ""):
            by_segment_id.setdefault(str(segment_id), []).append(entity)
        frame_id = row.get("frame_id")
        if frame_id not in (None, ""):
            by_frame_id.setdefault(str(frame_id), []).append(entity)
    return {"by_segment_id": by_segment_id, "by_frame_id": by_frame_id}


def _linked_visual_entities(
    document: dict[str, Any],
    *,
    visual_entity_context: dict[str, Any],
) -> list[dict[str, Any]]:
    by_segment_id = visual_entity_context["by_segment_id"]
    by_frame_id = visual_entity_context["by_frame_id"]
    linked: list[dict[str, Any]] = []
    seen: set[str] = set()

    segment_id = document.get("segment_id")
    if segment_id not in (None, ""):
        _append_unique_entities(linked, seen, by_segment_id.get(str(segment_id), []))
    for frame_id in _segment_frame_ids(document):
        _append_unique_entities(linked, seen, by_frame_id.get(frame_id, []))
    return linked


def _append_unique_entities(
    linked: list[dict[str, Any]],
    seen: set[str],
    candidates: list[dict[str, Any]],
) -> None:
    for entity in candidates:
        entity_id = str(entity.get("entity_id") or "")
        key = entity_id or json.dumps(entity, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        linked.append(entity)


def _segment_frame_ids(document: dict[str, Any]) -> list[str]:
    frame_refs = document.get("frame_refs")
    if not isinstance(frame_refs, list):
        return []
    frame_ids: list[str] = []
    seen: set[str] = set()
    for item in frame_refs:
        if isinstance(item, dict):
            value = item.get("frame_id")
        else:
            value = item
        if value in (None, ""):
            continue
        frame_id = str(value)
        if frame_id in seen:
            continue
        seen.add(frame_id)
        frame_ids.append(frame_id)
    return frame_ids


def _compact_visual_entity(row: dict[str, Any]) -> dict[str, Any]:
    entity = VisualEntity.from_dict(row)
    compact = {
        "entity_id": entity.entity_id,
        "frame_id": entity.frame_id,
        "text": entity.text,
        "visual_description": entity.visual_description,
        "entity_type": entity.entity_type,
        "confidence": entity.confidence,
        "source": entity.source,
        "source_model": entity.source_model,
    }
    segment_id = row.get("segment_id")
    if segment_id not in (None, ""):
        compact["segment_id"] = str(segment_id)
    video_id = row.get("video_id")
    if video_id not in (None, ""):
        compact["video_id"] = str(video_id)
    return compact


def _visual_entity_semantic_source_fields(document: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for field_name in ("text", "visual_description"):
        value = document.get(field_name)
        if isinstance(value, str) and value.strip():
            fields.append(field_name)
    return fields


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
