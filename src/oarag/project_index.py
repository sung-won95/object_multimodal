from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import default_paths
from .meili import LECTURE_SEGMENT_SETTINGS, MeiliClient


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
) -> dict[str, Any]:
    segments_path = segment_artifact_path(project_dir, segments=segments)

    if reset:
        client.wait_task(client.delete_index(index_uid))
    client.wait_task(client.create_index(index_uid, primary_key="segment_id"))
    client.wait_task(client.update_settings(index_uid, LECTURE_SEGMENT_SETTINGS))

    indexed_documents = 0
    indexed_batches = 0
    for batch in iter_batches(iter_jsonl_documents(segments_path), batch_size=batch_size):
        client.wait_task(client.add_documents(index_uid, batch))
        indexed_documents += len(batch)
        indexed_batches += 1

    return {
        "index": index_uid,
        "project_dir": str(project_dir),
        "segments_path": str(segments_path),
        "batch_size": batch_size,
        "reset": reset,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
    }
