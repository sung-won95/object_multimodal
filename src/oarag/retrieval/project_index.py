from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from oarag.core.config import default_paths
from oarag.core.io import write_json, write_jsonl
from oarag.embeddings.manifest import (
    NO_SEMANTIC_QUALITY_CLAIM,
    PROVIDER_EMBEDDING_QUALITY_CLAIM,
    LoadedVectorManifest,
    load_vector_manifest,
)
from oarag.integrations.meili import (
    DEFAULT_HYBRID_EMBEDDER_NAME,
    EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE,
    HYBRID_EMBEDDER_CUSTOM_SETTINGS_PROFILE,
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    MeiliClient,
)
from oarag.integrations.meili import (
    evidence_unit_settings,
    evidence_unit_settings_snapshot,
    hybrid_embedder_settings,
    hybrid_embedder_settings_snapshot,
    lecture_segment_settings,
    lecture_segment_settings_snapshot,
    lecture_window_settings,
    lecture_window_settings_snapshot,
    merge_hybrid_embedder_settings,
    normalize_hybrid_embedder_settings,
    normalize_query_vector,
    visual_entity_settings,
    visual_entity_settings_snapshot,
)
from oarag.core.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
    VisualEntity,
    ensure_lecture_segment_semantic_contract,
    slugify,
)
from oarag.retrieval.vectors import (
    LOCAL_HASH_VECTOR_SOURCE,
    deterministic_text_vector,
    text_from_document_fields,
)


LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH = Path("segments") / "lecture_windows.jsonl"
EVIDENCE_UNITS_ARTIFACT_RELATIVE_PATH = Path("segments") / "evidence_units.jsonl"
LOCAL_HASH_DOCUMENT_VECTOR_WARNING = (
    "local_hash_v1 document vectors are deterministic smoke-test fallback only; "
    "do not report semantic embedding quality without a provider-backed vector manifest."
)
TRANSCRIPT_KEYWORD_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "between",
    "from",
    "here",
    "into",
    "just",
    "like",
    "more",
    "next",
    "only",
    "over",
    "that",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "with",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "would",
}
UNVERIFIED_DOCUMENT_VECTOR_WARNING = (
    "Document vectors do not carry provider-backed embedding metadata; semantic "
    "quality claims are disabled for this index summary."
)


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


def window_artifact_path(project_dir: Path, windows: Path | None = None) -> Path:
    if windows is not None:
        candidate = windows.expanduser()
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Window artifact not found: {candidate}")
        return candidate

    default = project_dir / LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH
    if default.exists():
        return default
    raise FileNotFoundError(f"No window artifact found at {default}")


def evidence_unit_artifact_path(project_dir: Path, evidence_units: Path | None = None) -> Path:
    if evidence_units is not None:
        candidate = evidence_units.expanduser()
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Evidence unit artifact not found: {candidate}")
        return candidate

    default = project_dir / EVIDENCE_UNITS_ARTIFACT_RELATIVE_PATH
    if default.exists():
        return default
    raise FileNotFoundError(f"No evidence unit artifact found at {default}")


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


def build_window_index_documents(
    segments: list[dict[str, Any]],
    *,
    frames: list[dict[str, Any]] | None = None,
    visual_entities: list[dict[str, Any]] | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
) -> list[dict[str, Any]]:
    from oarag.retrieval.evidence import (
        frame_id,
        make_evidence_window,
        resolve_window_config,
        segment_sort_key,
        select_window_segments,
    )

    window_config = resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )
    frame_lookup = {frame_id(frame): frame for frame in frames or []}
    visual_entity_context = _visual_entity_context_from_rows(visual_entities or [])
    sorted_segments = [
        segment for _, segment in sorted(enumerate(segments), key=lambda item: segment_sort_key(item[1], item[0]))
    ]

    documents: list[dict[str, Any]] = []
    for target in sorted_segments:
        target_segment_id = str(target.get("segment_id") or "")
        if not target_segment_id:
            continue
        window_segments = select_window_segments(
            sorted_segments,
            target_segment_id=target_segment_id,
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        )
        evidence_window = make_evidence_window(
            target=target,
            window_segments=window_segments,
            frame_lookup=frame_lookup,
            window_config=window_config,
        ).to_dict()
        documents.append(
            _window_index_document(
                target=target,
                window_segments=window_segments,
                evidence_window=evidence_window,
                visual_entity_context=visual_entity_context,
                window_config=window_config,
            )
        )
    return documents


def build_project_windows(
    *,
    project_dir: Path,
    output_path: Path | None = None,
    segments: Path | None = None,
    frames_manifest: Path | None = None,
    visual_entities: Path | None = None,
    manifest_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    segments_path = segment_artifact_path(resolved_project_dir, segments=segments)
    frames_path = _optional_project_path(
        project_dir=resolved_project_dir,
        path=frames_manifest,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )
    visual_entities_path = _optional_visual_entity_artifact_path(
        project_dir=resolved_project_dir,
        visual_entities=visual_entities,
    )
    resolved_output_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=output_path,
        default=resolved_project_dir / LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
    )
    resolved_manifest_path = _resolve_project_output_path(
        project_dir=resolved_project_dir,
        path=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )

    segment_rows = list(iter_jsonl_documents(segments_path))
    frame_rows = list(iter_jsonl_documents(frames_path)) if frames_path.exists() else []
    visual_entity_rows = (
        list(iter_jsonl_documents(visual_entities_path)) if visual_entities_path else []
    )
    documents = build_window_index_documents(
        segment_rows,
        frames=frame_rows,
        visual_entities=visual_entity_rows,
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )
    write_jsonl(resolved_output_path, documents)

    summary = {
        "project_dir": str(resolved_project_dir),
        "paths": {
            "segments": str(segments_path),
            "frames_manifest": str(frames_path),
            "visual_entities": str(visual_entities_path) if visual_entities_path else None,
            "lecture_windows": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "window_config": documents[0]["window_config"] if documents else _empty_window_config(
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        ),
        "counts": {
            "segments_total": len(segment_rows),
            "windows_total": len(documents),
            "frames_total": len(frame_rows),
            "visual_entities_total": len(visual_entity_rows),
            "window_frame_refs": sum(len(_list_of_dicts(document.get("frame_refs"))) for document in documents),
            "window_visual_entities": sum(
                len(_list_of_dicts(document.get("visual_entities"))) for document in documents
            ),
        },
    }
    _update_project_window_manifest(
        manifest_path=resolved_manifest_path,
        windows_path=resolved_output_path,
        summary=summary,
    )
    return summary


def index_project_windows(
    client: MeiliClient,
    *,
    index_uid: str,
    project_dir: Path,
    batch_size: int = 500,
    reset: bool = False,
    configure_index: bool = True,
    windows: Path | None = None,
    segments: Path | None = None,
    frames_manifest: Path | None = None,
    visual_entities: Path | None = None,
    settings_profile: str = LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    hybrid_embedder_profile: str | None = None,
    hybrid_embedder_config: dict[str, Any] | None = None,
    hybrid_embedder_name: str = DEFAULT_HYBRID_EMBEDDER_NAME,
    hybrid_embedder_dimensions: int | None = None,
    hybrid_embedder_live_smoke: bool = False,
    vector_manifest: Path | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    build_requested = _window_build_inputs_requested(
        segments=segments,
        frames_manifest=frames_manifest,
        visual_entities=visual_entities,
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )
    windows_path = (
        window_artifact_path(resolved_project_dir, windows=windows)
        if windows is not None
        else None
    )
    if windows_path is None and not build_requested:
        windows_path = _optional_window_artifact_path(resolved_project_dir, windows=None)
    segments_path: Path | None = None
    frames_path: Path | None = None
    visual_entities_path: Path | None = None

    if windows_path is not None:
        documents: Iterable[dict[str, Any]] = (
            ensure_window_document_semantic_contract(document)
            for document in iter_jsonl_documents(windows_path)
        )
    else:
        segments_path = segment_artifact_path(resolved_project_dir, segments=segments)
        frames_path = _optional_project_path(
            project_dir=resolved_project_dir,
            path=frames_manifest,
            default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
        )
        visual_entities_path = _optional_visual_entity_artifact_path(
            project_dir=resolved_project_dir,
            visual_entities=visual_entities,
        )
        segment_rows = list(iter_jsonl_documents(segments_path))
        frame_rows = list(iter_jsonl_documents(frames_path)) if frames_path.exists() else []
        visual_entity_rows = (
            list(iter_jsonl_documents(visual_entities_path)) if visual_entities_path else []
        )
        documents = build_window_index_documents(
            segment_rows,
            frames=frame_rows,
            visual_entities=visual_entity_rows,
            window_seconds=window_seconds,
            neighbor_count=neighbor_count,
            previous_neighbor_count=previous_neighbor_count,
            next_neighbor_count=next_neighbor_count,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds,
        )

    hybrid_settings, hybrid_snapshot = _resolve_hybrid_embedder_settings(
        profile=hybrid_embedder_profile,
        config=hybrid_embedder_config,
        embedder_name=hybrid_embedder_name,
        dimensions=hybrid_embedder_dimensions,
        live_smoke=hybrid_embedder_live_smoke,
    )
    settings = merge_hybrid_embedder_settings(
        lecture_window_settings(settings_profile),
        hybrid_settings,
    )
    vector_specs = _user_provided_vector_specs(settings)
    vector_manifest_index = _load_index_vector_manifest(
        project_dir=resolved_project_dir,
        path=vector_manifest,
    )
    _validate_vector_manifest_for_specs(vector_specs, vector_manifest_index)
    vector_summary = _new_document_vector_summary(
        vector_specs,
        vector_manifest=vector_manifest_index,
    )
    settings_snapshot = lecture_window_settings_snapshot(
        settings,
        profile=settings_profile,
        redact_secrets=hybrid_snapshot is not None,
    )

    hybrid_live_smoke = None
    if configure_index:
        if reset:
            client.wait_task(client.delete_index(index_uid), ignored_error_codes={"index_not_found"})
        client.wait_task(
            client.create_index(index_uid, primary_key="window_id"),
            ignored_error_codes={"index_already_exists"},
        )
        _apply_index_settings(
            client,
            index_uid=index_uid,
            settings=settings,
            settings_profile=settings_profile,
            hybrid_snapshot=hybrid_snapshot,
        )
        hybrid_live_smoke = _run_hybrid_embedder_live_smoke(
            client,
            index_uid=index_uid,
            hybrid_snapshot=hybrid_snapshot,
            requested=hybrid_embedder_live_smoke,
        )

    indexed_documents = 0
    indexed_batches = 0
    semantic_source_field_counts: dict[str, int] = {}
    visual_entity_count = 0
    indexed_documents_iter = _documents_with_user_provided_vectors(
        documents,
        specs=vector_specs,
        summary=vector_summary,
        vector_manifest=vector_manifest_index,
        id_fields=("window_id", "target_segment_id", "segment_id", "sample_id"),
    )
    for batch in iter_batches(indexed_documents_iter, batch_size=batch_size):
        client.wait_task(client.add_documents(index_uid, batch))
        indexed_documents += len(batch)
        indexed_batches += 1
        for document in batch:
            visual_entity_count += len(_list_of_dicts(document.get("visual_entities")))
            for source_field in document.get(LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD, []):
                semantic_source_field_counts[str(source_field)] = (
                    semantic_source_field_counts.get(str(source_field), 0) + 1
                )
    _finalize_document_vector_summary(vector_summary)

    summary = {
        "index": index_uid,
        "project_dir": str(resolved_project_dir),
        "windows_path": str(windows_path) if windows_path else None,
        "segments_path": str(segments_path) if segments_path else None,
        "frames_manifest_path": str(frames_path) if frames_path else None,
        "visual_entities_path": str(visual_entities_path) if visual_entities_path else None,
        "batch_size": batch_size,
        "reset": reset,
        "configure_index": configure_index,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
        "embedded_visual_entities": visual_entity_count,
        "semantic_source_field_counts": semantic_source_field_counts,
        "settings_profile": settings_snapshot["profile"],
        "settings_hash": settings_snapshot["hash"],
        "settings_snapshot": settings_snapshot,
        "document_vectors": vector_summary,
    }
    _attach_embedding_backend_report(summary, vector_summary)
    _attach_hybrid_embedder_summary(
        summary,
        hybrid_snapshot=hybrid_snapshot,
        live_smoke=hybrid_live_smoke,
    )
    return summary


def index_project_evidence_units(
    client: MeiliClient,
    *,
    index_uid: str,
    project_dir: Path,
    batch_size: int = 500,
    reset: bool = False,
    configure_index: bool = True,
    evidence_units: Path | None = None,
    settings_profile: str = EVIDENCE_UNIT_DEFAULT_SETTINGS_PROFILE,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    evidence_units_path = evidence_unit_artifact_path(
        resolved_project_dir,
        evidence_units=evidence_units,
    )
    settings = evidence_unit_settings(settings_profile)
    settings_snapshot = evidence_unit_settings_snapshot(
        settings,
        profile=settings_profile,
    )

    if configure_index:
        if reset:
            client.wait_task(client.delete_index(index_uid), ignored_error_codes={"index_not_found"})
        client.wait_task(
            client.create_index(index_uid, primary_key="evidence_unit_id"),
            ignored_error_codes={"index_already_exists"},
        )
        _apply_index_settings(
            client,
            index_uid=index_uid,
            settings=settings,
            settings_profile=settings_profile,
            hybrid_snapshot=None,
        )

    indexed_documents = 0
    indexed_batches = 0
    alignment_status_counts: dict[str, int] = {}
    source_quality_counts = {
        "has_visual_state": 0,
        "has_visual_entity": 0,
        "has_vlm_entity": 0,
        "has_concept": 0,
        "has_concept_relation": 0,
        "has_verified_link": 0,
        "has_timestamp_fallback_link": 0,
    }
    documents = (
        _evidence_unit_index_document(document)
        for document in iter_jsonl_documents(evidence_units_path)
    )
    for batch in iter_batches(documents, batch_size=batch_size):
        client.wait_task(client.add_documents(index_uid, batch))
        indexed_documents += len(batch)
        indexed_batches += 1
        for document in batch:
            status = str(document.get("alignment_status") or "unknown")
            alignment_status_counts[status] = alignment_status_counts.get(status, 0) + 1
            source_quality = document.get("source_quality")
            if isinstance(source_quality, dict):
                for key in source_quality_counts:
                    if source_quality.get(key) is True:
                        source_quality_counts[key] += 1
            if _evidence_unit_has_concept_fields(document):
                source_quality_counts["has_concept"] += (
                    0
                    if isinstance(source_quality, dict) and source_quality.get("has_concept") is True
                    else 1
                )
            if _evidence_unit_has_concept_relation_fields(document):
                source_quality_counts["has_concept_relation"] += (
                    0
                    if isinstance(source_quality, dict)
                    and source_quality.get("has_concept_relation") is True
                    else 1
                )

    return {
        "index": index_uid,
        "project_dir": str(resolved_project_dir),
        "evidence_units_path": str(evidence_units_path),
        "batch_size": batch_size,
        "reset": reset,
        "configure_index": configure_index,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
        "alignment_status_counts": alignment_status_counts,
        "source_quality_counts": source_quality_counts,
        "settings_profile": settings_snapshot["profile"],
        "settings_hash": settings_snapshot["hash"],
        "settings_snapshot": settings_snapshot,
    }


def index_project_segments(
    client: MeiliClient,
    *,
    index_uid: str,
    project_dir: Path,
    batch_size: int = 500,
    reset: bool = False,
    configure_index: bool = True,
    segments: Path | None = None,
    visual_entities: Path | None = None,
    settings_profile: str = LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    hybrid_embedder_profile: str | None = None,
    hybrid_embedder_config: dict[str, Any] | None = None,
    hybrid_embedder_name: str = DEFAULT_HYBRID_EMBEDDER_NAME,
    hybrid_embedder_dimensions: int | None = None,
    hybrid_embedder_live_smoke: bool = False,
    vector_manifest: Path | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    segments_path = segment_artifact_path(resolved_project_dir, segments=segments)
    visual_entities_path = _optional_visual_entity_artifact_path(
        project_dir=resolved_project_dir,
        visual_entities=visual_entities,
    )
    visual_entity_context = _visual_entity_context(visual_entities_path)
    hybrid_settings, hybrid_snapshot = _resolve_hybrid_embedder_settings(
        profile=hybrid_embedder_profile,
        config=hybrid_embedder_config,
        embedder_name=hybrid_embedder_name,
        dimensions=hybrid_embedder_dimensions,
        live_smoke=hybrid_embedder_live_smoke,
    )
    settings = merge_hybrid_embedder_settings(
        lecture_segment_settings(settings_profile),
        hybrid_settings,
    )
    vector_specs = _user_provided_vector_specs(settings)
    vector_manifest_index = _load_index_vector_manifest(
        project_dir=resolved_project_dir,
        path=vector_manifest,
    )
    _validate_vector_manifest_for_specs(vector_specs, vector_manifest_index)
    vector_summary = _new_document_vector_summary(
        vector_specs,
        vector_manifest=vector_manifest_index,
    )
    settings_snapshot = lecture_segment_settings_snapshot(
        settings,
        profile=settings_profile,
        redact_secrets=hybrid_snapshot is not None,
    )

    hybrid_live_smoke = None
    if configure_index:
        if reset:
            client.wait_task(client.delete_index(index_uid), ignored_error_codes={"index_not_found"})
        client.wait_task(client.create_index(index_uid, primary_key="segment_id"))
        _apply_index_settings(
            client,
            index_uid=index_uid,
            settings=settings,
            settings_profile=settings_profile,
            hybrid_snapshot=hybrid_snapshot,
        )
        hybrid_live_smoke = _run_hybrid_embedder_live_smoke(
            client,
            index_uid=index_uid,
            hybrid_snapshot=hybrid_snapshot,
            requested=hybrid_embedder_live_smoke,
        )

    indexed_documents = 0
    indexed_batches = 0
    semantic_source_field_counts: dict[str, int] = {}
    embedded_visual_entity_count = 0
    documents = (
        _segment_index_document(document, visual_entity_context=visual_entity_context)
        for document in iter_jsonl_documents(segments_path)
    )
    indexed_documents_iter = _documents_with_user_provided_vectors(
        documents,
        specs=vector_specs,
        summary=vector_summary,
        vector_manifest=vector_manifest_index,
        id_fields=("segment_id", "sample_id"),
    )
    for batch in iter_batches(indexed_documents_iter, batch_size=batch_size):
        client.wait_task(client.add_documents(index_uid, batch))
        indexed_documents += len(batch)
        indexed_batches += 1
        for document in batch:
            embedded_visual_entity_count += len(_list_of_dicts(document.get("visual_entities")))
            for source_field in document.get("semantic_source_fields", []):
                semantic_source_field_counts[str(source_field)] = (
                    semantic_source_field_counts.get(str(source_field), 0) + 1
                )
    _finalize_document_vector_summary(vector_summary)

    summary = {
        "index": index_uid,
        "project_dir": str(resolved_project_dir),
        "segments_path": str(segments_path),
        "visual_entities_path": str(visual_entities_path) if visual_entities_path else None,
        "batch_size": batch_size,
        "reset": reset,
        "configure_index": configure_index,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
        "embedded_visual_entities": embedded_visual_entity_count,
        "semantic_source_field_counts": semantic_source_field_counts,
        "settings_profile": settings_snapshot["profile"],
        "settings_hash": settings_snapshot["hash"],
        "settings_snapshot": settings_snapshot,
        "document_vectors": vector_summary,
    }
    _attach_embedding_backend_report(summary, vector_summary)
    _attach_hybrid_embedder_summary(
        summary,
        hybrid_snapshot=hybrid_snapshot,
        live_smoke=hybrid_live_smoke,
    )
    return summary


def index_project_visual_entities(
    client: MeiliClient,
    *,
    index_uid: str,
    project_dir: Path,
    batch_size: int = 500,
    reset: bool = False,
    configure_index: bool = True,
    visual_entities: Path | None = None,
    settings_profile: str = VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    hybrid_embedder_profile: str | None = None,
    hybrid_embedder_config: dict[str, Any] | None = None,
    hybrid_embedder_name: str = DEFAULT_HYBRID_EMBEDDER_NAME,
    hybrid_embedder_dimensions: int | None = None,
    hybrid_embedder_live_smoke: bool = False,
    vector_manifest: Path | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    visual_entities_path = visual_entity_artifact_path(
        resolved_project_dir,
        visual_entities=visual_entities,
    )
    hybrid_settings, hybrid_snapshot = _resolve_hybrid_embedder_settings(
        profile=hybrid_embedder_profile,
        config=hybrid_embedder_config,
        embedder_name=hybrid_embedder_name,
        dimensions=hybrid_embedder_dimensions,
        live_smoke=hybrid_embedder_live_smoke,
    )
    settings = merge_hybrid_embedder_settings(
        visual_entity_settings(settings_profile),
        hybrid_settings,
    )
    vector_specs = _user_provided_vector_specs(settings)
    vector_manifest_index = _load_index_vector_manifest(
        project_dir=resolved_project_dir,
        path=vector_manifest,
    )
    _validate_vector_manifest_for_specs(vector_specs, vector_manifest_index)
    vector_summary = _new_document_vector_summary(
        vector_specs,
        vector_manifest=vector_manifest_index,
    )
    settings_snapshot = visual_entity_settings_snapshot(
        settings,
        profile=settings_profile,
        redact_secrets=hybrid_snapshot is not None,
    )

    hybrid_live_smoke = None
    if configure_index:
        if reset:
            client.wait_task(client.delete_index(index_uid), ignored_error_codes={"index_not_found"})
        client.wait_task(client.create_index(index_uid, primary_key="entity_id"))
        _apply_index_settings(
            client,
            index_uid=index_uid,
            settings=settings,
            settings_profile=settings_profile,
            hybrid_snapshot=hybrid_snapshot,
        )
        hybrid_live_smoke = _run_hybrid_embedder_live_smoke(
            client,
            index_uid=index_uid,
            hybrid_snapshot=hybrid_snapshot,
            requested=hybrid_embedder_live_smoke,
        )

    indexed_documents = 0
    indexed_batches = 0
    semantic_source_field_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    documents = (
        _visual_entity_index_document(document)
        for document in iter_jsonl_documents(visual_entities_path)
    )
    indexed_documents_iter = _documents_with_user_provided_vectors(
        documents,
        specs=vector_specs,
        summary=vector_summary,
        vector_manifest=vector_manifest_index,
        id_fields=("entity_id", "local_entity_id", "frame_id"),
    )
    for batch in iter_batches(indexed_documents_iter, batch_size=batch_size):
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
    _finalize_document_vector_summary(vector_summary)

    summary = {
        "index": index_uid,
        "project_dir": str(resolved_project_dir),
        "visual_entities_path": str(visual_entities_path),
        "batch_size": batch_size,
        "reset": reset,
        "configure_index": configure_index,
        "indexed_documents": indexed_documents,
        "indexed_batches": indexed_batches,
        "semantic_source_field_counts": semantic_source_field_counts,
        "source_counts": source_counts,
        "settings_profile": settings_snapshot["profile"],
        "settings_hash": settings_snapshot["hash"],
        "settings_snapshot": settings_snapshot,
        "document_vectors": vector_summary,
    }
    _attach_embedding_backend_report(summary, vector_summary)
    _attach_hybrid_embedder_summary(
        summary,
        hybrid_snapshot=hybrid_snapshot,
        live_smoke=hybrid_live_smoke,
    )
    return summary


def _resolve_hybrid_embedder_settings(
    *,
    profile: str | None,
    config: dict[str, Any] | None,
    embedder_name: str,
    dimensions: int | None,
    live_smoke: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    requested_name = str(embedder_name or "").strip()
    if not requested_name:
        raise ValueError("hybrid embedder name must not be empty")
    if profile is not None and config is not None:
        raise ValueError("Provide only one of hybrid_embedder_profile or hybrid_embedder_config")
    if profile is None and config is None:
        if dimensions is not None or live_smoke or requested_name != DEFAULT_HYBRID_EMBEDDER_NAME:
            raise ValueError(
                "Hybrid embedder options require hybrid_embedder_profile or "
                "hybrid_embedder_config"
            )
        return None, None
    if config is not None:
        if dimensions is not None:
            raise ValueError("hybrid_embedder_dimensions only applies to hybrid_embedder_profile")
        if requested_name != DEFAULT_HYBRID_EMBEDDER_NAME:
            raise ValueError("hybrid_embedder_name only applies to hybrid_embedder_profile")
        settings = normalize_hybrid_embedder_settings(config)
        return settings, hybrid_embedder_settings_snapshot(
            settings,
            profile=HYBRID_EMBEDDER_CUSTOM_SETTINGS_PROFILE,
        )

    if profile is None:
        raise ValueError("hybrid_embedder_profile is required")
    settings = hybrid_embedder_settings(
        profile,
        embedder_name=requested_name,
        dimensions=dimensions,
    )
    return settings, hybrid_embedder_settings_snapshot(settings, profile=profile)


def _apply_index_settings(
    client: MeiliClient,
    *,
    index_uid: str,
    settings: dict[str, Any],
    settings_profile: str,
    hybrid_snapshot: dict[str, Any] | None,
) -> None:
    context = f"settings profile '{settings_profile}'"
    if hybrid_snapshot is not None:
        context += f" with hybrid embedder profile '{hybrid_snapshot['profile']}'"
    try:
        client.wait_task(client.update_settings(index_uid, settings))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply Meilisearch {context} to index '{index_uid}': {exc}"
        ) from exc


def _run_hybrid_embedder_live_smoke(
    client: MeiliClient,
    *,
    index_uid: str,
    hybrid_snapshot: dict[str, Any] | None,
    requested: bool,
) -> dict[str, Any] | None:
    if hybrid_snapshot is None:
        return None
    if not requested:
        return {"enabled": False}
    if not hasattr(client, "get_settings"):
        raise RuntimeError(
            "Hybrid embedder live smoke requires a Meilisearch client with get_settings"
        )

    try:
        settings = client.get_settings(index_uid)
    except Exception as exc:
        raise RuntimeError(
            f"Hybrid embedder live smoke failed while reading settings for index "
            f"'{index_uid}': {exc}"
        ) from exc

    embedders = settings.get("embedders") if isinstance(settings, dict) else None
    if not isinstance(embedders, dict):
        raise RuntimeError(
            f"Hybrid embedder live smoke failed for index '{index_uid}': "
            "Meilisearch settings response did not include an embedders object"
        )

    expected_embedders = hybrid_snapshot["settings"]["embedders"]
    actual_snapshot = hybrid_embedder_settings_snapshot(
        {"embedders": embedders},
        profile=hybrid_snapshot["profile"],
    )
    mismatches = _hybrid_embedder_mismatches(
        expected_embedders,
        actual_snapshot["settings"]["embedders"],
    )
    if mismatches:
        raise RuntimeError(
            f"Hybrid embedder live smoke failed for index '{index_uid}': "
            + "; ".join(mismatches)
        )

    return {
        "enabled": True,
        "ok": True,
        "checked_embedder_names": sorted(str(name) for name in expected_embedders),
        "settings_hash": actual_snapshot["hash"],
    }


def _hybrid_embedder_mismatches(
    expected_embedders: dict[str, Any],
    actual_embedders: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for embedder_name, expected_config in expected_embedders.items():
        name = str(embedder_name)
        actual_config = actual_embedders.get(name)
        if not isinstance(actual_config, dict):
            mismatches.append(f"missing embedder '{name}'")
            continue
        if not isinstance(expected_config, dict):
            mismatches.append(f"expected embedder '{name}' config is not an object")
            continue
        for key, expected_value in expected_config.items():
            if actual_config.get(key) != expected_value:
                mismatches.append(f"embedder '{name}' field '{key}' did not match")
    return mismatches


def _attach_hybrid_embedder_summary(
    summary: dict[str, Any],
    *,
    hybrid_snapshot: dict[str, Any] | None,
    live_smoke: dict[str, Any] | None,
) -> None:
    if hybrid_snapshot is None:
        return
    summary["hybrid_embedder_profile"] = hybrid_snapshot["profile"]
    summary["hybrid_embedder_hash"] = hybrid_snapshot["hash"]
    summary["hybrid_embedder_snapshot"] = hybrid_snapshot
    summary["hybrid_embedder_live_smoke"] = live_smoke or {"enabled": False}


def _user_provided_vector_specs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    embedders = settings.get("embedders") if isinstance(settings, dict) else None
    if not isinstance(embedders, dict):
        return []
    specs: list[dict[str, Any]] = []
    for embedder_name, embedder in embedders.items():
        if not isinstance(embedder, dict) or embedder.get("source") != "userProvided":
            continue
        dimensions = embedder.get("dimensions")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            continue
        name = str(embedder_name or "").strip()
        if name:
            specs.append({"name": name, "dimensions": dimensions})
    return sorted(specs, key=lambda item: item["name"])


def _load_index_vector_manifest(
    *,
    project_dir: Path,
    path: Path | None,
) -> LoadedVectorManifest | None:
    if path is None:
        return None
    resolved_path = _optional_project_path(project_dir=project_dir, path=path, default=path)
    return load_vector_manifest(resolved_path)


def _validate_vector_manifest_for_specs(
    specs: list[dict[str, Any]],
    vector_manifest: LoadedVectorManifest | None,
) -> None:
    if vector_manifest is None:
        return
    if not specs:
        raise ValueError("vector_manifest requires a userProvided hybrid embedder")
    for spec in specs:
        name = str(spec["name"])
        expected_dimensions = int(spec["dimensions"])
        manifest_dimensions = vector_manifest.dimensions_by_embedder.get(name)
        if manifest_dimensions is None:
            available = ", ".join(vector_manifest.embedder_names)
            raise ValueError(
                f"vector manifest is missing embedder '{name}'. "
                f"Available embedders: {available}"
            )
        if manifest_dimensions != expected_dimensions:
            raise ValueError(
                f"vector manifest dimension mismatch for embedder '{name}': "
                f"manifest has {manifest_dimensions}, settings expect {expected_dimensions}"
            )


def _new_document_vector_summary(
    specs: list[dict[str, Any]],
    *,
    vector_manifest: LoadedVectorManifest | None = None,
) -> dict[str, Any]:
    using_manifest = vector_manifest is not None
    manifest_summary = vector_manifest.public_summary() if using_manifest else None
    quality_claim = (
        _backend_quality_claim(manifest_summary)
        if manifest_summary is not None
        else NO_SEMANTIC_QUALITY_CLAIM
        if specs
        else None
    )
    summary = {
        "enabled": bool(specs),
        "source": "userProvided" if specs else None,
        "generator": None if using_manifest else LOCAL_HASH_VECTOR_SOURCE if specs else None,
        "purpose": (
            "real_embedding_manifest"
            if using_manifest
            else "local_reproducibility_smoke_fallback"
            if specs
            else None
        ),
        "quality_claim": quality_claim,
        "embedder_names": [spec["name"] for spec in specs],
        "dimensions_by_embedder": {
            spec["name"]: spec["dimensions"] for spec in specs
        },
        "documents_seen": 0,
        "documents_with_vectors": 0,
        "generated_vector_count": 0,
        "existing_vector_count": 0,
    }
    if manifest_summary is not None:
        summary["manifest"] = manifest_summary
        summary["manifest_vector_count"] = 0
        summary["missing_vector_count"] = 0
    return summary


def _attach_embedding_backend_report(
    summary: dict[str, Any],
    vector_summary: dict[str, Any],
) -> None:
    report = _embedding_backend_report(vector_summary)
    summary["embedding_backend"] = report
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    if warnings:
        summary.setdefault("warnings", []).extend(warnings)


def _embedding_backend_report(vector_summary: dict[str, Any]) -> dict[str, Any]:
    if not vector_summary.get("enabled"):
        return {
            "configured": False,
            "quality_claim": None,
            "backend_contract": None,
            "warnings": [],
        }
    backend_contract = _document_vector_backend_contract(vector_summary)
    quality_claim = str(vector_summary.get("quality_claim") or NO_SEMANTIC_QUALITY_CLAIM)
    warnings: list[str] = []
    if vector_summary.get("generator") == LOCAL_HASH_VECTOR_SOURCE:
        warnings.append(LOCAL_HASH_DOCUMENT_VECTOR_WARNING)
    elif quality_claim != PROVIDER_EMBEDDING_QUALITY_CLAIM:
        warnings.append(UNVERIFIED_DOCUMENT_VECTOR_WARNING)
    return {
        "configured": True,
        "vector_source": vector_summary.get("generator") or vector_summary.get("source"),
        "quality_claim": quality_claim,
        "backend_contract": backend_contract,
        "warnings": warnings,
    }


def _document_vector_backend_contract(vector_summary: dict[str, Any]) -> dict[str, Any]:
    manifest = vector_summary.get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("backend_contract"), dict):
        return dict(manifest["backend_contract"])
    return {
        "source": LOCAL_HASH_VECTOR_SOURCE,
        "provider": None,
        "model": LOCAL_HASH_VECTOR_SOURCE,
        "source_model": LOCAL_HASH_VECTOR_SOURCE,
        "quality_claim": NO_SEMANTIC_QUALITY_CLAIM,
        "embedder_names": list(vector_summary.get("embedder_names") or []),
        "dimensions_by_embedder": dict(vector_summary.get("dimensions_by_embedder") or {}),
    }


def _backend_quality_claim(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict):
        return NO_SEMANTIC_QUALITY_CLAIM
    contract = summary.get("backend_contract")
    if isinstance(contract, dict) and contract.get("quality_claim"):
        return str(contract["quality_claim"])
    provider = summary.get("provider") if isinstance(summary.get("provider"), dict) else {}
    if provider.get("quality_claim"):
        return str(provider["quality_claim"])
    if provider.get("provider") and provider.get("provider") != "deterministic_fixture":
        return PROVIDER_EMBEDDING_QUALITY_CLAIM
    return NO_SEMANTIC_QUALITY_CLAIM


def _documents_with_user_provided_vectors(
    documents: Iterable[dict[str, Any]],
    *,
    specs: list[dict[str, Any]],
    summary: dict[str, Any],
    vector_manifest: LoadedVectorManifest | None = None,
    id_fields: Iterable[str] = (),
) -> Iterator[dict[str, Any]]:
    for document in documents:
        summary["documents_seen"] += 1
        indexed = _attach_user_provided_document_vectors(
            document,
            specs=specs,
            summary=summary,
            vector_manifest=vector_manifest,
            id_fields=id_fields,
        )
        yield indexed


def _attach_user_provided_document_vectors(
    document: dict[str, Any],
    *,
    specs: list[dict[str, Any]],
    summary: dict[str, Any],
    vector_manifest: LoadedVectorManifest | None = None,
    id_fields: Iterable[str] = (),
) -> dict[str, Any]:
    if not specs:
        if vector_manifest is not None:
            raise ValueError("vector_manifest requires a userProvided hybrid embedder")
        return document
    indexed = dict(document)
    vectors = indexed.get("_vectors")
    vectors = dict(vectors) if isinstance(vectors, dict) else {}
    text = _document_vector_text(indexed)
    record_ids = _document_vector_record_ids(indexed, id_fields=id_fields)
    document_had_vector = False
    for spec in specs:
        name = spec["name"]
        dimensions = spec["dimensions"]
        if vector_manifest is not None:
            manifest_record = vector_manifest.get(embedder=name, record_ids=record_ids)
            if manifest_record is None:
                summary["missing_vector_count"] += 1
                raise ValueError(
                    f"vector manifest is missing a vector for embedder '{name}' "
                    f"and document ids {record_ids or ['(none)']}"
                )
            vectors[name] = normalize_query_vector(
                manifest_record.vector,
                dimensions=dimensions,
                field_name=(
                    f"vector manifest record '{manifest_record.record_id}' "
                    f"for embedder '{name}'"
                ),
            )
            summary["manifest_vector_count"] += 1
        elif name in vectors:
            vectors[name] = normalize_query_vector(
                vectors[name],
                dimensions=dimensions,
                field_name=f"document vector for embedder '{name}'",
            )
            summary["existing_vector_count"] += 1
        else:
            vectors[name] = deterministic_text_vector(text, dimensions=dimensions)
            summary["generated_vector_count"] += 1
        document_had_vector = True
    indexed["_vectors"] = vectors
    if document_had_vector:
        summary["documents_with_vectors"] += 1
    return indexed


def _finalize_document_vector_summary(summary: dict[str, Any]) -> None:
    if "manifest" not in summary:
        return
    expected_vector_count = int(summary.get("documents_seen") or 0) * len(
        summary.get("embedder_names") or []
    )
    attached_vector_count = int(summary.get("manifest_vector_count") or 0)
    summary["expected_vector_count"] = expected_vector_count
    if attached_vector_count != expected_vector_count:
        raise ValueError(
            "vector manifest document/vector count mismatch: "
            f"expected {expected_vector_count}, attached {attached_vector_count}"
        )


def _document_vector_record_ids(document: dict[str, Any], *, id_fields: Iterable[str]) -> list[str]:
    record_ids: list[str] = []
    seen: set[str] = set()
    for field in id_fields:
        value = document.get(field)
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        record_ids.append(text)
    return record_ids


def _document_vector_text(document: dict[str, Any]) -> str:
    text = text_from_document_fields(
        document,
        [
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
            "transcript_window_text",
            "transcript_text",
            "normalized_text",
            "text",
            "visual_entities.text",
            "visual_description",
            "visual_entities.visual_description",
            "entity_type",
            "source_model",
            "video_name",
            "video_id",
            "frame_id",
            "segment_id",
            "window_id",
        ],
    )
    if text:
        return text
    return json.dumps(
        {
            "segment_id": document.get("segment_id"),
            "window_id": document.get("window_id"),
            "sample_id": document.get("sample_id"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def ensure_window_document_semantic_contract(document: dict[str, Any]) -> dict[str, Any]:
    indexed = dict(document)
    semantic_text, source_fields = _window_semantic_text(indexed)
    existing_text = _compact_text(str(indexed.get(LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD) or ""))
    existing_source_fields = _string_list(indexed.get(LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD))
    indexed[LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD] = existing_text or semantic_text
    indexed[LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD] = existing_source_fields or source_fields
    return indexed


def _evidence_unit_index_document(document: dict[str, Any]) -> dict[str, Any]:
    evidence_unit_id = _compact_text(str(document.get("evidence_unit_id") or ""))
    if not evidence_unit_id:
        raise ValueError("evidence unit document is missing evidence_unit_id")
    indexed = dict(document)
    indexed["evidence_unit_id"] = evidence_unit_id
    indexed.update(_evidence_unit_concept_search_fields(indexed))
    indexed.update(_evidence_unit_search_enrichment_fields(indexed))
    semantic_text = _compact_text(str(indexed.get("semantic_text") or ""))
    evidence_text = _compact_text(str(indexed.get("evidence_text") or ""))
    transcript_text = _compact_text(str(indexed.get("transcript_window_text") or ""))
    visual_text = text_from_document_fields(
        indexed,
        [
            "visual_states",
            "visual_entities",
        ],
    )
    concept_text = text_from_document_fields(
        indexed,
        [
            "concept_labels",
            "concept_aliases",
            "concept_relation_text",
            "concept_search_text",
        ],
    )
    transcript_keywords = " ".join(_string_list(indexed.get("transcript_keywords")))
    link_signal_text = _compact_text(
        " ".join(
            _unique_text_values(
                [
                    indexed.get("candidate_link_signal_summary"),
                    indexed.get("verified_link_signal_summary"),
                ]
            )
        )
    )
    if not evidence_text:
        evidence_text = _compact_text(
            " ".join(
                part
                for part in (
                    transcript_text,
                    transcript_keywords,
                    visual_text,
                    concept_text,
                    link_signal_text,
                )
                if part
            )
        )
        indexed["evidence_text"] = evidence_text
    if not semantic_text:
        indexed["semantic_text"] = _compact_text(
            " ".join(
                part
                for part in (
                    evidence_text,
                    transcript_text,
                    transcript_keywords,
                    visual_text,
                    concept_text,
                    link_signal_text,
                )
                if part
            )
        )
    return indexed


def _evidence_unit_search_enrichment_fields(document: dict[str, Any]) -> dict[str, Any]:
    transcript_text = _compact_text(str(document.get("transcript_window_text") or ""))
    transcript_keywords = _string_list(document.get("transcript_keywords"))
    if not transcript_keywords:
        transcript_keywords = _transcript_keywords(transcript_text)
    visual_state_text = _compact_text(str(document.get("visual_state_text") or ""))
    if not visual_state_text:
        visual_state_text = text_from_document_fields(document, ["visual_states.state_summary", "visual_states.detected_text"])
    visual_entity_text = _compact_text(str(document.get("visual_entity_text") or ""))
    if not visual_entity_text:
        visual_entity_text = text_from_document_fields(
            document,
            [
                "visual_entities.text",
                "visual_entities.visual_description",
                "visual_entities.entity_type",
                "visual_entities.detected_text",
                "visual_entities.position",
                "visual_entities.relations",
            ],
        )
    concept_search_text = _compact_text(str(document.get("concept_search_text") or ""))
    if not concept_search_text:
        concept_search_text = text_from_document_fields(
            document,
            [
                "concept_labels",
                "concept_aliases",
                "concept_relation_text",
                "concepts.canonical_label",
                "concepts.description",
                "concepts.definition",
                "concepts.example",
                "concepts.formula",
                "concepts.source_signals",
                "concept_relations.source_signals",
                "concept_relations.evidence_source_types",
            ],
        )
    source_quality = document.get("source_quality") if isinstance(document.get("source_quality"), dict) else {}
    candidate_link_signal_summary = _compact_text(str(document.get("candidate_link_signal_summary") or ""))
    if not candidate_link_signal_summary:
        candidate_link_signal_summary = _link_signal_summary_text(
            source_quality.get("candidate_link_signal_counts"),
            prefix="candidate",
        )
    verified_link_signal_summary = _compact_text(str(document.get("verified_link_signal_summary") or ""))
    if not verified_link_signal_summary:
        verified_link_signal_summary = _link_signal_summary_text(
            source_quality.get("verified_link_source_counts"),
            prefix="verified",
        )
    return {
        "transcript_keywords": transcript_keywords,
        "visual_state_text": visual_state_text,
        "visual_entity_text": visual_entity_text,
        "concept_search_text": concept_search_text,
        "candidate_link_signal_summary": candidate_link_signal_summary,
        "verified_link_signal_summary": verified_link_signal_summary,
    }


def _evidence_unit_concept_search_fields(document: dict[str, Any]) -> dict[str, Any]:
    concepts = _list_of_dicts(document.get("concepts"))
    relations = _list_of_dicts(document.get("concept_relations"))
    labels = _string_list(document.get("concept_labels"))
    aliases = _string_list(document.get("concept_aliases"))
    relation_text = _compact_text(str(document.get("concept_relation_text") or ""))

    if not labels:
        labels = _unique_text_values(
            [
                *[concept.get("label") for concept in concepts],
                *[relation.get("source_label") for relation in relations],
                *[relation.get("target_label") for relation in relations],
            ]
        )
    if not aliases:
        alias_values: list[str] = []
        for concept in concepts:
            alias_values.extend(_string_list(concept.get("aliases")))
        aliases = _unique_text_values(alias_values)
    if not relation_text:
        relation_text = _compact_text(
            " ".join(
                _unique_text_values(
                    [
                        *[relation.get("relation_text") for relation in relations],
                        *[relation.get("description") for relation in relations],
                    ]
                )
            )
        )
    return {
        "concept_labels": labels,
        "concept_aliases": aliases,
        "concept_relation_text": relation_text,
    }


def _evidence_unit_has_concept_fields(document: dict[str, Any]) -> bool:
    return bool(
        _string_list(document.get("concept_ids"))
        or _string_list(document.get("concept_labels"))
        or _string_list(document.get("concept_aliases"))
        or _list_of_dicts(document.get("concepts"))
    )


def _evidence_unit_has_concept_relation_fields(document: dict[str, Any]) -> bool:
    return bool(
        _compact_text(str(document.get("concept_relation_text") or ""))
        or _list_of_dicts(document.get("concept_relations"))
    )


def _window_index_document(
    *,
    target: dict[str, Any],
    window_segments: list[dict[str, Any]],
    evidence_window: dict[str, Any],
    visual_entity_context: dict[str, Any],
    window_config: dict[str, Any],
) -> dict[str, Any]:
    source_segment_ids = [
        str(segment.get("segment_id"))
        for segment in window_segments
        if segment.get("segment_id") not in (None, "")
    ]
    target_segment_id = str(target.get("segment_id") or "")
    transcript_window_text = _transcript_window_text(evidence_window)
    window_visual_entities = _window_visual_entities(
        window_segments=window_segments,
        visual_entity_context=visual_entity_context,
    )
    start_time = _optional_float(evidence_window.get("start_time"))
    end_time = _optional_float(evidence_window.get("end_time"))
    document: dict[str, Any] = {
        "window_id": _window_id(
            target_segment_id=target_segment_id,
            source_segment_ids=source_segment_ids,
            window_config=window_config,
        ),
        "target_segment_id": target_segment_id,
        "segment_id": target_segment_id,
        "project_id": target.get("project_id"),
        "dataset_name": target.get("dataset_name"),
        "subset_name": target.get("subset_name"),
        "split_name": target.get("split_name"),
        "sample_id": target.get("sample_id"),
        "sample_index": target.get("sample_index"),
        "video_id": target.get("video_id"),
        "video_name": target.get("video_name"),
        "source": target.get("source"),
        "start_time": start_time,
        "end_time": end_time,
        "timestamp_center": _window_timestamp_center(start_time=start_time, end_time=end_time, target=target),
        "target_start_time": _optional_float(target.get("start_time")),
        "target_end_time": _optional_float(target.get("end_time")),
        "target_timestamp_center": _optional_float(target.get("timestamp_center")),
        "source_segment_ids": source_segment_ids,
        "transcript_window_text": transcript_window_text,
        "transcript_text": transcript_window_text,
        "frame_refs": evidence_window.get("frame_refs", []),
        "visual_entities": window_visual_entities,
        "visual_entity_count": len(window_visual_entities),
        "evidence_window": evidence_window,
        "window_config": dict(window_config),
    }
    document = _drop_empty_window_fields(document)
    return ensure_window_document_semantic_contract(document)


def _visual_entity_index_document(document: dict[str, Any]) -> dict[str, Any]:
    entity = VisualEntity.from_dict(document)
    if not entity.entity_id:
        raise ValueError("visual entity document is missing entity_id")
    if not entity.frame_id:
        raise ValueError(f"visual entity document is missing frame_id: {entity.entity_id}")

    indexed = entity.to_dict()
    local_entity_id = entity.entity_id
    project_id = str(indexed.get("project_id") or "").strip()
    if project_id:
        indexed["local_entity_id"] = local_entity_id
        if not local_entity_id.startswith(f"{project_id}__"):
            indexed["entity_id"] = f"{project_id}__{local_entity_id}"
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
    if path is None:
        return {"by_segment_id": {}, "by_frame_id": {}}
    return _visual_entity_context_from_rows(iter_jsonl_documents(path))


def _visual_entity_context_from_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_segment_id: dict[str, list[dict[str, Any]]] = {}
    by_frame_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entity = _compact_visual_entity(row)
        segment_id = row.get("segment_id")
        if segment_id not in (None, ""):
            by_segment_id.setdefault(str(segment_id), []).append(entity)
        frame_id = row.get("frame_id")
        if frame_id not in (None, ""):
            by_frame_id.setdefault(str(frame_id), []).append(entity)
    return {"by_segment_id": by_segment_id, "by_frame_id": by_frame_id}


def _window_visual_entities(
    *,
    window_segments: list[dict[str, Any]],
    visual_entity_context: dict[str, Any],
) -> list[dict[str, Any]]:
    by_segment_id = visual_entity_context["by_segment_id"]
    by_frame_id = visual_entity_context["by_frame_id"]
    linked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment in window_segments:
        segment_id = segment.get("segment_id")
        if segment_id not in (None, ""):
            _append_unique_entities(linked, seen, by_segment_id.get(str(segment_id), []))
        for frame_ref in _segment_frame_ids(segment):
            _append_unique_entities(linked, seen, by_frame_id.get(frame_ref, []))
        inline_entities = segment.get("visual_entities")
        if isinstance(inline_entities, list):
            _append_unique_entities(
                linked,
                seen,
                [
                    _compact_visual_entity(entity)
                    for entity in inline_entities
                    if isinstance(entity, dict)
                ],
            )
    return sorted(linked, key=_entity_sort_key)


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


def _entity_sort_key(entity: dict[str, Any]) -> tuple[bool, float, str, str]:
    timestamp = _optional_float(entity.get("timestamp"))
    return (
        timestamp is None,
        timestamp or 0.0,
        str(entity.get("frame_id", "")),
        str(entity.get("entity_id", "")),
    )


def _transcript_window_text(evidence_window: dict[str, Any]) -> str:
    pieces: list[str] = []
    for segment in evidence_window.get("transcript_segments") or []:
        if not isinstance(segment, dict):
            continue
        text = _compact_text(str(segment.get("transcript_text") or ""))
        if text:
            pieces.append(text)
    return _compact_text(" ".join(pieces))


def _window_timestamp_center(
    *,
    start_time: float | None,
    end_time: float | None,
    target: dict[str, Any],
) -> float | None:
    if start_time is not None and end_time is not None:
        return round((start_time + end_time) / 2.0, 4)
    target_center = _optional_float(target.get("timestamp_center"))
    if target_center is not None:
        return target_center
    if start_time is not None:
        return start_time
    return end_time


def _window_id(
    *,
    target_segment_id: str,
    source_segment_ids: list[str],
    window_config: dict[str, Any],
) -> str:
    fingerprint = json.dumps(
        {
            "target_segment_id": target_segment_id,
            "source_segment_ids": source_segment_ids,
            "window_config": window_config,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"window_{slugify(target_segment_id)}_{digest}"


def _drop_empty_window_fields(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if value not in (None, "")
    }


def _window_semantic_text(document: dict[str, Any]) -> tuple[str, list[str]]:
    pieces: list[str] = []
    source_fields: list[str] = []

    transcript = _compact_text(str(document.get("transcript_window_text") or ""))
    if transcript:
        pieces.append(transcript)
        source_fields.append("transcript_window_text")

    visual_entities = _list_of_dicts(document.get("visual_entities"))
    for field_name, source_field in (
        ("text", "visual_entities.text"),
        ("visual_description", "visual_entities.visual_description"),
    ):
        values = _unique_text_values(entity.get(field_name) for entity in visual_entities)
        if not values:
            continue
        pieces.extend(values)
        source_fields.append(source_field)

    return _compact_text(" ".join(pieces)), source_fields


def _unique_text_values(values: Iterable[Any]) -> list[str]:
    text_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _compact_text(str(value or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        text_values.append(text)
    return text_values


def _transcript_keywords(text: str, *, limit: int = 32) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.casefold())
        if token not in TRANSCRIPT_KEYWORD_STOPWORDS
    ]
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, index)
    ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
    return ranked[: max(0, int(limit))]


def _link_signal_summary_text(value: Any, *, prefix: str) -> str:
    if not isinstance(value, dict):
        return ""
    labels = []
    for key, count in sorted(value.items()):
        try:
            parsed_count = int(count or 0)
        except (TypeError, ValueError):
            parsed_count = 0
        if parsed_count > 0:
            labels.append(f"{prefix} {_humanize_snake(str(key))}")
    return _compact_text(" ".join(labels))


def _humanize_snake(value: str) -> str:
    return _compact_text(value.replace("_", " "))


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_project_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _resolve_project_output_path(*, project_dir: Path, path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_dir / candidate).resolve()


def _optional_window_artifact_path(project_dir: Path, windows: Path | None) -> Path | None:
    if windows is not None:
        return window_artifact_path(project_dir, windows=windows)
    default = project_dir / LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH
    return default if default.exists() else None


def _window_build_inputs_requested(
    *,
    segments: Path | None,
    frames_manifest: Path | None,
    visual_entities: Path | None,
    window_seconds: float | None,
    neighbor_count: int,
    previous_neighbor_count: int | None,
    next_neighbor_count: int | None,
    window_before_seconds: float | None,
    window_after_seconds: float | None,
) -> bool:
    return any(
        (
            segments is not None,
            frames_manifest is not None,
            visual_entities is not None,
            window_seconds is not None,
            neighbor_count != 1,
            previous_neighbor_count is not None,
            next_neighbor_count is not None,
            window_before_seconds is not None,
            window_after_seconds is not None,
        )
    )


def _empty_window_config(
    *,
    window_seconds: float | None,
    neighbor_count: int,
    previous_neighbor_count: int | None,
    next_neighbor_count: int | None,
    window_before_seconds: float | None,
    window_after_seconds: float | None,
) -> dict[str, Any]:
    from oarag.retrieval.evidence import resolve_window_config

    return resolve_window_config(
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )


def _update_project_window_manifest(
    *,
    manifest_path: Path,
    windows_path: Path,
    summary: dict[str, Any],
) -> None:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts["lecture_windows"] = str(windows_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["lecture_windows"] = int(summary["counts"]["windows_total"])

    payload["window_indexing"] = {
        "window_config": summary["window_config"],
        "counts": summary["counts"],
    }
    write_json(manifest_path, payload)


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
