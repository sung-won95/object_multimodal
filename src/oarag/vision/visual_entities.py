from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from oarag.core.io import write_json, write_jsonl
from oarag.core.schemas import VLMVisualObservation, VisualEntity, slugify


OCR_MIN_CONFIDENCE = 0.40
OCR_SHORT_TEXT_MIN_CONFIDENCE = 0.85
OCR_SHORT_TEXT_MAX_ALNUM_CHARS = 1
DEFAULT_VISUAL_ENTITY_BACKEND = "vlm-first"
VLM_FIRST_BACKEND = "vlm-first"
AUTO_VISUAL_ENTITY_BACKEND = "auto"
VLM_OBSERVATIONS_BACKEND = "vlm-observations"
VLM_JSONL_BACKEND = "vlm-jsonl"
LOCAL_OCR_BACKEND = "local-ocr"
VLM_OBSERVATION_SUCCESS_STATUS = "success"


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    frame_path: str
    timestamp: float | None


class VisualEntityExtractor(Protocol):
    backend: str

    def extract(self, *, project_id: str, frame: FrameRecord) -> list[VisualEntity]:
        ...


@dataclass(frozen=True)
class StubVisualEntityExtractor:
    backend: str = "stub"

    def extract(self, *, project_id: str, frame: FrameRecord) -> list[VisualEntity]:
        return []


@dataclass(frozen=True)
class TesseractVisualEntityExtractor:
    language: str | None = None
    backend: str = "local-ocr"

    def extract(self, *, project_id: str, frame: FrameRecord) -> list[VisualEntity]:
        if shutil.which("tesseract") is None:
            raise RuntimeError("local-ocr backend requires the `tesseract` command in PATH")

        command = ["tesseract", frame.frame_path, "stdout"]
        if self.language:
            command.extend(["-l", self.language])
        command.append("tsv")

        result = _run(command)
        rows = _parse_tesseract_tsv(result.stdout)
        entities: list[VisualEntity] = []
        for index, row in enumerate(rows, start=1):
            text = row.get("text", "").strip()
            if not text:
                continue
            conf = _optional_float(row.get("conf"))
            if conf is not None and conf < 0:
                continue
            bbox = {
                "left": _optional_float(row.get("left")) or 0.0,
                "top": _optional_float(row.get("top")) or 0.0,
                "width": _optional_float(row.get("width")) or 0.0,
                "height": _optional_float(row.get("height")) or 0.0,
            }
            entities.append(
                VisualEntity(
                    entity_id=f"ent_{slugify(frame.frame_id)}_{index:04d}",
                    project_id=project_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    frame_path=frame.frame_path,
                    bbox=bbox,
                    text=text,
                    entity_type="ocr_text",
                    confidence=None if conf is None else conf / 100.0,
                    source="ocr:tesseract",
                )
            )
        return entities


@dataclass(frozen=True)
class VlmJsonlVisualEntityExtractor:
    jsonl_path: Path
    backend: str = "vlm-jsonl"
    _records_by_frame_id: dict[str, list[dict[str, Any]]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_records_by_frame_id",
            _read_vlm_jsonl_records(self.jsonl_path),
        )

    def extract(self, *, project_id: str, frame: FrameRecord) -> list[VisualEntity]:
        records = self._records_by_frame_id.get(frame.frame_id, [])
        entities: list[VisualEntity] = []
        for entity_index, record in enumerate(records, start=1):
            entity_payload = record["entity"]
            parser_version = _optional_str(
                entity_payload.get("parser_version")
            ) or record.get("parser_version")
            source_model = _optional_str(entity_payload.get("source_model")) or record.get(
                "source_model"
            )
            visual_description = _optional_str(
                entity_payload.get("visual_description")
            ) or _optional_str(entity_payload.get("description"))
            detected_text = _optional_str(
                entity_payload.get("detected_text") or entity_payload.get("visible_text")
            )
            text = _optional_str(entity_payload.get("text"))
            if text is None:
                text = (
                    detected_text
                    or visual_description
                    or _optional_str(entity_payload.get("label"))
                    or ""
                )
            entity_id = _optional_str(entity_payload.get("entity_id"))
            if entity_id is None:
                entity_id = f"ent_{slugify(frame.frame_id)}_{entity_index:04d}"
            source = _optional_str(entity_payload.get("source"))
            if source is None:
                source = f"vlm:{source_model}" if source_model else "vlm-jsonl"
            entity = VisualEntity(
                entity_id=entity_id,
                project_id=project_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                frame_path=frame.frame_path,
                bbox=_coerce_bbox(entity_payload.get("bbox")),
                text=text,
                entity_type=_optional_str(entity_payload.get("entity_type")) or "visual_object",
                confidence=_optional_float(entity_payload.get("confidence")),
                source=source,
                visual_description=visual_description,
                detected_text=detected_text,
                position=_coerce_mapping(entity_payload.get("position")),
                relations=_coerce_relations(entity_payload.get("relations")),
                parser_version=parser_version,
                source_model=source_model,
            )
            entities.append(entity)
        return entities

    @property
    def frame_ids(self) -> set[str]:
        return set(self._records_by_frame_id)


@dataclass(frozen=True)
class VlmObservationsVisualEntityExtractor:
    observations_path: Path
    backend: str = VLM_OBSERVATIONS_BACKEND
    _observations_by_frame_id: dict[str, list[VLMVisualObservation]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_observations_by_frame_id",
            _read_vlm_observation_records(self.observations_path),
        )

    def extract(self, *, project_id: str, frame: FrameRecord) -> list[VisualEntity]:
        observations = self._observations_by_frame_id.get(frame.frame_id, [])
        entities: list[VisualEntity] = []
        for observation_index, observation in enumerate(observations, start=1):
            status = _optional_str(observation.status)
            if status is not None and status.casefold() != VLM_OBSERVATION_SUCCESS_STATUS:
                continue

            detected_text = _optional_str(observation.detected_text)
            visual_description = _optional_str(observation.visual_description)
            text = detected_text or visual_description or ""
            source_model = _optional_str(observation.source_model)
            backend = _optional_str(observation.backend)
            if source_model is not None:
                source = f"vlm:{source_model}"
            elif backend is not None:
                source = f"vlm:{backend}"
            else:
                source = VLM_OBSERVATIONS_BACKEND
            observation_id = _optional_str(observation.observation_id)
            entity_id = (
                f"ent_{slugify(observation_id)}"
                if observation_id is not None
                else f"ent_{slugify(frame.frame_id)}_{observation_index:04d}"
            )

            entities.append(
                VisualEntity(
                    entity_id=entity_id,
                    project_id=project_id,
                    frame_id=frame.frame_id,
                    timestamp=(
                        frame.timestamp
                        if frame.timestamp is not None
                        else observation.timestamp
                    ),
                    frame_path=frame.frame_path,
                    bbox=observation.bbox,
                    text=text,
                    entity_type=_optional_str(observation.observation_type)
                    or "visual_observation",
                    confidence=observation.confidence,
                    source=source,
                    visual_description=visual_description,
                    detected_text=detected_text,
                    position=observation.position,
                    relations=observation.relations,
                    parser_version=(
                        _optional_str(observation.metadata.get("parser_version"))
                        or observation.schema_version
                    ),
                    source_model=source_model,
                )
            )
        return entities

    @property
    def frame_ids(self) -> set[str]:
        return set(self._observations_by_frame_id)


def extract_visual_entities(
    *,
    project_dir: Path,
    backend: str = DEFAULT_VISUAL_ENTITY_BACKEND,
    frames_manifest_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    ocr_language: str | None = None,
    vlm_jsonl_path: Path | None = None,
    vlm_observations_path: Path | None = None,
) -> dict[str, Any]:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_frames_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )
    resolved_output_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=output_path,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
    )
    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )

    frames = _read_frames_manifest(resolved_frames_manifest_path)
    project_id = _project_id_from_manifest_or_dir(resolved_manifest_path, resolved_project_dir)
    normalized_backend = backend.strip().lower()
    vlm_first_requested = normalized_backend in {
        VLM_FIRST_BACKEND,
        AUTO_VISUAL_ENTITY_BACKEND,
    }
    resolved_vlm_jsonl_path = (
        _resolve_path(
            project_dir=resolved_project_dir,
            candidate=vlm_jsonl_path,
            default=resolved_project_dir / "manifests" / "vlm_parser_output.jsonl",
        )
        if (
            vlm_jsonl_path is not None
            or normalized_backend == VLM_JSONL_BACKEND
            or vlm_first_requested
        )
        else None
    )
    resolved_vlm_observations_path = (
        _resolve_path(
            project_dir=resolved_project_dir,
            candidate=vlm_observations_path,
            default=resolved_project_dir / "manifests" / "vlm_visual_observations.jsonl",
        )
        if (
            vlm_observations_path is not None
            or normalized_backend == VLM_OBSERVATIONS_BACKEND
            or vlm_first_requested
        )
        else None
    )
    extractor = make_extractor(
        backend=backend,
        ocr_language=ocr_language,
        vlm_jsonl_path=resolved_vlm_jsonl_path,
        vlm_observations_path=resolved_vlm_observations_path,
    )
    if isinstance(extractor, VlmJsonlVisualEntityExtractor):
        manifest_frame_ids = {frame.frame_id for frame in frames}
        unknown_frame_ids = sorted(extractor.frame_ids - manifest_frame_ids)
        if unknown_frame_ids:
            preview = ", ".join(unknown_frame_ids[:5])
            raise ValueError(f"VLM JSONL contains frame_id not present in frames manifest: {preview}")
    if isinstance(extractor, VlmObservationsVisualEntityExtractor):
        manifest_frame_ids = {frame.frame_id for frame in frames}
        unknown_frame_ids = sorted(extractor.frame_ids - manifest_frame_ids)
        if unknown_frame_ids:
            preview = ", ".join(unknown_frame_ids[:5])
            raise ValueError(
                "VLM observations contain frame_id not present in frames manifest: "
                f"{preview}"
            )

    raw_entities: list[VisualEntity] = []
    for frame in frames:
        extracted = extractor.extract(project_id=project_id, frame=frame)
        for entity in extracted:
            _validate_entity_frame_consistency(entity=entity, frame=frame)
        raw_entities.extend(extracted)

    entities, filter_summary = filter_visual_entities(raw_entities)

    write_jsonl(resolved_output_path, [entity.to_dict() for entity in entities])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        visual_entities_path=resolved_output_path,
        requested_backend=normalized_backend,
        backend=extractor.backend,
        frames_total=len(frames),
        raw_entities_total=len(raw_entities),
        entities_total=len(entities),
        filter_summary=filter_summary,
    )

    return {
        "project_id": project_id,
        "requested_backend": normalized_backend,
        "backend": extractor.backend,
        "backend_role": _visual_entity_backend_role(extractor.backend),
        "fallback_policy": _visual_entity_fallback_policy(normalized_backend),
        "paths": {
            "project_dir": str(resolved_project_dir),
            "frames_manifest": str(resolved_frames_manifest_path),
            "visual_entities": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
            **(
                {"vlm_jsonl": str(resolved_vlm_jsonl_path)}
                if resolved_vlm_jsonl_path is not None
                else {}
            ),
            **(
                {"vlm_visual_observations": str(resolved_vlm_observations_path)}
                if resolved_vlm_observations_path is not None
                else {}
            ),
        },
        "counts": {
            "frames_total": len(frames),
            "raw_visual_entities": len(raw_entities),
            "visual_entities": len(entities),
            "dropped_visual_entities": filter_summary["dropped_visual_entities"],
            "filter_reasons": filter_summary["filter_reasons"],
        },
    }


def filter_visual_entities(
    entities: list[VisualEntity],
) -> tuple[list[VisualEntity], dict[str, Any]]:
    filtered: list[VisualEntity] = []
    seen: set[tuple[Any, ...]] = set()
    reasons = {
        "empty_text": 0,
        "low_confidence": 0,
        "no_alnum": 0,
        "short_low_confidence": 0,
        "duplicate": 0,
    }

    for entity in entities:
        reason = _visual_entity_drop_reason(entity)
        if reason is not None:
            reasons[reason] += 1
            continue

        key = _entity_duplicate_key(entity)
        if key in seen:
            reasons["duplicate"] += 1
            continue

        seen.add(key)
        filtered.append(entity)

    return filtered, {
        "raw_visual_entities": len(entities),
        "visual_entities": len(filtered),
        "dropped_visual_entities": len(entities) - len(filtered),
        "filter_reasons": reasons,
        "policy": {
            "ocr_min_confidence": OCR_MIN_CONFIDENCE,
            "ocr_short_text_min_confidence": OCR_SHORT_TEXT_MIN_CONFIDENCE,
            "ocr_short_text_max_alnum_chars": OCR_SHORT_TEXT_MAX_ALNUM_CHARS,
            "ocr_only_confidence_filters": True,
        },
    }


def make_extractor(
    *,
    backend: str,
    ocr_language: str | None = None,
    vlm_jsonl_path: Path | None = None,
    vlm_observations_path: Path | None = None,
) -> VisualEntityExtractor:
    normalized = backend.strip().lower()
    if normalized == "stub":
        return StubVisualEntityExtractor()
    if normalized == VLM_JSONL_BACKEND:
        if vlm_jsonl_path is None:
            raise ValueError("Backend vlm-jsonl requires --vlm-jsonl")
        return VlmJsonlVisualEntityExtractor(jsonl_path=vlm_jsonl_path)
    if normalized == VLM_OBSERVATIONS_BACKEND:
        if vlm_observations_path is None:
            raise ValueError("Backend vlm-observations requires --vlm-observations")
        return VlmObservationsVisualEntityExtractor(observations_path=vlm_observations_path)
    if normalized == LOCAL_OCR_BACKEND:
        if shutil.which("tesseract") is None:
            raise RuntimeError("Backend local-ocr requested, but `tesseract` command is not available")
        return TesseractVisualEntityExtractor(language=ocr_language)
    if normalized in {VLM_FIRST_BACKEND, AUTO_VISUAL_ENTITY_BACKEND}:
        if vlm_observations_path is not None and vlm_observations_path.exists():
            return VlmObservationsVisualEntityExtractor(observations_path=vlm_observations_path)
        if vlm_jsonl_path is not None and vlm_jsonl_path.exists():
            return VlmJsonlVisualEntityExtractor(jsonl_path=vlm_jsonl_path)
        if shutil.which("tesseract") is not None:
            return TesseractVisualEntityExtractor(language=ocr_language)
        return StubVisualEntityExtractor()
    raise ValueError(f"Unsupported visual entity backend: {backend}")


def _read_frames_manifest(path: Path) -> list[FrameRecord]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"frames_manifest.jsonl not found: {resolved}")

    rows: list[FrameRecord] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            frame_path = str(payload.get("frame_path", ""))
            frame_id = str(payload.get("frame_id") or Path(frame_path).stem)
            if not frame_id:
                raise ValueError(f"Missing frame_id in {resolved}:{line_number}")
            rows.append(
                FrameRecord(
                    frame_id=frame_id,
                    frame_path=frame_path,
                    timestamp=_optional_float(payload.get("timestamp")),
                )
            )
    return rows


def _project_id_from_manifest_or_dir(manifest_path: Path, project_dir: Path) -> str:
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            project_id = loaded.get("project_id")
            if project_id is not None:
                return str(project_id)
    return project_dir.name


def _validate_entity_frame_consistency(*, entity: VisualEntity, frame: FrameRecord) -> None:
    if entity.frame_id != frame.frame_id:
        raise ValueError(
            f"Entity frame_id mismatch for {entity.entity_id}: {entity.frame_id} != {frame.frame_id}"
        )
    if entity.frame_path != frame.frame_path:
        raise ValueError(
            f"Entity frame_path mismatch for {entity.entity_id}: {entity.frame_path} != {frame.frame_path}"
        )


def _visual_entity_drop_reason(entity: VisualEntity) -> str | None:
    normalized = _normalize_visual_entity_text(entity)
    if not normalized:
        return "empty_text"
    confidence = entity.confidence
    is_ocr = _is_ocr_entity(entity)
    if is_ocr and confidence is not None and confidence < OCR_MIN_CONFIDENCE:
        return "low_confidence"
    alnum_count = sum(1 for char in normalized if char.isalnum())
    if alnum_count == 0:
        return "no_alnum"
    if (
        is_ocr
        and alnum_count <= OCR_SHORT_TEXT_MAX_ALNUM_CHARS
        and (confidence is None or confidence < OCR_SHORT_TEXT_MIN_CONFIDENCE)
    ):
        return "short_low_confidence"
    return None


def _is_ocr_entity(entity: VisualEntity) -> bool:
    source = (entity.source or "").casefold()
    entity_type = (entity.entity_type or "").casefold()
    return source.startswith("ocr:") or entity_type == "ocr_text"


def _entity_duplicate_key(entity: VisualEntity) -> tuple[Any, ...]:
    bbox = entity.bbox or {}
    return (
        entity.frame_id,
        _normalize_visual_entity_text(entity).casefold(),
        round(_optional_float(bbox.get("left")) or 0.0),
        round(_optional_float(bbox.get("top")) or 0.0),
        round(_optional_float(bbox.get("width")) or 0.0),
        round(_optional_float(bbox.get("height")) or 0.0),
    )


def _normalize_visual_entity_text(entity: VisualEntity) -> str:
    return _normalize_ocr_text(
        entity.text or entity.visual_description or entity.detected_text or ""
    )


def _normalize_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _read_vlm_jsonl_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"VLM JSONL input not found: {resolved}")

    records_by_frame_id: dict[str, list[dict[str, Any]]] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            frame_id = _optional_str(payload.get("frame_id"))
            if frame_id is None:
                raise ValueError(f"Missing frame_id in {resolved}:{line_number}")
            raw_entities = payload.get("entities")
            if raw_entities is None:
                raw_entities = payload.get("visual_entities")
            if not isinstance(raw_entities, list):
                raise ValueError(
                    f"Expected entities list for frame_id={frame_id} in {resolved}:{line_number}"
                )
            for entity in raw_entities:
                if not isinstance(entity, dict):
                    raise ValueError(
                        f"Expected entity object for frame_id={frame_id} in {resolved}:{line_number}"
                    )
                entity_frame_id = _optional_str(entity.get("frame_id"))
                if entity_frame_id is not None and entity_frame_id != frame_id:
                    raise ValueError(
                        f"Entity frame_id mismatch in {resolved}:{line_number}: "
                        f"{entity_frame_id} != {frame_id}"
                    )
                records_by_frame_id.setdefault(frame_id, []).append(
                    {
                        "parser_version": _optional_str(payload.get("parser_version")),
                        "source_model": _optional_str(payload.get("source_model")),
                        "entity": entity,
                    }
                )
    return records_by_frame_id


def _read_vlm_observation_records(path: Path) -> dict[str, list[VLMVisualObservation]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"VLM observations input not found: {resolved}")

    observations_by_frame_id: dict[str, list[VLMVisualObservation]] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            frame_id = _optional_str(payload.get("frame_id"))
            if frame_id is None:
                raise ValueError(
                    f"Missing frame_id in VLM observations input {resolved}:{line_number}"
                )
            observation = VLMVisualObservation.from_dict(payload)
            if observation.frame_id != frame_id:
                raise ValueError(
                    f"Observation frame_id mismatch in {resolved}:{line_number}: "
                    f"{observation.frame_id} != {frame_id}"
                )
            observations_by_frame_id.setdefault(frame_id, []).append(observation)
    return observations_by_frame_id


def _coerce_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    bbox: dict[str, float] = {}
    for key, raw in value.items():
        parsed = _optional_float(raw)
        if parsed is not None:
            bbox[str(key)] = parsed
    return bbox or None


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _coerce_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _update_project_manifest(
    *,
    manifest_path: Path,
    visual_entities_path: Path,
    requested_backend: str,
    backend: str,
    frames_total: int,
    raw_entities_total: int,
    entities_total: int,
    filter_summary: dict[str, Any],
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
    artifacts["visual_entities"] = str(visual_entities_path)

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts["visual_entities"] = entities_total

    payload["visual_entity_extraction"] = {
        "requested_backend": requested_backend,
        "backend": backend,
        "backend_role": _visual_entity_backend_role(backend),
        "fallback_policy": _visual_entity_fallback_policy(requested_backend),
        "frames_total": frames_total,
        "raw_visual_entities": raw_entities_total,
        "visual_entities": entities_total,
        "dropped_visual_entities": filter_summary["dropped_visual_entities"],
        "filter_reasons": filter_summary["filter_reasons"],
        "filter_policy": filter_summary["policy"],
    }
    write_json(manifest_path, payload)


def _visual_entity_backend_role(backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized in {VLM_OBSERVATIONS_BACKEND, VLM_JSONL_BACKEND}:
        return "vlm_parser"
    if normalized == LOCAL_OCR_BACKEND:
        return "ocr_baseline_fallback"
    if normalized == "stub":
        return "empty_stub"
    return "custom"


def _visual_entity_fallback_policy(requested_backend: str) -> dict[str, Any]:
    normalized = requested_backend.strip().lower()
    if normalized in {VLM_FIRST_BACKEND, AUTO_VISUAL_ENTITY_BACKEND}:
        return {
            "mode": VLM_FIRST_BACKEND,
            "order": [VLM_OBSERVATIONS_BACKEND, VLM_JSONL_BACKEND, LOCAL_OCR_BACKEND, "stub"],
            "ocr_role": "baseline_or_fallback_only",
        }
    if normalized == LOCAL_OCR_BACKEND:
        return {"mode": "ocr_baseline", "order": [LOCAL_OCR_BACKEND]}
    if normalized in {VLM_OBSERVATIONS_BACKEND, VLM_JSONL_BACKEND}:
        return {"mode": "vlm_only", "order": [normalized]}
    return {"mode": normalized, "order": [normalized]}


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _parse_tesseract_tsv(tsv_text: str) -> list[dict[str, str]]:
    if not tsv_text.strip():
        return []
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    return [dict(row) for row in reader]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{stderr}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
