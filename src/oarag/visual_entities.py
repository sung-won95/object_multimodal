from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import write_json, write_jsonl
from .schemas import VisualEntity, slugify


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


def extract_visual_entities(
    *,
    project_dir: Path,
    backend: str = "auto",
    frames_manifest_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    ocr_language: str | None = None,
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
    extractor = make_extractor(backend=backend, ocr_language=ocr_language)

    entities: list[VisualEntity] = []
    for frame in frames:
        extracted = extractor.extract(project_id=project_id, frame=frame)
        for entity in extracted:
            _validate_entity_frame_consistency(entity=entity, frame=frame)
        entities.extend(extracted)

    write_jsonl(resolved_output_path, [entity.to_dict() for entity in entities])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        visual_entities_path=resolved_output_path,
        backend=extractor.backend,
        frames_total=len(frames),
        entities_total=len(entities),
    )

    return {
        "project_id": project_id,
        "backend": extractor.backend,
        "paths": {
            "project_dir": str(resolved_project_dir),
            "frames_manifest": str(resolved_frames_manifest_path),
            "visual_entities": str(resolved_output_path),
            "project_manifest": str(resolved_manifest_path),
        },
        "counts": {
            "frames_total": len(frames),
            "visual_entities": len(entities),
        },
    }


def make_extractor(*, backend: str, ocr_language: str | None = None) -> VisualEntityExtractor:
    normalized = backend.strip().lower()
    if normalized == "stub":
        return StubVisualEntityExtractor()
    if normalized == "local-ocr":
        if shutil.which("tesseract") is None:
            raise RuntimeError("Backend local-ocr requested, but `tesseract` command is not available")
        return TesseractVisualEntityExtractor(language=ocr_language)
    if normalized == "auto":
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


def _update_project_manifest(
    *,
    manifest_path: Path,
    visual_entities_path: Path,
    backend: str,
    frames_total: int,
    entities_total: int,
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
        "backend": backend,
        "frames_total": frames_total,
        "visual_entities": entities_total,
    }
    write_json(manifest_path, payload)


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
