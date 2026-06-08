from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from oarag.core.io import write_json


PUBLIC_SCHEMA_VERSION = "vlm-object-evidence-coverage-public-v1"
DEFAULT_VISUAL_ENTITIES_PATH = Path("manifests") / "visual_entities.jsonl"
DEFAULT_VLM_OBSERVATIONS_PATH = Path("manifests") / "vlm_visual_observations.jsonl"
DEFAULT_VLM_JSONL_PATH = Path("manifests") / "vlm_parser_output.jsonl"
DEFAULT_EVIDENCE_UNITS_PATH = Path("segments") / "evidence_units.jsonl"

_MOCK_OR_DETERMINISTIC_TOKENS = (
    "deterministic",
    "mock",
    "stub",
    "fixture",
    "fake",
    "dummy",
)
_DETERMINISTIC_DESCRIPTION_PREFIXES = (
    "deterministic vlm observation",
    "fixture observation",
)


def validate_vlm_object_evidence(
    *,
    project_dir: Path,
    visual_entities_path: Path | None = None,
    evidence_units_path: Path | None = None,
    vlm_observations_path: Path | None = None,
    vlm_jsonl_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Validate public-safe object-level VLM evidence coverage.

    The returned payload intentionally exposes only aggregate counts, hashed refs,
    and reason buckets. It never copies raw frame paths, transcripts, entity text,
    visual descriptions, or raw IDs into the report.
    """

    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_visual_entities_path = _project_path(
        project_dir=resolved_project_dir,
        candidate=visual_entities_path,
        default=DEFAULT_VISUAL_ENTITIES_PATH,
    )
    resolved_evidence_units_path = _project_path(
        project_dir=resolved_project_dir,
        candidate=evidence_units_path,
        default=DEFAULT_EVIDENCE_UNITS_PATH,
    )
    resolved_vlm_observations_path = _project_path(
        project_dir=resolved_project_dir,
        candidate=vlm_observations_path,
        default=DEFAULT_VLM_OBSERVATIONS_PATH,
    )
    resolved_vlm_jsonl_path = _project_path(
        project_dir=resolved_project_dir,
        candidate=vlm_jsonl_path,
        default=DEFAULT_VLM_JSONL_PATH,
    )

    visual_entities = (
        _read_jsonl(resolved_visual_entities_path)
        if resolved_visual_entities_path.exists()
        else []
    )
    evidence_units = (
        _read_jsonl(resolved_evidence_units_path)
        if resolved_evidence_units_path.exists()
        else []
    )

    visual_entity_coverage = summarize_visual_entity_coverage(visual_entities)
    evidence_unit_coverage = summarize_evidence_unit_coverage(evidence_units)
    artifact_presence = {
        "visual_entities": _artifact_presence(resolved_visual_entities_path),
        "vlm_visual_observations": _artifact_presence(resolved_vlm_observations_path),
        "vlm_parser_jsonl": _artifact_presence(resolved_vlm_jsonl_path),
        "evidence_units": _artifact_presence(resolved_evidence_units_path),
    }
    status = _validation_status(
        visual_entities_exists=resolved_visual_entities_path.exists(),
        vlm_artifact_exists=(
            resolved_vlm_observations_path.exists() or resolved_vlm_jsonl_path.exists()
        ),
        visual_entity_coverage=visual_entity_coverage,
    )
    payload = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "status": status["status"],
        "blocker": status["blocker"],
        "skip_reason": status["skip_reason"],
        "project_ref": _path_ref(resolved_project_dir),
        "privacy": _privacy_policy(),
        "artifacts": artifact_presence,
        "visual_entity_coverage": visual_entity_coverage,
        "evidence_unit_coverage": evidence_unit_coverage,
        "paper_quality_note": (
            "Only non-OCR, non-empty VLM-source entities that are not deterministic/mock "
            "and contain object-level visual evidence are counted as paper-quality VLM evidence."
        ),
    }

    if output_path is not None:
        write_json(_resolve_output_path(output_path), payload)
    if summary_path is not None:
        resolved_summary_path = _resolve_output_path(summary_path)
        resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_summary_path.write_text(_summary_markdown(payload), encoding="utf-8")
    return payload


def summarize_visual_entity_coverage(entities: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in entities if isinstance(row, dict)]
    classifications = [classify_visual_entity(row) for row in rows]
    vlm_source = [item for item in classifications if item["is_vlm_source"]]
    paper_quality = [item for item in classifications if item["is_paper_quality_vlm_entity"]]
    reason_counts: Counter[str] = Counter()
    for item in classifications:
        for reason in item["reasons"]:
            reason_counts[reason] += 1

    return {
        "visual_entities_total": len(rows),
        "vlm_source_entity_count": len(vlm_source),
        "paper_quality_vlm_entity_count": len(paper_quality),
        "rejected_vlm_entity_count": len(vlm_source) - len(paper_quality),
        "ocr_only_entity_count": sum(
            1
            for row, item in zip(rows, classifications, strict=True)
            if _is_ocr_entity(row) and not item["is_vlm_source"]
        ),
        "empty_invalid_entity_count": sum(
            1 for item in classifications if "empty_or_invalid_entity" in item["reasons"]
        ),
        "deterministic_or_mock_vlm_entity_count": sum(
            1 for item in classifications if "deterministic_or_mock_vlm_output" in item["reasons"]
        ),
        "source_counts": _counter_dict(classifications, "source_category"),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "vlm_source_field_coverage": _field_coverage(
            [row for row, item in zip(rows, classifications, strict=True) if item["is_vlm_source"]],
            denominator=len(vlm_source),
        ),
        "paper_quality_field_coverage": _field_coverage(
            [
                row
                for row, item in zip(rows, classifications, strict=True)
                if item["is_paper_quality_vlm_entity"]
            ],
            denominator=len(paper_quality),
        ),
        "paper_quality_entity_refs": [
            item["entity_ref"] for item in paper_quality[:5] if item["entity_ref"]
        ],
        "rejected_vlm_entity_refs": [
            {
                "ref": item["entity_ref"],
                "reasons": item["reasons"],
            }
            for item in vlm_source[:5]
            if not item["is_paper_quality_vlm_entity"] and item["entity_ref"]
        ],
    }


def summarize_evidence_unit_coverage(units: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in units if isinstance(row, dict)]
    with_context = [row for row in rows if isinstance(row.get("visual_entities"), list)]
    use_context = bool(with_context)

    units_with_vlm_entity = 0
    units_with_visual_description = 0
    units_with_detected_text = 0
    units_with_paper_quality_vlm_entity = 0
    units_with_rejected_vlm_entity = 0
    for row in rows:
        visual_entities = _list_of_dicts(row.get("visual_entities"))
        source_quality = _mapping(row.get("source_quality"))
        if visual_entities:
            classifications = [classify_visual_entity(entity) for entity in visual_entities]
            has_vlm_source = any(item["is_vlm_source"] for item in classifications)
            has_paper_quality = any(
                item["is_paper_quality_vlm_entity"] for item in classifications
            )
            has_rejected_vlm = has_vlm_source and not has_paper_quality
            has_visual_description = any(_has_visual_description(entity) for entity in visual_entities)
            has_detected_text = any(_has_detected_text(entity) for entity in visual_entities) or any(
                _has_detected_text(state) for state in _list_of_dicts(row.get("visual_states"))
            )
        else:
            has_vlm_source = bool(source_quality.get("has_vlm_entity"))
            has_paper_quality = False
            has_rejected_vlm = False
            has_visual_description = bool(source_quality.get("has_visual_description"))
            has_detected_text = bool(source_quality.get("has_detected_text"))

        units_with_vlm_entity += int(has_paper_quality if visual_entities else has_vlm_source)
        units_with_paper_quality_vlm_entity += int(has_paper_quality)
        units_with_rejected_vlm_entity += int(has_rejected_vlm)
        units_with_visual_description += int(has_visual_description)
        units_with_detected_text += int(has_detected_text)

    return {
        "status": "available" if rows else "not_available",
        "coverage_source": "visual_entity_context" if use_context else "source_quality_fallback",
        "evidence_units_total": len(rows),
        "units_with_visual_entity_context": len(with_context),
        "units_with_vlm_entity": units_with_vlm_entity,
        "units_with_paper_quality_vlm_entity": units_with_paper_quality_vlm_entity,
        "units_with_rejected_vlm_entity": units_with_rejected_vlm_entity,
        "units_with_visual_description": units_with_visual_description,
        "units_with_detected_text": units_with_detected_text,
        "ratios": {
            "units_with_vlm_entity": _ratio(units_with_vlm_entity, len(rows)),
            "units_with_visual_description": _ratio(units_with_visual_description, len(rows)),
            "units_with_detected_text": _ratio(units_with_detected_text, len(rows)),
        },
        "public_note": (
            "When embedded visual entity context is available, units_with_vlm_entity uses "
            "the strict paper-quality VLM classifier. Otherwise it reports legacy "
            "source_quality.has_vlm_entity as a fallback only."
        ),
    }


def classify_visual_entity(entity: dict[str, Any]) -> dict[str, Any]:
    is_ocr = _is_ocr_entity(entity)
    is_vlm_source = _is_vlm_source_entity(entity)
    has_object_evidence = _has_object_level_evidence(entity)
    has_detected_text = _has_detected_text(entity)
    empty_or_invalid = not _has_any_entity_content(entity)
    deterministic_or_mock = _is_deterministic_or_mock_entity(entity)

    reasons: list[str] = []
    if empty_or_invalid:
        reasons.append("empty_or_invalid_entity")
    if is_ocr and not is_vlm_source:
        reasons.append("ocr_only_entity")
    if is_vlm_source and deterministic_or_mock:
        reasons.append("deterministic_or_mock_vlm_output")
    if is_vlm_source and has_detected_text and not has_object_evidence:
        reasons.append("detected_text_only_not_object_evidence")
    if is_vlm_source and not has_object_evidence and not empty_or_invalid:
        reasons.append("missing_object_level_visual_evidence")

    is_paper_quality = bool(
        is_vlm_source
        and not is_ocr
        and not deterministic_or_mock
        and not empty_or_invalid
        and has_object_evidence
    )
    if is_vlm_source and not is_paper_quality and not reasons:
        reasons.append("not_paper_quality_vlm_evidence")

    return {
        "entity_ref": _id_ref(_text(entity.get("entity_id")), prefix="ent"),
        "source_category": _source_category(entity, is_vlm_source=is_vlm_source, is_ocr=is_ocr),
        "is_vlm_source": is_vlm_source,
        "is_paper_quality_vlm_entity": is_paper_quality,
        "reasons": reasons,
        "field_flags": {
            "has_visual_description": _has_visual_description(entity),
            "has_position": _has_position(entity),
            "has_relations": _has_relations(entity),
            "has_detected_text": has_detected_text,
        },
    }


def is_vlm_source_entity(entity: dict[str, Any]) -> bool:
    return classify_visual_entity(entity)["is_vlm_source"]


def is_paper_quality_vlm_entity(entity: dict[str, Any]) -> bool:
    return classify_visual_entity(entity)["is_paper_quality_vlm_entity"]


def _field_coverage(rows: list[dict[str, Any]], *, denominator: int) -> dict[str, Any]:
    counts = {
        "visual_description": sum(1 for row in rows if _has_visual_description(row)),
        "position": sum(1 for row in rows if _has_position(row)),
        "relations": sum(1 for row in rows if _has_relations(row)),
        "detected_text": sum(1 for row in rows if _has_detected_text(row)),
    }
    return {
        key: {
            "count": value,
            "ratio": _ratio(value, denominator),
        }
        for key, value in counts.items()
    }


def _validation_status(
    *,
    visual_entities_exists: bool,
    vlm_artifact_exists: bool,
    visual_entity_coverage: dict[str, Any],
) -> dict[str, Any]:
    if not visual_entities_exists:
        return {
            "status": "skipped",
            "blocker": True,
            "skip_reason": "missing_visual_entities_artifact",
        }
    if not vlm_artifact_exists and visual_entity_coverage["vlm_source_entity_count"] == 0:
        return {
            "status": "skipped",
            "blocker": True,
            "skip_reason": "missing_real_vlm_artifacts_from_issue_200",
        }
    if visual_entity_coverage["paper_quality_vlm_entity_count"] == 0:
        return {
            "status": "blocked",
            "blocker": True,
            "skip_reason": "no_paper_quality_vlm_object_evidence",
        }
    return {"status": "passed", "blocker": False, "skip_reason": None}


def _summary_markdown(payload: dict[str, Any]) -> str:
    entity = payload["visual_entity_coverage"]
    unit = payload["evidence_unit_coverage"]
    lines = [
        "# VLM Object Evidence Coverage",
        "",
        "This public-safe report redacts raw entity text, transcripts, local paths, and raw IDs.",
        "",
        f"- status: `{payload['status']}`",
        f"- blocker: `{payload['blocker']}`",
        f"- skip reason: `{payload.get('skip_reason')}`",
        f"- VLM source entities: `{entity['vlm_source_entity_count']}`",
        f"- paper-quality VLM entities: `{entity['paper_quality_vlm_entity_count']}`",
        f"- OCR-only entities: `{entity['ocr_only_entity_count']}`",
        f"- empty/invalid entities: `{entity['empty_invalid_entity_count']}`",
        f"- deterministic/mock VLM entities: `{entity['deterministic_or_mock_vlm_entity_count']}`",
        f"- evidence units with VLM entity: `{unit['units_with_vlm_entity']}`",
        f"- evidence units with visual description: `{unit['units_with_visual_description']}`",
        f"- evidence units with detected text: `{unit['units_with_detected_text']}`",
        "",
        "## VLM Source Field Coverage",
        "",
    ]
    for key, coverage in entity["vlm_source_field_coverage"].items():
        lines.append(f"- {key}: `{coverage['count']}` / `{entity['vlm_source_entity_count']}`")
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {path}:{line_number}")
            rows.append(payload)
    return rows


def _artifact_presence(path: Path) -> dict[str, Any]:
    return {
        "exists": path.exists(),
        "ref": _path_ref(path),
    }


def _privacy_policy() -> dict[str, Any]:
    return {
        "safe_to_publish": True,
        "raw_text_redacted": True,
        "raw_paths_redacted": True,
        "raw_ids_redacted": True,
    }


def _project_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    path = candidate if candidate is not None else default
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _resolve_output_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_vlm_source_entity(entity: dict[str, Any]) -> bool:
    if _is_ocr_entity(entity) and not _text(entity.get("source_model")):
        return False
    source = _text(entity.get("source")).casefold()
    source_model = _text(entity.get("source_model"))
    parser_version = _text(entity.get("parser_version")).casefold()
    backend = _text(entity.get("backend")).casefold()
    return bool(
        "vlm" in source
        or source_model
        or "vlm" in parser_version
        or backend in {"command", "jsonl"}
    )


def _is_ocr_entity(entity: dict[str, Any]) -> bool:
    source = _text(entity.get("source")).casefold()
    entity_type = _text(entity.get("entity_type")).casefold()
    return source.startswith("ocr") or source.startswith("local-ocr") or entity_type == "ocr_text"


def _is_deterministic_or_mock_entity(entity: dict[str, Any]) -> bool:
    metadata = _mapping(entity.get("metadata"))
    values = [
        entity.get("source"),
        entity.get("source_model"),
        entity.get("parser_version"),
        entity.get("backend"),
        metadata.get("backend"),
        metadata.get("parser_version"),
    ]
    haystack = " ".join(_text(value).casefold() for value in values if _text(value))
    if any(token in haystack for token in _MOCK_OR_DETERMINISTIC_TOKENS):
        return True
    description = _first_text_value(
        [entity.get("visual_description"), entity.get("text")]
    ).casefold()
    return any(description.startswith(prefix) for prefix in _DETERMINISTIC_DESCRIPTION_PREFIXES)


def _source_category(entity: dict[str, Any], *, is_vlm_source: bool, is_ocr: bool) -> str:
    if is_vlm_source:
        return "vlm_source"
    if is_ocr:
        return "ocr_only"
    if not _has_any_entity_content(entity):
        return "empty_or_invalid"
    return "other"


def _has_any_entity_content(entity: dict[str, Any]) -> bool:
    return bool(
        _has_visual_description(entity)
        or _has_detected_text(entity)
        or _first_text_value([entity.get("text")])
        or _has_position(entity)
        or _has_relations(entity)
        or _mapping(entity.get("bbox"))
    )


def _has_object_level_evidence(entity: dict[str, Any]) -> bool:
    return bool(
        _has_visual_description(entity)
        or _has_position(entity)
        or _has_relations(entity)
        or _mapping(entity.get("bbox"))
    )


def _has_visual_description(entity: dict[str, Any]) -> bool:
    return bool(_first_text_value([entity.get("visual_description"), entity.get("description")]))


def _has_detected_text(entity: dict[str, Any]) -> bool:
    return bool(_text_values([entity.get("detected_text")]))


def _has_position(entity: dict[str, Any]) -> bool:
    return bool(_mapping(entity.get("position")))


def _has_relations(entity: dict[str, Any]) -> bool:
    return bool(_list_of_dicts(entity.get("relations")))


def _counter_dict(items: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(item.get(key) or "unknown") for item in items)
    return dict(sorted(counts.items()))


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text_values(values: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            texts.extend(_text_values(value.values()))
        elif isinstance(value, list):
            texts.extend(_text_values(value))
        else:
            text = _text(value)
            if text:
                texts.append(text)
    return texts


def _first_text_value(values: Iterable[Any]) -> str:
    values = _text_values(values)
    return values[0] if values else ""


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return " ".join(str(value).split())


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _id_ref(value: str, *, prefix: str) -> str | None:
    if not value:
        return None
    return f"{prefix}:{_short_hash(value)}"


def _path_ref(path: Path) -> str:
    return f"path:{_short_hash(str(path.expanduser().resolve()))}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
