from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from oarag.core.io import write_jsonl


CONCEPT_GRAPH_SCHEMA_VERSION = "oarag-concept-graph-v1"
CONCEPT_GRAPH_ARTIFACT_NAME = "concept_graph"
CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH = Path("manifests") / "concept_graph.jsonl"

CONCEPT_NODE_RECORD_TYPE = "concept_node"
RELATION_EDGE_RECORD_TYPE = "relation_edge"

CONCEPT_RELATION_STATUSES = ("candidate", "verified")
TIMESTAMP_ONLY_SOURCE_SIGNALS = frozenset(
    {
        "time_overlap",
        "timestamp_overlap",
        "timestamp_fallback",
        "temporal_overlap",
    }
)

CONCEPT_SOURCE_SIGNAL_CONTRACT = {
    "transcript_statement": {
        "status": "candidate_or_verified",
        "description": "A relation or concept was stated in transcript text.",
    },
    "transcript_mention": {
        "status": "candidate_or_verified",
        "description": "A concept candidate was mentioned in transcript text.",
    },
    "slide_text": {
        "status": "candidate_or_verified",
        "description": "A concept was observed in slide or OCR text.",
    },
    "detected_text": {
        "status": "candidate_or_verified",
        "description": "A concept candidate was observed in detected visual text.",
    },
    "slide_cooccurrence": {
        "status": "candidate",
        "description": "Concepts appeared in the same slide or visual state.",
    },
    "vlm_visual_observation": {
        "status": "candidate_or_verified",
        "description": "A VLM observation described the concept or relation.",
    },
    "visual_object_relation": {
        "status": "candidate_or_verified",
        "description": "A visual object relation supports the graph relation.",
    },
    "lexical_alias_match": {
        "status": "candidate",
        "description": "A lexical alias rule matched two labels.",
    },
    "embedding_alias_match": {
        "status": "candidate",
        "description": "Embedding similarity suggested an alias or relation.",
    },
    "human_gold": {
        "status": "verified",
        "description": "A human/gold source verified the relation.",
    },
    "vlm_verifier": {
        "status": "verified",
        "description": "A verifier model explicitly verified the relation.",
    },
    "strict_deterministic_rule": {
        "status": "verified",
        "description": "A strict deterministic rule verified the relation.",
    },
    "timestamp_overlap": {
        "status": "candidate_only",
        "description": "Timestamp overlap is a candidate signal only.",
    },
    "time_overlap": {
        "status": "candidate_only",
        "description": "Time overlap is a candidate signal only.",
    },
    "timestamp_fallback": {
        "status": "candidate_only",
        "description": "Timestamp fallback is a candidate signal only.",
    },
    "temporal_overlap": {
        "status": "candidate_only",
        "description": "Temporal overlap is a candidate signal only.",
    },
}

CONCEPT_RELATION_TYPE_TO_GRAPH_EDGE_LABEL = {
    "mentions": "MENTIONS",
    "defines": "DEFINES",
    "explains": "EXPLAINS",
    "shows": "SHOWS",
    "alias_of": "ALIAS_OF",
    "prerequisite_of": "PREREQUISITE_OF",
    "uses": "USES",
    "contrasts_with": "CONTRASTS_WITH",
    "visually_depicts": "DEPICTS",
    "related_to": "RELATED_TO",
}

CONCEPT_GRAPH_DB_NODE_CONTRACT = {
    "Lecture": {
        "labels": ("GraphNode", "Lecture"),
        "key": "lecture:{lecture_id}",
        "required_properties": ("lecture_id",),
        "optional_properties": ("course_id", "title", "source"),
    },
    "EvidenceUnit": {
        "labels": ("GraphNode", "EvidenceUnit"),
        "key": "evidence_unit:{lecture_id}:{evidence_unit_id}",
        "required_properties": ("evidence_unit_id", "lecture_id"),
        "optional_properties": ("project_id", "video_id", "start_time", "end_time"),
    },
    "Concept": {
        "labels": ("GraphNode", "Concept"),
        "key": "concept:{lecture_id}:{concept_id}",
        "required_properties": (
            "concept_id",
            "lecture_id",
            "label",
            "concept_type",
            "confidence",
        ),
        "optional_properties": (
            "aliases",
            "description",
            "source_evidence_unit_ids",
            "evidence_sources",
            "metadata",
        ),
    },
    "VisualObject": {
        "labels": ("GraphNode", "VisualObject"),
        "key": "visual_object:{lecture_id}:{visual_object_id}",
        "required_properties": ("visual_object_id", "lecture_id", "label"),
        "optional_properties": ("source_visual_entity_id", "confidence", "metadata"),
    },
    "SlideState": {
        "labels": ("GraphNode", "SlideState"),
        "key": "slide_state:{lecture_id}:{visual_state_id}",
        "required_properties": ("visual_state_id", "lecture_id"),
        "optional_properties": ("start_time", "end_time", "state_summary", "metadata"),
    },
}

CONCEPT_GRAPH_DB_EDGE_CONTRACT = {
    "HAS_EVIDENCE": {
        "from": "Lecture",
        "to": "EvidenceUnit",
        "required_properties": ("lecture_id",),
        "optional_properties": ("source",),
    },
    "MENTIONS": {
        "from": "EvidenceUnit",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "DEFINES": {
        "from": "EvidenceUnit|Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "EXPLAINS": {
        "from": "EvidenceUnit|Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "SHOWS": {
        "from": "EvidenceUnit",
        "to": "VisualObject",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "ALIAS_OF": {
        "from": "Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "PREREQUISITE_OF": {
        "from": "Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "USES": {
        "from": "Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "CONTRASTS_WITH": {
        "from": "Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "DEPICTS": {
        "from": "VisualObject",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
    "RELATED_TO": {
        "from": "Concept",
        "to": "Concept",
        "required_properties": ("edge_id", "lecture_id", "evidence_unit_ids"),
        "optional_properties": ("source_signals", "confidence", "relation_status"),
    },
}

CONCEPT_GRAPH_ARTIFACT_CONTRACT = {
    "schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
    "artifact_name": CONCEPT_GRAPH_ARTIFACT_NAME,
    "path": str(CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH),
    "record_types": (CONCEPT_NODE_RECORD_TYPE, RELATION_EDGE_RECORD_TYPE),
    "timestamp_only_relation_contract": (
        "timestamp-only source signals are candidate signals only and must not be "
        "represented as verified object alignment."
    ),
}


@dataclass(frozen=True)
class ConceptGraphEvidenceSource:
    evidence_unit_id: str
    source_type: str
    source_signal: str
    source_id: str | None = None
    confidence: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    is_verified_object_alignment: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConceptGraphEvidenceSource":
        evidence_unit_id = _required_text(payload, "evidence_unit_id")
        source_type = _required_text(payload, "source_type")
        source_signal = _normalize_source_signal(_required_text(payload, "source_signal"))
        is_verified_object_alignment = bool(payload.get("is_verified_object_alignment", False))
        if is_verified_object_alignment and source_signal in TIMESTAMP_ONLY_SOURCE_SIGNALS:
            raise ValueError(
                "timestamp-only source signals cannot be verified object alignment"
            )
        return cls(
            evidence_unit_id=evidence_unit_id,
            source_type=source_type,
            source_signal=source_signal,
            source_id=_optional_text(payload.get("source_id")),
            confidence=_optional_confidence(payload.get("confidence"), "evidence_source.confidence"),
            start_time=_optional_float(payload.get("start_time")),
            end_time=_optional_float(payload.get("end_time")),
            is_verified_object_alignment=is_verified_object_alignment,
            metadata=_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence_unit_id": self.evidence_unit_id,
            "source_type": self.source_type,
            "source_signal": self.source_signal,
        }
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.start_time is not None:
            payload["start_time"] = self.start_time
        if self.end_time is not None:
            payload["end_time"] = self.end_time
        if self.is_verified_object_alignment:
            payload["is_verified_object_alignment"] = True
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class ConceptGraphConceptNode:
    concept_id: str
    lecture_id: str
    label: str
    concept_type: str
    source_evidence_unit_ids: tuple[str, ...]
    evidence_sources: tuple[ConceptGraphEvidenceSource, ...]
    confidence: float
    aliases: tuple[str, ...] = ()
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONCEPT_GRAPH_SCHEMA_VERSION
    record_type: str = CONCEPT_NODE_RECORD_TYPE

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConceptGraphConceptNode":
        _check_record_type(payload, CONCEPT_NODE_RECORD_TYPE)
        evidence_sources = tuple(
            ConceptGraphEvidenceSource.from_dict(row) for row in _dict_list(payload.get("evidence_sources"))
        )
        source_evidence_unit_ids = _string_tuple(
            payload.get("source_evidence_unit_ids") or payload.get("evidence_unit_ids")
        )
        if not source_evidence_unit_ids:
            source_evidence_unit_ids = _unique_tuple(
                source.evidence_unit_id for source in evidence_sources
            )
        if not source_evidence_unit_ids:
            raise ValueError("concept node must include source evidence unit ids")
        return cls(
            concept_id=_required_text(payload, "concept_id"),
            lecture_id=_required_text(payload, "lecture_id"),
            label=_required_text(payload, "label"),
            concept_type=_required_text(payload, "concept_type", fallback_key="type"),
            source_evidence_unit_ids=source_evidence_unit_ids,
            evidence_sources=evidence_sources,
            confidence=_required_confidence(payload.get("confidence"), "concept.confidence"),
            aliases=_string_tuple(payload.get("aliases")),
            description=_optional_text(payload.get("description")),
            metadata=_mapping(payload.get("metadata")),
            schema_version=str(payload.get("schema_version", CONCEPT_GRAPH_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "concept_id": self.concept_id,
            "lecture_id": self.lecture_id,
            "label": self.label,
            "aliases": list(self.aliases),
            "concept_type": self.concept_type,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "evidence_sources": [source.to_dict() for source in self.evidence_sources],
            "confidence": self.confidence,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def graph_db_labels(self) -> tuple[str, ...]:
        return CONCEPT_GRAPH_DB_NODE_CONTRACT["Concept"]["labels"]

    def graph_db_key(self) -> str:
        return f"concept:{self.lecture_id}:{self.concept_id}"

    def graph_db_properties(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("schema_version", None)
        payload.pop("record_type", None)
        return payload


@dataclass(frozen=True)
class ConceptGraphRelationEdge:
    edge_id: str
    lecture_id: str
    source_concept_id: str
    relation_type: str
    target_concept_id: str
    evidence_unit_ids: tuple[str, ...]
    source_signals: tuple[str, ...]
    evidence_sources: tuple[ConceptGraphEvidenceSource, ...]
    confidence: float
    relation_status: str = "candidate"
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONCEPT_GRAPH_SCHEMA_VERSION
    record_type: str = RELATION_EDGE_RECORD_TYPE

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConceptGraphRelationEdge":
        _check_record_type(payload, RELATION_EDGE_RECORD_TYPE)
        evidence_sources = tuple(
            ConceptGraphEvidenceSource.from_dict(row) for row in _dict_list(payload.get("evidence_sources"))
        )
        evidence_unit_ids = _string_tuple(payload.get("evidence_unit_ids"))
        if not evidence_unit_ids:
            evidence_unit_ids = _unique_tuple(source.evidence_unit_id for source in evidence_sources)
        if not evidence_unit_ids:
            raise ValueError("relation edge must include evidence_unit_ids")
        source_signals = _source_signals(payload.get("source_signals"), evidence_sources)
        relation_status = _relation_status(payload)
        if _timestamp_only(source_signals) and relation_status == "verified":
            raise ValueError(
                "timestamp-only relation edges must remain candidate and cannot be verified"
            )
        return cls(
            edge_id=_required_text(payload, "edge_id"),
            lecture_id=_required_text(payload, "lecture_id"),
            source_concept_id=_required_text(payload, "source_concept_id"),
            relation_type=_relation_type(
                _required_text(payload, "relation_type", fallback_key="relation")
            ),
            target_concept_id=_required_text(payload, "target_concept_id"),
            evidence_unit_ids=evidence_unit_ids,
            source_signals=source_signals,
            evidence_sources=evidence_sources,
            confidence=_required_confidence(payload.get("confidence"), "relation.confidence"),
            relation_status=relation_status,
            description=_optional_text(payload.get("description")),
            metadata=_mapping(payload.get("metadata")),
            schema_version=str(payload.get("schema_version", CONCEPT_GRAPH_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "edge_id": self.edge_id,
            "lecture_id": self.lecture_id,
            "source_concept_id": self.source_concept_id,
            "relation_type": self.relation_type,
            "target_concept_id": self.target_concept_id,
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "source_signals": list(self.source_signals),
            "evidence_sources": [source.to_dict() for source in self.evidence_sources],
            "confidence": self.confidence,
            "relation_status": self.relation_status,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def graph_db_relationship_type(self) -> str:
        return CONCEPT_RELATION_TYPE_TO_GRAPH_EDGE_LABEL[self.relation_type]

    def graph_db_key(self) -> str:
        rel_type = self.graph_db_relationship_type()
        return f"rel:{rel_type}:{self.lecture_id}:{self.edge_id}"

    def graph_db_properties(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("schema_version", None)
        payload.pop("record_type", None)
        payload["graph_db_relationship_type"] = self.graph_db_relationship_type()
        payload["timestamp_only_candidate"] = _timestamp_only(self.source_signals)
        payload["timestamp_fallback_counted_as_verified"] = False
        return payload


ConceptGraphRecord: TypeAlias = ConceptGraphConceptNode | ConceptGraphRelationEdge


def parse_concept_graph_record(payload: dict[str, Any]) -> ConceptGraphRecord:
    record_type = str(payload.get("record_type") or "")
    if record_type == CONCEPT_NODE_RECORD_TYPE:
        return ConceptGraphConceptNode.from_dict(payload)
    if record_type == RELATION_EDGE_RECORD_TYPE:
        return ConceptGraphRelationEdge.from_dict(payload)
    if "concept_id" in payload and "edge_id" not in payload:
        return ConceptGraphConceptNode.from_dict({**payload, "record_type": CONCEPT_NODE_RECORD_TYPE})
    if "edge_id" in payload:
        return ConceptGraphRelationEdge.from_dict({**payload, "record_type": RELATION_EDGE_RECORD_TYPE})
    raise ValueError(f"Unsupported concept graph record_type: {record_type!r}")


def load_concept_graph_artifact(path: Path) -> list[ConceptGraphRecord]:
    records: list[ConceptGraphRecord] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(parse_concept_graph_record(payload))
    validate_concept_graph_records(records)
    return records


def write_concept_graph_artifact(path: Path, records: list[ConceptGraphRecord]) -> int:
    validate_concept_graph_records(records)
    return write_jsonl(path, [record.to_dict() for record in records])


def validate_concept_graph_records(records: list[ConceptGraphRecord]) -> dict[str, Any]:
    concept_ids: set[str] = set()
    edge_ids: set[str] = set()
    candidate_relations = 0
    verified_relations = 0
    timestamp_only_candidate_relations = 0

    for record in records:
        if isinstance(record, ConceptGraphConceptNode):
            if record.concept_id in concept_ids:
                raise ValueError(f"Duplicate concept_id: {record.concept_id}")
            concept_ids.add(record.concept_id)
        elif isinstance(record, ConceptGraphRelationEdge):
            if record.edge_id in edge_ids:
                raise ValueError(f"Duplicate edge_id: {record.edge_id}")
            edge_ids.add(record.edge_id)
            if record.relation_status == "verified":
                verified_relations += 1
            else:
                candidate_relations += 1
            if _timestamp_only(record.source_signals):
                timestamp_only_candidate_relations += 1

    for record in records:
        if not isinstance(record, ConceptGraphRelationEdge):
            continue
        missing = [
            concept_id
            for concept_id in (record.source_concept_id, record.target_concept_id)
            if concept_id not in concept_ids
        ]
        if missing:
            raise ValueError(f"Relation edge {record.edge_id} references missing concepts: {missing}")

    return {
        "schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "concepts": len(concept_ids),
        "relations": len(edge_ids),
        "candidate_relations": candidate_relations,
        "verified_relations": verified_relations,
        "timestamp_only_candidate_relations": timestamp_only_candidate_relations,
        "timestamp_fallback_counted_as_verified": False,
    }


def _check_record_type(payload: dict[str, Any], expected: str) -> None:
    record_type = payload.get("record_type")
    if record_type is not None and str(record_type) != expected:
        raise ValueError(f"Expected record_type {expected!r}, got {record_type!r}")


def _required_text(payload: dict[str, Any], key: str, *, fallback_key: str | None = None) -> str:
    value = payload.get(key)
    if _blank(value) and fallback_key is not None:
        value = payload.get(fallback_key)
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{key} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    return _unique_tuple(str(item).strip() for item in raw_items if str(item).strip())


def _unique_tuple(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _required_confidence(value: Any, field_name: str) -> float:
    confidence = _optional_confidence(value, field_name)
    if confidence is None:
        raise ValueError(f"{field_name} is required")
    return confidence


def _optional_confidence(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = _optional_float(value)
    if parsed is None or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_signals(
    raw_signals: Any,
    evidence_sources: tuple[ConceptGraphEvidenceSource, ...],
) -> tuple[str, ...]:
    explicit = [_normalize_source_signal(item) for item in _string_tuple(raw_signals)]
    derived = [source.source_signal for source in evidence_sources]
    source_signals = _unique_tuple([*explicit, *derived])
    if not source_signals:
        raise ValueError("relation edge must include source_signals")
    return source_signals


def _normalize_source_signal(value: str) -> str:
    return _snake_case(value)


def _relation_type(value: str) -> str:
    relation_type = _snake_case(value)
    if relation_type not in CONCEPT_RELATION_TYPE_TO_GRAPH_EDGE_LABEL:
        raise ValueError(f"Unsupported relation_type: {value!r}")
    return relation_type


def _relation_status(payload: dict[str, Any]) -> str:
    raw_status = (
        payload.get("relation_status")
        or payload.get("evidence_status")
        or payload.get("status")
        or "candidate"
    )
    relation_status = _snake_case(str(raw_status))
    if relation_status not in CONCEPT_RELATION_STATUSES:
        raise ValueError(f"Unsupported relation_status: {raw_status!r}")
    return relation_status


def _timestamp_only(source_signals: tuple[str, ...]) -> bool:
    return bool(source_signals) and set(source_signals).issubset(TIMESTAMP_ONLY_SOURCE_SIGNALS)


def _snake_case(value: str) -> str:
    value = value.strip().replace("-", "_").replace(" ", "_")
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return re.sub(r"_+", "_", value).strip("_")


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""
