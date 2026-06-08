from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from oarag.core.config import (
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    ENV_NEO4J_DATABASE,
    ENV_NEO4J_PASSWORD,
    ENV_NEO4J_URI,
    ENV_NEO4J_USER,
    Neo4jConfig,
    load_neo4j_config,
)
from oarag.graph.concept_graph_schema import (
    CONCEPT_GRAPH_ARTIFACT_NAME,
    CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH,
    CONCEPT_GRAPH_DB_EDGE_CONTRACT,
    CONCEPT_GRAPH_DB_NODE_CONTRACT,
    CONCEPT_GRAPH_SCHEMA_VERSION,
    TIMESTAMP_ONLY_SOURCE_SIGNALS,
    ConceptGraphConceptNode,
    ConceptGraphEvidenceSource,
    ConceptGraphRecord,
    ConceptGraphRelationEdge,
    load_concept_graph_artifact,
    validate_concept_graph_records,
)
from oarag.graph.graph_ingest import (
    GraphIngestPlan,
    GraphIngestUnavailableError,
    build_graph_ingest_plan,
    classify_neo4j_error,
    create_neo4j_driver,
    run_graph_ingest_plan,
)


DriverFactory = Callable[[Neo4jConfig], Any]

VISUAL_OBJECT_SOURCE_TYPES = frozenset(
    {
        "detected_object",
        "object",
        "visual_entity",
        "visual_object",
    }
)
VISUAL_OBJECT_SOURCE_SIGNALS = frozenset({"visual_object_relation"})


def ingest_concept_graph(
    *,
    project_dir: Path | None = None,
    concept_graph_path: Path | None = None,
    project_id: str | None = None,
    config: Neo4jConfig | None = None,
    create_schema: bool = True,
    dry_run: bool = False,
    skip_unavailable_runtime: bool = True,
    driver_factory: DriverFactory = create_neo4j_driver,
) -> dict[str, Any]:
    """Load a concept graph artifact and idempotently merge it into Neo4j."""
    artifact_path = resolve_concept_graph_path(
        project_dir=project_dir,
        concept_graph_path=concept_graph_path,
    )
    records = load_concept_graph_artifact(artifact_path)
    document = build_concept_graph_document(records, project_id=project_id)
    plan = build_graph_ingest_plan(document, create_schema=create_schema)
    active_config = config or load_neo4j_config()
    started_at = time.monotonic()
    statements_executed = 0
    runtime = _runtime_summary(status="dry_run", attempted=False, skip_reason="dry_run")

    if not dry_run:
        runtime = _runtime_summary(status="connecting", attempted=True)
        driver = None
        try:
            driver = driver_factory(active_config)
            driver.verify_connectivity()
            with driver.session(database=active_config.database) as session:
                statements_executed = run_graph_ingest_plan(session, plan)
            runtime = _runtime_summary(status="ingested", attempted=True, available=True)
        except Exception as exc:
            classified = classify_neo4j_error(exc)
            if skip_unavailable_runtime and isinstance(classified, GraphIngestUnavailableError):
                runtime = _runtime_summary(
                    status="skipped_unavailable",
                    attempted=True,
                    skip_reason=_skip_reason(classified),
                )
            else:
                raise classified from exc
        finally:
            if driver is not None:
                driver.close()

    return summarize_concept_graph_ingest(
        document=document,
        records=records,
        plan=plan,
        elapsed_ms=round((time.monotonic() - started_at) * 1000, 2),
        dry_run=dry_run,
        statements_executed=statements_executed,
        runtime=runtime,
    )


def resolve_concept_graph_path(
    *,
    project_dir: Path | None = None,
    concept_graph_path: Path | None = None,
) -> Path:
    if project_dir is None and concept_graph_path is None:
        raise ValueError("Either project_dir or concept_graph_path is required")
    if concept_graph_path is not None:
        return concept_graph_path.expanduser()
    return project_dir.expanduser() / CONCEPT_GRAPH_ARTIFACT_RELATIVE_PATH  # type: ignore[union-attr]


def build_concept_graph_ingest_plan(
    records: list[ConceptGraphRecord],
    *,
    project_id: str | None = None,
    create_schema: bool = True,
) -> GraphIngestPlan:
    return build_graph_ingest_plan(
        build_concept_graph_document(records, project_id=project_id),
        create_schema=create_schema,
    )


def build_concept_graph_document(
    records: list[ConceptGraphRecord],
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    validation = validate_concept_graph_records(records)
    concept_records = [record for record in records if isinstance(record, ConceptGraphConceptNode)]
    relation_records = [
        record for record in records if isinstance(record, ConceptGraphRelationEdge)
    ]
    concept_by_id = {record.concept_id: record for record in concept_records}
    lecture_ids = sorted({record.lecture_id for record in records})
    evidence_units = _collect_evidence_units(records)
    visual_objects = _collect_visual_objects(records)

    nodes: list[dict[str, Any]] = []
    for lecture_id in lecture_ids:
        nodes.append(
            _node(
                key=f"lecture:{lecture_id}",
                labels=CONCEPT_GRAPH_DB_NODE_CONTRACT["Lecture"]["labels"],
                properties=_compact(
                    {
                        "lecture_id": lecture_id,
                        "project_id": project_id,
                        "source": CONCEPT_GRAPH_ARTIFACT_NAME,
                    }
                ),
            )
        )
    for item in sorted(evidence_units.values(), key=lambda row: row["key"]):
        nodes.append(_node(key=item["key"], labels=item["labels"], properties=item["properties"]))
    for concept in sorted(concept_records, key=lambda item: item.graph_db_key()):
        properties = dict(concept.graph_db_properties())
        if project_id is not None:
            properties["project_id"] = project_id
        nodes.append(
            _node(
                key=concept.graph_db_key(),
                labels=concept.graph_db_labels(),
                properties=properties,
            )
        )
    for item in sorted(visual_objects.values(), key=lambda row: row["key"]):
        nodes.append(_node(key=item["key"], labels=item["labels"], properties=item["properties"]))

    relationships: dict[str, dict[str, Any]] = {}
    for item in sorted(evidence_units.values(), key=lambda row: row["key"]):
        lecture_id = item["properties"]["lecture_id"]
        evidence_unit_id = item["properties"]["evidence_unit_id"]
        _merge_relationship(
            relationships,
            key=f"rel:HAS_EVIDENCE:{lecture_id}:{evidence_unit_id}",
            relationship_type="HAS_EVIDENCE",
            start_node_key=f"lecture:{lecture_id}",
            end_node_key=item["key"],
            properties={
                "lecture_id": lecture_id,
                "source": CONCEPT_GRAPH_ARTIFACT_NAME,
            },
        )

    for concept in concept_records:
        concept_key = concept.graph_db_key()
        for evidence_unit_id in concept.source_evidence_unit_ids:
            evidence_key = _evidence_unit_key(concept.lecture_id, evidence_unit_id)
            sources = [
                source
                for source in concept.evidence_sources
                if source.evidence_unit_id == evidence_unit_id
            ]
            _merge_relationship(
                relationships,
                key=f"rel:MENTIONS:{concept.lecture_id}:{evidence_unit_id}:{concept.concept_id}",
                relationship_type="MENTIONS",
                start_node_key=evidence_key,
                end_node_key=concept_key,
                properties={
                    "edge_id": f"mentions:{concept.lecture_id}:{evidence_unit_id}:{concept.concept_id}",
                    "lecture_id": concept.lecture_id,
                    "evidence_unit_ids": [evidence_unit_id],
                    "source_signals": _source_signals(sources),
                    "confidence": concept.confidence,
                    "relation_status": _source_relation_status(sources),
                },
            )
        for source in concept.evidence_sources:
            visual_object_id = _visual_object_id(source)
            if visual_object_id is None:
                continue
            visual_key = _visual_object_key(concept.lecture_id, visual_object_id)
            status = _source_relation_status([source])
            _merge_relationship(
                relationships,
                key=(
                    f"rel:SHOWS:{concept.lecture_id}:{source.evidence_unit_id}:"
                    f"{visual_object_id}"
                ),
                relationship_type="SHOWS",
                start_node_key=_evidence_unit_key(concept.lecture_id, source.evidence_unit_id),
                end_node_key=visual_key,
                properties={
                    "edge_id": (
                        f"shows:{concept.lecture_id}:{source.evidence_unit_id}:"
                        f"{visual_object_id}"
                    ),
                    "lecture_id": concept.lecture_id,
                    "evidence_unit_ids": [source.evidence_unit_id],
                    "source_signals": [source.source_signal],
                    "confidence": source.confidence,
                    "relation_status": status,
                },
            )
            _merge_relationship(
                relationships,
                key=f"rel:DEPICTS:{concept.lecture_id}:{visual_object_id}:{concept.concept_id}",
                relationship_type="DEPICTS",
                start_node_key=visual_key,
                end_node_key=concept_key,
                properties={
                    "edge_id": f"depicts:{concept.lecture_id}:{visual_object_id}:{concept.concept_id}",
                    "lecture_id": concept.lecture_id,
                    "evidence_unit_ids": [source.evidence_unit_id],
                    "source_signals": [source.source_signal],
                    "confidence": source.confidence,
                    "relation_status": status,
                },
            )

    for relation in relation_records:
        _merge_relationship(
            relationships,
            key=relation.graph_db_key(),
            relationship_type=relation.graph_db_relationship_type(),
            start_node_key=concept_by_id[relation.source_concept_id].graph_db_key(),
            end_node_key=concept_by_id[relation.target_concept_id].graph_db_key(),
            properties=relation.graph_db_properties(),
        )

    node_counts = _node_counts(nodes)
    relationship_counts = _relationship_counts(relationships.values())
    return {
        "schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "artifact_name": CONCEPT_GRAPH_ARTIFACT_NAME,
        "project_id": project_id,
        "nodes": nodes,
        "relationships": sorted(relationships.values(), key=lambda row: row["key"]),
        "counts": {
            "nodes": len(nodes),
            "relationships": len(relationships),
            "nodes_by_label": node_counts,
            "relationships_by_type": relationship_counts,
            "records": len(records),
            **validation,
        },
    }


def summarize_concept_graph_ingest(
    *,
    document: dict[str, Any],
    records: list[ConceptGraphRecord],
    plan: GraphIngestPlan,
    elapsed_ms: float,
    dry_run: bool,
    statements_executed: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    counts = document.get("counts") if isinstance(document.get("counts"), dict) else {}
    node_counts = counts.get("nodes_by_label") if isinstance(counts.get("nodes_by_label"), dict) else {}
    relationship_counts = (
        counts.get("relationships_by_type")
        if isinstance(counts.get("relationships_by_type"), dict)
        else {}
    )
    return {
        "schema_version": CONCEPT_GRAPH_SCHEMA_VERSION,
        "artifact": CONCEPT_GRAPH_ARTIFACT_NAME,
        "dry_run": dry_run,
        "runtime": runtime,
        "connection_contract": concept_graph_connection_contract(),
        "records": {
            "total": len(records),
            "concepts": int(counts.get("concepts") or 0),
            "relations": int(counts.get("relations") or 0),
            "candidate_relations": int(counts.get("candidate_relations") or 0),
            "verified_relations": int(counts.get("verified_relations") or 0),
            "timestamp_only_candidate_relations": int(
                counts.get("timestamp_only_candidate_relations") or 0
            ),
            "timestamp_fallback_counted_as_verified": False,
        },
        "nodes": {
            "total": int(counts.get("nodes") or 0),
            "by_label": dict(sorted(node_counts.items())),
        },
        "relationships": {
            "total": int(counts.get("relationships") or 0),
            "by_type": dict(sorted(relationship_counts.items())),
            "skipped_missing_endpoint": len(plan.skipped_relationships),
        },
        "plan": {
            "schema_statements": len(plan.schema_statements),
            "node_merge_statements": len(plan.node_statements),
            "relationship_merge_statements": len(plan.relationship_statements),
            "statements_total": len(plan.statements),
        },
        "statements_executed": statements_executed,
        "elapsed_ms": elapsed_ms,
        "public_safe_summary": {
            "level": "counts_status_and_contract_only",
            "omits": ["raw_transcript", "local_paths", "frame_paths", "secrets"],
        },
    }


def concept_graph_connection_contract() -> dict[str, Any]:
    return {
        "backend": "neo4j",
        "driver": "neo4j-python",
        "config_env": {
            "uri": ENV_NEO4J_URI,
            "user": ENV_NEO4J_USER,
            "password": ENV_NEO4J_PASSWORD,
            "database": ENV_NEO4J_DATABASE,
        },
        "defaults": {
            "uri": DEFAULT_NEO4J_URI,
            "user": DEFAULT_NEO4J_USER,
            "password": "***",
            "database": DEFAULT_NEO4J_DATABASE,
        },
        "schema": {
            "node_key": "GraphNode.key",
            "relationship_key_property": "key",
            "node_contract": CONCEPT_GRAPH_DB_NODE_CONTRACT,
            "edge_contract": CONCEPT_GRAPH_DB_EDGE_CONTRACT,
        },
    }


def _collect_evidence_units(records: list[ConceptGraphRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if isinstance(record, ConceptGraphConceptNode):
            evidence_unit_ids = record.source_evidence_unit_ids
        else:
            evidence_unit_ids = record.evidence_unit_ids
        for evidence_unit_id in evidence_unit_ids:
            _ensure_evidence_unit(
                rows,
                lecture_id=record.lecture_id,
                evidence_unit_id=evidence_unit_id,
            )
        for source in record.evidence_sources:
            item = _ensure_evidence_unit(
                rows,
                lecture_id=record.lecture_id,
                evidence_unit_id=source.evidence_unit_id,
            )
            _merge_interval(item["properties"], source)
    return rows


def _collect_visual_objects(records: list[ConceptGraphRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for source in record.evidence_sources:
            visual_object_id = _visual_object_id(source)
            if visual_object_id is None:
                continue
            key = (record.lecture_id, visual_object_id)
            item = rows.get(key)
            if item is None:
                item = _node(
                    key=_visual_object_key(record.lecture_id, visual_object_id),
                    labels=CONCEPT_GRAPH_DB_NODE_CONTRACT["VisualObject"]["labels"],
                    properties=_compact(
                        {
                            "visual_object_id": visual_object_id,
                            "lecture_id": record.lecture_id,
                            "label": _visual_object_label(source, visual_object_id),
                            "source_visual_entity_id": _source_visual_entity_id(source),
                            "confidence": source.confidence,
                            "metadata": source.metadata or None,
                        }
                    ),
                )
                rows[key] = item
            else:
                properties = item["properties"]
                properties["confidence"] = _max_optional_float(
                    properties.get("confidence"),
                    source.confidence,
                )
    return rows


def _ensure_evidence_unit(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    lecture_id: str,
    evidence_unit_id: str,
) -> dict[str, Any]:
    key = (lecture_id, evidence_unit_id)
    item = rows.get(key)
    if item is None:
        item = _node(
            key=_evidence_unit_key(lecture_id, evidence_unit_id),
            labels=CONCEPT_GRAPH_DB_NODE_CONTRACT["EvidenceUnit"]["labels"],
            properties={
                "evidence_unit_id": evidence_unit_id,
                "lecture_id": lecture_id,
            },
        )
        rows[key] = item
    return item


def _merge_interval(properties: dict[str, Any], source: ConceptGraphEvidenceSource) -> None:
    if source.start_time is not None:
        current = properties.get("start_time")
        properties["start_time"] = (
            source.start_time if current is None else min(float(current), source.start_time)
        )
    if source.end_time is not None:
        current = properties.get("end_time")
        properties["end_time"] = source.end_time if current is None else max(float(current), source.end_time)


def _merge_relationship(
    relationships: dict[str, dict[str, Any]],
    *,
    key: str,
    relationship_type: str,
    start_node_key: str,
    end_node_key: str,
    properties: dict[str, Any],
) -> None:
    existing = relationships.get(key)
    if existing is None:
        relationships[key] = {
            "key": key,
            "type": relationship_type,
            "start_node_key": start_node_key,
            "end_node_key": end_node_key,
            "properties": _compact(properties),
        }
        return

    existing_properties = existing["properties"]
    for field in ("evidence_unit_ids", "source_signals"):
        merged = _unique_strings(
            [
                *list(existing_properties.get(field) or []),
                *list(properties.get(field) or []),
            ]
        )
        if merged:
            existing_properties[field] = merged
    existing_properties["confidence"] = _max_optional_float(
        existing_properties.get("confidence"),
        properties.get("confidence"),
    )
    if (
        existing_properties.get("relation_status") != "verified"
        and properties.get("relation_status") == "verified"
    ):
        existing_properties["relation_status"] = "verified"


def _node(*, key: str, labels: tuple[str, ...], properties: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "labels": list(labels), "properties": properties}


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _source_signals(sources: list[ConceptGraphEvidenceSource]) -> list[str]:
    return _unique_strings(source.source_signal for source in sources)


def _source_relation_status(sources: list[ConceptGraphEvidenceSource]) -> str:
    for source in sources:
        if (
            source.is_verified_object_alignment
            and source.source_signal not in TIMESTAMP_ONLY_SOURCE_SIGNALS
        ):
            return "verified"
    return "candidate"


def _visual_object_id(source: ConceptGraphEvidenceSource) -> str | None:
    metadata_id = _text(source.metadata.get("visual_object_id"))
    if metadata_id is not None:
        return metadata_id
    if not _is_visual_object_source(source):
        return None
    return _text(source.source_id)


def _is_visual_object_source(source: ConceptGraphEvidenceSource) -> bool:
    source_type = _snake_case(source.source_type)
    source_signal = _snake_case(source.source_signal)
    return source_type in VISUAL_OBJECT_SOURCE_TYPES or source_signal in VISUAL_OBJECT_SOURCE_SIGNALS


def _visual_object_label(source: ConceptGraphEvidenceSource, visual_object_id: str) -> str:
    for key in ("label", "object_label", "name"):
        value = _text(source.metadata.get(key))
        if value is not None:
            return value
    return visual_object_id


def _source_visual_entity_id(source: ConceptGraphEvidenceSource) -> str | None:
    value = _text(source.metadata.get("source_visual_entity_id"))
    if value is not None:
        return value
    if _snake_case(source.source_type) == "visual_entity":
        return _text(source.source_id)
    return None


def _evidence_unit_key(lecture_id: str, evidence_unit_id: str) -> str:
    return f"evidence_unit:{lecture_id}:{evidence_unit_id}"


def _visual_object_key(lecture_id: str, visual_object_id: str) -> str:
    return f"visual_object:{lecture_id}:{visual_object_id}"


def _node_counts(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in nodes:
        labels = [str(label) for label in node.get("labels") or []]
        public_label = next((label for label in labels if label != "GraphNode"), None)
        if public_label is not None:
            counts[public_label] += 1
    return dict(sorted(counts.items()))


def _relationship_counts(relationships: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for relationship in relationships:
        relationship_type = str(relationship.get("type") or "")
        if relationship_type:
            counts[relationship_type] += 1
    return dict(sorted(counts.items()))


def _runtime_summary(
    *,
    status: str,
    attempted: bool,
    available: bool = False,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    return _compact(
        {
            "backend": "neo4j",
            "attempted": attempted,
            "available": available,
            "status": status,
            "skip_reason": skip_reason,
        }
    )


def _skip_reason(error: GraphIngestUnavailableError) -> str:
    message = str(error).casefold()
    if "not installed" in message or "no module named" in message:
        return "neo4j_driver_missing"
    return "neo4j_runtime_unavailable"


def _max_optional_float(left: Any, right: Any) -> float | None:
    values = [value for value in (_optional_float(left), _optional_float(right)) if value is not None]
    return max(values) if values else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _snake_case(value: str) -> str:
    value = value.strip().replace("-", "_").replace(" ", "_")
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return re.sub(r"_+", "_", value).strip("_")
