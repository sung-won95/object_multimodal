from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import Neo4jConfig, load_neo4j_config
from .graph_document import build_graph_document


class GraphIngestError(RuntimeError):
    pass


class GraphIngestUnavailableError(GraphIngestError):
    pass


class GraphIngestAuthError(GraphIngestError):
    pass


class GraphIngestDatabaseError(GraphIngestError):
    pass


@dataclass(frozen=True)
class CypherStatement:
    cypher: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphIngestPlan:
    schema_statements: tuple[CypherStatement, ...]
    node_statements: tuple[CypherStatement, ...]
    relationship_statements: tuple[CypherStatement, ...]
    skipped_relationships: tuple[dict[str, Any], ...]
    missing_artifacts: tuple[str, ...]

    @property
    def statements(self) -> tuple[CypherStatement, ...]:
        return self.schema_statements + self.node_statements + self.relationship_statements


class Neo4jSession(Protocol):
    def run(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        ...


class Neo4jDriver(Protocol):
    def session(self, **kwargs: Any) -> Any:
        ...

    def verify_connectivity(self) -> None:
        ...

    def close(self) -> None:
        ...


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ingest_project_graph(
    *,
    project_dir: Path | None = None,
    graph_document_path: Path | None = None,
    config: Neo4jConfig | None = None,
    create_schema: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    if project_dir is None and graph_document_path is None:
        raise ValueError("Either project_dir or graph_document_path is required")
    document = (
        load_graph_document(graph_document_path)
        if graph_document_path
        else build_graph_document(project_dir=project_dir or Path()).to_dict()
    )
    plan = build_graph_ingest_plan(document, create_schema=create_schema)
    started_at = time.monotonic()
    statements_executed = 0
    active_config = config or load_neo4j_config()
    if not dry_run:
        driver = create_neo4j_driver(active_config)
        try:
            driver.verify_connectivity()
            with driver.session(database=active_config.database) as session:
                statements_executed = run_graph_ingest_plan(session, plan)
        except Exception as exc:
            raise classify_neo4j_error(exc) from exc
        finally:
            driver.close()
    return summarize_graph_ingest(
        document,
        plan,
        elapsed_ms=round((time.monotonic() - started_at) * 1000, 2),
        dry_run=dry_run,
        statements_executed=statements_executed,
    )


def load_graph_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected graph document JSON object in {path}")
    return payload


def build_graph_ingest_plan(document: dict[str, Any], *, create_schema: bool = True) -> GraphIngestPlan:
    nodes = [_normalize_node(row) for row in _object_rows(document.get("nodes"), "nodes")]
    node_keys = {node["key"] for node in nodes}
    relationships = [
        _normalize_relationship(row)
        for row in _object_rows(document.get("relationships"), "relationships")
    ]
    valid_relationships: list[dict[str, Any]] = []
    skipped_relationships: list[dict[str, Any]] = []
    for relationship in relationships:
        missing = [
            key
            for key in (relationship["start_node_key"], relationship["end_node_key"])
            if key not in node_keys
        ]
        if missing:
            skipped_relationships.append(
                {
                    "key": relationship["key"],
                    "type": relationship["type"],
                    "missing_node_keys": missing,
                }
            )
            continue
        valid_relationships.append(relationship)

    relationship_types = sorted({relationship["type"] for relationship in valid_relationships})
    return GraphIngestPlan(
        schema_statements=tuple(schema_statements(relationship_types)) if create_schema else (),
        node_statements=tuple(node_merge_statements(nodes)),
        relationship_statements=tuple(relationship_merge_statements(valid_relationships)),
        skipped_relationships=tuple(skipped_relationships),
        missing_artifacts=tuple(missing_artifacts(document)),
    )


def schema_statements(relationship_types: list[str]) -> list[CypherStatement]:
    statements = [
        CypherStatement(
            "CREATE CONSTRAINT graph_node_key IF NOT EXISTS "
            "FOR (n:GraphNode) REQUIRE n.key IS UNIQUE"
        ),
        CypherStatement(
            "CREATE INDEX graph_node_project_id IF NOT EXISTS "
            "FOR (n:GraphNode) ON (n.project_id)"
        ),
    ]
    for relationship_type in relationship_types:
        statements.append(
            CypherStatement(
                f"CREATE INDEX graph_relationship_{relationship_type.lower()}_key IF NOT EXISTS "
                f"FOR ()-[r:{relationship_type}]-() ON (r.key)"
            )
        )
    return statements


def node_merge_statements(nodes: list[dict[str, Any]]) -> list[CypherStatement]:
    statements: list[CypherStatement] = []
    for labels, rows in _group_by_labels(nodes).items():
        label_clause = "".join(f":{label}" for label in labels)
        statements.append(
            CypherStatement(
                "UNWIND $nodes AS row\n"
                "MERGE (n:GraphNode {key: row.key})\n"
                f"SET n{label_clause}\n"
                "SET n += row.properties\n"
                "SET n.key = row.key",
                {"nodes": rows},
            )
        )
    return statements


def relationship_merge_statements(relationships: list[dict[str, Any]]) -> list[CypherStatement]:
    statements: list[CypherStatement] = []
    for relationship_type, rows in _group_by_type(relationships).items():
        statements.append(
            CypherStatement(
                "UNWIND $relationships AS row\n"
                "MATCH (start:GraphNode {key: row.start_node_key})\n"
                "MATCH (end:GraphNode {key: row.end_node_key})\n"
                f"MERGE (start)-[r:{relationship_type} {{key: row.key}}]->(end)\n"
                "SET r += row.properties\n"
                "SET r.key = row.key",
                {"relationships": rows},
            )
        )
    return statements


def run_graph_ingest_plan(session: Neo4jSession, plan: GraphIngestPlan) -> int:
    count = 0
    for statement in plan.statements:
        session.run(statement.cypher, statement.parameters)
        count += 1
    return count


def summarize_graph_ingest(
    document: dict[str, Any],
    plan: GraphIngestPlan,
    *,
    elapsed_ms: float,
    dry_run: bool,
    statements_executed: int,
) -> dict[str, Any]:
    counts = document.get("counts") if isinstance(document.get("counts"), dict) else {}
    return {
        "project_id": document.get("project_id"),
        "dry_run": dry_run,
        "nodes": {
            "total": len(_object_rows(document.get("nodes"), "nodes")),
            "by_label": counts.get("nodes_by_label") or _count_nodes_by_label(document),
        },
        "relationships": {
            "total": len(_object_rows(document.get("relationships"), "relationships")),
            "by_type": counts.get("relationships_by_type") or _count_relationships_by_type(document),
            "skipped_missing_endpoint": len(plan.skipped_relationships),
        },
        "missing_artifacts": {
            "count": len(plan.missing_artifacts),
            "names": list(plan.missing_artifacts),
        },
        "schema_statements": len(plan.schema_statements),
        "node_merge_statements": len(plan.node_statements),
        "relationship_merge_statements": len(plan.relationship_statements),
        "statements_executed": statements_executed,
        "elapsed_ms": elapsed_ms,
    }


def missing_artifacts(document: dict[str, Any]) -> list[str]:
    availability = document.get("availability")
    if not isinstance(availability, dict):
        return []
    artifacts = availability.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    return sorted(
        name
        for name, artifact in artifacts.items()
        if isinstance(artifact, dict) and artifact.get("available") is False
    )


def create_neo4j_driver(config: Neo4jConfig) -> Neo4jDriver:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise GraphIngestUnavailableError(
            "Neo4j driver is not installed. Install project dependencies before graph-ingest."
        ) from exc
    return GraphDatabase.driver(config.uri, auth=(config.user, config.password))


def classify_neo4j_error(exc: Exception) -> GraphIngestError:
    try:
        from neo4j.exceptions import AuthError, ClientError, DatabaseError, ServiceUnavailable
    except ImportError:
        return GraphIngestUnavailableError(str(exc))
    if isinstance(exc, AuthError):
        return GraphIngestAuthError(f"Neo4j authentication failed: {exc}")
    if isinstance(exc, ServiceUnavailable):
        return GraphIngestUnavailableError(f"Neo4j is unavailable: {exc}")
    if isinstance(exc, DatabaseError):
        return GraphIngestDatabaseError(f"Neo4j database error: {exc}")
    if isinstance(exc, ClientError) and "database" in str(exc).casefold():
        return GraphIngestDatabaseError(f"Neo4j database is unavailable: {exc}")
    if isinstance(exc, GraphIngestError):
        return exc
    return GraphIngestError(str(exc))


def _normalize_node(row: dict[str, Any]) -> dict[str, Any]:
    key = str(row.get("key") or "")
    if not key:
        raise ValueError("Graph node is missing key")
    labels = row.get("labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError(f"Graph node {key} must include at least one label")
    normalized_labels = tuple(_safe_identifier(str(label), "node label") for label in labels)
    properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    return {"key": key, "labels": normalized_labels, "properties": _json_properties(properties)}


def _normalize_relationship(row: dict[str, Any]) -> dict[str, Any]:
    relationship_type = _safe_identifier(str(row.get("type") or ""), "relationship type")
    start_node_key = str(row.get("start_node_key") or "")
    end_node_key = str(row.get("end_node_key") or "")
    if not start_node_key or not end_node_key:
        raise ValueError(f"Graph relationship {row.get('key')} is missing endpoint keys")
    key = str(row.get("key") or f"rel:{relationship_type}:{start_node_key}:{end_node_key}")
    properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    return {
        "key": key,
        "type": relationship_type,
        "start_node_key": start_node_key,
        "end_node_key": end_node_key,
        "properties": _json_properties(properties),
    }


def _safe_identifier(value: str, kind: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid Neo4j {kind}: {value!r}")
    return value


def _object_rows(value: Any, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Graph document {name} must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"Graph document {name}[{index}] must be an object")
        rows.append(row)
    return rows


def _group_by_labels(nodes: list[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for node in nodes:
        grouped.setdefault(node["labels"], []).append(
            {"key": node["key"], "properties": node["properties"]}
        )
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _group_by_type(relationships: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for relationship in relationships:
        grouped.setdefault(relationship["type"], []).append(relationship)
    return dict(sorted(grouped.items()))


def _json_properties(properties: dict[str, Any]) -> dict[str, Any]:
    jsonable = json.loads(json.dumps(properties, ensure_ascii=False))
    return {str(key): _neo4j_property_value(value) for key, value in jsonable.items()}


def _neo4j_property_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value
    return value


def _count_nodes_by_label(document: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in _object_rows(document.get("nodes"), "nodes"):
        labels = node.get("labels")
        if isinstance(labels, list) and labels:
            label = str(labels[0])
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _count_relationships_by_type(document: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relationship in _object_rows(document.get("relationships"), "relationships"):
        relationship_type = str(relationship.get("type") or "")
        if relationship_type:
            counts[relationship_type] = counts.get(relationship_type, 0) + 1
    return dict(sorted(counts.items()))
