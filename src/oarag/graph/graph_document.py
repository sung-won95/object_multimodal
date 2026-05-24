from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from oarag.core.domain_lexicon import DomainLexicon, load_domain_lexicon
from oarag.retrieval.project_index import segment_artifact_path
from oarag.vision.reference_resolution import REFERENCE_RESOLUTION_SOURCE, resolve_references
from oarag.core.schemas import EntityLink, VisualEntity, slugify

GRAPH_DOCUMENT_SCHEMA_VERSION = "graph-document-v1"


@dataclass(frozen=True)
class GraphNode:
    key: str
    labels: tuple[str, ...]
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "labels": list(self.labels),
            "properties": self.properties,
        }


@dataclass(frozen=True)
class GraphRelationship:
    key: str
    type: str
    start_node_key: str
    end_node_key: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type,
            "start_node_key": self.start_node_key,
            "end_node_key": self.end_node_key,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class GraphDocument:
    schema_version: str
    project_id: str
    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]
    availability: dict[str, Any]
    counts: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "availability": self.availability,
            "counts": self.counts,
            "warnings": list(self.warnings),
        }


def build_graph_document(
    *,
    project_dir: Path,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    manifest_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
) -> GraphDocument:
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    resolved_segments_path = _optional_segment_path(resolved_project_dir, segments_path)
    resolved_frames_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )
    resolved_visual_entities_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=visual_entities_path,
        default=resolved_project_dir / "manifests" / "visual_entities.jsonl",
    )
    resolved_entity_links_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=entity_links_path,
        default=resolved_project_dir / "manifests" / "entity_links.jsonl",
    )

    segments = _read_optional_jsonl(resolved_segments_path)
    frames = _read_optional_jsonl(resolved_frames_manifest_path)
    entities = [
        VisualEntity.from_dict(row) for row in _read_optional_jsonl(resolved_visual_entities_path)
    ]
    links = [EntityLink.from_dict(row) for row in _read_optional_jsonl(resolved_entity_links_path)]
    manifest = _read_optional_json(resolved_manifest_path)
    domain_lexicon = _load_optional_domain_lexicon(
        project_dir=resolved_project_dir,
        domain_lexicon_path=domain_lexicon_path,
    )
    project_id = _project_id(
        manifest=manifest,
        project_dir=resolved_project_dir,
        segments=segments,
        frames=frames,
        entities=entities,
    )

    builder = _GraphDocumentBuilder(project_id=project_id, domain_lexicon=domain_lexicon)
    builder.add_project(manifest)
    builder.add_segments(segments)
    builder.add_frames(frames)
    builder.add_visual_entities(entities)
    builder.add_entity_links(links)
    builder.add_reference_resolutions(segments=segments, entities=entities, links=links)
    builder.add_temporal_edges()

    availability = _availability(
        paths={
            "project_manifest": resolved_manifest_path,
            "segments": resolved_segments_path,
            "frames_manifest": resolved_frames_manifest_path,
            "visual_entities": resolved_visual_entities_path,
            "entity_links": resolved_entity_links_path,
            "domain_lexicon": domain_lexicon.source_path,
        },
        counts={
            "segments": len(segments),
            "frames": len(frames),
            "visual_entities": len(entities),
            "entity_links": len(links),
        },
        selected_segments_path=resolved_segments_path,
    )
    warnings = _warnings(availability)
    nodes = tuple(sorted(builder.nodes.values(), key=lambda node: node.key))
    relationships = tuple(sorted(builder.relationships.values(), key=lambda rel: rel.key))

    return GraphDocument(
        schema_version=GRAPH_DOCUMENT_SCHEMA_VERSION,
        project_id=project_id,
        nodes=nodes,
        relationships=relationships,
        availability=availability,
        counts=_counts(nodes=nodes, relationships=relationships),
        warnings=tuple(warnings),
    )


class _GraphDocumentBuilder:
    def __init__(self, *, project_id: str, domain_lexicon: DomainLexicon) -> None:
        self.project_id = project_id
        self.domain_lexicon = domain_lexicon
        self.nodes: dict[str, GraphNode] = {}
        self.relationships: dict[str, GraphRelationship] = {}
        self.segments_by_video: dict[str, list[dict[str, Any]]] = {}
        self.frames_by_id: dict[str, dict[str, Any]] = {}
        self.entities_by_id: dict[str, VisualEntity] = {}

    def add_project(self, manifest: dict[str, Any]) -> None:
        properties = _compact(
            {
                "project_id": self.project_id,
                "title": manifest.get("title") or manifest.get("project_name"),
                "source": manifest.get("source"),
            }
        )
        self._add_node("project", self.project_id, ("Project",), properties)

    def add_segments(self, segments: list[dict[str, Any]]) -> None:
        for segment in _sorted_segments(segments):
            video_id = _video_id_from_segment(segment)
            self._add_video(video_id, video_name=str(segment.get("video_name") or video_id))
            segment_id = str(segment.get("segment_id", ""))
            if not segment_id:
                continue
            segment_key = self._add_node(
                "segment",
                segment_id,
                ("Segment",),
                _compact(
                    {
                        "segment_id": segment_id,
                        "project_id": self.project_id,
                        "video_id": video_id,
                        "sample_id": segment.get("sample_id"),
                        "start_time": _optional_float(segment.get("start_time")),
                        "end_time": _optional_float(segment.get("end_time")),
                        "timestamp_center": _optional_float(segment.get("timestamp_center")),
                        "transcript_text": segment.get("transcript_text"),
                        "source": segment.get("source"),
                    }
                ),
            )
            self._add_relationship("HAS_SEGMENT", self._node_key("video", video_id), segment_key)
            self.segments_by_video.setdefault(video_id, []).append(segment)
            for concept in _concepts_from_segment(segment, self.domain_lexicon):
                concept_key = self._add_concept(concept)
                self._add_relationship(
                    "MENTIONS",
                    segment_key,
                    concept_key,
                    {"source": "segment"},
                    identity=concept,
                )

    def add_frames(self, frames: list[dict[str, Any]]) -> None:
        for frame in _sorted_frames(frames):
            frame_id = _frame_id(frame)
            if not frame_id:
                continue
            video_id = _video_id_from_frame(frame, fallback=self._fallback_video_id())
            self._add_video(video_id)
            frame_key = self._add_node(
                "frame",
                frame_id,
                ("Frame",),
                _compact(
                    {
                        "frame_id": frame_id,
                        "project_id": self.project_id,
                        "video_id": video_id,
                        "timestamp": _optional_float(frame.get("timestamp")),
                        "frame_path": frame.get("frame_path"),
                    }
                ),
            )
            self.frames_by_id[frame_id] = frame
            self._add_relationship("HAS_FRAME", self._node_key("video", video_id), frame_key)
        self._add_segment_frame_alignment()

    def add_visual_entities(self, entities: list[VisualEntity]) -> None:
        for entity in sorted(entities, key=lambda item: (item.frame_id, item.entity_id)):
            if not entity.entity_id:
                continue
            entity_key = self._add_node(
                "visual_entity",
                entity.entity_id,
                ("VisualEntity",),
                _compact(
                    {
                        "entity_id": entity.entity_id,
                        "project_id": self.project_id,
                        "frame_id": entity.frame_id,
                        "timestamp": entity.timestamp,
                        "text": entity.text,
                        "entity_type": entity.entity_type,
                        "confidence": entity.confidence,
                        "source": entity.source,
                        "visual_description": entity.visual_description,
                        "bbox": entity.bbox,
                        "position": entity.position,
                    }
                ),
            )
            self.entities_by_id[entity.entity_id] = entity
            if entity.frame_id:
                frame_key = self._node_key("frame", entity.frame_id)
                if frame_key in self.nodes:
                    self._add_relationship("CONTAINS", frame_key, entity_key)
            for concept in _concepts_from_entity(entity, self.domain_lexicon):
                concept_key = self._add_concept(concept)
                self._add_relationship(
                    "REPRESENTS",
                    entity_key,
                    concept_key,
                    {"source": "visual_entity"},
                    identity=concept,
                )

    def add_entity_links(self, links: list[EntityLink]) -> None:
        for link in sorted(links, key=lambda item: item.link_id):
            segment_key = self._node_key("segment", link.segment_id)
            entity_key = self._node_key("visual_entity", link.entity_id)
            if segment_key not in self.nodes or entity_key not in self.nodes:
                continue
            self._add_relationship(
                "LINKED_TO",
                segment_key,
                entity_key,
                _compact(
                    {
                        "link_id": link.link_id,
                        "link_type": link.link_type,
                        "score": link.score,
                        "evidence": link.evidence,
                        "time_overlap": link.time_overlap,
                        "lexical_match": link.lexical_match,
                        "mention_candidate": link.mention_candidate,
                        "frame_id": link.frame_id,
                    }
                ),
                identity=link.link_id,
            )

    def add_reference_resolutions(
        self,
        *,
        segments: list[dict[str, Any]],
        entities: list[VisualEntity],
        links: list[EntityLink],
    ) -> None:
        for resolution in resolve_references(
            segments=_segments_with_string_frame_refs(segments),
            domain_lexicon=self.domain_lexicon,
            entities=entities,
            links=links,
        ):
            segment_key = self._node_key("segment", resolution.segment_id)
            reference_key = self._add_node(
                "reference_mention",
                resolution.reference_id,
                ("ReferenceMention",),
                _compact(
                    {
                        "reference_id": resolution.reference_id,
                        "project_id": self.project_id,
                        "segment_id": resolution.segment_id,
                        "hint_text": resolution.hint_text,
                        "source": REFERENCE_RESOLUTION_SOURCE,
                        "score": resolution.score,
                        "reason": resolution.reason,
                        "evidence": list(resolution.evidence),
                        "lookback_segment_id": resolution.lookback_segment_id,
                        "lookback_segment_count": resolution.lookback_segment_count,
                        "lookback_seconds": resolution.lookback_seconds,
                    }
                ),
            )
            self._add_relationship(
                "REFERS_TO",
                segment_key,
                reference_key,
                _compact(
                    {
                        "source": REFERENCE_RESOLUTION_SOURCE,
                        "score": resolution.score,
                        "reason": resolution.reason,
                        "evidence": list(resolution.evidence),
                        "lookback_segment_id": resolution.lookback_segment_id,
                        "lookback_segment_count": resolution.lookback_segment_count,
                        "lookback_seconds": resolution.lookback_seconds,
                    }
                ),
                identity=resolution.reference_id,
            )
            for target in resolution.targets:
                target_key = self._node_key(target.node_namespace, target.local_id)
                self._add_relationship(
                    "RESOLVES_TO",
                    reference_key,
                    target_key,
                    _compact(
                        {
                            "source": REFERENCE_RESOLUTION_SOURCE,
                            "target_type": target.target_type,
                            "score": target.score,
                            "reason": target.reason,
                            "evidence": list(target.evidence),
                            "lookback_segment_id": resolution.lookback_segment_id,
                            "lookback_segment_count": resolution.lookback_segment_count,
                            "lookback_seconds": resolution.lookback_seconds,
                        }
                    ),
                    identity=f"{resolution.reference_id}:{target.target_type}:{target.local_id}",
                )

    def add_temporal_edges(self) -> None:
        for video_id, segments in sorted(self.segments_by_video.items()):
            ordered = _sorted_segments(segments)
            for previous, current in zip(ordered, ordered[1:]):
                previous_id = str(previous.get("segment_id", ""))
                current_id = str(current.get("segment_id", ""))
                if not previous_id or not current_id:
                    continue
                self._add_relationship(
                    "NEXT_SEGMENT",
                    self._node_key("segment", previous_id),
                    self._node_key("segment", current_id),
                    {"video_id": video_id},
                )

    def _add_segment_frame_alignment(self) -> None:
        for segment in [item for values in self.segments_by_video.values() for item in values]:
            segment_id = str(segment.get("segment_id", ""))
            segment_key = self._node_key("segment", segment_id)
            for frame_id in _aligned_frame_ids(segment, self.frames_by_id):
                frame_key = self._node_key("frame", frame_id)
                if segment_key in self.nodes and frame_key in self.nodes:
                    self._add_relationship("ALIGNED_WITH", segment_key, frame_key)
            for frame_id in _temporally_near_frame_ids(segment, self.frames_by_id):
                frame_key = self._node_key("frame", frame_id)
                if segment_key in self.nodes and frame_key in self.nodes:
                    self._add_relationship("TEMPORALLY_NEAR", segment_key, frame_key)

    def _add_video(self, video_id: str, video_name: str | None = None) -> str:
        video_key = self._add_node(
            "video",
            video_id,
            ("Video",),
            _compact({"video_id": video_id, "project_id": self.project_id, "video_name": video_name}),
        )
        self._add_relationship("HAS_VIDEO", self._node_key("project", self.project_id), video_key)
        return video_key

    def _add_concept(self, canonical: str) -> str:
        aliases = self.domain_lexicon.aliases_by_canonical.get(canonical, ())
        return self._add_node(
            "concept",
            canonical,
            ("Concept",),
            _compact(
                {
                    "concept_id": canonical,
                    "canonical": canonical,
                    "aliases": list(aliases) if aliases else None,
                }
            ),
        )

    def _add_node(
        self,
        namespace: str,
        local_id: str,
        labels: tuple[str, ...],
        properties: dict[str, Any],
    ) -> str:
        key = self._node_key(namespace, local_id)
        if key not in self.nodes:
            self.nodes[key] = GraphNode(key=key, labels=labels, properties=properties)
        return key

    def _add_relationship(
        self,
        relationship_type: str,
        start_node_key: str,
        end_node_key: str,
        properties: dict[str, Any] | None = None,
        *,
        identity: str | None = None,
    ) -> str:
        if start_node_key not in self.nodes or end_node_key not in self.nodes:
            return ""
        key = _relationship_key(
            relationship_type=relationship_type,
            start_node_key=start_node_key,
            end_node_key=end_node_key,
            identity=identity,
        )
        if key not in self.relationships:
            self.relationships[key] = GraphRelationship(
                key=key,
                type=relationship_type,
                start_node_key=start_node_key,
                end_node_key=end_node_key,
                properties=properties or {},
            )
        return key

    def _fallback_video_id(self) -> str:
        if len(self.segments_by_video) == 1:
            return next(iter(self.segments_by_video))
        return self.project_id

    def _node_key(self, namespace: str, local_id: str) -> str:
        scoped_namespaces = {"frame", "visual_entity"}
        local_key = slugify(str(local_id))
        if namespace in scoped_namespaces:
            return f"{namespace}:{slugify(self.project_id)}:{local_key}"
        return f"{namespace}:{local_key}"


def _optional_segment_path(project_dir: Path, segments_path: Path | None) -> Path:
    if segments_path is not None:
        return _resolve_path(
            project_dir=project_dir,
            candidate=segments_path,
            default=project_dir / "segments" / "lecture_segments.jsonl",
        )
    try:
        return segment_artifact_path(project_dir)
    except FileNotFoundError:
        return project_dir / "segments" / "lecture_segments.jsonl"


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_optional_domain_lexicon(
    *,
    project_dir: Path,
    domain_lexicon_path: Path | None,
) -> DomainLexicon:
    try:
        return load_domain_lexicon(project_dir=project_dir, domain_lexicon_path=domain_lexicon_path)
    except FileNotFoundError:
        if domain_lexicon_path is None:
            return DomainLexicon()
        raise


def _project_id(
    *,
    manifest: dict[str, Any],
    project_dir: Path,
    segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    entities: list[VisualEntity],
) -> str:
    if manifest.get("project_id") is not None:
        return str(manifest["project_id"])
    for segment in segments:
        if segment.get("project_id") is not None:
            return str(segment["project_id"])
    for frame in frames:
        if frame.get("project_id") is not None:
            return str(frame["project_id"])
    for entity in entities:
        if entity.project_id:
            return entity.project_id
    return project_dir.name


def _availability(
    *,
    paths: dict[str, Path | None],
    counts: dict[str, int],
    selected_segments_path: Path,
) -> dict[str, Any]:
    artifacts = {
        name: {
            "available": path is not None and path.exists(),
            "path": str(path) if path is not None else None,
        }
        for name, path in sorted(paths.items())
    }
    artifacts["segments"]["selected"] = selected_segments_path.name
    return {"artifacts": artifacts, "counts": counts}


def _warnings(availability: dict[str, Any]) -> list[str]:
    artifacts = availability["artifacts"]
    warnings: list[str] = []
    if not artifacts["segments"]["available"]:
        warnings.append("segments artifact is missing; graph contains only project-level available artifacts")
    for artifact in ("frames_manifest", "visual_entities", "entity_links", "domain_lexicon"):
        if not artifacts[artifact]["available"]:
            warnings.append(f"{artifact} artifact is missing; related graph elements were skipped")
    return warnings


def _counts(*, nodes: tuple[GraphNode, ...], relationships: tuple[GraphRelationship, ...]) -> dict[str, Any]:
    return {
        "nodes": len(nodes),
        "relationships": len(relationships),
        "nodes_by_label": _count_by(nodes, lambda node: node.labels[0]),
        "relationships_by_type": _count_by(relationships, lambda rel: rel.type),
    }


def _count_by(items: tuple[Any, ...], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = key_fn(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _sorted_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        segments,
        key=lambda segment: (
            _video_id_from_segment(segment),
            _optional_float(segment.get("start_time"))
            if _optional_float(segment.get("start_time")) is not None
            else _optional_float(segment.get("timestamp_center")) or 0.0,
            str(segment.get("segment_id", "")),
        ),
    )


def _sorted_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        frames,
        key=lambda frame: (
            str(frame.get("video_id", "")),
            _optional_float(frame.get("timestamp")) is None,
            _optional_float(frame.get("timestamp")) or 0.0,
            _frame_id(frame),
        ),
    )


def _video_id_from_segment(segment: dict[str, Any]) -> str:
    return str(segment.get("video_id") or segment.get("video_name") or "video")


def _video_id_from_frame(frame: dict[str, Any], *, fallback: str) -> str:
    return str(frame.get("video_id") or frame.get("video_name") or fallback)


def _frame_id(frame: dict[str, Any]) -> str:
    if frame.get("frame_id") is not None:
        return str(frame["frame_id"])
    if frame.get("frame_path") is not None:
        return Path(str(frame["frame_path"])).stem
    return ""


def _aligned_frame_ids(
    segment: dict[str, Any],
    frames_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        {frame_id for frame_id in _segment_frame_ids(segment) if frame_id in frames_by_id}
    )


def _segments_with_string_frame_refs(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment.get("frame_refs"), list):
            normalized.append(segment)
            continue
        normalized_segment = dict(segment)
        normalized_segment["frame_refs"] = _segment_frame_ids(segment)
        normalized.append(normalized_segment)
    return normalized


def _segment_frame_ids(segment: dict[str, Any]) -> list[str]:
    frame_refs = segment.get("frame_refs")
    if not isinstance(frame_refs, list):
        return []
    return sorted(
        {
            frame_id
            for frame_id in (_frame_ref_id(frame_ref) for frame_ref in frame_refs)
            if frame_id
        }
    )


def _frame_ref_id(frame_ref: Any) -> str:
    if isinstance(frame_ref, dict):
        frame_id = frame_ref.get("frame_id")
        if frame_id is None:
            return ""
        return str(frame_id)
    if frame_ref is None:
        return ""
    return str(frame_ref)


def _temporally_near_frame_ids(
    segment: dict[str, Any],
    frames_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    aligned = set(_aligned_frame_ids(segment, frames_by_id))
    window = _segment_window(segment)
    if window is None:
        return []
    near = []
    for frame_id, frame in frames_by_id.items():
        if frame_id in aligned:
            continue
        timestamp = _optional_float(frame.get("timestamp"))
        if timestamp is not None and window[0] <= timestamp <= window[1]:
            near.append(frame_id)
    return sorted(near)


def _segment_window(segment: dict[str, Any]) -> tuple[float, float] | None:
    start = _optional_float(segment.get("start_time"))
    end = _optional_float(segment.get("end_time"))
    center = _optional_float(segment.get("timestamp_center"))
    if start is None and end is None:
        if center is None:
            return None
        return (center, center)
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def _concepts_from_segment(segment: dict[str, Any], domain_lexicon: DomainLexicon) -> list[str]:
    terms = _candidate_terms(str(segment.get("transcript_text", "")))
    candidates = segment.get("mention_candidates")
    if isinstance(candidates, list):
        terms.update(str(candidate) for candidate in candidates)
    return _canonical_concepts(terms, domain_lexicon)


def _concepts_from_entity(entity: VisualEntity, domain_lexicon: DomainLexicon) -> list[str]:
    text = " ".join(
        item
        for item in (entity.text, entity.visual_description or "")
        if item
    )
    return _canonical_concepts(_candidate_terms(text), domain_lexicon)


def _candidate_terms(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in re.finditer(r"[0-9a-zA-Z가-힣]+", text)
        if len(match.group(0)) >= 2
    }


def _canonical_concepts(terms: set[str], domain_lexicon: DomainLexicon) -> list[str]:
    concepts = {
        domain_lexicon.canonicalize(term)
        for term in terms
        if _is_concept_term(domain_lexicon.canonicalize(term))
    }
    return sorted(concepts)


def _is_concept_term(term: str) -> bool:
    return bool(term) and not term.isdigit() and term not in {
        "and",
        "are",
        "for",
        "is",
        "of",
        "on",
        "or",
        "the",
        "this",
        "that",
        "to",
        "with",
    }


def _relationship_key(
    *,
    relationship_type: str,
    start_node_key: str,
    end_node_key: str,
    identity: str | None,
) -> str:
    identity_part = f":{slugify(identity)}" if identity else ""
    return f"rel:{relationship_type}:{slugify(start_node_key)}:{slugify(end_node_key)}{identity_part}"


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


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if value == "":
            continue
        compacted[key] = _jsonable(value)
    return compacted


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
