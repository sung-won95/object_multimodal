from __future__ import annotations

from pathlib import Path
from typing import Any

from oarag.integrations.meili import MeiliClient
from oarag.retrieval.project_query import query_project


def ask_project(
    *,
    client: MeiliClient,
    index_uid: str,
    project_dir: Path,
    query: str,
    visual_index_uid: str | None = None,
    limit: int = 5,
    segments_path: Path | None = None,
    frames_manifest_path: Path | None = None,
    visual_entities_path: Path | None = None,
    entity_links_path: Path | None = None,
    domain_lexicon_path: Path | None = None,
    window_seconds: float | None = None,
    neighbor_count: int = 1,
    previous_neighbor_count: int | None = None,
    next_neighbor_count: int | None = None,
    window_before_seconds: float | None = None,
    window_after_seconds: float | None = None,
    rerank: bool = False,
    rerank_time_hint: str | None = None,
) -> dict[str, Any]:
    response = query_project(
        client=client,
        index_uid=index_uid,
        visual_index_uid=visual_index_uid,
        project_dir=project_dir,
        query=query,
        limit=limit,
        segments_path=segments_path,
        frames_manifest_path=frames_manifest_path,
        visual_entities_path=visual_entities_path,
        entity_links_path=entity_links_path,
        domain_lexicon_path=domain_lexicon_path,
        window_seconds=window_seconds,
        neighbor_count=neighbor_count,
        previous_neighbor_count=previous_neighbor_count,
        next_neighbor_count=next_neighbor_count,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
        rerank=rerank,
        rerank_time_hint=rerank_time_hint,
    )
    answer_text = build_extractive_answer(response)
    response["answer_text"] = answer_text
    response["answer"] = {
        "mode": "extractive",
        "text": answer_text,
        "source_bundle_count": len(response.get("bundles", [])),
    }
    return response


def build_extractive_answer(query_response: dict[str, Any]) -> str:
    bundles = query_response.get("bundles", [])
    if not bundles:
        return "No answer could be extracted because no evidence bundles were retrieved."

    top_bundle = bundles[0]
    evidence_window = top_bundle.get("evidence_window", {})
    target = evidence_window.get("target_segment", {})
    transcript = str(target.get("transcript_text") or "").strip()
    if not transcript:
        transcript = str(top_bundle.get("candidate", {}).get("text") or "").strip()

    entity_texts = _linked_entity_texts(top_bundle)
    parts = []
    if transcript:
        parts.append(transcript)
    if entity_texts:
        parts.append("Linked visual evidence: " + ", ".join(entity_texts))
    if not parts:
        return "The top evidence bundle did not contain extractable answer text."
    return " ".join(parts)


def _linked_entity_texts(bundle: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for linked in bundle.get("linked_entities", []):
        entity = linked.get("entity") if isinstance(linked, dict) else None
        if not isinstance(entity, dict):
            continue
        text = str(entity.get("text") or "").strip()
        if text and text not in seen:
            texts.append(text)
            seen.add(text)
    return texts
