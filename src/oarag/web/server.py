from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from oarag.integrations.meili import MeiliClient
from oarag.retrieval.answer import ask_project


ClientFactory = Callable[[], Any]


def build_ask_project_response(
    *,
    client: Any,
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
    return ask_project(
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


def build_ask_project_response_from_payload(
    *,
    payload: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    index_uid = str(payload["index"])
    project_dir = Path(str(payload["project_dir"]))
    query = str(payload["query"])
    return build_ask_project_response(
        client=client,
        index_uid=index_uid,
        visual_index_uid=_optional_str(payload.get("visual_index")),
        project_dir=project_dir,
        query=query,
        limit=int(payload.get("limit", 5)),
        segments_path=_optional_path(payload.get("segments")),
        frames_manifest_path=_optional_path(payload.get("frames_manifest")),
        visual_entities_path=_optional_path(payload.get("visual_entities")),
        entity_links_path=_optional_path(payload.get("entity_links")),
        domain_lexicon_path=_optional_path(payload.get("domain_lexicon")),
        window_seconds=_optional_float(payload.get("window_seconds")),
        neighbor_count=int(payload.get("neighbor_count", 1)),
        previous_neighbor_count=_optional_int(payload.get("previous_neighbor_count")),
        next_neighbor_count=_optional_int(payload.get("next_neighbor_count")),
        window_before_seconds=_optional_float(payload.get("window_before_seconds")),
        window_after_seconds=_optional_float(payload.get("window_after_seconds")),
        rerank=bool(payload.get("rerank", False)),
        rerank_time_hint=_optional_str(payload.get("rerank_time_hint")),
    )


def create_app(
    *,
    client_factory: ClientFactory | None = None,
    default_url: str = "http://127.0.0.1:7700",
    default_api_key: str | None = None,
) -> Any:
    try:
        from fastapi import FastAPI
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is required to create the web app") from exc

    app = FastAPI(title="Object-Aligned RAG")
    factory = client_factory or (lambda: MeiliClient(base_url=default_url, api_key=default_api_key))

    @app.post("/ask-project")
    def ask_project_route(payload: dict[str, Any]) -> dict[str, Any]:
        return build_ask_project_response_from_payload(payload=payload, client=factory())

    return app


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
