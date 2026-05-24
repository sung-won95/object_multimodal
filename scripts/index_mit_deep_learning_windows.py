#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL
from oarag.integrations.meili import (
    DEFAULT_HYBRID_EMBEDDER_NAME,
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    MeiliClient,
)
from oarag.retrieval.project_index import (
    LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
    build_project_windows,
    index_project_windows,
)


DEFAULT_MANIFEST = REPO_ROOT / "eval" / "mit_deep_learning_stt" / "benchmark_matrix_manifest.json"
DEFAULT_INDEX = "mit_deep_learning_stt_windows"


def main() -> None:
    args = parse_args()
    manifest_path = _resolve_input_path(args.manifest)
    manifest = _read_manifest(manifest_path)
    suites = _matrix_suites(manifest)
    projects = _projects_from_suites(suites=suites, manifest_dir=manifest_path.parent)

    client = None if args.build_only else MeiliClient(base_url=args.url, api_key=args.api_key)
    project_summaries: list[dict[str, Any]] = []
    for index, project in enumerate(projects):
        build_summary = None
        if not args.index_only:
            build_summary = build_project_windows(
                project_dir=project["project_dir"],
                window_seconds=args.window_seconds,
                neighbor_count=args.neighbor_count,
                previous_neighbor_count=args.previous_neighbor_count,
                next_neighbor_count=args.next_neighbor_count,
                window_before_seconds=args.window_before_seconds,
                window_after_seconds=args.window_after_seconds,
            )

        index_summary = None
        if client is not None:
            index_summary = index_project_windows(
                client,
                index_uid=args.index,
                project_dir=project["project_dir"],
                batch_size=args.batch_size,
                reset=args.reset and index == 0,
                windows=LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
                settings_profile=args.settings_profile,
                hybrid_embedder_profile=args.hybrid_embedder_profile,
                hybrid_embedder_name=args.hybrid_embedder_name,
                hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
                hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
            )

        project_summaries.append(
            _public_project_summary(
                project=project,
                build_summary=build_summary,
                index_summary=index_summary,
            )
        )

    summary = {
        "manifest": _display_path(manifest_path),
        "suite_count": len(suites),
        "project_count": len(projects),
        "window_artifact": LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH.as_posix(),
        "index": args.index,
        "build_enabled": not args.index_only,
        "index_enabled": not args.build_only,
        "reset_requested": args.reset,
        "projects": project_summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MIT Deep Learning lecture_windows.jsonl artifacts and index them "
            "into a shared Meilisearch window index."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="Only write segments/lecture_windows.jsonl files; do not contact Meilisearch.",
    )
    mode.add_argument(
        "--index-only",
        action="store_true",
        help="Only index existing segments/lecture_windows.jsonl files.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--url", default=DEFAULT_MEILI_URL)
    parser.add_argument("--api-key", default=DEFAULT_MEILI_API_KEY)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--settings-profile",
        default=LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
        help="Meilisearch settings profile for lecture window documents.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        help="Include segments whose timestamps overlap this many seconds around each target.",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=1,
        help="Neighboring segments to include on each side when --window-seconds is omitted.",
    )
    parser.add_argument(
        "--previous-neighbor-count",
        type=int,
        help="Neighboring segments to include before the target when using neighbor mode.",
    )
    parser.add_argument(
        "--next-neighbor-count",
        type=int,
        help="Neighboring segments to include after the target when using neighbor mode.",
    )
    parser.add_argument(
        "--window-before-seconds",
        type=float,
        help="Seconds to include before each target when using time-window mode.",
    )
    parser.add_argument(
        "--window-after-seconds",
        type=float,
        help="Seconds to include after each target when using time-window mode.",
    )
    parser.add_argument(
        "--hybrid-embedder-profile",
        help="Optional hybrid embedder settings profile for window_hybrid retrieval.",
    )
    parser.add_argument(
        "--hybrid-embedder-name",
        default=DEFAULT_HYBRID_EMBEDDER_NAME,
        help="Meilisearch embedder name used by hybrid retrieval.",
    )
    parser.add_argument(
        "--hybrid-embedder-dimensions",
        type=int,
        help="Optional userProvided embedder dimensions.",
    )
    parser.add_argument(
        "--hybrid-embedder-live-smoke",
        action="store_true",
        help="Ask Meilisearch to validate the configured hybrid embedder after settings update.",
    )
    return parser.parse_args()


def _resolve_input_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in manifest: {path}")
    return payload


def _matrix_suites(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    suites = [
        suite
        for suite in manifest.get("suites", [])
        if isinstance(suite, dict) and suite.get("type") == "retrieval_answer_matrix"
    ]
    if not suites:
        raise ValueError("No retrieval_answer_matrix suites found in manifest.")
    return suites


def _projects_from_suites(
    *, suites: list[dict[str, Any]], manifest_dir: Path
) -> list[dict[str, Any]]:
    projects_by_dir: dict[Path, dict[str, Any]] = {}
    for suite in suites:
        project_dir_value = suite.get("project_dir")
        if project_dir_value is None:
            raise ValueError(f"Suite {suite.get('suite_id')} is missing project_dir.")
        project_dir = Path(str(project_dir_value)).expanduser()
        if not project_dir.is_absolute():
            project_dir = (manifest_dir / project_dir).resolve()
        else:
            project_dir = project_dir.resolve()
        if project_dir not in projects_by_dir:
            projects_by_dir[project_dir] = {
                "project_dir": project_dir,
                "suite_ids": [],
                "video_ids": [],
            }
        projects_by_dir[project_dir]["suite_ids"].append(str(suite.get("suite_id") or ""))
        if suite.get("video_id"):
            projects_by_dir[project_dir]["video_ids"].append(str(suite["video_id"]))
    return list(projects_by_dir.values())


def _public_project_summary(
    *,
    project: dict[str, Any],
    build_summary: dict[str, Any] | None,
    index_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "project_ref": _display_path(project["project_dir"]),
        "suite_ids": project["suite_ids"],
        "video_ids": project["video_ids"],
    }
    if build_summary is not None:
        counts = build_summary.get("counts") if isinstance(build_summary.get("counts"), dict) else {}
        summary["build"] = {
            "windows_total": counts.get("windows_total"),
            "segments_total": counts.get("segments_total"),
            "frames_total": counts.get("frames_total"),
            "visual_entities_total": counts.get("visual_entities_total"),
        }
    if index_summary is not None:
        summary["indexing"] = {
            "indexed_documents": index_summary.get("indexed_documents"),
            "indexed_batches": index_summary.get("indexed_batches"),
            "settings_profile": index_summary.get("settings_profile"),
            "settings_hash": index_summary.get("settings_hash"),
            "hybrid_embedder_profile": index_summary.get("hybrid_embedder_profile"),
            "hybrid_embedder_hash": index_summary.get("hybrid_embedder_hash"),
        }
    return summary


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"path:{_short_hash(path.resolve().as_posix())}"


def _short_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
