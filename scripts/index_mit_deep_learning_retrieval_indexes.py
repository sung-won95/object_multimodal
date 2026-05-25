#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL  # noqa: E402
from oarag.integrations.meili import DEFAULT_HYBRID_EMBEDDER_NAME, MeiliClient  # noqa: E402
from oarag.retrieval.project_index import (  # noqa: E402
    LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
    build_project_windows,
    index_project_segments,
    index_project_visual_entities,
    index_project_windows,
)


DEFAULT_MANIFEST = REPO_ROOT / "eval" / "mit_deep_learning_stt" / "benchmark_matrix_manifest.json"
DEFAULT_SEGMENT_INDEX = "mit_deep_learning_stt_segments"
DEFAULT_WINDOW_INDEX = "mit_deep_learning_stt_windows"
DEFAULT_VISUAL_INDEX = "mit_deep_learning_stt_visual_entities"
STAGES = ("segment", "window", "visual")
WINDOW_OPTION_FIELDS = (
    "window_seconds",
    "neighbor_count",
    "previous_neighbor_count",
    "next_neighbor_count",
    "window_before_seconds",
    "window_after_seconds",
)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    manifest_path = _resolve_input_path(args.manifest)
    manifest = _read_manifest(manifest_path)
    suites = _matrix_suites(manifest)
    projects = _projects_from_suites(suites=suites, manifest_dir=manifest_path.parent)

    client = MeiliClient(base_url=args.url, api_key=args.api_key)
    configured_once = {stage: False for stage in STAGES}
    project_summaries: list[dict[str, Any]] = []
    for project in projects:
        project_summary: dict[str, Any] = {
            "project_ref": _display_path(project["project_dir"]),
            "suite_ids": project["suite_ids"],
            "video_ids": project["video_ids"],
            "stages": {},
        }
        if "segment" in args.stages:
            configure_index = not configured_once["segment"]
            index_summary = index_project_segments(
                client,
                index_uid=args.segment_index,
                project_dir=project["project_dir"],
                batch_size=args.batch_size,
                reset=args.reset and configure_index,
                configure_index=configure_index,
                hybrid_embedder_profile=args.hybrid_embedder_profile,
                hybrid_embedder_name=args.hybrid_embedder_name,
                hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
                hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke and configure_index,
                vector_manifest=_project_vector_manifest(
                    project=project,
                    explicit_path=args.segment_vector_manifest,
                    default_relative=args.segment_vector_manifest_relative,
                ),
            )
            configured_once["segment"] = True
            project_summary["stages"]["segment"] = _public_indexing_summary(index_summary)

        if "window" in args.stages:
            build_summary = build_project_windows(
                project_dir=project["project_dir"],
                **project["window_options"],
            )
            configure_index = not configured_once["window"]
            index_summary = index_project_windows(
                client,
                index_uid=args.window_index,
                project_dir=project["project_dir"],
                batch_size=args.batch_size,
                reset=args.reset and configure_index,
                configure_index=configure_index,
                windows=LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
                hybrid_embedder_profile=args.hybrid_embedder_profile,
                hybrid_embedder_name=args.hybrid_embedder_name,
                hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
                hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke and configure_index,
                vector_manifest=_project_vector_manifest(
                    project=project,
                    explicit_path=args.window_vector_manifest,
                    default_relative=args.window_vector_manifest_relative,
                ),
            )
            configured_once["window"] = True
            project_summary["stages"]["window"] = {
                "build": _public_window_build_summary(build_summary),
                "indexing": _public_indexing_summary(index_summary),
            }

        if "visual" in args.stages:
            configure_index = not configured_once["visual"]
            index_summary = index_project_visual_entities(
                client,
                index_uid=args.visual_index,
                project_dir=project["project_dir"],
                batch_size=args.batch_size,
                reset=args.reset and configure_index,
                configure_index=configure_index,
                hybrid_embedder_profile=args.hybrid_embedder_profile,
                hybrid_embedder_name=args.hybrid_embedder_name,
                hybrid_embedder_dimensions=args.hybrid_embedder_dimensions,
                hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke and configure_index,
                vector_manifest=_project_vector_manifest(
                    project=project,
                    explicit_path=args.visual_vector_manifest,
                    default_relative=args.visual_vector_manifest_relative,
                ),
            )
            configured_once["visual"] = True
            project_summary["stages"]["visual"] = _public_indexing_summary(index_summary)

        project_summaries.append(project_summary)

    summary = {
        "manifest": _display_path(manifest_path),
        "suite_count": len(suites),
        "project_count": len(projects),
        "stages": list(args.stages),
        "indexes": {
            "segment": args.segment_index,
            "window": args.window_index,
            "visual": args.visual_index,
        },
        "batch_size": args.batch_size,
        "reset_requested": args.reset,
        "index_configured_once": {
            stage: configured_once[stage] for stage in args.stages
        },
        "hybrid_embedder": _public_hybrid_request_summary(args),
        "window_artifact": LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH.as_posix(),
        "projects": project_summaries,
        "totals": _aggregate_totals(project_summaries),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index all MIT Deep Learning manifest projects into shared segment, "
            "window, and visual Meilisearch retrieval indexes."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--url", default=DEFAULT_MEILI_URL)
    parser.add_argument("--api-key", default=DEFAULT_MEILI_API_KEY)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--segment-index", default=DEFAULT_SEGMENT_INDEX)
    parser.add_argument("--window-index", default=DEFAULT_WINDOW_INDEX)
    parser.add_argument("--visual-index", default=DEFAULT_VISUAL_INDEX)
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=STAGES,
        help=(
            "Comma-separated stages to run: segment, window, visual, or all. "
            "Default: segment,window,visual."
        ),
    )
    parser.add_argument(
        "--hybrid-embedder-profile",
        help="Optional hybrid embedder settings profile for shared vector retrieval.",
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
        help="Validate the configured hybrid embedder once per shared index.",
    )
    parser.add_argument(
        "--segment-vector-manifest",
        type=Path,
        help="Single real embedding vector manifest for all segment documents.",
    )
    parser.add_argument(
        "--window-vector-manifest",
        type=Path,
        help="Single real embedding vector manifest for all window documents.",
    )
    parser.add_argument(
        "--visual-vector-manifest",
        type=Path,
        help="Single real embedding vector manifest for all visual entity documents.",
    )
    parser.add_argument(
        "--segment-vector-manifest-relative",
        type=Path,
        help="Per-project segment vector manifest path relative to each project dir.",
    )
    parser.add_argument(
        "--window-vector-manifest-relative",
        type=Path,
        help="Per-project window vector manifest path relative to each project dir.",
    )
    parser.add_argument(
        "--visual-vector-manifest-relative",
        type=Path,
        help="Per-project visual vector manifest path relative to each project dir.",
    )
    return parser.parse_args(argv)


def parse_stages(value: str) -> tuple[str, ...]:
    aliases = {
        "segment": ("segment",),
        "segments": ("segment",),
        "window": ("window",),
        "windows": ("window",),
        "visual": ("visual",),
        "visuals": ("visual",),
        "visual_entity": ("visual",),
        "visual_entities": ("visual",),
        "all": STAGES,
    }
    selected: list[str] = []
    for raw_token in value.split(","):
        token = raw_token.strip().lower().replace("-", "_")
        if not token:
            continue
        stages = aliases.get(token)
        if stages is None:
            valid = ", ".join(sorted(aliases))
            raise argparse.ArgumentTypeError(f"Unknown stage '{raw_token}'. Valid stages: {valid}")
        for stage in stages:
            if stage not in selected:
                selected.append(stage)
    if not selected:
        raise argparse.ArgumentTypeError("At least one stage is required.")
    return tuple(selected)


def _resolve_input_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _project_vector_manifest(
    *,
    project: dict[str, Any],
    explicit_path: Path | None,
    default_relative: Path | None,
) -> Path | None:
    if explicit_path is not None:
        return _resolve_input_path(explicit_path)
    if default_relative is None:
        return None
    candidate = default_relative.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project["project_dir"] / candidate).resolve()


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
        window_options = _suite_window_options(suite)
        if project_dir not in projects_by_dir:
            projects_by_dir[project_dir] = {
                "project_dir": project_dir,
                "suite_ids": [],
                "video_ids": [],
                "window_options": window_options,
            }
        elif projects_by_dir[project_dir]["window_options"] != window_options:
            raise ValueError(
                f"Project {_display_path(project_dir)} has conflicting window options "
                "across manifest suites."
            )
        _append_unique(projects_by_dir[project_dir]["suite_ids"], str(suite.get("suite_id") or ""))
        if suite.get("video_id"):
            _append_unique(projects_by_dir[project_dir]["video_ids"], str(suite["video_id"]))
    return list(projects_by_dir.values())


def _suite_window_options(suite: dict[str, Any]) -> dict[str, Any]:
    return {
        field: suite[field]
        for field in WINDOW_OPTION_FIELDS
        if suite.get(field) is not None
    }


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _public_window_build_summary(build_summary: dict[str, Any]) -> dict[str, Any]:
    counts = build_summary.get("counts") if isinstance(build_summary.get("counts"), dict) else {}
    return {
        "windows_total": counts.get("windows_total"),
        "segments_total": counts.get("segments_total"),
        "frames_total": counts.get("frames_total"),
        "visual_entities_total": counts.get("visual_entities_total"),
        "window_frame_refs": counts.get("window_frame_refs"),
        "window_visual_entities": counts.get("window_visual_entities"),
        "window_config": build_summary.get("window_config"),
    }


def _public_indexing_summary(index_summary: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "index": index_summary.get("index"),
        "indexed_documents": index_summary.get("indexed_documents"),
        "indexed_batches": index_summary.get("indexed_batches"),
        "configure_index": index_summary.get("configure_index"),
        "reset": index_summary.get("reset"),
        "settings_profile": index_summary.get("settings_profile"),
        "settings_hash": index_summary.get("settings_hash"),
    }
    for count_key in (
        "embedded_visual_entities",
        "semantic_source_field_counts",
        "source_counts",
    ):
        if count_key in index_summary:
            summary[count_key] = index_summary[count_key]
    vector_summary = index_summary.get("document_vectors")
    if isinstance(vector_summary, dict):
        summary["document_vectors"] = _public_vector_summary(vector_summary)
    if "hybrid_embedder_profile" in index_summary:
        summary["hybrid_embedder_profile"] = index_summary.get("hybrid_embedder_profile")
        summary["hybrid_embedder_hash"] = index_summary.get("hybrid_embedder_hash")
        summary["hybrid_embedder_live_smoke"] = index_summary.get("hybrid_embedder_live_smoke")
    return summary


def _public_vector_summary(vector_summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "enabled",
        "source",
        "generator",
        "purpose",
        "quality_claim",
        "embedder_names",
        "dimensions_by_embedder",
        "documents_seen",
        "documents_with_vectors",
        "generated_vector_count",
        "existing_vector_count",
        "manifest_vector_count",
        "missing_vector_count",
        "expected_vector_count",
        "manifest",
    )
    return {key: vector_summary.get(key) for key in keys}


def _public_hybrid_request_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile": args.hybrid_embedder_profile,
        "embedder_name": args.hybrid_embedder_name,
        "dimensions": args.hybrid_embedder_dimensions,
        "live_smoke_requested": args.hybrid_embedder_live_smoke,
    }


def _aggregate_totals(project_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for project_summary in project_summaries:
        stages = project_summary.get("stages")
        if not isinstance(stages, dict):
            continue
        for stage_name, stage_summary in stages.items():
            if stage_name == "window":
                build_summary = stage_summary.get("build") if isinstance(stage_summary, dict) else {}
                indexing_summary = (
                    stage_summary.get("indexing") if isinstance(stage_summary, dict) else {}
                )
                stage_totals = totals.setdefault(
                    stage_name,
                    {"built_documents": 0, "indexed_documents": 0, "indexed_batches": 0},
                )
                stage_totals["built_documents"] += _int_or_zero(build_summary.get("windows_total"))
                stage_totals["indexed_documents"] += _int_or_zero(
                    indexing_summary.get("indexed_documents")
                )
                stage_totals["indexed_batches"] += _int_or_zero(
                    indexing_summary.get("indexed_batches")
                )
                continue
            stage_totals = totals.setdefault(
                stage_name,
                {"indexed_documents": 0, "indexed_batches": 0},
            )
            if isinstance(stage_summary, dict):
                stage_totals["indexed_documents"] += _int_or_zero(
                    stage_summary.get("indexed_documents")
                )
                stage_totals["indexed_batches"] += _int_or_zero(
                    stage_summary.get("indexed_batches")
                )
    return totals


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"path:{_short_hash(path.resolve().as_posix())}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
