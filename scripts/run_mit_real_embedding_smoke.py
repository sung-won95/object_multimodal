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

from oarag.core.config import DEFAULT_MEILI_API_KEY, DEFAULT_MEILI_URL  # noqa: E402
from oarag.core.io import write_json  # noqa: E402
from oarag.embeddings.cache import JsonlEmbeddingCache  # noqa: E402
from oarag.embeddings.manifest import build_vector_manifest, load_input_records  # noqa: E402
from oarag.embeddings.providers import (  # noqa: E402
    DEFAULT_EMBEDDING_API_KEY_ENV,
    DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
    DeterministicFixtureEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
)
from oarag.evaluation.benchmark import run_benchmark  # noqa: E402
from oarag.integrations.meili import DEFAULT_HYBRID_EMBEDDER_NAME, MeiliClient  # noqa: E402
from oarag.retrieval.project_index import (  # noqa: E402
    LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
    build_project_windows,
    index_project_segments,
    index_project_visual_entities,
    index_project_windows,
)


DEFAULT_MANIFEST = REPO_ROOT / "eval" / "mit_deep_learning_stt" / "benchmark_matrix_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "mit_deep_learning_eval" / "real_embedding_smoke"
REAL_PROVIDER_DENYLIST = {"deterministic_fixture", "local_hash_v1"}
HYBRID_VARIANTS = {"hybrid", "window_hybrid"}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    manifest_path = _resolve_input_path(args.manifest)
    matrix_manifest = _read_json(manifest_path)
    selected_suites = _select_suites(
        _matrix_suites(matrix_manifest),
        suite_id=args.suite_id,
        max_suites=args.max_suites,
    )
    if not selected_suites:
        raise ValueError("no suites selected")

    output_root = _resolve_output_root(args.output_root)
    vector_root = output_root / "vector_manifests"
    cache_dir = output_root / "embedding_cache"
    benchmark_dir = output_root / "benchmark"
    vector_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    provider = _provider_from_args(args)
    provider_summary = provider.public_config()
    client = MeiliClient(base_url=args.url, api_key=args.api_key)
    manifest_dir = manifest_path.parent
    indexed_projects: list[dict[str, Any]] = []
    suite_entries: list[dict[str, Any]] = []
    for suite in selected_suites:
        project_dir = _resolve_suite_project_dir(suite, manifest_dir=manifest_dir)
        suite_id = str(suite.get("suite_id") or project_dir.name)
        suite_vector_dir = vector_root / suite_id
        suite_vector_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{suite_id}.jsonl"

        window_summary = build_project_windows(
            project_dir=project_dir,
            **_suite_window_options(suite),
        )
        segment_manifest = suite_vector_dir / "segment_vectors.json"
        window_manifest = suite_vector_dir / "window_vectors.json"
        visual_manifest = suite_vector_dir / "visual_vectors.json"
        query_manifest = suite_vector_dir / "query_vectors.json"

        segment_summary = build_vector_manifest(
            records=load_input_records(
                project_dir / "segments" / "lecture_segments_aligned.jsonl",
                input_format="jsonl",
            ),
            provider=provider,
            output_path=segment_manifest,
            id_field="segment_id",
            text_fields=["semantic_text", "transcript_text", "visual_entities.visual_description"],
            kind="document",
            embedder=args.embedder_name,
            batch_size=args.embedding_batch_size,
            cache=JsonlEmbeddingCache(cache_path),
            source_label="mitdl_segment",
        )
        window_summary_vectors = build_vector_manifest(
            records=load_input_records(
                project_dir / LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
                input_format="jsonl",
            ),
            provider=provider,
            output_path=window_manifest,
            id_field="window_id",
            text_fields=["semantic_text", "transcript_window_text", "visual_entities.visual_description"],
            kind="document",
            embedder=args.embedder_name,
            batch_size=args.embedding_batch_size,
            cache=JsonlEmbeddingCache(cache_path),
            source_label="mitdl_window",
        )
        visual_summary = build_vector_manifest(
            records=load_input_records(
                project_dir / "manifests" / "visual_entities.jsonl",
                input_format="jsonl",
            ),
            provider=provider,
            output_path=visual_manifest,
            id_field="entity_id",
            text_fields=["text", "visual_description", "entity_type"],
            kind="document",
            embedder=args.embedder_name,
            batch_size=args.embedding_batch_size,
            cache=JsonlEmbeddingCache(cache_path),
            source_label="mitdl_visual",
        )
        query_summary = build_vector_manifest(
            records=load_input_records(_resolve_suite_queries(suite, manifest_dir=manifest_dir)),
            provider=provider,
            output_path=query_manifest,
            id_field="query_id",
            text_fields=["query_text"],
            kind="query",
            embedder=args.embedder_name,
            batch_size=args.embedding_batch_size,
            cache=JsonlEmbeddingCache(cache_path),
            source_label="mitdl_query",
        )

        index_prefix = f"{args.index_prefix}_{_safe_id(suite_id)}"
        segment_index = f"{index_prefix}_segments"
        window_index = f"{index_prefix}_windows"
        visual_index = f"{index_prefix}_visual"
        segment_index_summary = index_project_segments(
            client,
            index_uid=segment_index,
            project_dir=project_dir,
            batch_size=args.index_batch_size,
            reset=True,
            hybrid_embedder_profile=args.hybrid_embedder_profile,
            hybrid_embedder_name=args.embedder_name,
            hybrid_embedder_dimensions=args.dimensions,
            hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
            vector_manifest=segment_manifest,
        )
        window_index_summary = index_project_windows(
            client,
            index_uid=window_index,
            project_dir=project_dir,
            batch_size=args.index_batch_size,
            reset=True,
            windows=LECTURE_WINDOW_ARTIFACT_RELATIVE_PATH,
            hybrid_embedder_profile=args.hybrid_embedder_profile,
            hybrid_embedder_name=args.embedder_name,
            hybrid_embedder_dimensions=args.dimensions,
            hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
            vector_manifest=window_manifest,
        )
        visual_index_summary = index_project_visual_entities(
            client,
            index_uid=visual_index,
            project_dir=project_dir,
            batch_size=args.index_batch_size,
            reset=True,
            hybrid_embedder_profile=args.hybrid_embedder_profile,
            hybrid_embedder_name=args.embedder_name,
            hybrid_embedder_dimensions=args.dimensions,
            hybrid_embedder_live_smoke=args.hybrid_embedder_live_smoke,
            vector_manifest=visual_manifest,
        )

        indexed_projects.append(
            {
                "suite_id": suite_id,
                "project_ref": f"project:{_short_hash(str(project_dir))}",
                "window_build": _public_window_summary(window_summary),
                "vector_manifests": {
                    "segment": segment_summary,
                    "window": window_summary_vectors,
                    "visual": visual_summary,
                    "query": query_summary,
                },
                "indexes": {
                    "segment": _public_index_summary(segment_index_summary),
                    "window": _public_index_summary(window_index_summary),
                    "visual": _public_index_summary(visual_index_summary),
                },
            }
        )
        suite_entries.append(
            _benchmark_suite_entry(
                suite,
                project_dir=project_dir,
                queries_path=_resolve_suite_queries(suite, manifest_dir=manifest_dir),
                segment_index=segment_index,
                window_index=window_index,
                visual_index=visual_index,
                query_manifest=query_manifest,
                embedder_name=args.embedder_name,
                dimensions=args.dimensions,
            )
        )

    benchmark_manifest_path = output_root / "benchmark_manifest.json"
    benchmark_manifest = {
        "schema_version": "retrieval-answer-ablation-matrix-v1",
        "run_id": args.run_id,
        "deltas": matrix_manifest.get("deltas", [5, 10, 15]),
        "suites": suite_entries,
    }
    write_json(benchmark_manifest_path, benchmark_manifest)
    benchmark = run_benchmark(
        client=client,
        manifest_path=benchmark_manifest_path,
        output_dir=benchmark_dir,
        repo_root=REPO_ROOT,
    )
    validation = validate_run(
        indexed_projects=indexed_projects,
        metrics_path=benchmark.metrics_path,
        query_results_path=benchmark.query_results_path,
        semantic_smoke_path=benchmark.semantic_smoke_path,
        require_real_provider=not args.allow_fixture_provider,
    )
    paper_ready = validation["ok"] and not args.allow_fixture_provider
    summary = {
        "schema_version": "mit-real-embedding-smoke-v1",
        "run_id": args.run_id,
        "paper_ready": paper_ready,
        "plumbing_ready": validation["ok"],
        "provider": _public_provider_summary(provider_summary),
        "suite_count": len(suite_entries),
        "artifacts": _public_artifact_summary(
            output_root=output_root,
            benchmark_manifest_path=benchmark_manifest_path,
            metrics_path=benchmark.metrics_path,
            query_results_path=benchmark.query_results_path,
            semantic_smoke_path=benchmark.semantic_smoke_path,
            summary_path=benchmark.summary_path,
        ),
        "indexed_projects": indexed_projects,
        "validation": validation,
        "privacy": {
            "raw_vectors": "excluded_from_summary",
            "credential": "excluded",
            "raw_transcripts": "excluded_from_summary",
            "fixture_mode": "not_paper_ready" if args.allow_fixture_provider else "disabled",
        },
    }
    summary_path = output_root / "real_embedding_smoke_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not validation["ok"]:
        raise SystemExit(2)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MIT Deep Learning RAG smoke/eval with explicit embedding manifests."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite-id", help="Run only one suite id.")
    parser.add_argument("--max-suites", type=int, help="Run the first N suites after filtering.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="mit_real_embedding_smoke")
    parser.add_argument("--url", default=DEFAULT_MEILI_URL)
    parser.add_argument("--api-key", default=DEFAULT_MEILI_API_KEY)
    parser.add_argument(
        "--provider",
        choices=["openai-compatible", "sentence-transformers", "deterministic-fixture"],
        default="openai-compatible",
    )
    parser.add_argument("--model", help="Embedding model. Required for openai-compatible unless env is set.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", default=DEFAULT_EMBEDDING_API_KEY_ENV)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="For sentence-transformers, load only locally cached model files.",
    )
    parser.add_argument("--embedder-name", default=DEFAULT_HYBRID_EMBEDDER_NAME)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--index-batch-size", type=int, default=500)
    parser.add_argument("--index-prefix", default="mit_real_embedding")
    parser.add_argument("--hybrid-embedder-profile", default="manual_user_provided_v1")
    parser.add_argument("--hybrid-embedder-live-smoke", action="store_true")
    parser.add_argument(
        "--allow-fixture-provider",
        action="store_true",
        help="Allow deterministic-fixture provider for plumbing tests. Not paper-ready.",
    )
    return parser.parse_args(argv)


def validate_run(
    *,
    indexed_projects: list[dict[str, Any]],
    metrics_path: Path,
    query_results_path: Path,
    semantic_smoke_path: Path,
    require_real_provider: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    provider_ids: set[str] = set()
    for project in indexed_projects:
        vector_manifests = project.get("vector_manifests") if isinstance(project, dict) else {}
        if isinstance(vector_manifests, dict):
            for name, summary in vector_manifests.items():
                provider = _provider_id_from_summary(summary)
                if provider:
                    provider_ids.add(provider)
                if require_real_provider and provider in REAL_PROVIDER_DENYLIST:
                    failures.append(f"{name} manifest uses non-real provider '{provider}'")
        indexes = project.get("indexes") if isinstance(project, dict) else {}
        if isinstance(indexes, dict):
            for name, summary in indexes.items():
                vector_summary = summary.get("document_vectors") if isinstance(summary, dict) else {}
                failures.extend(_validate_document_vector_summary(name, vector_summary))

    semantic_smoke = _read_json(semantic_smoke_path)
    semantic_live = semantic_smoke.get("semantic_live_smoke")
    semantic_smoke_ok = isinstance(semantic_live, dict) and semantic_live.get("ok") is True
    if not semantic_smoke_ok:
        failures.append("semantic_smoke did not pass")
    if int((semantic_smoke.get("counts") or {}).get("query_vector_configured_count") or 0) <= 0:
        failures.append("semantic_smoke has no configured query vectors")

    query_vector_sources: set[str] = set()
    with query_results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_vector = (row.get("config") or {}).get("query_vector")
            if isinstance(query_vector, dict) and query_vector.get("configured") is True:
                source = str(query_vector.get("source") or "")
                query_vector_sources.add(source)
                if source != "manifest":
                    failures.append(f"query vector source is '{source}', expected manifest")

    metrics = _read_json(metrics_path)
    return {
        "ok": not failures,
        "failures": failures,
        "provider_ids": sorted(provider_ids),
        "query_vector_sources": sorted(query_vector_sources),
        "semantic_smoke_ok": semantic_smoke_ok,
        "query_count": metrics.get("query_count"),
    }


def _validate_document_vector_summary(name: str, vector_summary: Any) -> list[str]:
    failures: list[str] = []
    if not isinstance(vector_summary, dict):
        return [f"{name} index is missing document vector summary"]
    if vector_summary.get("generator") is not None:
        failures.append(f"{name} index used generator {vector_summary.get('generator')}")
    if vector_summary.get("purpose") != "real_embedding_manifest":
        failures.append(f"{name} index purpose is not real_embedding_manifest")
    if vector_summary.get("generated_vector_count") != 0:
        failures.append(f"{name} index generated fallback vectors")
    if vector_summary.get("manifest_vector_count") != vector_summary.get("expected_vector_count"):
        failures.append(f"{name} index manifest vector count mismatch")
    if vector_summary.get("missing_vector_count") not in (0, None):
        failures.append(f"{name} index has missing vectors")
    return failures


def _provider_from_args(args: argparse.Namespace):
    if args.provider == "deterministic-fixture":
        return DeterministicFixtureEmbeddingProvider(
            model=args.model or "deterministic_fixture_v1",
            dimensions=args.dimensions,
        )
    if args.provider == "sentence-transformers":
        return SentenceTransformersEmbeddingProvider(
            model=args.model or DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
            dimensions=args.dimensions,
            local_files_only=args.local_files_only,
        )
    return OpenAICompatibleEmbeddingProvider.from_env(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        dimensions=args.dimensions,
    )


def _benchmark_suite_entry(
    suite: dict[str, Any],
    *,
    project_dir: Path,
    queries_path: Path,
    segment_index: str,
    window_index: str,
    visual_index: str,
    query_manifest: Path,
    embedder_name: str,
    dimensions: int,
) -> dict[str, Any]:
    entry = dict(suite)
    entry["project_dir"] = str(project_dir)
    entry["queries"] = str(queries_path)
    entry["index"] = segment_index
    entry["window_index"] = window_index
    entry["visual_index"] = visual_index
    entry["hybrid_query_vector_manifest"] = str(query_manifest)
    entry["hybrid_query_vector_name_field"] = "query_id"
    entry["hybrid_query_vector_embedder"] = embedder_name
    entry["hybrid_query_vector_dimensions"] = dimensions
    variants = entry.get("variants")
    if not isinstance(variants, list) or not variants:
        variants = ["segment_lexical", "hybrid", "window_hybrid"]
    entry["variants"] = [variant for variant in variants if str(variant) in HYBRID_VARIANTS or str(variant) == "segment_lexical"]
    if not any(str(variant) in HYBRID_VARIANTS for variant in entry["variants"]):
        entry["variants"].append("hybrid")
    return entry


def _select_suites(
    suites: list[dict[str, Any]],
    *,
    suite_id: str | None,
    max_suites: int | None,
) -> list[dict[str, Any]]:
    selected = [
        suite
        for suite in suites
        if suite_id is None or str(suite.get("suite_id") or "") == suite_id
    ]
    if max_suites is not None:
        if max_suites <= 0:
            raise ValueError("max_suites must be positive")
        selected = selected[:max_suites]
    return selected


def _matrix_suites(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        suite
        for suite in manifest.get("suites", [])
        if isinstance(suite, dict) and suite.get("type") == "retrieval_answer_matrix"
    ]


def _resolve_suite_project_dir(suite: dict[str, Any], *, manifest_dir: Path) -> Path:
    value = suite.get("project_dir")
    if value is None:
        raise ValueError(f"suite {suite.get('suite_id')} is missing project_dir")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def _resolve_suite_queries(suite: dict[str, Any], *, manifest_dir: Path) -> Path:
    value = suite.get("queries")
    if value is None:
        raise ValueError(f"suite {suite.get('suite_id')} is missing queries")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def _suite_window_options(suite: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "window_seconds",
        "neighbor_count",
        "previous_neighbor_count",
        "next_neighbor_count",
        "window_before_seconds",
        "window_after_seconds",
    )
    return {field: suite[field] for field in fields if suite.get(field) is not None}


def _public_index_summary(index_summary: dict[str, Any]) -> dict[str, Any]:
    vector_summary = index_summary.get("document_vectors")
    return {
        "index_ref": f"index:{_short_hash(str(index_summary.get('index')))}",
        "indexed_documents": index_summary.get("indexed_documents"),
        "indexed_batches": index_summary.get("indexed_batches"),
        "document_vectors": _public_vector_summary(vector_summary)
        if isinstance(vector_summary, dict)
        else None,
    }


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
    return {key: vector_summary.get(key) for key in keys if key in vector_summary}


def _public_window_summary(summary: dict[str, Any]) -> dict[str, Any]:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    return {
        "segments_total": counts.get("segments_total"),
        "windows_total": counts.get("windows_total"),
        "frames_total": counts.get("frames_total"),
        "visual_entities_total": counts.get("visual_entities_total"),
    }


def _public_provider_summary(provider: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": provider.get("provider"),
        "model": provider.get("model"),
        "dimensions": provider.get("dimensions"),
        "api_base": provider.get("api_base"),
        "local_files_only": provider.get("local_files_only"),
    }


def _public_artifact_summary(
    *,
    output_root: Path,
    benchmark_manifest_path: Path,
    metrics_path: Path,
    query_results_path: Path,
    semantic_smoke_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "output_ref": f"artifact:{_short_hash(str(output_root))}",
        "benchmark_manifest": benchmark_manifest_path.name,
        "metrics": metrics_path.name,
        "query_results": query_results_path.name,
        "semantic_smoke": semantic_smoke_path.name,
        "summary": summary_path.name,
    }


def _provider_id_from_summary(summary: Any) -> str | None:
    if not isinstance(summary, dict):
        return None
    provider = summary.get("provider")
    if not isinstance(provider, dict):
        return None
    value = provider.get("provider")
    return str(value) if value not in (None, "") else None


def _resolve_input_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_output_root(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _safe_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in cleaned.split("_") if part)[:48] or "suite"


def _short_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
