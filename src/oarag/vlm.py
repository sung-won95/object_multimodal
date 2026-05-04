from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .io import write_json, write_jsonl
from .schemas import (
    VLM_FRAME_CANDIDATES_ARTIFACT,
    VLM_PROJECT_MANIFEST_SECTION,
    VLM_VISUAL_OBSERVATIONS_ARTIFACT,
    VLMFrameCandidate,
    VLMVisualObservation,
    build_vlm_project_manifest_fields,
    slugify,
)


DEFAULT_VLM_BACKEND = "deterministic"
VLM_SUCCESS_STATUS = "success"
VLM_SKIPPED_RESUMED_STATUS = "skipped_resumed"
VLM_BACKEND_FAILURE_STATUS = "backend_failure"
VLM_PARSE_FAILURE_STATUS = "parse_failure"


class VLMParseError(ValueError):
    """Raised when a backend response cannot be converted into observation records."""


@dataclass(frozen=True)
class VLMFrameInput:
    project_id: str
    video_id: str
    frame_id: str
    frame_path: str | None = None
    timestamp: float | None = None
    segment_id: str | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMRunConfig:
    backend: str
    model: str
    device: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: f"vlm_{int(time.time())}")

    @property
    def model_version(self) -> str | None:
        value = self.options.get("model_version")
        return str(value) if value is not None else None


class VLMBackend(Protocol):
    backend: str

    def run(
        self,
        *,
        frames: list[VLMFrameInput],
        config: VLMRunConfig,
    ) -> list[VLMVisualObservation]:
        ...


@dataclass(frozen=True)
class DeterministicVLMBackend:
    backend: str = DEFAULT_VLM_BACKEND

    def run(
        self,
        *,
        frames: list[VLMFrameInput],
        config: VLMRunConfig,
    ) -> list[VLMVisualObservation]:
        confidence = _optional_float(config.options.get("confidence"))
        if confidence is None:
            confidence = 1.0
        observation_type = str(config.options.get("observation_type") or "frame_summary")
        description_prefix = str(
            config.options.get("description_prefix") or "Deterministic VLM observation"
        )
        detected_text = _optional_str(config.options.get("detected_text"))
        backend_failure_frame_ids = _option_str_set(
            config.options.get("backend_fail_frame_ids") or config.options.get("fail_frame_ids")
        )
        parse_failure_frame_ids = _option_str_set(
            config.options.get("parse_fail_frame_ids")
            or config.options.get("parse_failure_frame_ids")
        )

        observations: list[VLMVisualObservation] = []
        for frame in frames:
            if frame.frame_id in parse_failure_frame_ids:
                raise VLMParseError(
                    f"Deterministic parse failure requested for frame_id={frame.frame_id}"
                )
            if frame.frame_id in backend_failure_frame_ids:
                raise RuntimeError(
                    f"Deterministic backend failure requested for frame_id={frame.frame_id}"
                )

            timestamp_text = "unknown" if frame.timestamp is None else f"{frame.timestamp:.3f}s"
            observations.append(
                VLMVisualObservation(
                    observation_id=f"obs_{slugify(frame.frame_id)}_0001",
                    project_id=frame.project_id,
                    video_id=frame.video_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    segment_id=frame.segment_id,
                    backend=self.backend,
                    source_model=config.model,
                    model_version=config.model_version,
                    confidence=confidence,
                    status=VLM_SUCCESS_STATUS,
                    observation_type=observation_type,
                    visual_description=(
                        f"{description_prefix}: {frame.frame_id} at {timestamp_text}"
                    ),
                    detected_text=detected_text,
                    attributes={
                        "run_id": config.run_id,
                        "device": config.device,
                        "frame_path": frame.frame_path,
                        "rank": frame.rank,
                    },
                    metadata={"backend_options": dict(config.options)},
                )
            )
        return observations


@dataclass(frozen=True)
class CommandVLMBackend:
    backend: str = "command"

    def run(
        self,
        *,
        frames: list[VLMFrameInput],
        config: VLMRunConfig,
    ) -> list[VLMVisualObservation]:
        raw_command = config.options.get("command")
        if raw_command is None:
            raise ValueError("VLM command backend requires vlm_options.command")

        observations: list[VLMVisualObservation] = []
        for frame in frames:
            command = _format_command(raw_command, frame=frame, config=config)
            result = _run_command(command, frame_id=frame.frame_id)
            observations.extend(
                _observations_from_command_stdout(
                    result.stdout,
                    frame=frame,
                    config=config,
                    backend=self.backend,
                )
            )
        return observations


BackendFactory = Callable[[], VLMBackend]


VLM_BACKEND_REGISTRY: dict[str, BackendFactory] = {
    DEFAULT_VLM_BACKEND: DeterministicVLMBackend,
    "command": CommandVLMBackend,
}


def available_vlm_backends() -> tuple[str, ...]:
    return tuple(sorted(VLM_BACKEND_REGISTRY))


def make_vlm_backend(backend: str) -> VLMBackend:
    normalized = backend.strip().lower()
    factory = VLM_BACKEND_REGISTRY.get(normalized)
    if factory is None:
        available = ", ".join(available_vlm_backends())
        raise ValueError(f"Unsupported VLM backend: {backend}. Available backends: {available}")
    return factory()


def run_vlm(
    *,
    project_dir: Path,
    backend: str = DEFAULT_VLM_BACKEND,
    model: str | None,
    device: str | None = None,
    options: Mapping[str, Any] | None = None,
    frames_manifest_path: Path | None = None,
    frame_candidates_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    resolved_model = _validate_model(model)
    resolved_project_dir = project_dir.expanduser().resolve()
    if not resolved_project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {resolved_project_dir}")

    resolved_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=manifest_path,
        default=resolved_project_dir / "manifests" / "project_manifest.json",
    )
    resolved_frames_manifest_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=frames_manifest_path,
        default=resolved_project_dir / "manifests" / "frames_manifest.jsonl",
    )
    default_frame_candidates_path = (
        resolved_project_dir / "manifests" / "vlm_frame_candidates.jsonl"
    )
    resolved_frame_candidates_path = (
        _resolve_path(
            project_dir=resolved_project_dir,
            candidate=frame_candidates_path,
            default=default_frame_candidates_path,
        )
        if frame_candidates_path is not None or default_frame_candidates_path.exists()
        else None
    )
    resolved_output_path = _resolve_path(
        project_dir=resolved_project_dir,
        candidate=output_path,
        default=resolved_project_dir / "manifests" / "vlm_visual_observations.jsonl",
    )

    manifest = _read_json_object(resolved_manifest_path) if resolved_manifest_path.exists() else {}
    project_id = _project_id_from_manifest_or_dir(manifest, resolved_project_dir)
    video_id = _video_id_from_manifest(manifest, project_id)
    frames = _load_frame_inputs(
        frames_manifest_path=resolved_frames_manifest_path,
        frame_candidates_path=resolved_frame_candidates_path,
        project_id=project_id,
        video_id=video_id,
    )

    runner = make_vlm_backend(backend)
    config = VLMRunConfig(
        backend=runner.backend,
        model=resolved_model,
        device=_optional_str(device),
        options=dict(options or {}),
    )

    started_at = time.perf_counter()
    existing_observations = (
        _read_existing_observations(resolved_output_path) if resume else []
    )
    existing_frame_ids = {observation.frame_id for observation in existing_observations}
    new_observations: list[VLMVisualObservation] = []
    frame_statuses: Counter[str] = Counter()
    frame_latencies: list[float] = []

    for frame in frames:
        if frame.frame_id in existing_frame_ids:
            frame_statuses[VLM_SKIPPED_RESUMED_STATUS] += 1
            continue

        frame_started_at = time.perf_counter()
        try:
            frame_observations = _normalize_success_observations(
                runner.run(frames=[frame], config=config)
            )
        except VLMParseError as exc:
            frame_observations = [
                _failure_observation(
                    frame=frame,
                    config=config,
                    backend=runner.backend,
                    status=VLM_PARSE_FAILURE_STATUS,
                    reason=str(exc),
                )
            ]
            frame_statuses[VLM_PARSE_FAILURE_STATUS] += 1
        except Exception as exc:
            frame_observations = [
                _failure_observation(
                    frame=frame,
                    config=config,
                    backend=runner.backend,
                    status=VLM_BACKEND_FAILURE_STATUS,
                    reason=str(exc),
                )
            ]
            frame_statuses[VLM_BACKEND_FAILURE_STATUS] += 1
        else:
            frame_statuses[VLM_SUCCESS_STATUS] += 1

        frame_latencies.append(time.perf_counter() - frame_started_at)
        new_observations.extend(frame_observations)

    observations = existing_observations + new_observations
    elapsed_seconds = round(time.perf_counter() - started_at, 4)
    average_frame_latency_seconds = (
        round(sum(frame_latencies) / len(frame_latencies), 4) if frame_latencies else None
    )
    failure_reasons = {
        VLM_BACKEND_FAILURE_STATUS: frame_statuses[VLM_BACKEND_FAILURE_STATUS],
        VLM_PARSE_FAILURE_STATUS: frame_statuses[VLM_PARSE_FAILURE_STATUS],
    }
    failure_count = sum(failure_reasons.values())
    skipped_count = frame_statuses[VLM_SKIPPED_RESUMED_STATUS]
    manifest_status = "completed_with_errors" if failure_count else "completed"

    write_jsonl(resolved_output_path, [observation.to_dict() for observation in observations])
    _update_project_manifest(
        manifest_path=resolved_manifest_path,
        backend=runner.backend,
        model=resolved_model,
        model_version=config.model_version,
        status=manifest_status,
        device=config.device,
        options=config.options,
        run_id=config.run_id,
        output_path=resolved_output_path,
        frame_candidates_path=resolved_frame_candidates_path,
        frame_candidate_count=len(frames) if resolved_frame_candidates_path else 0,
        observation_count=len(observations),
        failures={"count": failure_count, "reasons": failure_reasons},
        skips={"count": skipped_count, "reasons": {"resume": skipped_count}},
        frame_status_counts=dict(frame_statuses),
        elapsed_seconds=elapsed_seconds,
        average_frame_latency_seconds=average_frame_latency_seconds,
    )

    return {
        "project_id": project_id,
        "video_id": video_id,
        "backend": runner.backend,
        "source_model": resolved_model,
        "model_version": config.model_version,
        "device": config.device,
        "run_id": config.run_id,
        "status": manifest_status,
        "paths": {
            "project_dir": str(resolved_project_dir),
            "frames_manifest": str(resolved_frames_manifest_path),
            "project_manifest": str(resolved_manifest_path),
            "vlm_visual_observations": str(resolved_output_path),
            **(
                {"vlm_frame_candidates": str(resolved_frame_candidates_path)}
                if resolved_frame_candidates_path is not None
                else {}
            ),
        },
        "counts": {
            "frames_total": len(frames),
            "frames_processed": len(frames) - skipped_count,
            "frames_succeeded": frame_statuses[VLM_SUCCESS_STATUS],
            "frames_failed": failure_count,
            "frames_skipped_resumed": skipped_count,
            "vlm_visual_observations": len(observations),
        },
        "frame_status_counts": dict(frame_statuses),
        "elapsed_seconds": elapsed_seconds,
        "average_frame_latency_seconds": average_frame_latency_seconds,
    }


def _validate_model(model: str | None) -> str:
    resolved = _optional_str(model)
    if resolved is None:
        raise ValueError("--vlm-model is required for VLM execution")
    return resolved


def _load_frame_inputs(
    *,
    frames_manifest_path: Path,
    frame_candidates_path: Path | None,
    project_id: str,
    video_id: str,
) -> list[VLMFrameInput]:
    frame_lookup = {
        frame.frame_id: frame
        for frame in _read_frames_manifest(
            frames_manifest_path,
            project_id=project_id,
            video_id=video_id,
        )
    }
    if frame_candidates_path is None:
        return list(frame_lookup.values())

    candidates = _read_frame_candidates(frame_candidates_path)
    frames: list[VLMFrameInput] = []
    for candidate in candidates:
        frame = frame_lookup.get(candidate.frame_id)
        frames.append(
            VLMFrameInput(
                project_id=candidate.project_id or project_id,
                video_id=candidate.video_id or video_id,
                frame_id=candidate.frame_id,
                frame_path=candidate.frame_path or (frame.frame_path if frame else None),
                timestamp=candidate.timestamp if candidate.timestamp is not None else (
                    frame.timestamp if frame else None
                ),
                segment_id=candidate.segment_id,
                rank=candidate.rank,
                metadata={"candidate_status": candidate.status},
            )
        )
    return frames


def _read_frames_manifest(
    path: Path,
    *,
    project_id: str,
    video_id: str,
) -> list[VLMFrameInput]:
    rows = _read_jsonl(path, missing_message="frames_manifest.jsonl not found")
    frames: list[VLMFrameInput] = []
    for payload in rows:
        frame_path = _optional_str(payload.get("frame_path"))
        frame_id = _optional_str(payload.get("frame_id")) or (
            Path(frame_path).stem if frame_path else None
        )
        if frame_id is None:
            raise ValueError(f"Missing frame_id in {path}")
        frames.append(
            VLMFrameInput(
                project_id=_optional_str(payload.get("project_id")) or project_id,
                video_id=_optional_str(payload.get("video_id")) or video_id,
                frame_id=frame_id,
                frame_path=frame_path,
                timestamp=_optional_float(payload.get("timestamp")),
                segment_id=_optional_str(payload.get("segment_id")),
            )
        )
    return frames


def _read_frame_candidates(path: Path) -> list[VLMFrameCandidate]:
    return [VLMFrameCandidate.from_dict(row) for row in _read_jsonl(path)]


def _read_existing_observations(path: Path) -> list[VLMVisualObservation]:
    if not path.exists():
        return []
    return [VLMVisualObservation.from_dict(row) for row in _read_jsonl(path)]


def _normalize_success_observations(
    observations: list[VLMVisualObservation],
) -> list[VLMVisualObservation]:
    return [
        replace(observation, status=VLM_SUCCESS_STATUS)
        if observation.status in {"", "completed"}
        else observation
        for observation in observations
    ]


def _failure_observation(
    *,
    frame: VLMFrameInput,
    config: VLMRunConfig,
    backend: str,
    status: str,
    reason: str,
) -> VLMVisualObservation:
    return VLMVisualObservation(
        observation_id=f"obs_{slugify(frame.frame_id)}_{status}",
        project_id=frame.project_id,
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        segment_id=frame.segment_id,
        backend=backend,
        source_model=config.model,
        model_version=config.model_version,
        confidence=None,
        status=status,
        observation_type="frame_error",
        visual_description="",
        attributes={
            "run_id": config.run_id,
            "device": config.device,
            "frame_path": frame.frame_path,
            "rank": frame.rank,
        },
        metadata={"failure_reason": reason},
    )


def _observations_from_command_stdout(
    stdout: str,
    *,
    frame: VLMFrameInput,
    config: VLMRunConfig,
    backend: str,
) -> list[VLMVisualObservation]:
    stripped = stdout.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise VLMParseError(
            f"VLM command returned invalid JSON for frame_id={frame.frame_id}"
        ) from exc

    if isinstance(payload, dict) and isinstance(payload.get("observations"), list):
        raw_observations = payload["observations"]
    elif isinstance(payload, list):
        raw_observations = payload
    else:
        raw_observations = [payload]

    observations: list[VLMVisualObservation] = []
    for index, raw in enumerate(raw_observations, start=1):
        if not isinstance(raw, dict):
            raise VLMParseError(
                f"VLM command returned non-object observation for {frame.frame_id}"
            )
        frame_id = _optional_str(raw.get("frame_id"))
        if frame_id is not None and frame_id != frame.frame_id:
            raise VLMParseError(
                f"VLM command frame_id mismatch for {frame.frame_id}: returned {frame_id}"
            )
        observations.append(
            VLMVisualObservation(
                observation_id=_optional_str(raw.get("observation_id"))
                or f"obs_{slugify(frame.frame_id)}_{index:04d}",
                project_id=_optional_str(raw.get("project_id")) or frame.project_id,
                video_id=_optional_str(raw.get("video_id")) or frame.video_id,
                frame_id=frame.frame_id,
                timestamp=_optional_float(raw.get("timestamp"))
                if raw.get("timestamp") is not None
                else frame.timestamp,
                segment_id=_optional_str(raw.get("segment_id")) or frame.segment_id,
                backend=_optional_str(raw.get("backend")) or backend,
                source_model=_optional_str(raw.get("source_model")) or config.model,
                model_version=_optional_str(raw.get("model_version")) or config.model_version,
                confidence=_optional_float(raw.get("confidence")),
                status=_optional_str(raw.get("status")) or VLM_SUCCESS_STATUS,
                observation_type=_optional_str(raw.get("observation_type")) or "frame_summary",
                visual_description=str(raw.get("visual_description") or ""),
                detected_text=_optional_str(raw.get("detected_text")),
                bbox=_float_mapping_or_none(raw.get("bbox")),
                position=_optional_mapping(raw.get("position")),
                attributes=_mapping(raw.get("attributes")),
                relations=_dict_list(raw.get("relations")),
                metadata={
                    **_mapping(raw.get("metadata")),
                    "run_id": config.run_id,
                },
            )
        )
    return observations


def _format_command(
    raw_command: Any,
    *,
    frame: VLMFrameInput,
    config: VLMRunConfig,
) -> list[str]:
    if isinstance(raw_command, str):
        parts = shlex.split(raw_command)
    elif isinstance(raw_command, list):
        parts = [str(part) for part in raw_command]
    else:
        raise ValueError("vlm_options.command must be a string or list")
    if not parts:
        raise ValueError("vlm_options.command must not be empty")

    values = {
        "project_id": frame.project_id,
        "video_id": frame.video_id,
        "frame_id": frame.frame_id,
        "frame_path": frame.frame_path or "",
        "timestamp": "" if frame.timestamp is None else str(frame.timestamp),
        "segment_id": frame.segment_id or "",
        "model": config.model,
        "device": config.device or "",
        "run_id": config.run_id,
    }
    try:
        return [part.format(**values) for part in parts]
    except KeyError as exc:
        raise ValueError(f"Unknown VLM command placeholder: {exc.args[0]}") from exc


def _run_command(command: list[str], *, frame_id: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"VLM backend command not found for frame_id={frame_id}: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(
            f"VLM backend command failed for frame_id={frame_id}: "
            f"{' '.join(command)} exited with {exc.returncode}\n{stderr}"
        ) from exc


def _update_project_manifest(
    *,
    manifest_path: Path,
    backend: str,
    model: str,
    model_version: str | None,
    status: str,
    device: str | None,
    options: dict[str, Any],
    run_id: str,
    output_path: Path,
    frame_candidates_path: Path | None,
    frame_candidate_count: int,
    observation_count: int,
    failures: dict[str, Any] | None,
    skips: dict[str, Any] | None,
    frame_status_counts: dict[str, int],
    elapsed_seconds: float,
    average_frame_latency_seconds: float | None,
) -> None:
    payload = _read_json_object(manifest_path) if manifest_path.exists() else {}
    artifact_paths = {VLM_VISUAL_OBSERVATIONS_ARTIFACT: str(output_path)}
    if frame_candidates_path is not None:
        artifact_paths[VLM_FRAME_CANDIDATES_ARTIFACT] = str(frame_candidates_path)

    vlm_fields = build_vlm_project_manifest_fields(
        backend=backend,
        source_model=model,
        model_version=model_version,
        status=status,
        settings={
            "run_id": run_id,
            "device": device,
            "options": options,
        },
        artifact_paths=artifact_paths,
        counts={
            VLM_FRAME_CANDIDATES_ARTIFACT: frame_candidate_count,
            VLM_VISUAL_OBSERVATIONS_ARTIFACT: observation_count,
        },
        failures=failures,
        skips=skips,
    )
    section = vlm_fields[VLM_PROJECT_MANIFEST_SECTION]
    section["frame_status_counts"] = frame_status_counts
    section["elapsed_seconds"] = elapsed_seconds
    section["average_frame_latency_seconds"] = average_frame_latency_seconds

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    artifacts.update(vlm_fields["artifacts"])

    counts = payload.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        payload["counts"] = counts
    counts.update(vlm_fields["counts"])

    payload[VLM_PROJECT_MANIFEST_SECTION] = section
    write_json(manifest_path, payload)


def _read_jsonl(path: Path, *, missing_message: str = "JSONL input not found") -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{missing_message}: {resolved}")

    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {resolved}:{line_number}")
            rows.append(payload)
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def _resolve_path(*, project_dir: Path, candidate: Path | None, default: Path) -> Path:
    if candidate is None:
        return default
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def _project_id_from_manifest_or_dir(manifest: dict[str, Any], project_dir: Path) -> str:
    return _optional_str(manifest.get("project_id")) or project_dir.name


def _video_id_from_manifest(manifest: dict[str, Any], project_id: str) -> str:
    return _optional_str(manifest.get("video_id")) or project_id


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _option_str_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _float_mapping_or_none(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        number = _optional_float(raw)
        if number is not None:
            parsed[str(key)] = number
    return parsed or None


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
