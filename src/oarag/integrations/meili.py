from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from oarag.core.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
)


LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE = "lecture_segments_legacy_v0"
LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE = "lecture_segments_default_v1"
LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE = "lecture_segments_default_v2"
LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE = "lecture_windows_default_v1"
VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE = "visual_entities_default_v1"
DEFAULT_HYBRID_EMBEDDER_NAME = "default"
HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE = "manual_user_provided_v1"
HYBRID_EMBEDDER_CUSTOM_SETTINGS_PROFILE = "custom"
REDACTED_SETTINGS_VALUE = "<redacted>"

HYBRID_EMBEDDER_SETTINGS_PROFILES: dict[str, dict[str, Any]] = {
    HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE: {
        "source": "userProvided",
        "dimensions": 384,
    },
}

LECTURE_SEGMENT_SETTINGS_PROFILES: dict[str, dict[str, Any]] = {
    LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE: {
        "searchableAttributes": [
            "transcript_text",
            "normalized_text",
            "mention_candidates",
            "video_name",
        ],
        "filterableAttributes": [
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "video_id",
            "video_name",
            "source",
        ],
        "sortableAttributes": [
            "timestamp_center",
            "sample_index",
        ],
        "displayedAttributes": ["*"],
    },
    LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE: {
        "searchableAttributes": [
            "transcript_text",
            "normalized_text",
            "mention_candidates",
            "video_name",
            "video_id",
            "slide_id",
            "sample_id",
        ],
        "filterableAttributes": [
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "video_id",
            "video_name",
            "source",
            "segment_id",
            "sample_id",
            "slide_id",
            "sample_index",
            "start_time",
            "end_time",
            "timestamp_center",
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
        ],
        "sortableAttributes": [
            "timestamp_center",
            "start_time",
            "end_time",
            "sample_index",
        ],
        "displayedAttributes": [
            "segment_id",
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "sample_id",
            "sample_index",
            "video_id",
            "video_name",
            "start_time",
            "end_time",
            "timestamp_center",
            "timestamp_points",
            "transcript_text",
            "normalized_text",
            "slide_id",
            "frame_refs",
            "visual_entities",
            "mention_candidates",
            "source",
        ],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
        "stopWords": [],
        "synonyms": {},
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {
                "oneTypo": 5,
                "twoTypos": 9,
            },
            "disableOnAttributes": [
                "video_id",
                "slide_id",
                "sample_id",
            ],
            "disableOnWords": [],
        },
    },
    LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE: {
        "searchableAttributes": [
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
            "transcript_text",
            "normalized_text",
            "mention_candidates",
            "video_name",
            "video_id",
            "slide_id",
            "sample_id",
        ],
        "filterableAttributes": [
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "video_id",
            "video_name",
            "source",
            "segment_id",
            "sample_id",
            "slide_id",
            "sample_index",
            "start_time",
            "end_time",
            "timestamp_center",
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
        ],
        "sortableAttributes": [
            "timestamp_center",
            "start_time",
            "end_time",
            "sample_index",
        ],
        "displayedAttributes": [
            "segment_id",
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "sample_id",
            "sample_index",
            "video_id",
            "video_name",
            "start_time",
            "end_time",
            "timestamp_center",
            "timestamp_points",
            "transcript_text",
            "normalized_text",
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
            "slide_id",
            "frame_refs",
            "visual_entities",
            "mention_candidates",
            "source",
        ],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
        "stopWords": [],
        "synonyms": {},
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {
                "oneTypo": 5,
                "twoTypos": 9,
            },
            "disableOnAttributes": [
                "video_id",
                "slide_id",
                "sample_id",
            ],
            "disableOnWords": [],
        },
    },
}

VISUAL_ENTITY_SETTINGS_PROFILES: dict[str, dict[str, Any]] = {
    VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE: {
        "searchableAttributes": [
            "text",
            "visual_description",
            "entity_type",
            "frame_id",
            "source_model",
        ],
        "filterableAttributes": [
            "project_id",
            "frame_id",
            "entity_id",
            "entity_type",
            "source",
            "source_model",
            "semantic_source_fields",
            "timestamp",
        ],
        "sortableAttributes": [
            "timestamp",
            "confidence",
        ],
        "displayedAttributes": [
            "entity_id",
            "project_id",
            "frame_id",
            "timestamp",
            "frame_path",
            "bbox",
            "text",
            "entity_type",
            "confidence",
            "source",
            "semantic_source_fields",
            "visual_description",
            "position",
            "relations",
            "parser_version",
            "source_model",
        ],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
        "stopWords": [],
        "synonyms": {},
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {
                "oneTypo": 5,
                "twoTypos": 9,
            },
            "disableOnAttributes": [
                "frame_id",
                "entity_id",
                "source_model",
            ],
            "disableOnWords": [],
        },
    },
}

LECTURE_WINDOW_SETTINGS_PROFILES: dict[str, dict[str, Any]] = {
    LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE: {
        "searchableAttributes": [
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
            "transcript_window_text",
            "transcript_text",
            "visual_entities.text",
            "visual_entities.visual_description",
            "video_name",
            "video_id",
            "sample_id",
        ],
        "filterableAttributes": [
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "video_id",
            "video_name",
            "source",
            "window_id",
            "target_segment_id",
            "segment_id",
            "sample_id",
            "source_segment_ids",
            "start_time",
            "end_time",
            "timestamp_center",
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
        ],
        "sortableAttributes": [
            "timestamp_center",
            "start_time",
            "end_time",
        ],
        "displayedAttributes": [
            "window_id",
            "target_segment_id",
            "segment_id",
            "project_id",
            "dataset_name",
            "subset_name",
            "split_name",
            "sample_id",
            "sample_index",
            "video_id",
            "video_name",
            "start_time",
            "end_time",
            "timestamp_center",
            "target_start_time",
            "target_end_time",
            "target_timestamp_center",
            "source_segment_ids",
            "transcript_window_text",
            "transcript_text",
            LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
            LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
            "frame_refs",
            "visual_entities",
            "evidence_window",
            "window_config",
            "source",
        ],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
        "stopWords": [],
        "synonyms": {},
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {
                "oneTypo": 5,
                "twoTypos": 9,
            },
            "disableOnAttributes": [
                "video_id",
                "window_id",
                "target_segment_id",
                "segment_id",
                "sample_id",
            ],
            "disableOnWords": [],
        },
    },
}


def lecture_segment_settings(
    profile: str = LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
) -> dict[str, Any]:
    if profile not in LECTURE_SEGMENT_SETTINGS_PROFILES:
        valid = ", ".join(sorted(LECTURE_SEGMENT_SETTINGS_PROFILES))
        raise ValueError(f"Unknown lecture segment settings profile: {profile}. Valid profiles: {valid}")
    return copy.deepcopy(LECTURE_SEGMENT_SETTINGS_PROFILES[profile])


def lecture_segment_settings_profile_names() -> list[str]:
    return sorted(LECTURE_SEGMENT_SETTINGS_PROFILES)


def lecture_segment_settings_hash(
    settings: dict[str, Any] | None = None,
    *,
    profile: str = LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
) -> str:
    payload = lecture_segment_settings(profile) if settings is None else settings
    encoded = _canonical_settings_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lecture_segment_settings_snapshot(
    settings: dict[str, Any] | None = None,
    *,
    profile: str = LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE,
    redact_secrets: bool = False,
) -> dict[str, Any]:
    payload = lecture_segment_settings(profile) if settings is None else copy.deepcopy(settings)
    if redact_secrets:
        payload = sanitize_meili_settings_for_snapshot(payload)
    return {
        "profile": profile,
        "hash": lecture_segment_settings_hash(payload),
        "settings": payload,
    }


def visual_entity_settings(
    profile: str = VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
) -> dict[str, Any]:
    if profile not in VISUAL_ENTITY_SETTINGS_PROFILES:
        valid = ", ".join(sorted(VISUAL_ENTITY_SETTINGS_PROFILES))
        raise ValueError(f"Unknown visual entity settings profile: {profile}. Valid profiles: {valid}")
    return copy.deepcopy(VISUAL_ENTITY_SETTINGS_PROFILES[profile])


def visual_entity_settings_profile_names() -> list[str]:
    return sorted(VISUAL_ENTITY_SETTINGS_PROFILES)


def visual_entity_settings_hash(
    settings: dict[str, Any] | None = None,
    *,
    profile: str = VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
) -> str:
    payload = visual_entity_settings(profile) if settings is None else settings
    encoded = _canonical_settings_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def visual_entity_settings_snapshot(
    settings: dict[str, Any] | None = None,
    *,
    profile: str = VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE,
    redact_secrets: bool = False,
) -> dict[str, Any]:
    payload = visual_entity_settings(profile) if settings is None else copy.deepcopy(settings)
    if redact_secrets:
        payload = sanitize_meili_settings_for_snapshot(payload)
    return {
        "profile": profile,
        "hash": visual_entity_settings_hash(payload),
        "settings": payload,
    }


def lecture_window_settings(
    profile: str = LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
) -> dict[str, Any]:
    if profile not in LECTURE_WINDOW_SETTINGS_PROFILES:
        valid = ", ".join(sorted(LECTURE_WINDOW_SETTINGS_PROFILES))
        raise ValueError(f"Unknown lecture window settings profile: {profile}. Valid profiles: {valid}")
    return copy.deepcopy(LECTURE_WINDOW_SETTINGS_PROFILES[profile])


def lecture_window_settings_profile_names() -> list[str]:
    return sorted(LECTURE_WINDOW_SETTINGS_PROFILES)


def lecture_window_settings_hash(
    settings: dict[str, Any] | None = None,
    *,
    profile: str = LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
) -> str:
    payload = lecture_window_settings(profile) if settings is None else settings
    encoded = _canonical_settings_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lecture_window_settings_snapshot(
    settings: dict[str, Any] | None = None,
    *,
    profile: str = LECTURE_WINDOW_DEFAULT_SETTINGS_PROFILE,
    redact_secrets: bool = False,
) -> dict[str, Any]:
    payload = lecture_window_settings(profile) if settings is None else copy.deepcopy(settings)
    if redact_secrets:
        payload = sanitize_meili_settings_for_snapshot(payload)
    return {
        "profile": profile,
        "hash": lecture_window_settings_hash(payload),
        "settings": payload,
    }


def hybrid_embedder_settings(
    profile: str = HYBRID_EMBEDDER_MANUAL_SETTINGS_PROFILE,
    *,
    embedder_name: str = DEFAULT_HYBRID_EMBEDDER_NAME,
    dimensions: int | None = None,
) -> dict[str, Any]:
    if profile not in HYBRID_EMBEDDER_SETTINGS_PROFILES:
        valid = ", ".join(sorted(HYBRID_EMBEDDER_SETTINGS_PROFILES))
        raise ValueError(f"Unknown hybrid embedder settings profile: {profile}. Valid profiles: {valid}")
    name = str(embedder_name or "").strip()
    if not name:
        raise ValueError("hybrid embedder name must not be empty")
    embedder = copy.deepcopy(HYBRID_EMBEDDER_SETTINGS_PROFILES[profile])
    if dimensions is not None:
        embedder["dimensions"] = _validate_positive_int(
            dimensions,
            field_name="hybrid embedder dimensions",
        )
    settings = {"embedders": {name: embedder}}
    _validate_hybrid_embedder_settings(settings)
    return settings


def hybrid_embedder_settings_profile_names() -> list[str]:
    return sorted(HYBRID_EMBEDDER_SETTINGS_PROFILES)


def normalize_hybrid_embedder_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("Hybrid embedder settings must be a JSON object")
    if "embedders" in settings:
        payload = {"embedders": copy.deepcopy(settings["embedders"])}
    else:
        payload = {"embedders": copy.deepcopy(settings)}
    _validate_hybrid_embedder_settings(payload)
    return payload


def load_hybrid_embedder_settings(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse hybrid embedder config JSON at {path}: {exc}") from exc
    return normalize_hybrid_embedder_settings(loaded)


def normalize_query_vector(
    vector: Sequence[Any],
    *,
    dimensions: int | None = None,
    field_name: str = "query vector",
) -> list[float]:
    if isinstance(vector, str | bytes) or not isinstance(vector, Sequence):
        raise ValueError(f"{field_name} must be a JSON array of finite numbers")

    expected_dimensions: int | None = None
    if dimensions is not None:
        expected_dimensions = _validate_positive_int(
            dimensions,
            field_name=f"{field_name} dimensions",
        )

    normalized: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field_name} item {index} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} item {index} must be a finite number")
        normalized.append(number)

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if expected_dimensions is not None and len(normalized) != expected_dimensions:
        raise ValueError(
            f"{field_name} dimension mismatch: expected {expected_dimensions}, "
            f"got {len(normalized)}"
        )
    return normalized


def merge_hybrid_embedder_settings(
    settings: dict[str, Any],
    hybrid_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(settings)
    if hybrid_settings is None:
        return merged
    normalized = normalize_hybrid_embedder_settings(hybrid_settings)
    merged["embedders"] = normalized["embedders"]
    return merged


def hybrid_embedder_settings_hash(settings: dict[str, Any]) -> str:
    payload = sanitize_meili_settings_for_snapshot(normalize_hybrid_embedder_settings(settings))
    encoded = _canonical_settings_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hybrid_embedder_settings_snapshot(
    settings: dict[str, Any],
    *,
    profile: str = HYBRID_EMBEDDER_CUSTOM_SETTINGS_PROFILE,
) -> dict[str, Any]:
    payload = sanitize_meili_settings_for_snapshot(normalize_hybrid_embedder_settings(settings))
    return {
        "profile": profile,
        "hash": hybrid_embedder_settings_hash(payload),
        "settings": payload,
    }


def sanitize_meili_settings_for_snapshot(settings: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_setting_value(copy.deepcopy(settings))
    if not isinstance(sanitized, dict):
        raise ValueError("Meilisearch settings snapshot must be a JSON object")
    return sanitized


def _canonical_settings_json(settings: dict[str, Any]) -> str:
    return json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_hybrid_embedder_settings(settings: dict[str, Any]) -> None:
    embedders = settings.get("embedders")
    if not isinstance(embedders, dict) or not embedders:
        raise ValueError("Hybrid embedder settings require a non-empty 'embedders' object")
    for embedder_name, embedder in embedders.items():
        name = str(embedder_name or "").strip()
        if not name:
            raise ValueError("Hybrid embedder settings contain an empty embedder name")
        if not isinstance(embedder, dict):
            raise ValueError(f"Hybrid embedder '{name}' settings must be a JSON object")
        source = embedder.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Hybrid embedder '{name}' requires a non-empty 'source'")
        if source == "userProvided":
            if "documentTemplate" in embedder or "documentTemplateMaxBytes" in embedder:
                raise ValueError(
                    f"Hybrid embedder '{name}' uses source 'userProvided', which cannot "
                    "include documentTemplate or documentTemplateMaxBytes"
                )
            _validate_positive_int(
                embedder.get("dimensions"),
                field_name=f"hybrid embedder '{name}' dimensions",
            )


def _validate_positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _sanitize_setting_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_secret_settings_key(key):
        return REDACTED_SETTINGS_VALUE
    if isinstance(value, dict):
        return {str(item_key): _sanitize_setting_value(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_setting_value(item) for item in value]
    return value


def _is_secret_settings_key(key: str) -> bool:
    normalized = key.replace("_", "").replace("-", "").lower()
    return normalized in {
        "apikey",
        "authorization",
        "bearertoken",
        "password",
        "secret",
        "token",
    }


LECTURE_SEGMENT_SETTINGS = lecture_segment_settings()


@dataclass(frozen=True)
class MeiliTask:
    uid: int
    status: str | None = None


class MeiliTaskError(RuntimeError):
    def __init__(self, task_uid: int, status: str, payload: dict[str, Any]) -> None:
        self.task_uid = task_uid
        self.status = status
        self.payload = payload
        super().__init__(f"Meilisearch task {task_uid} ended with status {status}: {payload}")


class MeiliClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def create_index(self, uid: str, primary_key: str) -> MeiliTask | None:
        try:
            payload = self._request("POST", "/indexes", {"uid": uid, "primaryKey": primary_key})
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                return None
            raise
        return MeiliTask(uid=int(payload["taskUid"]))

    def delete_index(self, uid: str) -> MeiliTask | None:
        try:
            payload = self._request("DELETE", f"/indexes/{quote(uid)}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return MeiliTask(uid=int(payload["taskUid"]))

    def update_settings(self, index_uid: str, settings: dict[str, Any]) -> MeiliTask:
        payload = self._request("PATCH", f"/indexes/{quote(index_uid)}/settings", settings)
        return MeiliTask(uid=int(payload["taskUid"]))

    def get_settings(self, index_uid: str) -> dict[str, Any]:
        return self._request("GET", f"/indexes/{quote(index_uid)}/settings")

    def add_documents(self, index_uid: str, documents: list[dict[str, Any]]) -> MeiliTask:
        payload = self._request("POST", f"/indexes/{quote(index_uid)}/documents", documents)
        return MeiliTask(uid=int(payload["taskUid"]))

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int = 10,
        show_ranking_score: bool = True,
        hybrid: dict[str, Any] | None = None,
        vector: list[float] | None = None,
        show_ranking_score_details: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "q": query,
            "limit": limit,
            "showRankingScore": show_ranking_score,
        }
        if hybrid is not None:
            payload["hybrid"] = hybrid
        if vector is not None:
            payload["vector"] = normalize_query_vector(vector)
        if show_ranking_score_details:
            payload["showRankingScoreDetails"] = True
        return self._request("POST", f"/indexes/{quote(index_uid)}/search", payload)

    def wait_task(
        self,
        task: MeiliTask | None,
        timeout_seconds: float = 60.0,
        ignored_error_codes: Iterable[str] = (),
    ) -> dict[str, Any] | None:
        if task is None:
            return None
        ignored_error_codes = set(ignored_error_codes)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            payload = self._request("GET", f"/tasks/{task.uid}")
            status = payload.get("status")
            if status in {"succeeded", "failed", "canceled"}:
                if status != "succeeded":
                    if _task_error_code(payload) in ignored_error_codes:
                        return payload
                    raise MeiliTaskError(task.uid, status, payload)
                return payload
            time.sleep(0.2)
        raise TimeoutError(f"Timed out waiting for Meilisearch task {task.uid}")

    def _request(self, method: str, path: str, payload: Any | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))


def _task_error_code(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    return None


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")
