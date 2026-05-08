from __future__ import annotations

import copy
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from oarag.core.schemas import (
    LECTURE_SEGMENT_SEMANTIC_SOURCE_FIELDS_FIELD,
    LECTURE_SEGMENT_SEMANTIC_TEXT_FIELD,
)


LECTURE_SEGMENT_LEGACY_SETTINGS_PROFILE = "lecture_segments_legacy_v0"
LECTURE_SEGMENT_PRE_SEMANTIC_SETTINGS_PROFILE = "lecture_segments_default_v1"
LECTURE_SEGMENT_DEFAULT_SETTINGS_PROFILE = "lecture_segments_default_v2"
VISUAL_ENTITY_DEFAULT_SETTINGS_PROFILE = "visual_entities_default_v1"

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
) -> dict[str, Any]:
    payload = lecture_segment_settings(profile) if settings is None else copy.deepcopy(settings)
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
) -> dict[str, Any]:
    payload = visual_entity_settings(profile) if settings is None else copy.deepcopy(settings)
    return {
        "profile": profile,
        "hash": visual_entity_settings_hash(payload),
        "settings": payload,
    }


def _canonical_settings_json(settings: dict[str, Any]) -> str:
    return json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


LECTURE_SEGMENT_SETTINGS = lecture_segment_settings()


@dataclass(frozen=True)
class MeiliTask:
    uid: int
    status: str | None = None


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
    ) -> dict[str, Any]:
        payload = {
            "q": query,
            "limit": limit,
            "showRankingScore": show_ranking_score,
        }
        return self._request("POST", f"/indexes/{quote(index_uid)}/search", payload)

    def wait_task(self, task: MeiliTask | None, timeout_seconds: float = 60.0) -> dict[str, Any] | None:
        if task is None:
            return None
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            payload = self._request("GET", f"/tasks/{task.uid}")
            status = payload.get("status")
            if status in {"succeeded", "failed", "canceled"}:
                if status != "succeeded":
                    raise RuntimeError(f"Meilisearch task {task.uid} ended with status {status}: {payload}")
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


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")
