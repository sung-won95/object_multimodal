from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


LECTURE_SEGMENT_SETTINGS = {
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
}


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

