from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from oarag.core.schemas import EduVidQARecord, LectureSegment


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def iter_records(path: Path, limit: int | None = None) -> Iterator[EduVidQARecord]:
    for idx, payload in enumerate(iter_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        yield EduVidQARecord.from_dict(payload)


def iter_lecture_segments(
    path: Path,
    project_id: str = "eduvidqa",
    limit: int | None = None,
) -> Iterator[LectureSegment]:
    for record in iter_records(path, limit=limit):
        yield LectureSegment.from_eduvidqa(record, project_id=project_id)


def load_records_by_id(path: Path, limit: int | None = None) -> dict[str, EduVidQARecord]:
    return {record.sample_id: record for record in iter_records(path, limit=limit)}
