# MIT Deep Learning STT Seed Evaluation Set

이 폴더는 STT로 구축한 MIT 6.7960 Deep Learning RAG를 평가하기 위한 1차 seed 평가셋이다.
각 row는 질문, gold segment timestamp, modality label, frame/link evidence 힌트를 포함한다.

## Files

- `queries.csv`: 통합 평가셋
- `queries.jsonl`: 동일 내용을 JSONL로 저장한 버전
- `suites/*.csv`: `benchmark-retrieval`이 강의별로 읽는 CSV
- `benchmark_manifest.json`: 바로 실행 가능한 retrieval benchmark manifest
- `benchmark_matrix_manifest.json`: segment/window retrieval-answer matrix manifest
- `summary.json`: row count, split, modality 분포 요약

## Scope

- Queries: 48
- Videos: 24
- Splits: {'dev': 24, 'test': 24}
- Modalities: {'audio': 16, 'both': 32}
- Question types: {'audio_reasoning': 3, 'concept_explanation': 7, 'definition_lookup': 6, 'formula_table_lookup': 2, 'multimodal_grounded': 29, 'visual_object_reference': 1}

## Use

```bash
python -m oarag benchmark-retrieval \
  --manifest eval/mit_deep_learning_stt/benchmark_manifest.json
```

## Window Matrix Use

Matrix benchmark는 24개 lecture suite를 `segment_lexical`, `window`, `window_hybrid` 변형으로 실행한다.
실행 전 Meilisearch가 떠 있어야 하며, manifest가 가리키는 MIT project artifacts와 기존 segment/visual entity index가 준비되어 있어야 한다.

```bash
python scripts/index_mit_deep_learning_windows.py --build-only
python scripts/index_mit_deep_learning_windows.py --reset
python -m oarag benchmark-retrieval \
  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json
```

`--build-only`는 각 project artifact 아래 `segments/lecture_windows.jsonl`만 만든다.
두 번째 명령은 같은 window artifact를 `mit_deep_learning_stt_windows` Meilisearch index로 적재한다.
`window_hybrid`를 실제로 쓰려면 window index의 Meilisearch embedder 설정이 segment index와 호환되어야 하므로, 필요한 경우 스크립트의 `--hybrid-embedder-profile`과 `--hybrid-embedder-dimensions`를 함께 지정한다.

## Caveat

이 데이터셋은 STT segment와 대표 frame alignment에서 만든 seed label이다.
정식 논문 수치로 쓰기 전에는 사람이 timestamp, modality, visual entity를 한 번 더 검수하는 것이 좋다.

Source: MIT OpenCourseWare 6.7960 Deep Learning, Fall 2024
License: CC BY-NC-SA 4.0
