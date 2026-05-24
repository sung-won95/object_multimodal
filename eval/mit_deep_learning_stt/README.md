# MIT Deep Learning STT Seed Evaluation Set

이 폴더는 STT로 구축한 MIT 6.7960 Deep Learning RAG를 평가하기 위한 1차 seed 평가셋이다.
각 row는 질문, gold segment timestamp, modality label, frame/link evidence 힌트를 포함한다.

## Files

- `queries.csv`: 통합 평가셋
- `queries.jsonl`: 동일 내용을 JSONL로 저장한 버전
- `suites/*.csv`: `benchmark-retrieval`이 강의별로 읽는 CSV
- `benchmark_manifest.json`: 바로 실행 가능한 retrieval benchmark manifest
- `domain_lexicon.json`: MIT Deep Learning 질의 확장을 위한 보수적 약어/동의어 seed
- `benchmark_matrix_manifest.json`: 6-variant retrieval-answer matrix manifest
- `paper_bundle_manifest.json`: `run-paper-bundle`용 MIT paper matrix manifest
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

공통 lexicon을 쓰려면 각 suite에 `"domain_lexicon": "domain_lexicon.json"`을 추가한다.
이 seed는 MIT OpenCourseWare 6.7960 Deep Learning 강의자료와 STT seed labels에서 파생된 보조 lexicon이며, 원자료와 동일하게 CC BY-NC-SA 4.0 출처 조건을 따른다.

Matrix benchmark에서 lexicon variant를 쓰려면 manifest 또는 suite에 `"domain_lexicon": "domain_lexicon.json"`을 두고, variant의 `"use_domain_lexicon"`을 `true`로 둔다.

## Paper Matrix Use

Matrix benchmark는 24개 lecture suite를 `segment_lexical`, `domain_lexicon`, `hybrid`, `window`, `window_hybrid`, `rerank` 변형으로 실행한다.
실행 전 Meilisearch가 떠 있어야 하며, manifest가 가리키는 MIT project artifacts와 기존 segment/visual entity index가 준비되어 있어야 한다.
`rerank` 변형은 최종 반환 `limit`과 별개로 더 깊은 후보 풀을 조회해 deterministic reranker가 rank 밖 후보를 재정렬할 수 있게 한다.
`hybrid`와 `window_hybrid`는 Meilisearch vector store가 켜져 있고, segment/window/visual index에 `userProvided` embedder와 `_vectors.default`가 있어야 한다.
로컬 paper docker에서는 먼저 다음처럼 vector store를 켠 뒤, segment/window/visual index를 `manual_user_provided_v1` profile로 다시 적재한다.

```bash
docker compose up -d meilisearch
curl -X PATCH 'http://127.0.0.1:7700/experimental-features/' \
  -H 'Authorization: Bearer dev-master-key' \
  -H 'Content-Type: application/json' \
  --data-binary '{"vectorStore": true}'
```

MIT manifest의 `hybrid_query_vector_dimensions: 384`는 `local_hash_v1` query vector fallback을 사용해 1-suite/all-variant smoke를 재현 가능하게 만든다.
이 deterministic hash vector는 semantic 품질 claim이 아니라 local reproducibility/smoke 전용 fallback이며, 공개 artifact에는 `purpose: local_reproducibility_smoke_fallback`, `quality_claim: none` metadata만 남기고 raw vector 값은 남기지 않는다.

Partial smoke를 먼저 돌릴 때는 공개 출력 원칙을 유지한 채 suite 수만 줄인 임시 manifest를 만들 수 있다.

```bash
jq '.suites |= .[:1]' \
  eval/mit_deep_learning_stt/benchmark_matrix_manifest.json \
  > /tmp/mit_deep_learning_matrix_1suite.json
python -m oarag benchmark-retrieval \
  --manifest /tmp/mit_deep_learning_matrix_1suite.json \
  --output-dir reports/mit_deep_learning_eval/paper_matrix_1suite_smoke
```

```bash
python scripts/index_mit_deep_learning_retrieval_indexes.py --reset \
  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json \
  --stages segment,window,visual \
  --hybrid-embedder-profile manual_user_provided_v1 \
  --hybrid-embedder-dimensions 384 \
  --hybrid-embedder-live-smoke
python -m oarag benchmark-retrieval \
  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json
```

`index_mit_deep_learning_retrieval_indexes.py`는 manifest의 모든 project를 순회하며 shared
`mit_deep_learning_stt_segments`, `mit_deep_learning_stt_windows`,
`mit_deep_learning_stt_visual_entities` index를 준비한다. 각 stage의 첫 project에서만 index
생성/settings/live smoke를 실행하고 이후 project는 같은 shared index에 document만 append한다.
window stage는 각 project artifact 아래 `segments/lecture_windows.jsonl`을 만든 뒤 적재한다.
스크립트 출력은 project별 aggregate count, vector summary, settings hash, display-safe project ref만
포함하며 raw transcript/query/vector나 로컬 절대 경로는 남기지 않는다.

Full paper bundle skeleton:

```bash
python -m oarag run-paper-bundle \
  --manifest eval/mit_deep_learning_stt/paper_bundle_manifest.json \
  --output-dir reports/mit_deep_learning_eval/paper_matrix_v1 \
  --gate-config reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json \
  --baseline-variant-id segment_lexical
```

`reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json` is a
bundle-completeness guard, not a final paper performance threshold.

## Caveat

이 데이터셋은 STT segment와 대표 frame alignment에서 만든 seed label이다.
정식 논문 수치로 쓰기 전에는 사람이 timestamp, modality, visual entity를 한 번 더 검수하는 것이 좋다.

Source: MIT OpenCourseWare 6.7960 Deep Learning, Fall 2024
License: CC BY-NC-SA 4.0
