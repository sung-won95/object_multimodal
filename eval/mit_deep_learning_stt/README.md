# MIT Deep Learning STT Seed Evaluation Set

이 폴더는 STT로 구축한 MIT 6.7960 Deep Learning RAG를 평가하기 위한 1차 seed 평가셋이다.
각 row는 질문, gold segment timestamp, modality label, frame/link evidence 힌트를 포함한다.

## Files

- `queries.csv`: 통합 평가셋
- `queries.jsonl`: 동일 내용을 JSONL로 저장한 버전
- `suites/*.csv`: `benchmark-retrieval`이 강의별로 읽는 CSV
- `benchmark_manifest.json`: 바로 실행 가능한 retrieval benchmark manifest
- `domain_lexicon.json`: MIT Deep Learning 질의 확장을 위한 보수적 약어/동의어 seed
- `benchmark_matrix_manifest.json`: 9-variant retrieval-answer matrix manifest
- `public_safe_eval_slice_schema.json`: public-safe evidence-unit evaluation slice schema
- `mit_lec01_02_public_safe_eval_slice.json`: 2-lecture, 24-target public-safe slice for target-rank diagnostics
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

## Human Audit Completion Gate

Issue #270 must not be closed with automatically generated labels alone. Before
closing that issue, replace or extend this seed set with a human-audited
evaluation CSV and run:

```bash
python scripts/validate_mit_deep_learning_eval_dataset.py \
  --queries eval/mit_deep_learning_stt/queries.csv \
  --readme eval/mit_deep_learning_stt/README.md \
  --require-readme-audit-summary
```

The gate requires at least 240 queries across the 24 lecture videos listed in
`benchmark_matrix_manifest.json`, at least 10 queries per video, dev/test split
balance, required question-type coverage, and at least 20% human audit coverage.
A human-audited label is a row whose
`annotator_id` is non-empty and does not start with `codex_`; a CSV where every
annotator is `codex_*` must fail. The validator also checks numeric timestamps
and `end >= start` without printing raw query text, reference answers, or
transcripts.

When the audited expansion lands, this README must include a short human audit
result summary with audited label counts, timestamp correction policy, and mean
timestamp correction if corrections were made. The current 48-row seed is
expected to fail this completion gate until a real person audits the expanded
labels.

## Expanded Candidate Packet

Create the automatic 240-row candidate packet for human audit with:

```bash
python scripts/prepare_mit_deep_learning_eval_expansion_candidates.py \
  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json \
  --output-dir eval/mit_deep_learning_stt/candidates/expanded_seed_v1
```

The generated `candidates/expanded_seed_v1/queries_candidate.csv` is a
pre-audit candidate set and does not replace `queries.csv`. It should satisfy
the row/video/per-video/question-type shape requirements, but it is expected to
fail the final validator gate because all labels are `codex_*` generated and no
human audit summary exists yet. Use `human_audit_sample.csv` as the initial
48-row audit packet, then create a separate human-audited expanded CSV before
using the expansion in benchmark claims or closing #270.

## Public-Safe Evidence-Unit Slice

`mit_lec01_02_public_safe_eval_slice.json` fixes 24 query targets across two MIT
lecture projects for accuracy-improvement diagnostics. It intentionally stores
query IDs, concept aliases, expected modality/query type, target segment IDs, and
timestamp buckets only. Raw query text, transcript excerpts, reference answers,
raw evidence text, and private local paths are excluded from the manifest and
from public smoke outputs. Candidate-depth diagnostics use a 100-hit Meili-only
pool and report target found buckets (`top1`, `top5`, `top10`, `top50`,
`top100`, `not_found`), `target_found@50`, `target_found@100`, and aggregate
public-safe not_found reason codes.

Dry-run wiring check:

```bash
python -m oarag evidence-unit-smoke \
  --manifest eval/mit_deep_learning_stt/mit_lec01_02_public_safe_eval_slice.json \
  --dry-run
```

When private artifacts and Meilisearch indexes are available, the same manifest
can be run without `--dry-run` to produce public-safe target-rank diagnostics.
The dry-run baseline is 24 configured targets with no query execution; live
baseline recall values are recorded only in the generated public-safe report.

## Paper Matrix Use

Matrix benchmark는 24개 lecture suite를 `segment_lexical`, `domain_lexicon`, `hybrid`, `window`, `window_hybrid`, `rerank`, `evidence_unit_candidate`, `evidence_unit_verified`, `evidence_unit_quality_rerank` 변형으로 실행한다.
실행 전 Meilisearch가 떠 있어야 하며, manifest가 가리키는 MIT project artifacts와 기존 segment/visual entity index가 준비되어 있어야 한다.
`rerank` 변형은 최종 반환 `limit`과 별개로 더 깊은 후보 풀을 조회해 deterministic reranker가 rank 밖 후보를 재정렬할 수 있게 한다.
`hybrid`와 `window_hybrid`는 Meilisearch vector store가 켜져 있고, segment/window/visual index에 `userProvided` embedder와 `_vectors.default`가 있어야 한다.
`evidence_unit_*` 변형은 같은 query set에서 `segments/evidence_units.jsonl` 기반 object-aligned 후보를 비교한다. evidence-unit artifact 또는 `mit_deep_learning_stt_evidence_units` index가 준비되지 않은 suite는 variant row를 삭제하지 않고 `skipped_count`와 public-safe `skip_reason`으로 남기며, missing VLM/verified evidence는 0-count로 집계한다.
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

## Real Embedding Smoke

논문용 실제 embedding run은 `local_hash_v1` fallback이 아니라 명시적인 vector manifest를 생성한 뒤
segment/window/visual/query 경로가 모두 manifest를 쓰는지 검증해야 한다. 다음 스크립트는 선택한
MIT suite에 대해 window artifact, vector manifest, shared smoke index, benchmark manifest,
semantic smoke gate를 한 번에 실행한다.

```bash
python scripts/run_mit_real_embedding_smoke.py \
  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json \
  --suite-id mitdl_lec01 \
  --output-root reports/mit_deep_learning_eval/real_embedding_smoke/mitdl_lec01_local_st \
  --provider sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --dimensions 384 \
  --local-files-only \
  --index-prefix mit_real_embedding_local_st \
  --hybrid-embedder-profile manual_user_provided_v1
```

외부 API credential이 있는 경우에는 OpenAI-compatible provider를 쓸 수 있다.

```bash
export OARAG_EMBEDDING_API_KEY=...
export OARAG_EMBEDDING_MODEL=...

python scripts/run_mit_real_embedding_smoke.py \
  --manifest eval/mit_deep_learning_stt/benchmark_matrix_manifest.json \
  --suite-id mitdl_lec01 \
  --output-root reports/mit_deep_learning_eval/real_embedding_smoke/mitdl_lec01 \
  --provider openai-compatible \
  --dimensions 1536 \
  --index-prefix mit_real_embedding \
  --hybrid-embedder-profile manual_user_provided_v1
```

`sentence-transformers` provider는 로컬 neural embedding 모델을 사용한다. `--local-files-only`를
주면 Hugging Face cache에 있는 모델만 로드하므로 네트워크 없이 paper-ready smoke를 재현할 수 있다.
새 환경에서는 `object-aligned-rag[embeddings]` extra로 optional dependency를 설치한다.
`--allow-fixture-provider`는 배관 테스트 전용이다. 이 옵션을 쓰면 `deterministic_fixture` provider도
통과하지만 paper-ready claim으로 해석하면 안 된다. 기본 real mode에서는 `deterministic_fixture`나
`local_hash_v1`가 발견되면 validation이 실패한다.

성공 조건은 다음과 같다.

- document index summary의 `generator`가 `null`
- `purpose == real_embedding_manifest`
- `generated_vector_count == 0`
- `manifest_vector_count == expected_vector_count`
- hybrid query 결과의 query vector source가 `manifest`
- `semantic_smoke.json`의 `semantic_live_smoke.ok == true`

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

Run the retrieval quality gate as a required post-benchmark step with the
`semantic_smoke.json` from the same output directory:

```bash
python -m oarag check-retrieval-gate \
  --metrics reports/mit_deep_learning_eval/full_provider_matrix_v1/metrics.json \
  --config reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json \
  --semantic-smoke reports/mit_deep_learning_eval/full_provider_matrix_v1/semantic_smoke.json
```

`reports/mit_deep_learning_eval/paper_matrix_v1/retrieval_quality_gate.json` is a
#267-derived regression guard. Suite aggregate thresholds cover retrieval quality
floors, and variant-specific thresholds preserve observed variant behavior; zero
variant thresholds require an explicit public-safe allowance.

## Caveat

이 데이터셋은 STT segment와 대표 frame alignment에서 만든 seed label이다.
정식 논문 수치로 쓰기 전에는 사람이 timestamp, modality, visual entity를 한 번 더 검수하는 것이 좋다.

Source: MIT OpenCourseWare 6.7960 Deep Learning, Fall 2024
License: CC BY-NC-SA 4.0
