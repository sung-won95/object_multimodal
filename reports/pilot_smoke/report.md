# Object-Aligned Pipeline Pilot Smoke-Test Report

## Scope

- 목적: formal evaluation claim 이전에 object-aligned multimodal RAG 파이프라인이 실제 로컬 강의 영상에서 끝까지 동작하는지 확인한다.
- 사용 영상: `2_Matrices`, `10_Deviating_From_The_Charts`
- 질의 수: presentation-safe pilot query 6개
- 주의: 이 문서는 정식 성능 비교가 아니라 feasibility/pilot 기록이다.
- 경로 정책: tracked 산출물에는 로컬 절대경로를 넣지 않고 repo-relative artifact path만 남겼다.

## Query Set

질의 목록은 [pilot_queries.csv](./pilot_queries.csv)에 저장했다.

대표 질의:

- `matrices_what_is_matrix`: 매트릭스 소개 구간 retrieval
- `matrices_offsuit_location`: matrix left-side offsuit 설명 retrieval
- `matrices_77_vs_ak_equity`: late-timestamp equity 설명 retrieval
- `charts_ajo_by_position`: AJo position-dependent example retrieval
- `charts_utg_3x_aj`: UTG 3x 대응 retrieval
- `charts_co_ajo_3bet`: CO AJo 3-bet rationale retrieval

## Command Sequence

아래 시퀀스로 재현했다. 실제 환경에서는 placeholder만 자신의 경로로 바꾸면 된다.

```bash
REPO_ROOT=/path/to/object_aligned_rag
DATA_ROOT=/path/to/paper_materials_object_aligned_multimodal_rag/data

cd "$REPO_ROOT"

docker compose up -d meilisearch

PYTHONPATH=src python3 -m oarag ingest-video \
  --video "$DATA_ROOT/업스윙 포커/2. 매트릭스(V)/2. Matrices.mp4" \
  --srt "$DATA_ROOT/업스윙 포커/2. 매트릭스(V)/2. Matrices.srt" \
  --project-id pilot_issue8_matrices_safe \
  --frame-rate 1 \
  --max-frames 120

PYTHONPATH=src python3 -m oarag ingest-video \
  --video "$DATA_ROOT/업스윙 포커/10. 차트에서 벗어나 전략 수정하기 (v)/10. Deviating From The Charts.mp4" \
  --srt "$DATA_ROOT/업스윙 포커/10. 차트에서 벗어나 전략 수정하기 (v)/10. Deviating From The Charts.srt" \
  --project-id pilot_issue8_charts_safe \
  --frame-rate 1 \
  --max-frames 120

PYTHONPATH=src python3 -m oarag align-frames --project-id pilot_issue8_matrices_safe --margin-seconds 0.5
PYTHONPATH=src python3 -m oarag align-frames --project-id pilot_issue8_charts_safe --margin-seconds 0.5

PYTHONPATH=src python3 -m oarag extract-visual-entities \
  --project-id pilot_issue8_matrices_safe \
  --backend local-ocr \
  --ocr-language eng

PYTHONPATH=src python3 -m oarag extract-visual-entities \
  --project-id pilot_issue8_charts_safe \
  --backend local-ocr \
  --ocr-language eng

PYTHONPATH=src python3 -m oarag link-entities --project-id pilot_issue8_matrices_safe
PYTHONPATH=src python3 -m oarag link-entities --project-id pilot_issue8_charts_safe

PYTHONPATH=src python3 -m oarag index-project \
  --project-id pilot_issue8_matrices_safe \
  --index pilot_issue8_matrices_segments_safe

PYTHONPATH=src python3 -m oarag index-project \
  --project-id pilot_issue8_charts_safe \
  --index pilot_issue8_charts_segments_safe

PYTHONPATH=src python3 -m oarag query-project \
  --project-id pilot_issue8_matrices_safe \
  --index pilot_issue8_matrices_segments_safe \
  --query "매트릭스가 무엇인지" \
  --limit 3 \
  --neighbor-count 1

PYTHONPATH=src python3 -m oarag query-project \
  --project-id pilot_issue8_charts_safe \
  --index pilot_issue8_charts_segments_safe \
  --query "AJo 전략은 포지션에 따라 어떻게 바뀌나요" \
  --limit 3 \
  --neighbor-count 1
```

각 query의 sanitize된 JSONL 결과는 [query_outputs.jsonl](./query_outputs.jsonl)에 저장했다.

## Pipeline Health

| project_id | lecture_segments | sampled_frames | segments_with_frames | visual_entities | entity_links | indexed_documents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilot_issue8_matrices_safe` | 94 | 120 | 34 | 3877 | 4869 | 94 |
| `pilot_issue8_charts_safe` | 96 | 120 | 30 | 415 | 499 | 96 |

관찰:

- 두 영상 모두 ingest, alignment, OCR, weak linking, indexing, query 단계가 끝까지 실행됐다.
- 모든 pilot query에서 transcript hit는 반환됐다.
- early segment query는 frame-backed evidence bundle까지 반환됐다.
- 일부 late segment query는 transcript-only fallback으로 내려갔다.

## Representative Hits

### 1. Matrix introduction

- query: `매트릭스가 무엇인지`
- top hit: `seg_2_Matrices_000002` at `6.06s-10.02s`
- transcript evidence: `오늘은 매트릭스가 무엇인지 알아볼 겁니다`
- frame preview:
  - `artifacts/projects/pilot_issue8_matrices_safe/frames/frame_000002.jpg`
  - `artifacts/projects/pilot_issue8_matrices_safe/frames/frame_000003.jpg`
  - `artifacts/projects/pilot_issue8_matrices_safe/frames/frame_000004.jpg`
- visual/entity preview: `Card`, `matroc`, `Predefined`
- note: frame-backed multimodal bundle은 성공했지만 OCR label 자체는 일부 noisy token을 포함한다.

### 2. Offsuit location explanation

- query: `오프수딧 핸드는 어디에 있나요`
- top hit: `seg_2_Matrices_000021` at `69.47s-74.04s`
- transcript evidence: `이 핸드들은 두개의 카드가 문양이 다른 '오프수딧' 이라고 합니다`
- frame preview:
  - `artifacts/projects/pilot_issue8_matrices_safe/frames/frame_000066.jpg`
  - `artifacts/projects/pilot_issue8_matrices_safe/frames/frame_000067.jpg`
  - `artifacts/projects/pilot_issue8_matrices_safe/frames/frame_000068.jpg`
- visual/entity preview: `WET`, `Poke-stretegy.com`, `Equila`
- note: frame는 붙지만 OCR가 UI 텍스트와 partial token을 많이 뽑아서 entity explanation 품질은 낮다.

### 3. AJo position-dependent example

- query: `AJo 전략은 포지션에 따라 어떻게 바뀌나요`
- top hit: `seg_10_Deviating_From_The_Charts_000009` at `34.27s-36.16s`
- transcript evidence: `AJo 에 대해서 이야기 해봅시다`
- frame preview:
  - `artifacts/projects/pilot_issue8_charts_safe/frames/frame_000028.jpg`
  - `artifacts/projects/pilot_issue8_charts_safe/frames/frame_000029.jpg`
  - `artifacts/projects/pilot_issue8_charts_safe/frames/frame_000030.jpg`
- visual/entity preview: none in top bundle summary
- note: charts 영상에서도 early query는 frame-backed retrieval이 가능했다.

### 4. Transcript-only fallback examples

- `77이 AK 상대로 가지는 에퀴티`
  - top hit: `seg_2_Matrices_000041` at `142.34s-147.50s`
  - transcript evidence: `77 이 AK 을 상대로 55%의 에퀴티를 가지는 것을 알 수 있습니다`
  - frame preview: none
- `UTG 3x를 마주하면 AJ를 어떻게 해야 하나요`
  - top hit: `seg_10_Deviating_From_The_Charts_000032` at `126.10s-130.75s`
  - transcript evidence: `HJ 에서 레귤러의 UTG 3x 를 마주하고 있고`
  - frame preview: none
- `CO에서 AJo로 3벳을 좋아하는 이유`
  - top hit: `seg_10_Deviating_From_The_Charts_000038` at `144.51s-151.44s`
  - transcript evidence: `전 CO 에 AJo 를 들고 있습니다 갑작스럽게도 지금 이 핸드는 모든 카테고리에 들어가는 핸드입니다`
  - frame preview: none

이 세 질의는 transcript retrieval 자체는 성공했지만, sampled frame cap 때문에 object-aligned evidence가 따라오지 못한 사례로 기록한다.

## Known Failure Cases

1. `--max-frames 120`와 `1fps` 조합에서는 약 120초 이후 구간이 frame-free가 되기 쉬웠다.
2. local OCR는 chart/UI-heavy frame에서 token을 과다 추출했고, 의미 없는 partial string이 많았다.
3. 이번 run의 weak entity linking은 두 프로젝트 모두 `lexical_links=0`, `mention_links=0`이었다.
4. 따라서 entity evidence는 존재하더라도 실제 explanation 품질은 아직 낮고, 대부분 `time_overlap`에 머물렀다.

## Smoke-Test Takeaways

- 최소 두 개의 로컬 강의 영상에서 ingest -> align -> OCR -> weak linking -> Meilisearch indexing -> multimodal query 흐름이 끝까지 실행되었다.
- transcript evidence retrieval은 여섯 개의 pilot query 전부에서 동작했다.
- early segment에서는 frame-backed bundle이 생성되어 object-aligned evidence 구조가 실제로 반환되는 것을 확인했다.
- formal evaluation 이전에 필요한 다음 보완점은 `frame sampling coverage 개선`, `OCR noise filtering`, `entity link precision 개선`이다.
