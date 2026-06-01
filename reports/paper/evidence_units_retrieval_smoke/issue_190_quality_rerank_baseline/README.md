# Issue 190 Evidence Unit Quality-Rerank Baseline

Date: 2026-06-01
Runner: `evidence-unit-smoke --quality-rerank`

## Scope

This public-safe follow-up adds a deterministic quality-aware rerank baseline to the
evidence-unit smoke runner. It is smoke/report only; it does not change the production
query path and it does not use expected target labels in the rerank score.

The runner output redacts raw query text, transcripts, local paths, and raw
segment/entity/evidence-unit IDs. Timestamp-only fallback remains a flag/penalty only
and is never counted as verified object alignment.

## Rerank Signals

The score is deterministic and explainable. It preserves original retrieval order
signals through rank and Meilisearch score components, then adds public-safe quality
signals:

| Component | Meaning |
| --- | --- |
| rank_preservation | keeps original rank as a signal |
| meili_score | keeps Meilisearch ranking score when available |
| evidence_text_available | evidence text exists, but content is redacted |
| semantic_text_available | semantic text exists, but content is redacted |
| visual_state_count | capped visual-state count boost |
| visual_entity_count | capped visual-entity count boost |
| candidate_entity_link_count | capped candidate-link count boost |
| vlm_entity_presence | boost for VLM-backed entity presence |
| verified_link_presence | boost only for explicit verified links |
| transcript_only_penalty | penalty for transcript-only evidence units |
| timestamp_fallback_penalty | fallback flag penalty; not a verified boost |

## Public Fixture Smoke

The committed test fixture exercises the base-vs-reranked diagnostics with a
public-safe fake Meilisearch client:

| Metric | Base | Reranked |
| --- | ---: | ---: |
| configured queries | 1 | 1 |
| top-hit target match | 0 | 1 |
| target rank bucket top1 | 0 | 1 |
| target rank bucket top5 | 1 | 0 |
| top result changed | 0 | 1 |

The reranked top hit in the fixture has VLM-entity presence and candidate visual
support. It also carries a timestamp-fallback flag, which contributes only a penalty
and is not counted as verified alignment. `verified_link_presence` remains zero.

## MIT Carry-Forward From Issue 188

The latest committed 1-2 lecture MIT smoke report remains
`reports/paper/evidence_units_retrieval_smoke/issue_188_mit_target_rank/`.
Those base target-rank buckets are carried forward here for comparison context:

| Suite | configured | found top-k | base top1 | base top5 | base top10 | base top50 | base not_found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | 2 | 1 | 0 | 0 | 0 | 1 | 1 |
| mit_lec04 | 2 | 1 | 0 | 0 | 1 | 0 | 1 |
| total | 4 | 2 | 0 | 0 | 1 | 1 | 2 |

The private MIT artifacts are not committed, so this PR does not fabricate reranked
MIT bucket counts. After rerunning the private manifest with `--quality-rerank`, the
runner will emit:

- per-query `rerank_diagnostics.base_target_rank_bucket`
- per-query `rerank_diagnostics.reranked_target_rank_bucket`
- per-query base vs reranked top-hit match flags
- public-safe rerank score components for the reranked top hit
- suite/global base and reranked target-rank bucket counts

## Verification

Commands used in this PR:

```bash
python3 -m pytest tests/test_evidence_unit_smoke.py tests/test_cli.py -q
```

The fixture confirms that target labels are used only after scoring for diagnostics,
not in the rerank score itself.
