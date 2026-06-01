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

## MIT Quality-Rerank Smoke

The runner was also applied to the same two MIT lecture smoke artifacts used in
Issue 188, with Meilisearch available and `--quality-rerank` enabled. The private
manifest and copied lecture artifacts stayed outside the repository; the committed
outputs are public-safe.

| Metric | Value |
| --- | ---: |
| suites | 2 |
| queries | 4 |
| queried | 4 |
| skipped | 0 |
| indexed evidence units | 2474 |
| rerank-enabled queries | 4 |
| base top-hit target matches | 0 |
| reranked top-hit target matches | 0 |
| reranked top result changed | 0 |

## Base vs Reranked Target Rank

| Suite | base top10 | base top50 | base not_found | reranked top5 | reranked top10 | reranked not_found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mit_lec01 | 0 | 1 | 1 | 1 | 0 | 1 |
| mit_lec04 | 1 | 0 | 1 | 0 | 1 | 1 |
| total | 1 | 1 | 2 | 1 | 1 | 2 |

The quality-aware baseline improved one query from `top50` to `top5`, but did not
turn any query into a top-hit match. The top result also did not change for any of
the four queries. On this slice, the deterministic metadata-only rerank helps
target rank slightly but is not enough to solve retrieval quality.

## Public Fixture Smoke

The committed test fixture exercises the base-vs-reranked diagnostics with a
public-safe fake Meilisearch client and confirms that the reranker can change the
top result when the metadata strongly favors the target. In that fixture,
`verified_link_presence` remains zero and timestamp fallback contributes only a
penalty/flag.

## Verification

Commands used in this PR:

```bash
python3 -m pytest tests/test_evidence_unit_smoke.py tests/test_cli.py -q
python3 -m pytest tests/test_evidence_unit_smoke.py tests/test_evidence_units.py tests/test_cli.py -q
python3 -m pytest -q
```

The fixture confirms that target labels are used only after scoring for diagnostics,
not in the rerank score itself.

## Artifacts

- `metrics.json`: public-safe aggregate metrics, suite summaries, and rerank
  diagnostics.
- `query_results.jsonl`: one sanitized row per query with base and reranked
  diagnostics.
- `summary.md`: runner-generated public summary.
