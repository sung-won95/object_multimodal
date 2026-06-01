# Evidence Units Smoke Report

Issue: #180
Date: 2026-06-01

## Scope

This smoke checked whether the new `evidence_units` artifact path can be built for a
small MIT Deep Learning slice and whether the resulting storage looks safe to feed into
RAG. It intentionally does not treat timestamp-only overlap as verified object
alignment.

Inputs were two local MIT lecture project artifacts copied into a temporary shadow
workspace before running the CLI. The report below is public-safe: it omits raw
transcript text, private media paths, local absolute paths, and frame contents.

## Commands

```bash
PYTHONPATH=src python3 -c 'from oarag.cli import main; main()' \
  build-project-evidence-units --project-dir <tmp-shadow-project-lec01>

PYTHONPATH=src python3 -c 'from oarag.cli import main; main()' \
  build-project-evidence-units --project-dir <tmp-shadow-project-lec04>

PYTHONPATH=src python3 -c 'from oarag.cli import main; main()' health
```

The first `python3 -m oarag.cli ...` attempt was a no-op because this module does not
invoke `main()` when executed with `-m`; the smoke used the package entrypoint function
directly.

## Build Results

| Lecture | evidence_units total | alignment_status verified | alignment_status candidate | alignment_status transcript_only | units_with_visual_state | units_with_visual_entity | units_with_vlm_entity | units_with_verified_link | timestamp_fallback_links | candidate_links | verified_links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MIT 6.7960 lecture 01 | 1023 | 0 | 1011 | 12 | 1011 | 774 | 0 | 0 | 1946 | 126 | 0 |
| MIT 6.7960 lecture 04 | 1451 | 0 | 1431 | 20 | 1431 | 831 | 0 | 0 | 2256 | 144 | 0 |

Interpretation:

- `build-project-evidence-units` successfully produced one evidence unit per aligned
  lecture segment for both lectures.
- Visual-state coverage is high after converting sampled frames into rough intervals.
- Visual entity attachment exists, but the current local artifacts are OCR-only for
  this slice: `units_with_vlm_entity = 0`.
- No verified object links were produced: `verified_links = 0` and
  `units_with_verified_link = 0`.
- Timestamp fallback links are present and were counted separately from verified links.
  They should not be used as object-alignment success evidence.

## Query And Index Smoke

Meilisearch was not running locally during this smoke:

```text
health: connection refused on localhost:7700
```

Per the issue instructions, the smoke did not start or reconfigure Docker/services.
Therefore:

- `index-project-evidence-units`: skipped, blocked by unavailable Meilisearch.
- `query-project-evidence-units`: skipped, blocked by unavailable Meilisearch.
- RAG input inspectability from actual query output: not verified in this run.

The generated evidence unit rows themselves are inspectable as RAG inputs because the
artifact contains `evidence_text`, `semantic_text`, source segment IDs, visual state
IDs, visual entity IDs, link IDs, `alignment_status`, and `source_quality`. However,
actual retrieval ranking could not be judged without the index/query step.

## Segment Baseline Comparison Notes

Existing segment baseline metrics for the same two lecture suites show mixed behavior:

| Lecture suite | baseline queries | Hit@10s | MRR note | qualitative note |
| --- | ---: | ---: | --- | --- |
| MIT lecture 01 | 2 | 0/2 | both MRR 0.0 | Segment top hits missed the gold windows even when some hits were frame/link backed. |
| MIT lecture 04 | 2 | 1/2 | one MRR 1.0, one MRR 0.0 | One query was retrieved close to the gold window; one missed badly despite frame/link-backed candidates. |

Gold-target evidence unit spot checks:

| Query topic class | evidence unit status | visual states | visual entities | verified links | candidate/timestamp note |
| --- | --- | ---: | ---: | ---: | --- |
| Lecture 01 audio concept | candidate | 2 | 45 | 0 | no candidate or fallback links on the gold unit |
| Lecture 01 slide-grounded concept | candidate | 1 | 8 | 0 | all linked evidence on the gold unit is timestamp fallback |
| Lecture 04 slide-grounded concept | candidate | 1 | 6 | 0 | one non-fallback candidate link plus timestamp fallback links |
| Lecture 04 audio concept | candidate | 2 | 11 | 0 | no candidate or fallback links on the gold unit |

Qualitative judgment:

- The evidence unit representation is more inspectable than isolated segment hits
  because it packages transcript window, visual state, entity IDs, and link quality in
  one retrievable row.
- This smoke does not show verified object alignment. The strongest result is storage
  readiness for candidate-level RAG, not paper-claim readiness.
- Segment baseline top-hit/miss behavior suggests the next useful comparison is not
  another aggregate frame-backed ratio. It should compare actual evidence-unit search
  top hits against the same four query topics once Meilisearch is available.

## Follow-Up Issue Candidates

1. #183: Add a public-safe evidence-unit retrieval smoke runner that builds, indexes, queries,
   and emits sanitized top-hit summaries, with a clear skip when Meilisearch is absent.
2. Add or generate VLM-first visual entities for the 1-2 lecture slice; current OCR-only
   visual entities keep `units_with_vlm_entity` at zero.
3. Add a verified-link source for the seed multimodal query rows, or explicitly keep
   these as candidate-only until a verifier/human annotation path exists.
4. Investigate why the CLI module can be executed with `python -m oarag.cli` without
   invoking `main()`; this can make manual smoke commands look successful while doing
   nothing.

## Conclusion

The build side of the evidence-unit storage path works on two local MIT lecture
projects. The indexed/queryable RAG smoke is blocked by unavailable Meilisearch in this
environment. Current storage quality is candidate-level only: timestamp fallback links
remain separated, and no VLM or verified object-alignment signal is present for this
slice.
