# Evidence Unit Retrieval Smoke

Run ID: `issue_192_mit_semantic_coverage`

This public-safe report redacts raw query text, transcripts, local paths, and raw IDs.
Timestamp-only overlap is not counted as verified object alignment.

## Meilisearch

- available: `True`
- status: `available`

## Suites

### mit_lec01

- evidence units: `1023`
- alignment statuses: `{"candidate": 1011, "transcript_only": 12}`
- link counts: `{"candidate_links": 126, "timestamp_fallback_links": 1946, "verified_links": 0}`
- index status: `indexed`
- RAG input inspectable top hits: `2`
- target rank buckets: `{"not_found": 1, "top50": 1}`
- found target query-term buckets: `{"high": 1}`
- top-vs-target coverage flags: `{"target_has_longer_semantic_text": 1, "target_has_more_query_term_matches": 1, "target_has_more_visual_entities": 1}`
- reranked target rank buckets: `{"not_found": 1, "top5": 1}`
- reranked top-hit matches: `0`

### mit_lec04

- evidence units: `1451`
- alignment statuses: `{"candidate": 1431, "transcript_only": 20}`
- link counts: `{"candidate_links": 144, "timestamp_fallback_links": 2256, "verified_links": 0}`
- index status: `indexed`
- RAG input inspectable top hits: `2`
- target rank buckets: `{"not_found": 1, "top10": 1}`
- found target query-term buckets: `{"very_high": 1}`
- top-vs-target coverage flags: `{"target_has_longer_semantic_text": 1, "target_has_more_query_term_matches": 1}`
- reranked target rank buckets: `{"not_found": 1, "top10": 1}`
- reranked top-hit matches: `0`
