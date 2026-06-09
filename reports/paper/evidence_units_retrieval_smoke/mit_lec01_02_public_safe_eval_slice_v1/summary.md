# Evidence Unit Retrieval Smoke

Run ID: `mit_lec01_02_public_safe_eval_slice_v1`

This public-safe report redacts raw query text, transcripts, local paths, and raw IDs.
Timestamp-only overlap is not counted as verified object alignment.
Visual-state interval overlap is reported as candidate support, not verified object alignment.

## Meilisearch

- available: `False`
- status: `dry_run`
- skip reason: `dry_run_requested`

## Evaluation Slice

- slice id: `mit_lec01_02_public_safe_v1`
- query targets: `24`
- targets configured: `24`
- expected modalities: `{"audio": 12, "both": 12}`
- query types: `{"audio_reasoning": 3, "concept_explanation": 6, "definition_lookup": 4, "formula_table_lookup": 2, "multimodal_grounded": 6, "visual_object_reference": 3}`
- queries with concept aliases: `24`
- concept alias total: `71`
- privacy: `raw_query_text/transcripts/answers/evidence_text/local_paths_excluded`

## Suites

### mitdl_lec01_public_safe_slice

- evidence units: `0`
- alignment statuses: `{}`
- link counts: `{}`
- candidate visual support: `{"candidate_link_signal_counts": {"lexical_overlap": 0, "mention_deictic_hook": 0, "semantic_domain_hint": 0, "spatial_position": 0, "temporal_overlap": 0, "timestamp_fallback": 0, "visual_text_overlap": 0, "vlm_object_visual_description_overlap": 0}, "candidate_links": 0, "paper_claim_eligible": false, "timestamp_fallback_links": 0, "units_with_candidate_link": 0, "units_with_candidate_visual_support": 0, "units_with_timestamp_fallback_link": 0}`
- verified object alignment: `{"paper_claim_eligible_units": 0, "timestamp_fallback_counted_as_verified": false, "units_with_verified_object_alignment": 0, "verified_link_source_counts": {"explicit_verified_flag": 0, "explicit_verified_status": 0, "human_gold": 0, "strict_deterministic_rule": 0, "unspecified_verified": 0, "vlm_verifier": 0}, "verified_links": 0}`
- visual state source: `unknown`
- visual state coverage: `0`/`0`
- concept field coverage: `0`/`0`
- visual state duration buckets: `{}`
- visual state gate: `not_configured`
- index status: `dry_run`
- RAG input inspectable top hits: `0`
- target rank buckets: `{"not_queried": 12}`
- target found buckets: `{"not_found": 0, "top1": 0, "top10": 0, "top100": 0, "top5": 0, "top50": 0}`
- target_found@50: `None`
- target_found@100: `None`
- not_found reason codes: `{"candidate_recall_failure": 0, "index_settings_issue": 0, "modality_evidence_missing": 0, "text_coverage_failure": 0}`
- target evidence-text buckets: `{}`
- target semantic-text buckets: `{}`
- target feature coverage: `{}`
- found target query-term buckets: `{}`
- found target candidate link signals: `{}`
- found target verified link sources: `{}`
- top-vs-target coverage flags: `{}`
- reranked target rank buckets: `{}`
- reranked top-hit matches: `0`
- modality-aware query types: `{}`
- modality-aware failure modes: `{}`
- modality-aware top changes: `0`
- VLM evidence status: `dry_run`
- paper-quality VLM entities: `0`
- OCR-only entities: `0`
- units with VLM entity: `0`
- units with visual description: `0`
- units with detected text: `0`

### mitdl_lec02_public_safe_slice

- evidence units: `0`
- alignment statuses: `{}`
- link counts: `{}`
- candidate visual support: `{"candidate_link_signal_counts": {"lexical_overlap": 0, "mention_deictic_hook": 0, "semantic_domain_hint": 0, "spatial_position": 0, "temporal_overlap": 0, "timestamp_fallback": 0, "visual_text_overlap": 0, "vlm_object_visual_description_overlap": 0}, "candidate_links": 0, "paper_claim_eligible": false, "timestamp_fallback_links": 0, "units_with_candidate_link": 0, "units_with_candidate_visual_support": 0, "units_with_timestamp_fallback_link": 0}`
- verified object alignment: `{"paper_claim_eligible_units": 0, "timestamp_fallback_counted_as_verified": false, "units_with_verified_object_alignment": 0, "verified_link_source_counts": {"explicit_verified_flag": 0, "explicit_verified_status": 0, "human_gold": 0, "strict_deterministic_rule": 0, "unspecified_verified": 0, "vlm_verifier": 0}, "verified_links": 0}`
- visual state source: `unknown`
- visual state coverage: `0`/`0`
- concept field coverage: `0`/`0`
- visual state duration buckets: `{}`
- visual state gate: `not_configured`
- index status: `dry_run`
- RAG input inspectable top hits: `0`
- target rank buckets: `{"not_queried": 12}`
- target found buckets: `{"not_found": 0, "top1": 0, "top10": 0, "top100": 0, "top5": 0, "top50": 0}`
- target_found@50: `None`
- target_found@100: `None`
- not_found reason codes: `{"candidate_recall_failure": 0, "index_settings_issue": 0, "modality_evidence_missing": 0, "text_coverage_failure": 0}`
- target evidence-text buckets: `{}`
- target semantic-text buckets: `{}`
- target feature coverage: `{}`
- found target query-term buckets: `{}`
- found target candidate link signals: `{}`
- found target verified link sources: `{}`
- top-vs-target coverage flags: `{}`
- reranked target rank buckets: `{}`
- reranked top-hit matches: `0`
- modality-aware query types: `{}`
- modality-aware failure modes: `{}`
- modality-aware top changes: `0`
- VLM evidence status: `dry_run`
- paper-quality VLM entities: `0`
- OCR-only entities: `0`
- units with VLM entity: `0`
- units with visual description: `0`
- units with detected text: `0`

