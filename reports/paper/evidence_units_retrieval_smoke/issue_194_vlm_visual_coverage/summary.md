# Issue 194 VLM Visual Coverage Preflight

Run ID: `issue_194_vlm_visual_coverage_preflight`

This public-safe report redacts raw queries, transcripts, local paths, raw IDs, and
frame contents.

## Outcome

- Real VLM input available: `false`
- Retrieval smoke rerun: `false`
- Coverage improvement over #193: `none`
- Blocker: the checked two-lecture slice has no real VLM observations or structured
  VLM parser output.

## Counts

- suites checked: `2`
- VLM observations files present: `0`
- VLM parser files present: `0`
- current visual entities checked: `885`
- OCR visual entities: `885`
- VLM visual entities: `0`
- visual-description entities: `0`
- detected-text entities: `0`
- verified-like links: `0`

## Checks

- CLI help checks: `extract-visual-entities`, `run-vlm`, `run-vlm-alignment`,
  `evidence-unit-smoke`
- focused pytest subset: `30 passed`

## Next Step

Generate real VLM observations or parser JSONL for the same two lecture projects,
then rerun VLM-first visual entity extraction and evidence-unit smoke. Mock or
deterministic VLM output should remain excluded from paper-quality coverage claims.
