# Issue 281 VLM entity-link verifier smoke

This public-safe sample records the expected shape of the VLM entity-link
verifier workflow. It uses synthetic fixture-style inputs only and does not
expose raw transcripts, visual text, frame image bytes, local paths, private
video content, or private evaluation data.

Smoke command shape with a fixture backend:

```bash
PYTHONPATH=src python -m oarag verify-entity-links-vlm \
  --project-dir <project_dir> \
  --entity-links manifests/entity_links.jsonl \
  --segments segments/lecture_segments_aligned.jsonl \
  --visual-entities manifests/visual_entities.jsonl \
  --frames-manifest manifests/frames_manifest.jsonl \
  --vlm-backend jsonl \
  --vlm-model fixture-vlm \
  --vlm-options jsonl_path=manifests/vlm_link_decisions.jsonl \
  --output manifests/entity_links.vlm_verified.jsonl \
  --cache manifests/entity_links.vlm_verifier_cache.json \
  --report reports/vlm_entity_link_verifier.json \
  --human-audit-template reports/vlm_entity_link_verifier_human_audit_template.jsonl

PYTHONPATH=src python -m oarag build-project-evidence-units \
  --project-dir <project_dir> \
  --entity-links manifests/entity_links.vlm_verified.jsonl
```

Expected smoke properties:

- `verified` decisions are promoted with `verification_source=vlm_verifier`
- `rejected` and `uncertain` decisions remain candidate links
- evidence-unit rebuild has
  `verified_link_source_counts.vlm_verifier > 0` when at least one verified
  decision is present
- cache stores link-id keyed public-safe decisions and reason codes only
- backend unavailability writes an explicit skip report and never promotes all
  candidates as a fallback

Issue #281 still requires a real human audit before closure: at least 50
random VLM decision rows must be checked directly against video by a human and
an agreement rate must be recorded. The automated template supports that audit,
but this repository artifact does not claim that the human audit has been
completed.
