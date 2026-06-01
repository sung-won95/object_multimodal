# Evidence Unit Smoke Summary

Run ID: `issue_196_real_vlm_observations_blocker`

Real VLM observation generation was not executed because the checked two-lecture slice
has no existing real VLM observation/parser JSONL artifacts and no configured real
`command` backend. Coverage therefore did not improve over PR 195.

Candidate-frame dry-run planning is available for the next real run: 12 frames for
`mit_lec01` and 12 frames for `mit_lec04`, capped to avoid full-corpus or high-cost
processing.

No deterministic or mock VLM output was used as evidence.
