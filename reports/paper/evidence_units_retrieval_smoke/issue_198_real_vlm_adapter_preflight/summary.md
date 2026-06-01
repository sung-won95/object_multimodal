# Issue 198 Summary

- Added a repo-level real VLM command adapter: `oarag.vision.vlm_command_adapter`.
- Added `preflight_command` support for the command backend so missing real
  credential/config can fail before frame processing.
- The adapter reads `vlm-command-request-v1` JSON from stdin and writes accepted
  `observations` JSON to stdout.
- Current environment is blocked by missing `OARAG_VLM_API_KEY`/`OPENAI_API_KEY`
  and `OARAG_VLM_MODEL`/request model config.
- No real 2-lecture VLM smoke was executed, and no deterministic/mock VLM output
  was used as paper-quality evidence.
- Full tests passed: `362 passed, 4 skipped`.
