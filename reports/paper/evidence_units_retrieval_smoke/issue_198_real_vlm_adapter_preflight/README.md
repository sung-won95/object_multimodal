# Issue 198 Real VLM Command Adapter Preflight

Date: 2026-06-01
Runner: repo-level VLM command adapter/preflight

## Scope

This public-safe report follows Issue 196 / PR 197. The goal is not to improve
ranking or claim new object evidence from mock output. The goal is to make the
existing `run-vlm-alignment --vlm-backend command` path concrete enough that the
next bounded 2-lecture run can either call a real VLM or fail before frame
processing with a clear blocker.

No raw transcripts, raw queries, local absolute paths, raw frame IDs, raw segment
IDs, raw entity IDs, raw evidence-unit IDs, frame images, raw VLM responses, or
secret values are included here. Timestamp-only overlap remains candidate/fallback
evidence only.

## Result

A real command adapter now exists in the repo:

- module path: `oarag.vision.vlm_command_adapter`
- script path: `scripts/oarag_vlm_command_adapter.py`
- stdin contract: `vlm-command-request-v1`
- stdout contract: `{"observations": [...]}`
- preflight mode: `--preflight`

The current execution environment did not have the required real VLM config, so
the bounded 2-lecture real VLM run was not executed. No deterministic or mock VLM
output was generated or presented as real object evidence.

| Check | Status |
| --- | --- |
| command adapter added | passed |
| adapter accepts `vlm-command-request-v1` JSON stdin | passed by focused test |
| adapter outputs accepted `observations` JSON | passed by focused test |
| confidence validation | implemented |
| missing credential/config preflight | blocked before frame processing |
| real 2-lecture VLM run | not executed |
| evidence-unit smoke rerun | not executed |

## Missing Config

The preflight blocker is:

- set `OARAG_VLM_API_KEY` or `OPENAI_API_KEY`
- set `OARAG_VLM_MODEL` or pass a real `--vlm-model` so the preflight command can
  receive `--model {model}`

Optional config:

- `OARAG_VLM_BASE_URL` for OpenAI-compatible non-default endpoints
- `OARAG_VLM_TIMEOUT_SECONDS` for request timeout
- adapter `--frame-root <private-frame-root>` when frame paths in the private
  manifest are relative

## Command Shape

Use this shape for each private 2-lecture project once credential/config exists:

```bash
PYTHONPATH=src python3 -m oarag run-vlm-alignment \
  --project-dir <private-project-dir> \
  --vlm-backend command \
  --vlm-model <real-vlm-model-id> \
  --vlm-options '{"command":["python3","-m","oarag.vision.vlm_command_adapter","--frame-root","<private-frame-root>"],"input_mode":"json-stdin","preflight_command":["python3","-m","oarag.vision.vlm_command_adapter","--preflight","--model","{model}"],"prompt_template_version":"vlm-visual-parser-v1"}' \
  --max-vlm-frames 12 \
  --candidate-max-per-segment 1 \
  --candidate-max-per-window 4 \
  --resume
```

The adapter sends an OpenAI-compatible chat-completions request and asks the model
to return strict JSON observations. Each successful observation must include
`visual_description` or `detected_text`; `confidence`, if present, must be between
0 and 1.

## Why The Real Run Did Not Execute

The real run requires a credential-backed VLM endpoint and model. The preflight
reported missing credential/model config before any frame processing, which is the
intended safe failure mode. Running the deterministic or mock backend here would
inflate coverage without real visual evidence and would contaminate the paper
evidence trail.

## Checks Run

```bash
python3 -m pytest tests/test_vlm_command_adapter.py tests/test_vlm.py tests/test_vlm_alignment_pipeline.py -q
python3 -m pytest tests/test_vlm_command_adapter.py tests/test_vlm.py tests/test_vlm_alignment_pipeline.py tests/test_visual_entities.py tests/test_evidence_units.py tests/test_evidence_unit_smoke.py tests/test_cli.py -q
python3 -m pytest -q
PYTHONPATH=src python3 -m oarag run-vlm --help
PYTHONPATH=src python3 -m oarag run-vlm-alignment --help
PYTHONPATH=src python3 -c 'from oarag.cli import main; main(["extract-visual-entities", "--help"])'
python3 scripts/oarag_vlm_command_adapter.py --preflight --api-key-env OARAG_TEST_MISSING_VLM_API_KEY_198 --fallback-api-key-env OARAG_TEST_MISSING_OPENAI_API_KEY_198 --model fixture-vlm
python3 scripts/oarag_vlm_command_adapter.py --preflight
```

Focused adapter/VLM tests passed: `22 passed`.
Expanded focused tests passed: `97 passed`.
Full tests passed: `362 passed, 4 skipped`.
