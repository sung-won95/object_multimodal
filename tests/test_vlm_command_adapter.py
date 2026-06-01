import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from oarag.vision import vlm_command_adapter as adapter

import pytest


def test_vlm_command_adapter_preflight_reports_public_missing_config() -> None:
    env = _adapter_env()
    for key in (
        "OARAG_VLM_API_KEY",
        "OPENAI_API_KEY",
        "OARAG_VLM_MODEL",
        "OARAG_TEST_VLM_API_KEY",
        "OARAG_TEST_OPENAI_API_KEY",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oarag.vision.vlm_command_adapter",
            "--preflight",
            "--api-key-env",
            "OARAG_TEST_VLM_API_KEY",
            "--fallback-api-key-env",
            "OARAG_TEST_OPENAI_API_KEY",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["ok"] is False
    assert payload["status"] == "blocked_missing_config"
    assert "OARAG_TEST_VLM_API_KEY or OARAG_TEST_OPENAI_API_KEY" in payload["missing"]
    assert payload["safe_to_publish"] is True
    assert "sk-" not in result.stdout


def test_vlm_command_adapter_reads_request_and_outputs_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8\xff\xe0fixture-image")
    request = {
        "schema_version": "vlm-command-request-v1",
        "model": "fixture-vlm",
        "prompt_template_version": "fixture-template",
        "frame": {
            "project_id": "project",
            "video_id": "video",
            "frame_id": "frame_000001",
            "frame_path": str(frame_path),
            "timestamp": 3.5,
        },
    }
    captured: dict[str, object] = {}

    def fake_call_chat_completions(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "observations": [
                                    {
                                        "observation_type": "diagram",
                                        "visual_description": "A real adapter fixture chart",
                                        "detected_text": "Loss",
                                        "confidence": 0.82,
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setenv("OARAG_TEST_VLM_API_KEY", "test-api-key")
    monkeypatch.setattr(adapter, "_call_chat_completions", fake_call_chat_completions)
    args = argparse.Namespace(
        api_key_env="OARAG_TEST_VLM_API_KEY",
        fallback_api_key_env="OARAG_TEST_OPENAI_API_KEY",
        base_url_env="OARAG_TEST_VLM_BASE_URL",
        frame_root=None,
    )

    observations = adapter.run_adapter_request(request=request, args=args)

    assert observations[0]["visual_description"] == "A real adapter fixture chart"
    assert observations[0]["detected_text"] == "Loss"
    assert observations[0]["confidence"] == 0.82
    assert observations[0]["parser_version"] == (
        "oarag-openai-compatible-command-v1"
    )
    assert captured["api_key"] == "test-api-key"
    assert str(captured["image_url"]).startswith("data:image/jpeg;base64,")


def _adapter_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    return env
