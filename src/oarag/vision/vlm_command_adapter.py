from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


ENV_API_KEY = "OARAG_VLM_API_KEY"
ENV_API_KEY_FALLBACK = "OPENAI_API_KEY"
ENV_BASE_URL = "OARAG_VLM_BASE_URL"
ENV_MODEL = "OARAG_VLM_MODEL"
ENV_TIMEOUT_SECONDS = "OARAG_VLM_TIMEOUT_SECONDS"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
PARSER_VERSION = "oarag-openai-compatible-command-v1"
VERIFIER_PARSER_VERSION = "oarag-openai-compatible-vlm-link-verifier-v1"
OBSERVATION_REQUEST_SCHEMA_VERSION = "vlm-command-request-v1"
VERIFIER_REQUEST_SCHEMA_VERSION = "vlm-entity-link-verifier-request-v1"
GENERIC_REASON_BY_DECISION = {
    "verified": "vlm_decision_verified",
    "rejected": "vlm_decision_rejected",
    "uncertain": "vlm_decision_uncertain",
}
PUBLIC_REASON_TEXT_BY_CODE = {
    "speaker_refers_to_visible_chart": "VLM judged that the speaker refers to the visible entity.",
    "speaker_refers_to_visible_curve": "VLM judged that the speaker refers to the visible entity.",
    "visual_entity_not_referenced": "VLM judged that the visual entity is not referenced.",
    "not_referenced": "VLM judged that the visual entity is not referenced.",
    "ambiguous_deictic_reference": "VLM judged that the reference is ambiguous.",
    "ambiguous": "VLM judged that the reference is ambiguous.",
    "configured_default_decision": "VLM verifier used a configured default decision.",
    "vlm_decision_verified": "VLM judged this link as verified.",
    "vlm_decision_rejected": "VLM judged this link as rejected.",
    "vlm_decision_uncertain": "VLM judged this link as uncertain.",
}
PUBLIC_REASON_CODE_ALLOWLIST = frozenset(PUBLIC_REASON_TEXT_BY_CODE)


class AdapterConfigError(RuntimeError):
    pass


class AdapterResponseError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preflight:
        return _run_preflight(args)
    return _run_request(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI-compatible VLM command adapter for oarag command backend. "
            "Normal mode reads a vlm-command-request-v1 JSON object from stdin "
            "and writes {'observations': [...]} JSON to stdout."
        )
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate credential/model configuration and exit before frame processing.",
    )
    parser.add_argument(
        "--model",
        help=f"Model override for preflight mode. Defaults to ${ENV_MODEL}.",
    )
    parser.add_argument(
        "--api-key-env",
        default=ENV_API_KEY,
        help=f"Primary API key environment variable name. Defaults to {ENV_API_KEY}.",
    )
    parser.add_argument(
        "--fallback-api-key-env",
        default=ENV_API_KEY_FALLBACK,
        help=f"Fallback API key environment variable name. Defaults to {ENV_API_KEY_FALLBACK}.",
    )
    parser.add_argument(
        "--base-url-env",
        default=ENV_BASE_URL,
        help=f"Base URL environment variable name. Defaults to {ENV_BASE_URL}.",
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        help="Optional root used to resolve relative frame_path values from stdin.",
    )
    return parser


def _run_preflight(args: argparse.Namespace) -> int:
    missing = _missing_config(args=args, request_model=args.model)
    if missing:
        _write_json(
            {
                "ok": False,
                "status": "blocked_missing_config",
                "message": "Set the required environment/config before running real VLM frames.",
                "missing": missing,
                "safe_to_publish": True,
            }
        )
        return 2

    api_key_env = _selected_api_key_env(args)
    _write_json(
        {
            "ok": True,
            "status": "configured",
            "message": "Real VLM adapter config is present.",
            "api_key_env": api_key_env,
            "base_url_env": args.base_url_env,
            "model_configured": True,
            "safe_to_publish": True,
        }
    )
    return 0


def _run_request(args: argparse.Namespace) -> int:
    try:
        request = _read_request()
        _validate_request(request)
        missing = _missing_config(args=args, request_model=_request_model(request))
        if missing:
            raise AdapterConfigError(
                "Missing required real VLM config: " + ", ".join(missing)
            )
        if request.get("schema_version") == VERIFIER_REQUEST_SCHEMA_VERSION:
            decision = run_verifier_adapter_request(request=request, args=args)
            _write_json(decision)
            return 0
        observations = run_adapter_request(request=request, args=args)
    except AdapterConfigError as exc:
        _write_error(exc, status="blocked_missing_config")
        return 2
    except Exception as exc:
        _write_error(exc, status="adapter_error")
        return 1

    _write_json({"observations": observations})
    return 0


def run_adapter_request(*, request: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    frame = _request_frame(request)
    image_url = _frame_image_data_url(frame=frame, frame_root=args.frame_root)
    response_payload = _call_chat_completions(
        api_key=_api_key(args),
        base_url=_base_url(args),
        model=_request_model(request),
        image_url=image_url,
        prompt=_prompt_from_request(request),
        timeout_seconds=_timeout_seconds(),
    )
    observations = _observations_from_response(response_payload)
    return [_normalize_observation(observation) for observation in observations]


def run_verifier_adapter_request(
    *,
    request: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame = _request_frame(request)
    image_url = _frame_image_data_url(frame=frame, frame_root=args.frame_root)
    response_payload = _call_chat_completions(
        api_key=_api_key(args),
        base_url=_base_url(args),
        model=_request_model(request),
        image_url=image_url,
        prompt=_verifier_prompt_from_request(request),
        timeout_seconds=_timeout_seconds(),
    )
    return _normalize_verifier_decision(_verifier_decision_from_response(response_payload))


def _read_request() -> dict[str, Any]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise AdapterResponseError("stdin must contain vlm-command-request-v1 JSON") from exc
    if not isinstance(payload, dict):
        raise AdapterResponseError("stdin JSON must be an object")
    return payload


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") not in {
        OBSERVATION_REQUEST_SCHEMA_VERSION,
        VERIFIER_REQUEST_SCHEMA_VERSION,
    }:
        raise AdapterResponseError(
            "expected schema_version=vlm-command-request-v1 "
            "or vlm-entity-link-verifier-request-v1"
        )
    frame = request.get("frame")
    if not isinstance(frame, dict):
        raise AdapterResponseError("request.frame is required")


def _missing_config(*, args: argparse.Namespace, request_model: str | None) -> list[str]:
    missing: list[str] = []
    if _selected_api_key_env(args) is None:
        missing.append(f"{args.api_key_env} or {args.fallback_api_key_env}")
    if _non_empty(request_model) is None and _non_empty(os.environ.get(ENV_MODEL)) is None:
        missing.append(f"{ENV_MODEL} or request.model")
    return missing


def _selected_api_key_env(args: argparse.Namespace) -> str | None:
    if _non_empty(os.environ.get(args.api_key_env)) is not None:
        return str(args.api_key_env)
    if _non_empty(os.environ.get(args.fallback_api_key_env)) is not None:
        return str(args.fallback_api_key_env)
    return None


def _api_key(args: argparse.Namespace) -> str:
    selected = _selected_api_key_env(args)
    if selected is None:
        raise AdapterConfigError("missing real VLM API key environment variable")
    return os.environ[selected]


def _base_url(args: argparse.Namespace) -> str:
    return (_non_empty(os.environ.get(args.base_url_env)) or DEFAULT_BASE_URL).rstrip("/")


def _timeout_seconds() -> float:
    raw = _non_empty(os.environ.get(ENV_TIMEOUT_SECONDS))
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError as exc:
        raise AdapterConfigError(f"{ENV_TIMEOUT_SECONDS} must be numeric") from exc


def _request_model(request: Mapping[str, Any]) -> str | None:
    return _non_empty(request.get("model")) or _non_empty(os.environ.get(ENV_MODEL))


def _request_frame(request: Mapping[str, Any]) -> Mapping[str, Any]:
    frame = request.get("frame")
    if not isinstance(frame, Mapping):
        raise AdapterResponseError("request.frame must be an object")
    return frame


def _frame_image_data_url(*, frame: Mapping[str, Any], frame_root: Path | None) -> str:
    frame_path_text = _non_empty(frame.get("frame_path"))
    if frame_path_text is None:
        raise AdapterResponseError("request.frame.frame_path is required")
    path = Path(frame_path_text).expanduser()
    if not path.is_absolute():
        if frame_root is None:
            raise AdapterResponseError(
                "relative frame_path requires --frame-root in adapter command"
            )
        path = frame_root.expanduser() / path
    resolved = path.resolve()
    if not resolved.exists():
        raise AdapterResponseError("frame_path does not exist")
    mime_type = mimetypes.guess_type(resolved.name)[0] or "image/jpeg"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _prompt_from_request(request: Mapping[str, Any]) -> str:
    template = _non_empty(request.get("prompt_template"))
    if template is not None:
        return template
    version = _non_empty(request.get("prompt_template_version")) or "vlm-visual-parser-v1"
    return (
        f"Prompt template: {version}\n"
        "Parse the lecture frame into concise object-aligned visual evidence. "
        "Return strict JSON with an observations array. Each observation must include "
        "visual_description or detected_text, optional confidence between 0 and 1, "
        "and may include observation_type, position, and relations. Do not include "
        "raw transcript, local paths, or secrets."
    )


def _verifier_prompt_from_request(request: Mapping[str, Any]) -> str:
    template = _non_empty(request.get("prompt_template"))
    if template is not None:
        return template
    version = _non_empty(request.get("prompt_template_version")) or "vlm-entity-link-verifier-v1"
    segment = request.get("segment") if isinstance(request.get("segment"), Mapping) else {}
    entity = request.get("visual_entity") if isinstance(request.get("visual_entity"), Mapping) else {}
    link = request.get("link") if isinstance(request.get("link"), Mapping) else {}
    return (
        f"Prompt template: {version}\n"
        "Decide whether the speaker in this transcript segment is actually referring "
        "to or explaining the visual entity in the frame. Return strict JSON only.\n"
        "Allowed decisions: verified, rejected, uncertain.\n"
        "Return fields: decision, reason_code, public_reason, confidence.\n"
        "The public_reason must be short and must not quote the raw transcript, expose "
        "local paths, or include private identifiers.\n"
        f"Link evidence: {json.dumps(link.get('evidence', []), ensure_ascii=False)}\n"
        f"Transcript segment: {segment.get('transcript_text', '')}\n"
        f"Visual entity metadata: {json.dumps(entity, ensure_ascii=False)}"
    )


def _call_chat_completions(
    *,
    api_key: str,
    base_url: str,
    model: str | None,
    image_url: str,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if model is None:
        raise AdapterConfigError("missing VLM model")
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise AdapterResponseError(f"VLM API HTTP error: {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AdapterResponseError(f"VLM API request failed: {exc.reason}") from exc

    try:
        loaded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AdapterResponseError("VLM API returned non-JSON response") from exc
    if not isinstance(loaded, dict):
        raise AdapterResponseError("VLM API response must be a JSON object")
    return loaded


def _observations_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = _chat_message_content(response)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdapterResponseError("VLM response message was not JSON") from exc
    if isinstance(parsed, dict) and isinstance(parsed.get("observations"), list):
        observations = parsed["observations"]
    elif isinstance(parsed, dict):
        observations = [parsed]
    elif isinstance(parsed, list):
        observations = parsed
    else:
        raise AdapterResponseError("VLM response JSON must be object or list")
    if not observations:
        raise AdapterResponseError("VLM response produced no observations")
    if not all(isinstance(observation, dict) for observation in observations):
        raise AdapterResponseError("VLM observations must be JSON objects")
    return observations


def _verifier_decision_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    content = _chat_message_content(response)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdapterResponseError("VLM verifier response message was not JSON") from exc
    if isinstance(parsed, Mapping) and isinstance(parsed.get("decision"), Mapping):
        parsed = parsed["decision"]
    if not isinstance(parsed, Mapping):
        raise AdapterResponseError("VLM verifier response JSON must be an object")
    return dict(parsed)


def _chat_message_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AdapterResponseError("VLM API response is missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise AdapterResponseError("VLM API choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise AdapterResponseError("VLM API choice is missing message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise AdapterResponseError("VLM API message content must be a JSON string")


def _normalize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    visual_description = _non_empty(
        observation.get("visual_description") or observation.get("description")
    )
    detected_text = _non_empty(observation.get("detected_text"))
    if visual_description is None and detected_text is None:
        raise AdapterResponseError("observation requires visual_description or detected_text")
    confidence = observation.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise AdapterResponseError("observation confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise AdapterResponseError("observation confidence must be between 0 and 1")

    normalized = dict(observation)
    normalized["parser_version"] = _non_empty(observation.get("parser_version")) or PARSER_VERSION
    if visual_description is not None:
        normalized["visual_description"] = visual_description
    if detected_text is not None:
        normalized["detected_text"] = detected_text
    if confidence is not None:
        normalized["confidence"] = confidence
    normalized.setdefault("observation_type", "frame_summary")
    return normalized


def _normalize_verifier_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    value = _non_empty(decision.get("decision") or decision.get("status"))
    if value is None:
        raise AdapterResponseError("verifier decision is required")
    normalized_decision = value.casefold()
    if normalized_decision not in {"verified", "rejected", "uncertain"}:
        raise AdapterResponseError("verifier decision must be verified, rejected, or uncertain")
    confidence = decision.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise AdapterResponseError("verifier confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise AdapterResponseError("verifier confidence must be between 0 and 1")
    reason_code = _safe_reason_code(
        decision.get("reason_code")
        or decision.get("public_reason_code")
        or decision.get("reason")
        or GENERIC_REASON_BY_DECISION[normalized_decision],
        decision=normalized_decision,
    )
    normalized = {
        "parser_version": _non_empty(decision.get("parser_version")) or VERIFIER_PARSER_VERSION,
        "decision": normalized_decision,
        "reason_code": reason_code,
        "public_reason": _safe_public_reason(
            reason_code=reason_code,
            decision=normalized_decision,
        ),
    }
    if confidence is not None:
        normalized["confidence"] = confidence
    return normalized


def _public_code(value: Any) -> str:
    text = (_non_empty(value) or "unspecified_public_reason").casefold()
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "unspecified_public_reason"


def _safe_reason_code(value: Any, *, decision: str) -> str:
    code = _public_code(value)
    if code in PUBLIC_REASON_CODE_ALLOWLIST:
        return code
    return GENERIC_REASON_BY_DECISION.get(decision, "vlm_decision_uncertain")


def _safe_public_reason(*, reason_code: str, decision: str) -> str:
    return PUBLIC_REASON_TEXT_BY_CODE.get(
        reason_code,
        PUBLIC_REASON_TEXT_BY_CODE[
            GENERIC_REASON_BY_DECISION.get(decision, "vlm_decision_uncertain")
        ],
    )


def _public_text(value: Any) -> str:
    text = " ".join((_non_empty(value) or "unspecified_public_reason").split())
    return text[:180]


def _write_error(exc: Exception, *, status: str) -> None:
    payload = {
        "ok": False,
        "status": status,
        "message": str(exc),
        "safe_to_publish": True,
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def _write_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _non_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
