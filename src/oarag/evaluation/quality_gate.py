from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from oarag.evaluation.benchmark import DEFAULT_MATRIX_VARIANTS, MATRIX_SCHEMA_VERSION


RETRIEVAL_QUALITY_GATE_SCHEMA_VERSION = "retrieval-quality-gate-v1"
RETRIEVAL_QUALITY_GATE_RESULT_SCHEMA_VERSION = "retrieval-quality-gate-result-v1"
PUBLIC_FIXTURE_GATE_NOTE = (
    "Public fixture regression guard only; thresholds are not paper claims."
)
DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS = [
    str(variant["variant_id"]) for variant in DEFAULT_MATRIX_VARIANTS
]


def check_retrieval_quality_gate(*, metrics_path: Path, config_path: Path) -> dict[str, Any]:
    """Load aggregate benchmark metrics and evaluate the retrieval quality gate."""
    metrics = _read_json(metrics_path)
    config = _read_json(config_path)
    return evaluate_retrieval_quality_gate(metrics=metrics, config=config)


def evaluate_retrieval_quality_gate(
    *,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate aggregate retrieval benchmark metrics against public fixture thresholds."""
    gate_id = str(config.get("gate_id") or "retrieval_quality_gate")
    note = str(config.get("note") or PUBLIC_FIXTURE_GATE_NOTE)
    suites = _list_of_dicts(metrics.get("suites"))
    configured_suites = _configured_suites(config)
    failures: list[dict[str, Any]] = []
    checked_suites: list[dict[str, Any]] = []

    if not suites:
        failures.append(
            _failure(
                code="metrics_suites_missing",
                message="metrics payload missing suites list; expected aggregate benchmark metrics.json",
            )
        )

    for suite_config in configured_suites:
        suite_result = _evaluate_suite_gate(
            suites=suites,
            suite_config=suite_config,
            failures=failures,
        )
        checked_suites.append(suite_result)

    return {
        "schema_version": RETRIEVAL_QUALITY_GATE_RESULT_SCHEMA_VERSION,
        "gate_id": gate_id,
        "passed": not failures,
        "note": note,
        "suite_count": len(suites),
        "checked_suite_count": len(checked_suites),
        "failure_count": len(failures),
        "failures": failures,
        "checked_suites": checked_suites,
        "privacy": {
            "payload": "aggregate_metrics_only",
            "query_content": "excluded",
            "transcript_content": "excluded",
            "evidence_content": "excluded",
        },
    }


def _evaluate_suite_gate(
    *,
    suites: list[dict[str, Any]],
    suite_config: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    selector = _suite_selector(suite_config)
    expected_schema = str(suite_config.get("schema_version") or MATRIX_SCHEMA_VERSION)
    thresholds = _metric_thresholds(suite_config)
    required_variants = _required_variants(suite_config)
    suite = _find_suite(suites=suites, selector=selector)
    suite_result: dict[str, Any] = {
        "suite_id": selector.get("suite_id"),
        "suite_type": selector.get("suite_type"),
        "expected_schema_version": expected_schema,
        "required_variants": required_variants,
        "thresholds": thresholds,
        "found": suite is not None,
        "checked_variants": [],
    }

    if suite is None:
        failures.append(
            _failure(
                code="suite_missing",
                message=(
                    f"suite={selector.get('suite_id') or '*'} "
                    f"type={selector.get('suite_type') or '*'} missing from metrics payload"
                ),
                suite_id=selector.get("suite_id"),
                suite_type=selector.get("suite_type"),
            )
        )
        return suite_result

    suite_id = _optional_text(suite.get("suite_id"))
    suite_type = _optional_text(suite.get("suite_type"))
    suite_result["suite_id"] = suite_id
    suite_result["suite_type"] = suite_type
    suite_result["schema_version"] = _optional_text(suite.get("schema_version"))
    suite_result["query_count"] = _optional_int(suite.get("query_count"))
    suite_result["variant_count"] = _optional_int(suite.get("variant_count"))

    if "schema_version" not in suite:
        failures.append(
            _failure(
                code="suite_schema_missing",
                message=(
                    f"suite={suite_id} type={suite_type} missing schema_version "
                    f"(expected {expected_schema})"
                ),
                suite_id=suite_id,
                suite_type=suite_type,
            )
        )
    elif suite.get("schema_version") != expected_schema:
        failures.append(
            _failure(
                code="suite_schema_mismatch",
                message=(
                    f"suite={suite_id} type={suite_type} schema_version={suite.get('schema_version')} "
                    f"does not match expected {expected_schema}"
                ),
                suite_id=suite_id,
                suite_type=suite_type,
            )
        )

    variant_metrics = _variant_metrics_by_id(suite)
    if not variant_metrics:
        failures.append(
            _failure(
                code="variant_metrics_missing",
                message=f"suite={suite_id} type={suite_type} missing variant_metrics/variants aggregate block",
                suite_id=suite_id,
                suite_type=suite_type,
            )
        )

    checked_variants: list[dict[str, Any]] = []
    for variant_id in required_variants:
        variant_metric = variant_metrics.get(variant_id)
        variant_result: dict[str, Any] = {
            "variant_id": variant_id,
            "found": variant_metric is not None,
            "metrics": {},
        }
        if variant_metric is None:
            failures.append(
                _failure(
                    code="variant_missing",
                    message=f"suite={suite_id} missing required variant={variant_id}",
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                )
            )
            checked_variants.append(variant_result)
            continue

        for metric_name, threshold in thresholds.items():
            if metric_name not in variant_metric:
                failures.append(
                    _failure(
                        code="metric_missing",
                        message=(
                            f"suite={suite_id} variant={variant_id} missing metric={metric_name}"
                        ),
                        suite_id=suite_id,
                        suite_type=suite_type,
                        variant_id=variant_id,
                        metric=metric_name,
                        threshold=threshold,
                    )
                )
                variant_result["metrics"][metric_name] = {
                    "threshold": threshold,
                    "value": None,
                    "passed": False,
                }
                continue

            value = variant_metric.get(metric_name)
            if not _is_number(value):
                failures.append(
                    _failure(
                        code="metric_not_numeric",
                        message=(
                            f"suite={suite_id} variant={variant_id} metric={metric_name} "
                            "is not numeric"
                        ),
                        suite_id=suite_id,
                        suite_type=suite_type,
                        variant_id=variant_id,
                        metric=metric_name,
                        threshold=threshold,
                    )
                )
                variant_result["metrics"][metric_name] = {
                    "threshold": threshold,
                    "value": None,
                    "passed": False,
                }
                continue

            numeric_value = round(float(value), 4)
            passed = numeric_value >= threshold
            variant_result["metrics"][metric_name] = {
                "threshold": threshold,
                "value": numeric_value,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    _failure(
                        code="metric_below_threshold",
                        message=(
                            f"suite={suite_id} variant={variant_id} metric={metric_name} "
                            f"value={numeric_value} below public fixture guard threshold={threshold}"
                        ),
                        suite_id=suite_id,
                        suite_type=suite_type,
                        variant_id=variant_id,
                        metric=metric_name,
                        value=numeric_value,
                        threshold=threshold,
                    )
                )

        checked_variants.append(variant_result)

    suite_result["checked_variants"] = checked_variants
    return suite_result


def _configured_suites(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_suites = config.get("suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ValueError("retrieval quality gate config requires a non-empty suites list")
    suites = _list_of_dicts(raw_suites)
    if len(suites) != len(raw_suites):
        raise ValueError("retrieval quality gate config suites must be JSON objects")
    return suites


def _suite_selector(suite_config: Mapping[str, Any]) -> dict[str, str | None]:
    suite_id = _optional_text(suite_config.get("suite_id"))
    suite_type = _optional_text(suite_config.get("suite_type"))
    if not suite_id and not suite_type:
        raise ValueError("retrieval quality gate suite config requires suite_id or suite_type")
    return {"suite_id": suite_id, "suite_type": suite_type}


def _metric_thresholds(suite_config: Mapping[str, Any]) -> dict[str, float]:
    raw_thresholds = suite_config.get("thresholds", suite_config.get("metric_thresholds"))
    if not isinstance(raw_thresholds, Mapping) or not raw_thresholds:
        raise ValueError("retrieval quality gate suite config requires metric thresholds")
    thresholds: dict[str, float] = {}
    for raw_name, raw_value in raw_thresholds.items():
        metric_name = str(raw_name).strip()
        if not metric_name:
            raise ValueError("retrieval quality gate metric threshold has an empty metric name")
        if not _is_number(raw_value):
            raise ValueError(f"retrieval quality gate threshold for {metric_name} must be numeric")
        thresholds[metric_name] = round(float(raw_value), 4)
    return thresholds


def _required_variants(suite_config: Mapping[str, Any]) -> list[str]:
    raw_variants = suite_config.get("required_variants") or DEFAULT_RETRIEVAL_ANSWER_MATRIX_VARIANTS
    if isinstance(raw_variants, str):
        variants = [item.strip() for item in raw_variants.split(",")]
    elif isinstance(raw_variants, list):
        variants = [str(item).strip() for item in raw_variants]
    else:
        raise ValueError("retrieval quality gate required_variants must be a list or comma string")
    variants = [item for item in variants if item]
    if not variants:
        raise ValueError("retrieval quality gate requires at least one variant")
    return variants


def _find_suite(
    *,
    suites: list[dict[str, Any]],
    selector: Mapping[str, str | None],
) -> dict[str, Any] | None:
    for suite in suites:
        suite_id = selector.get("suite_id")
        suite_type = selector.get("suite_type")
        if suite_id and suite.get("suite_id") != suite_id:
            continue
        if suite_type and suite.get("suite_type") != suite_type:
            continue
        return suite
    return None


def _variant_metrics_by_id(suite: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_metrics = suite.get("variant_metrics")
    if isinstance(raw_metrics, Mapping):
        return {
            str(variant_id): dict(metric)
            for variant_id, metric in raw_metrics.items()
            if isinstance(metric, Mapping)
        }
    raw_variants = suite.get("variants")
    if isinstance(raw_variants, list):
        return {
            str(item["variant_id"]): dict(item)
            for item in raw_variants
            if isinstance(item, Mapping) and item.get("variant_id")
        }
    return {}


def _failure(
    *,
    code: str,
    message: str,
    suite_id: str | None = None,
    suite_type: str | None = None,
    variant_id: str | None = None,
    metric: str | None = None,
    value: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    for key, item in (
        ("suite_id", suite_id),
        ("suite_type", suite_type),
        ("variant_id", variant_id),
        ("metric", metric),
        ("value", value),
        ("threshold", threshold),
    ):
        if item is not None:
            payload[key] = item
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
