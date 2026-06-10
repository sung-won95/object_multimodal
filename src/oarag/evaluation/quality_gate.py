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


def check_retrieval_quality_gate(
    *,
    metrics_path: Path,
    config_path: Path,
    semantic_smoke_path: Path | None = None,
) -> dict[str, Any]:
    """Load aggregate benchmark metrics and evaluate the retrieval quality gate."""
    metrics = _read_json(metrics_path)
    config = _read_json(config_path)
    semantic_smoke = _read_json(semantic_smoke_path) if semantic_smoke_path is not None else None
    return evaluate_retrieval_quality_gate(
        metrics=metrics,
        config=config,
        semantic_smoke=semantic_smoke,
    )


def evaluate_retrieval_quality_gate(
    *,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    semantic_smoke: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate aggregate retrieval benchmark metrics against regression thresholds."""
    gate_id = str(config.get("gate_id") or "retrieval_quality_gate")
    note = str(config.get("note") or PUBLIC_FIXTURE_GATE_NOTE)
    suites = _list_of_dicts(metrics.get("suites"))
    configured_suites = _configured_suites(config)
    failures: list[dict[str, Any]] = []
    checked_suites: list[dict[str, Any]] = []
    semantic_smoke_result = _evaluate_semantic_smoke_gate(
        suites=suites,
        config=config,
        semantic_smoke=semantic_smoke,
        failures=failures,
    )

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
        "semantic_smoke": semantic_smoke_result,
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
    thresholds = _threshold_specs(
        suite_config.get("thresholds", suite_config.get("metric_thresholds")),
        scope="suite",
        failures=failures,
        suite_id=selector.get("suite_id"),
        suite_type=selector.get("suite_type"),
    )
    variant_thresholds = _variant_threshold_specs(
        suite_config.get("variant_thresholds"),
        failures=failures,
        suite_id=selector.get("suite_id"),
        suite_type=selector.get("suite_type"),
    )
    required_variants = _required_variants(suite_config)
    suite = _find_suite(suites=suites, selector=selector)
    suite_result: dict[str, Any] = {
        "suite_id": selector.get("suite_id"),
        "suite_type": selector.get("suite_type"),
        "expected_schema_version": expected_schema,
        "required_variants": required_variants,
        "thresholds": _threshold_values(thresholds),
        "variant_thresholds": {
            variant_id: _threshold_values(specs)
            for variant_id, specs in variant_thresholds.items()
        },
        "found": suite is not None,
        "checked_metrics": {},
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
    suite_result["skipped_count"] = _optional_int(suite.get("skipped_count"))

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

    suite_result["checked_metrics"] = _evaluate_metric_thresholds(
        metric_payload=suite,
        thresholds=thresholds,
        failures=failures,
        suite_id=suite_id,
        suite_type=suite_type,
        variant_id=None,
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

        skipped_count = _optional_int(variant_metric.get("skipped_count")) or 0
        variant_result["skipped_count"] = skipped_count
        if skipped_count > 0:
            failures.append(
                _failure(
                    code="required_variant_skipped",
                    message=(
                        f"suite={suite_id} required variant={variant_id} "
                        f"has skipped_count={skipped_count}"
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    value=float(skipped_count),
                    threshold=0.0,
                )
            )

        variant_result["metrics"] = _evaluate_metric_thresholds(
            metric_payload=variant_metric,
            thresholds=variant_thresholds.get(variant_id, {}),
            failures=failures,
            suite_id=suite_id,
            suite_type=suite_type,
            variant_id=variant_id,
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


def _threshold_specs(
    raw_thresholds: Any,
    *,
    scope: str,
    failures: list[dict[str, Any]],
    suite_id: str | None,
    suite_type: str | None,
    variant_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_thresholds, Mapping) or not raw_thresholds:
        if scope == "suite":
            failures.append(
                _failure(
                    code="thresholds_missing",
                    message="retrieval quality gate suite config requires metric thresholds",
                    suite_id=suite_id,
                    suite_type=suite_type,
                )
            )
        return {}
    thresholds: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in raw_thresholds.items():
        metric_name = str(raw_name).strip()
        if not metric_name:
            failures.append(
                _failure(
                    code="threshold_metric_name_empty",
                    message="retrieval quality gate metric threshold has an empty metric name",
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                )
            )
            continue

        if isinstance(raw_spec, Mapping):
            raw_value = raw_spec.get("threshold")
            allow_zero = raw_spec.get("allow_zero_threshold") is True
            reason = _optional_text(raw_spec.get("reason") or raw_spec.get("note"))
        else:
            raw_value = raw_spec
            allow_zero = False
            reason = None

        if not _is_number(raw_value):
            failures.append(
                _failure(
                    code="threshold_not_numeric",
                    message=f"retrieval quality gate threshold for {metric_name} must be numeric",
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                )
            )
            continue

        threshold = round(float(raw_value), 4)
        if threshold < 0:
            failures.append(
                _failure(
                    code="threshold_negative",
                    message=(
                        f"suite={suite_id} variant={variant_id or '*'} metric={metric_name} "
                        f"has negative threshold={threshold}"
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                    threshold=threshold,
                )
            )
        if threshold == 0 and (not allow_zero or scope != "variant"):
            reason_suffix = (
                " without allow_zero_threshold=true"
                if not allow_zero
                else "; zero thresholds are only allowed for variant-specific metrics"
            )
            failures.append(
                _failure(
                    code="zero_threshold_not_allowed",
                    message=(
                        f"suite={suite_id} variant={variant_id or '*'} metric={metric_name} "
                        f"uses threshold=0{reason_suffix}"
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                    threshold=threshold,
                )
            )
        if threshold == 0 and allow_zero and scope == "variant" and not reason:
            failures.append(
                _failure(
                    code="zero_threshold_reason_missing",
                    message=(
                        f"suite={suite_id} variant={variant_id or '*'} metric={metric_name} "
                        "allows threshold=0 but is missing a public-safe reason"
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                    threshold=threshold,
                )
            )

        thresholds[metric_name] = {
            "threshold": threshold,
            "allow_zero_threshold": allow_zero,
            "reason": reason,
        }
    return thresholds


def _variant_threshold_specs(
    raw_variant_thresholds: Any,
    *,
    failures: list[dict[str, Any]],
    suite_id: str | None,
    suite_type: str | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    if raw_variant_thresholds is None:
        return {}
    if not isinstance(raw_variant_thresholds, Mapping):
        failures.append(
            _failure(
                code="variant_thresholds_not_object",
                message="retrieval quality gate variant_thresholds must be a JSON object",
                suite_id=suite_id,
                suite_type=suite_type,
            )
        )
        return {}
    return {
        str(raw_variant_id): _threshold_specs(
            raw_thresholds,
            scope="variant",
            failures=failures,
            suite_id=suite_id,
            suite_type=suite_type,
            variant_id=str(raw_variant_id),
        )
        for raw_variant_id, raw_thresholds in raw_variant_thresholds.items()
    }


def _threshold_values(thresholds: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {
        metric_name: float(spec["threshold"])
        for metric_name, spec in thresholds.items()
        if _is_number(spec.get("threshold"))
    }


def _evaluate_metric_thresholds(
    *,
    metric_payload: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, Any]],
    failures: list[dict[str, Any]],
    suite_id: str | None,
    suite_type: str | None,
    variant_id: str | None,
) -> dict[str, dict[str, Any]]:
    checked: dict[str, dict[str, Any]] = {}
    for metric_name, spec in thresholds.items():
        threshold = float(spec["threshold"])
        if metric_name not in metric_payload:
            failures.append(
                _failure(
                    code="metric_missing",
                    message=_metric_message(
                        suite_id=suite_id,
                        variant_id=variant_id,
                        metric_name=metric_name,
                        suffix="missing metric",
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                    threshold=threshold,
                )
            )
            checked[metric_name] = {
                "threshold": threshold,
                "value": None,
                "passed": False,
            }
            continue

        value = metric_payload.get(metric_name)
        if not _is_number(value):
            failures.append(
                _failure(
                    code="metric_not_numeric",
                    message=_metric_message(
                        suite_id=suite_id,
                        variant_id=variant_id,
                        metric_name=metric_name,
                        suffix="is not numeric",
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                    threshold=threshold,
                )
            )
            checked[metric_name] = {
                "threshold": threshold,
                "value": None,
                "passed": False,
            }
            continue

        numeric_value = round(float(value), 4)
        passed = numeric_value >= threshold
        checked[metric_name] = {
            "threshold": threshold,
            "value": numeric_value,
            "passed": passed,
        }
        if not passed:
            failures.append(
                _failure(
                    code="metric_below_threshold",
                    message=(
                        f"suite={suite_id} variant={variant_id or '*'} metric={metric_name} "
                        f"value={numeric_value} below regression threshold={threshold}"
                    ),
                    suite_id=suite_id,
                    suite_type=suite_type,
                    variant_id=variant_id,
                    metric=metric_name,
                    value=numeric_value,
                    threshold=threshold,
                )
            )
    return checked


def _metric_message(
    *,
    suite_id: str | None,
    variant_id: str | None,
    metric_name: str,
    suffix: str,
) -> str:
    if variant_id is None:
        return f"suite={suite_id} metric={metric_name} {suffix}"
    return f"suite={suite_id} variant={variant_id} metric={metric_name} {suffix}"


def _evaluate_semantic_smoke_gate(
    *,
    suites: list[dict[str, Any]],
    config: Mapping[str, Any],
    semantic_smoke: Mapping[str, Any] | None,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provided": semantic_smoke is not None,
        "provider_backed": None,
        "hybrid_metric_reported": False,
        "hybrid_variant_ids": [],
    }
    if semantic_smoke is None:
        return result

    smoke_block = semantic_smoke.get("semantic_live_smoke")
    provider_backed = (
        isinstance(smoke_block, Mapping)
        and smoke_block.get("provider_backed") is True
    )
    result["provider_backed"] = provider_backed
    hybrid_variant_ids = _hybrid_variant_ids(
        suites=suites,
        config=config,
        semantic_smoke=semantic_smoke,
    )
    result["hybrid_variant_ids"] = sorted(hybrid_variant_ids)
    hybrid_metric_reported = _hybrid_metric_reported(
        suites=suites,
        hybrid_variant_ids=hybrid_variant_ids,
    )
    result["hybrid_metric_reported"] = hybrid_metric_reported

    if provider_backed is False and hybrid_metric_reported:
        failures.append(
            _failure(
                code="semantic_smoke_provider_unbacked_hybrid_metrics",
                message=(
                    "semantic_smoke provider_backed=false but hybrid variant metrics "
                    "are present in metrics payload"
                ),
            )
        )
    return result


def _hybrid_variant_ids(
    *,
    suites: list[dict[str, Any]],
    config: Mapping[str, Any],
    semantic_smoke: Mapping[str, Any],
) -> set[str]:
    variant_ids: set[str] = set()
    raw_smoke_ids = semantic_smoke.get("hybrid_variant_ids")
    if isinstance(raw_smoke_ids, list):
        variant_ids.update(str(item) for item in raw_smoke_ids if str(item).strip())
    raw_config_ids = config.get("hybrid_variant_ids")
    if isinstance(raw_config_ids, list):
        variant_ids.update(str(item) for item in raw_config_ids if str(item).strip())
    for suite in suites:
        for variant_id in _variant_metrics_by_id(suite):
            if "hybrid" in variant_id:
                variant_ids.add(variant_id)
    return variant_ids


def _hybrid_metric_reported(
    *,
    suites: list[dict[str, Any]],
    hybrid_variant_ids: set[str],
) -> bool:
    if not hybrid_variant_ids:
        return False
    metric_names = {
        "hit_at_10s",
        "mrr_at_max_delta",
        "grounded_answer_ratio",
        "expected_citation_hit_ratio",
    }
    for suite in suites:
        for variant_id, variant_metric in _variant_metrics_by_id(suite).items():
            if variant_id not in hybrid_variant_ids:
                continue
            if any(_is_number(variant_metric.get(metric_name)) for metric_name in metric_names):
                return True
    return False


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
