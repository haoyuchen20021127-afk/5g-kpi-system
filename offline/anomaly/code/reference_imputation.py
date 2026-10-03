"""Per-feature imputation fitted only on explicitly supplied reference data.

This module does not read files, select data phases, clip values, or train a
detector. Callers must pass fit-reference values only to either fitting function.
The saved ``fit_observed_n`` is the reference evidence count before imputation;
filled repetitions must not make an otherwise undersized reference eligible.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


DISCRETE_FEATURES = frozenset({"cqi", "ri", "pcmax"})


def _as_1d_float(values) -> np.ndarray:
    """Keep each feature/reference group separate; reject accidental matrices."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Expected one-dimensional values for one feature/reference group")
    if np.iscomplexobj(raw):
        raise ValueError("Complex values are not supported")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Values must be convertible to real floating-point numbers") from error


def _finite_median(observed: np.ndarray) -> float:
    """Return the ordinary median without overflowing an even-pair average."""
    middle = len(observed) // 2
    if len(observed) % 2:
        return float(np.partition(observed, middle)[middle])
    pair = np.partition(observed, (middle - 1, middle))
    low, high = float(pair[middle - 1]), float(pair[middle])
    total = low + high
    # Half the sum gives the usual result (including subnormal rounding).
    # Only same-sign, very large finite pairs need a scaled sum instead.
    return total * 0.5 if math.isfinite(total) else low * 0.5 + high * 0.5


def fit_median_imputer(fit_values) -> dict[str, Any]:
    """Fit a scalar imputation value using finite, fit-reference observations.

    NaN and both infinities count as missing. If no finite observation exists,
    the JSON-safe result contains ``status='not_available'`` and ``median=None``.
    No calibration or test observations should be supplied to this function.
    """
    values = _as_1d_float(fit_values)
    observed_mask = np.isfinite(values)
    observed_n = int(observed_mask.sum())
    median = _finite_median(values[observed_mask]) if observed_n else None
    return {
        "version": 1,
        "status": "ok" if observed_n else "not_available",
        "reason": "finite_fit_reference_available" if observed_n else "no_finite_fit_reference",
        "median": median,
        "strategy": "median",
        "fill_value": median,
        "median_role": "imputation_value",
        "fit_total_n": int(values.size),
        "fit_observed_n": observed_n,
        "fit_missing_n": int(values.size - observed_n),
        "rule": "median_of_finite_provided_fit_values",
        "statistics_source": "provided_fit_reference_only",
        "missing_definition": "nan_positive_infinity_or_negative_infinity",
        "finite_value_policy": "preserve_without_clipping",
        "minimum_reference_count_basis": "fit_observed_n_before_imputation",
    }


def fit_reference_imputer(fit_values, feature=None) -> dict[str, Any]:
    """Use mode for CQI/RI/pcmax and median for other scalar feature groups.

    Feature matching is case-insensitive and exact. Mode ties are resolved by
    the smallest observed finite value, so filling cannot invent a new level.
    ``median`` remains descriptive metadata for mode, not its imputation value.
    """
    params = fit_median_imputer(fit_values)
    if feature is None or str(feature).lower() not in DISCRETE_FEATURES:
        return params
    values = _as_1d_float(fit_values)
    observed = values[np.isfinite(values)]
    fill_value = None
    if observed.size:
        levels, counts = np.unique(observed, return_counts=True)
        fill_value = float(levels[np.flatnonzero(counts == counts.max())[0]])
    params.update({
        "strategy": "mode",
        "fill_value": fill_value,
        "rule": "mode_of_finite_provided_fit_values",
        "mode_tie_break": "smallest_observed_finite_value",
        "median_role": "descriptive_only_not_imputation_value",
    })
    return params


def _validate_params(params: Mapping[str, Any]) -> tuple[str, float | None]:
    status = params.get("status")
    if status not in {"ok", "not_available"}:
        raise ValueError("Unknown imputer status")
    counts = []
    for name in ("fit_total_n", "fit_observed_n", "fit_missing_n"):
        value = params.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        counts.append(int(value))
    total, observed, missing = counts
    if total != observed + missing:
        raise ValueError("Imputer reference counts are inconsistent")
    strategy = params.get("strategy", "median")
    if strategy not in {"median", "mode"}:
        raise ValueError("Unknown imputer strategy")
    if "fill_value" in params:
        fill_value = params["fill_value"]
    elif strategy == "median":
        fill_value = params.get("median")
    else:
        raise ValueError("Mode imputer requires an explicit fill_value")
    if status == "not_available":
        if observed != 0 or fill_value is not None or params.get("median") is not None:
            raise ValueError("Unavailable imputer must have zero observations and no filling value")
        return status, None
    if observed == 0 or isinstance(fill_value, (bool, str)):
        raise ValueError("Available imputer requires a finite reference fill_value")
    try:
        fill_value = float(fill_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Available imputer requires a finite reference fill_value") from error
    if not math.isfinite(fill_value):
        raise ValueError("Available imputer requires a finite reference fill_value")
    if "median" in params:
        median = params["median"]
        if isinstance(median, (bool, str)):
            raise ValueError("Stored median must be finite numeric metadata")
        try:
            median = float(median)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Stored median must be finite numeric metadata") from error
        if not math.isfinite(median) or (strategy == "median" and median != fill_value):
            raise ValueError("Stored median and filling metadata are inconsistent")
    return status, fill_value


def transform(values, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return a filled copy and the original missing-value mask, without refitting.

    The mask is True for each original NaN/inf/-inf. Existing finite values are
    untouched. If the reference has no usable filling value, missing positions stay
    NaN; existing finite values are still retained, but no detector eligibility
    can be inferred from these later values. Neither input nor params is mutated.
    """
    values = _as_1d_float(values)
    status, fill_value = _validate_params(params)
    missing_mask = ~np.isfinite(values)
    filled = values.copy()
    filled[missing_mask] = fill_value if status == "ok" else np.nan
    return filled, missing_mask
