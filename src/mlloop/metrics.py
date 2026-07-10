"""Metric registry and bootstrap noise-floor estimation.

The noise floor answers "how big must a delta be before it is evidence rather
than noise?" — estimated by bootstrap-resampling the prediction rows. It is a
cheap stand-in for multi-seed reruns and is refined by them when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable
    needs_proba: bool
    task: str  # 'classification' | 'regression'


def _f1_average(y_true) -> str:
    return "binary" if len(np.unique(y_true)) <= 2 else "macro"


_REGISTRY: dict[str, MetricSpec] = {}


def _register(names: tuple[str, ...], fn: Callable, needs_proba: bool, task: str) -> None:
    for name in names:
        _REGISTRY[name] = MetricSpec(names[0], fn, needs_proba, task)


_register(("auc", "roc_auc"), lambda yt, yp, pr: skm.roc_auc_score(yt, pr), True, "classification")
_register(("accuracy", "acc"), lambda yt, yp, pr: skm.accuracy_score(yt, yp), False, "classification")
_register(("f1",), lambda yt, yp, pr: skm.f1_score(yt, yp, average=_f1_average(yt)), False, "classification")
_register(
    ("precision",),
    lambda yt, yp, pr: skm.precision_score(yt, yp, average=_f1_average(yt), zero_division=0),
    False,
    "classification",
)
_register(
    ("recall",),
    lambda yt, yp, pr: skm.recall_score(yt, yp, average=_f1_average(yt), zero_division=0),
    False,
    "classification",
)
_register(("logloss", "log_loss"), lambda yt, yp, pr: skm.log_loss(yt, pr), True, "classification")
_register(("rmse",), lambda yt, yp, pr: float(np.sqrt(skm.mean_squared_error(yt, yp))), False, "regression")
_register(("mae",), lambda yt, yp, pr: skm.mean_absolute_error(yt, yp), False, "regression")
_register(("r2",), lambda yt, yp, pr: skm.r2_score(yt, yp), False, "regression")

FALLBACKS = {"classification": "accuracy", "regression": "rmse"}


def resolve_metric(name: str, task_type: str) -> tuple[MetricSpec, bool]:
    """Return (spec, is_fallback). Unknown names fall back to a computable default."""
    spec = _REGISTRY.get((name or "").strip().lower())
    if spec is not None:
        return spec, False
    return _REGISTRY[FALLBACKS[task_type]], True


def compute(spec: MetricSpec, y_true, y_pred, proba=None) -> float | None:
    try:
        value = spec.fn(np.asarray(y_true), np.asarray(y_pred), None if proba is None else np.asarray(proba))
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def bootstrap_noise_floor(
    spec: MetricSpec, y_true, y_pred, proba=None, n_boot: int = 200, seed: int = 0
) -> dict | None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = None if proba is None else np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        value = compute(spec, y_true[idx], y_pred[idx], None if proba is None else proba[idx])
        if value is not None:
            values.append(value)
    if len(values) < max(20, n_boot // 4):
        return None
    arr = np.asarray(values)
    std = float(arr.std(ddof=1))
    return {
        "metric": spec.name,
        "n_boot": len(values),
        "mean": float(arr.mean()),
        "std": std,
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "min_significant_delta": 2 * std,
        "note": (
            "Bootstrap over prediction rows; deltas below ~2 std are indistinguishable "
            "from noise. Multi-seed reruns refine this."
        ),
    }
