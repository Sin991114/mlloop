"""Ensemble-opportunity probe: price ensembling with zero training.

Every finished run already shipped held-out predictions. Averaging them on the
common rows and scoring against the best single run — paired, on identical
rows — answers "would combining what we already have beat the best model?"
before anyone trains an ensemble. The probe generates a hypothesis; a proper
ensemble run confirms it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diagnostics import positive_proba

MIN_COMMON_ROWS = 50
MIN_COMMON_FRACTION = 0.6


def run_ensemble_probe(
    *,
    member_predictions: dict[str, pd.DataFrame],
    metric_compute,
    metric_name: str,
    direction: str,
    task_type: str,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """member_predictions: run_id -> predictions frame (row_id, y_true, y_pred, proba_*)."""
    ids = list(member_predictions)
    common = None
    for df in member_predictions.values():
        rows = set(df["row_id"].tolist())
        common = rows if common is None else (common & rows)
    smallest = min(len(df) for df in member_predictions.values())
    if common is None or len(common) < max(MIN_COMMON_ROWS, int(MIN_COMMON_FRACTION * smallest)):
        return {
            "ok": False,
            "error": (
                f"the runs share only {0 if common is None else len(common)} held-out rows — "
                "ensembling can only be priced on a common evaluation set. Use runs that "
                "share the same split, or ship cv_predictions covering all rows."
            ),
            "runs": ids,
        }
    order = sorted(common)
    aligned = {
        run_id: df.set_index("row_id").loc[order].reset_index()
        for run_id, df in member_predictions.items()
    }

    reference = aligned[ids[0]]["y_true"].to_numpy()
    for run_id, df in aligned.items():
        if not np.array_equal(df["y_true"].to_numpy(), reference):
            return {
                "ok": False,
                "error": (
                    f"y_true disagrees between {ids[0]} and {run_id} on shared rows — "
                    "the runs were evaluated against different labels or encodings."
                ),
                "runs": ids,
            }

    # Build the ensemble frame.
    probas = {rid: positive_proba(df) for rid, df in aligned.items()}
    ensemble = aligned[ids[0]][["row_id", "y_true"]].copy()
    if task_type == "classification" and all(p is not None for p in probas.values()):
        mean_proba = np.mean([probas[rid] for rid in ids], axis=0)
        # Preserve the members' selection-rate semantics (matters for metrics
        # like AMS that key off a selection cut, harmless for the rest).
        positive = np.max(reference)
        rate = float(np.mean([(aligned[rid]["y_pred"] == positive).mean() for rid in ids]))
        rate = min(max(rate, 0.01), 0.99)
        threshold = float(np.quantile(mean_proba, 1 - rate))
        ensemble["y_pred"] = np.where(mean_proba >= threshold, positive, np.min(reference))
        ensemble["proba_1"] = mean_proba
    elif task_type == "classification":
        votes = np.stack([aligned[rid]["y_pred"].to_numpy() for rid in ids])
        positive = np.max(reference)
        ensemble["y_pred"] = np.where(
            (votes == positive).mean(axis=0) >= 0.5, positive, np.min(reference)
        )
    else:
        ensemble["y_pred"] = np.mean(
            [aligned[rid]["y_pred"].to_numpy(dtype=float) for rid in ids], axis=0
        )

    per_run = {}
    for rid in ids:
        try:
            per_run[rid] = round(float(metric_compute(aligned[rid])), 4)
        except Exception:
            per_run[rid] = None
    scored = {rid: v for rid, v in per_run.items() if v is not None}
    if len(scored) < 2:
        return {"ok": False, "error": "fewer than two member runs could be scored", "runs": ids}
    sign = 1.0 if direction == "maximize" else -1.0
    best_id = max(scored, key=lambda rid: sign * scored[rid])
    ensemble_value = float(metric_compute(ensemble))
    gain = sign * (ensemble_value - scored[best_id])

    # Paired bootstrap: ensemble vs best single on identical resampled rows.
    rng = np.random.default_rng(seed)
    n = len(order)
    deltas = []
    best_frame = aligned[best_id]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            d = sign * (
                float(metric_compute(ensemble.iloc[idx].reset_index(drop=True)))
                - float(metric_compute(best_frame.iloc[idx].reset_index(drop=True)))
            )
        except Exception:
            continue
        if np.isfinite(d):
            deltas.append(d)
    if len(deltas) >= 20:
        paired_std = float(np.std(deltas, ddof=1))
        bar = max(2 * paired_std, 1e-6)
        significant = bool(gain > bar)
    else:
        paired_std, bar, significant = None, None, False

    worth = significant
    if worth:
        conclusion = (
            f"averaging {len(ids)} runs on {n} shared rows scores {ensemble_value:.4f} "
            f"{metric_name} vs best single {best_id} {scored[best_id]:.4f} — gain "
            f"{gain:+.4f} clears the paired 2-sigma bar ({bar:.4f}). Register an ensembling "
            "hypothesis and confirm with a proper run."
        )
    else:
        conclusion = (
            f"averaging {len(ids)} runs scores {ensemble_value:.4f} {metric_name} vs best "
            f"single {best_id} {scored[best_id]:.4f} (gain {gain:+.4f}"
            + (f", paired 2-sigma bar {bar:.4f}" if bar is not None else "")
            + ") — these members are too correlated to pay; try more diverse model families "
            "before concluding ensembling is dead."
        )
    return {
        "ok": True,
        "runs": ids,
        "n_common_rows": n,
        "per_run": per_run,
        "best_single": {"id": best_id, "value": scored[best_id]},
        "ensemble_value": round(ensemble_value, 4),
        "gain": round(gain, 4),
        "paired_std": None if paired_std is None else round(paired_std, 4),
        "significance_bar": None if bar is None else round(bar, 4),
        "verdict": "ensemble_worth_testing" if worth else "ensemble_unlikely_to_help",
        "conclusion": conclusion,
        "note": (
            "Zero-training screening from stored predictions; a real ensemble (bagging, "
            "stacking, diverse families) can do better than a plain average. Confirm with "
            "a hypothesis-driven run."
        ),
    }
