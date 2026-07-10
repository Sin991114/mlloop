"""Post-run diagnostics battery (DESIGN.md §7).

Every diagnostic is computed from the artifact contract plus the goal dataset —
never from the user's training code. Each item returns a plain-language
conclusion, structured details, and optionally an SVG chart file name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import viz
from .metrics import bootstrap_noise_floor, resolve_metric

MIN_BUCKET = 10


def positive_proba(predictions: pd.DataFrame) -> np.ndarray | None:
    """Positive-class probability for binary tasks; None when not derivable."""
    proba_cols = [c for c in predictions.columns if c.startswith("proba_")]
    if len(proba_cols) == 1:
        return predictions[proba_cols[0]].to_numpy(dtype=float)
    if len(proba_cols) == 2:
        # Convention: the lexically greatest class label is the positive one.
        return predictions[sorted(proba_cols)[-1]].to_numpy(dtype=float)
    return None


def run_diagnostics(
    *,
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    primary_metric: str,
    metric_direction: str,
    predictions: pd.DataFrame,
    run_metrics: dict,
    out_dir: Path | str,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_id = predictions["row_id"].to_numpy()
    if len(row_id) and (row_id.min() < 0 or row_id.max() >= len(df)):
        raise ValueError(
            "predictions row_id out of range for the goal dataset — row_id must be "
            "the 0-based row index into the dataset registered at goal_define"
        )
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    proba = positive_proba(predictions)
    features = df.drop(columns=[target_column]).iloc[row_id].reset_index(drop=True)

    if task_type == "classification":
        per_row_error = (y_true != y_pred).astype(float)
        error_name = "error rate"
    else:
        per_row_error = np.abs(y_true.astype(float) - y_pred.astype(float))
        error_name = "absolute error"
    overall_error = float(per_row_error.mean())

    items: dict[str, dict] = {}
    items["error_slices"] = _error_slices(features, per_row_error, overall_error, error_name, out_dir)
    items["noise_floor"] = _noise_floor(primary_metric, task_type, y_true, y_pred, proba)
    if task_type == "classification":
        items["confusion"] = _confusion(y_true, y_pred, out_dir)
        if proba is not None and len(np.unique(y_true)) == 2:
            items["calibration"] = _calibration(y_true, proba, out_dir)
        items["class_balance"] = _class_balance(y_true)
    else:
        items["residuals"] = _residuals(y_true.astype(float), y_pred.astype(float), out_dir)
    items["overfit_gap"] = _overfit_gap(primary_metric, metric_direction, run_metrics)

    return {"overall_error": overall_error, "error_name": error_name, "items": items}


def _error_slices(features, per_row_error, overall, error_name, out_dir) -> dict:
    n = len(per_row_error)
    min_bucket = max(MIN_BUCKET, int(0.02 * n))
    err = pd.Series(per_row_error)
    slices: list[dict] = []
    for col in list(features.columns)[:30]:
        series = features[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            try:
                buckets = pd.qcut(series, 4, duplicates="drop").astype(str)
            except (ValueError, TypeError):
                continue
        else:
            as_str = series.astype(str)
            top = as_str.value_counts().index[:8]
            buckets = as_str.where(as_str.isin(top), other="(other)")
        for bucket, group in err.groupby(buckets):
            if len(group) < min_bucket:
                continue
            rate = float(group.mean())
            if overall > 0 and rate > overall:
                slices.append(
                    {
                        "feature": col,
                        "bucket": str(bucket),
                        "n": int(len(group)),
                        "error": round(rate, 4),
                        "lift": round(rate / overall, 2),
                    }
                )
    slices.sort(key=lambda s: s["lift"], reverse=True)
    slices = slices[:8]

    chart = None
    if slices:
        fig, ax = viz.new_fig(6.4, 0.8 + 0.45 * len(slices))
        labels = [f"{s['feature']} = {s['bucket']}" for s in slices][::-1]
        ax.barh(labels, [s["lift"] for s in slices][::-1], color=viz.ACCENT)
        ax.axvline(1.0, color=viz.MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel(f"{error_name} lift vs overall")
        chart = viz.save_svg(fig, out_dir / "error_slices.svg")

    if slices:
        worst = slices[0]
        conclusion = (
            f"worst slice: {worst['feature']} = {worst['bucket']} has {worst['lift']}x "
            f"the overall {error_name} (n={worst['n']})"
        )
    else:
        conclusion = (
            f"no slice with {error_name} meaningfully above overall "
            f"(buckets under {min_bucket} rows ignored)"
        )
    return {
        "conclusion": conclusion,
        "details": {"overall": overall, "error_name": error_name, "slices": slices},
        "chart": chart,
    }


def _noise_floor(primary_metric, task_type, y_true, y_pred, proba) -> dict:
    spec, is_fallback = resolve_metric(primary_metric, task_type)
    if spec.needs_proba and proba is None:
        return {
            "conclusion": (
                f"cannot bootstrap '{spec.name}' without probabilities — add proba_<class> "
                "columns to predictions.parquet"
            ),
            "details": {},
            "chart": None,
        }
    floor = bootstrap_noise_floor(spec, y_true, y_pred, proba if spec.needs_proba else None)
    if floor is None:
        return {
            "conclusion": f"bootstrap failed for '{spec.name}' (too few valid resamples)",
            "details": {},
            "chart": None,
        }
    fallback_note = (
        f" (primary metric '{primary_metric}' is not in the registry; bootstrapped "
        f"'{spec.name}' instead)"
        if is_fallback
        else ""
    )
    conclusion = (
        f"{spec.name} noise floor: std ±{floor['std']:.4f}; treat deltas below "
        f"~{floor['min_significant_delta']:.4f} as noise{fallback_note}"
    )
    return {"conclusion": conclusion, "details": floor, "chart": None}


def _confusion(y_true, y_pred, out_dir) -> dict:
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                pairs.append({"true": str(true_label), "predicted": str(pred_label), "n": int(matrix[i, j])})
    pairs.sort(key=lambda p: p["n"], reverse=True)

    chart = None
    if len(labels) <= 12:
        fig, ax = viz.new_fig(4.6, 4.0)
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)), [str(l) for l in labels])
        ax.set_yticks(range(len(labels)), [str(l) for l in labels])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
        chart = viz.save_svg(fig, out_dir / "confusion.svg")

    conclusion = (
        f"most confused: true {pairs[0]['true']} predicted as {pairs[0]['predicted']} "
        f"({pairs[0]['n']} rows)"
        if pairs
        else "no misclassifications in the held-out predictions"
    )
    return {
        "conclusion": conclusion,
        "details": {"labels": [str(l) for l in labels], "matrix": matrix.tolist(), "top_pairs": pairs[:5]},
        "chart": chart,
    }


def _calibration(y_true, proba, out_dir) -> dict:
    y_true = np.asarray(y_true)
    positive = np.max(y_true)
    is_positive = (y_true == positive).astype(float)
    bins = np.clip((proba * 10).astype(int), 0, 9)
    ece = 0.0
    xs, ys = [], []
    n = len(proba)
    for b in range(10):
        mask = bins == b
        if not mask.any():
            continue
        confidence = float(proba[mask].mean())
        accuracy = float(is_positive[mask].mean())
        ece += abs(accuracy - confidence) * mask.sum() / n
        xs.append(confidence)
        ys.append(accuracy)

    fig, ax = viz.new_fig(4.6, 4.0)
    ax.plot([0, 1], [0, 1], color=viz.MUTED, linestyle="--", linewidth=1)
    ax.plot(xs, ys, marker="o", color=viz.ACCENT)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    chart = viz.save_svg(fig, out_dir / "calibration.svg")

    quality = "well calibrated" if ece < 0.05 else ("moderately miscalibrated" if ece < 0.15 else "badly miscalibrated")
    return {
        "conclusion": f"ECE = {ece:.3f} ({quality})",
        "details": {"ece": float(ece), "bin_confidence": xs, "bin_accuracy": ys},
        "chart": chart,
    }


def _class_balance(y_true) -> dict:
    values, counts = np.unique(np.asarray(y_true), return_counts=True)
    dist = {str(v): int(c) for v, c in zip(values, counts)}
    ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
    level = "balanced" if ratio < 2 else ("imbalanced" if ratio < 10 else "severely imbalanced")
    return {
        "conclusion": f"class ratio {ratio:.1f}:1 in held-out labels ({level})",
        "details": {"counts": dist, "imbalance_ratio": ratio},
        "chart": None,
    }


def _residuals(y_true, y_pred, out_dir) -> dict:
    residuals = y_true - y_pred
    bias = float(residuals.mean())
    with np.errstate(invalid="ignore"):
        het = float(np.corrcoef(y_pred, np.abs(residuals))[0, 1]) if len(y_pred) > 2 else 0.0
    if not np.isfinite(het):
        het = 0.0

    sample = np.random.default_rng(0).choice(len(y_pred), min(len(y_pred), 2000), replace=False)
    fig, ax = viz.new_fig()
    ax.scatter(y_pred[sample], residuals[sample], s=8, alpha=0.5, color=viz.ACCENT)
    ax.axhline(0, color=viz.MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("prediction")
    ax.set_ylabel("residual (true - pred)")
    chart = viz.save_svg(fig, out_dir / "residuals.svg")

    notes = []
    if abs(bias) > 0.05 * (np.std(y_true) + 1e-12):
        notes.append(f"systematic bias {bias:+.4g}")
    if abs(het) > 0.3:
        notes.append(f"heteroscedastic (|residual| vs prediction corr {het:.2f})")
    conclusion = "; ".join(notes) if notes else "residuals look unbiased and homoscedastic"
    return {
        "conclusion": conclusion,
        "details": {"bias": bias, "residual_std": float(residuals.std()), "heteroscedasticity_corr": het},
        "chart": chart,
    }


def _overfit_gap(primary_metric, metric_direction, run_metrics) -> dict:
    train_key = f"train_{primary_metric}"
    if train_key not in (run_metrics or {}):
        return {
            "conclusion": (
                f"not measurable — report '{train_key}' alongside '{primary_metric}' in "
                "run_finish metrics to enable"
            ),
            "details": {},
            "chart": None,
        }
    train = float(run_metrics[train_key])
    val = float(run_metrics[primary_metric])
    gap = train - val if metric_direction == "maximize" else val - train
    label = "large train/validation gap — likely overfitting" if gap > 0.05 else "small train/validation gap"
    return {
        "conclusion": f"{label} (train {train:.4f} vs validation {val:.4f})",
        "details": {"train": train, "validation": val, "gap": gap},
        "chart": None,
    }
