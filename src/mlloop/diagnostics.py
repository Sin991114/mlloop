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
        binary = len(np.unique(y_true)) == 2
        if proba is not None and binary:
            items["calibration"] = _calibration(y_true, proba, out_dir)
            items["operating_curve"] = _operating_curve(y_true, proba, out_dir)
        items["class_balance"] = _class_balance(y_true)
        if binary:
            try:
                items["missed_positives"] = _missed_positives(
                    df, target_column, predictions, y_true, y_pred, out_dir
                )
            except Exception as exc:  # the explanation must never sink the whole diagnosis
                items["missed_positives"] = {
                    "conclusion": f"missed-positive explanation failed: {exc}",
                    "details": {},
                    "chart": None,
                }
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


def _operating_curve(y_true, proba, out_dir) -> dict:
    """Overkill (share of good flagged) vs catch rate (recall) across thresholds.

    Makes the AUC operational — and exposes degenerate predictions (e.g. a
    constant score) that a bare AUC number can hide.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    positive = np.max(y_true)
    y_bin = (y_true == positive).astype(int)
    n_distinct = int(len(np.unique(proba)))
    if n_distinct < 3:
        return {
            "conclusion": (
                f"predicted probabilities are (nearly) constant ({n_distinct} distinct value(s)) — "
                "the model does not rank rows, so AUC and any threshold choice are meaningless "
                "for this run"
            ),
            "details": {"n_distinct_probabilities": n_distinct},
            "chart": None,
        }

    fpr, tpr, _ = roc_curve(y_bin, proba)

    def recall_at(overkill: float) -> float:
        return float(tpr[max(np.searchsorted(fpr, overkill, side="right") - 1, 0)])

    def overkill_at(recall: float) -> float | None:
        if tpr.max() < recall:
            return None
        return float(fpr[np.argmax(tpr >= recall)])

    points = {
        "recall_at_5pct_overkill": round(recall_at(0.05), 3),
        "recall_at_10pct_overkill": round(recall_at(0.10), 3),
        "recall_at_20pct_overkill": round(recall_at(0.20), 3),
        "overkill_for_80pct_recall": overkill_at(0.80),
        "overkill_for_90pct_recall": overkill_at(0.90),
    }

    fig, ax = viz.new_fig(5.4, 4.6)
    ax.plot(fpr, tpr, color=viz.ACCENT)
    ax.plot([0, 1], [0, 1], color=viz.MUTED, linestyle="--", linewidth=1)
    for x in (0.05, 0.10, 0.20):
        r = recall_at(x)
        ax.scatter([x], [r], color=viz.WARN, zorder=3, s=25)
        ax.annotate(f"{r:.0%} @ {x:.0%}", (x, r), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("overkill — share of good flagged (FPR)")
    ax.set_ylabel("catch rate — share of bad caught (recall)")
    chart = viz.save_svg(fig, out_dir / "operating_curve.svg")

    ov80 = points["overkill_for_80pct_recall"]
    cost = f"catching 80% of positives costs {ov80:.0%} overkill" if ov80 is not None else "80% recall is unreachable"
    conclusion = f"{cost}; at 10% overkill you catch {points['recall_at_10pct_overkill']:.0%} of positives"
    points = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in points.items()}
    return {"conclusion": conclusion, "details": {"operating_points": points}, "chart": chart}


def _missed_positives(df, target_column, predictions, y_true, y_pred, out_dir) -> dict:
    """Explain the uncaught positives: feature-limited or model-limited?

    An out-of-fold reference model scores every missed positive. Rows the
    reference model also scores as negative look like 'good data' to the
    current features — no model can catch them. SHAP shows which features
    pull them toward the negative class.
    """
    try:
        import shap
    except ImportError:
        return {
            "conclusion": "shap is not installed — `pip install shap` enables missed-positive explanations",
            "details": {},
            "chart": None,
        }
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from .tabular import prepare

    positive = np.max(y_true)
    fn_mask = (y_true == positive) & (y_pred != positive)
    n_caught = int(((y_true == positive) & (y_pred == positive)).sum())
    fn_rows = predictions["row_id"].to_numpy()[fn_mask]
    if fn_rows.size == 0:
        return {
            "conclusion": "no missed positives in the held-out predictions — nothing to explain",
            "details": {"n_missed": 0, "n_caught": n_caught},
            "chart": None,
        }

    prep = prepare(df, target_column, "classification")
    position_of = {int(r): i for i, r in enumerate(prep["original_row_index"])}
    fn_positions = np.array([position_of[int(r)] for r in fn_rows if int(r) in position_of])
    X, y = prep["X"], prep["y"]
    positive_code = int(np.bincount(y[fn_positions]).argmax())  # FN rows are true positives

    reference = HistGradientBoostingClassifier(
        random_state=0, max_iter=200, learning_rate=0.05, min_samples_leaf=30
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(reference, X, y, cv=cv, method="predict_proba")[:, positive_code]
    fn_oof = oof[fn_positions]
    fraction_feature_limited = float((fn_oof < 0.5).mean())

    reference.fit(X, y)
    sample = fn_positions
    if len(sample) > 400:
        sample = np.random.default_rng(0).choice(fn_positions, 400, replace=False)
    values = shap.TreeExplainer(reference)(X[sample]).values
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, positive_code]
    elif positive_code == 0:
        values = -values
    names = prep["feature_names"]
    pulling_negative = sorted(zip(names, values.mean(axis=0)), key=lambda t: t[1])[:5]

    import matplotlib.pyplot as plt

    shap.plots.beeswarm(
        shap.Explanation(values=values, base_values=np.zeros(len(values)),
                         data=X[sample], feature_names=names),
        max_display=12, show=False,
    )
    chart = viz.save_svg(plt.gcf(), out_dir / "missed_positives_shap.svg")

    top = ", ".join(f"{name} ({value:+.2f})" for name, value in pulling_negative[:3])
    conclusion = (
        f"of {fn_rows.size} missed positives, {fraction_feature_limited:.0%} look like negatives even to an "
        f"out-of-fold reference model — feature-limited rows no model can catch with these features; "
        f"{1 - fraction_feature_limited:.0%} are learnable misses. "
        f"Features pulling the missed toward 'good': {top}"
    )
    return {
        "conclusion": conclusion,
        "details": {
            "n_missed": int(fn_rows.size),
            "n_caught": n_caught,
            "fraction_feature_limited": round(fraction_feature_limited, 3),
            "fraction_learnable": round(1 - fraction_feature_limited, 3),
            "mean_reference_oof_proba_on_missed": round(float(fn_oof.mean()), 3),
            "features_pulling_toward_negative": [
                {"feature": name, "mean_shap": round(float(value), 3)}
                for name, value in pulling_negative
            ],
        },
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
