"""Data forensics battery (DESIGN.md §8).

Answers one question with independent lines of evidence: is the performance
ceiling set by the data, or by the modeling? The probes train quick reference
models internally — they never touch the run ledger and are not candidate
solutions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    learning_curve,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from . import viz
from .tabular import prepare

NOISE_RATE_ALARM = 0.08  # estimated label-noise rate above this flags a data problem


def run_forensics(
    *,
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    out_dir: Path | str,
    quick: bool = False,
    seed: int = 0,
    max_rows: int = 20000,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = prepare(df, target_column, task_type, max_rows=max_rows, seed=seed)
    X, y = prep["X"], prep["y"]
    classification = task_type == "classification"
    binary = classification and len(prep["classes"]) == 2
    scoring = "roc_auc" if binary else ("accuracy" if classification else "r2")
    n_splits = 3 if quick else 5
    if classification:
        min_class = int(np.bincount(y).min())
        n_splits = max(2, min(n_splits, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = HistGradientBoostingClassifier(random_state=seed, max_iter=60 if quick else 200)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = HistGradientBoostingRegressor(random_state=seed, max_iter=60 if quick else 200)

    items: dict[str, dict] = {}
    items["signal_check"] = _signal_check(model, X, y, cv, scoring, out_dir, quick, seed)
    has_signal = items["signal_check"]["details"]["has_signal"]
    if classification:
        items["label_noise"] = _label_noise(model, X, y, cv, prep, has_signal)
    items["conflict_rate"] = _conflict_rate(df, target_column, task_type)
    items["learning_curve"] = _learning_curve(model, X, y, cv, scoring, out_dir, quick, seed)
    items["feature_signal"] = _feature_signal(X, y, prep["feature_names"], classification, binary, model, cv, out_dir, seed)
    items["simple_model_bound"] = _simple_model_bound(X, y, cv, scoring, classification, seed, quick)

    verdict = _synthesize(items, task_type, scoring)
    return {
        "scoring": scoring,
        "n_rows_used": prep["n_rows"],
        "subsampled": prep["subsampled"],
        "dropped_columns": prep["dropped_columns"],
        "classes": prep["classes"],
        "items": items,
        "verdict": verdict,
    }


def _signal_check(model, X, y, cv, scoring, out_dir, quick, seed) -> dict:
    real = float(cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())
    rng = np.random.default_rng(seed)
    shuffled_scores = []
    for _ in range(2 if quick else 3):
        y_shuffled = rng.permutation(y)
        shuffled_scores.append(float(cross_val_score(model, X, y_shuffled, cv=cv, scoring=scoring).mean()))
    shuffled_mean = float(np.mean(shuffled_scores))
    shuffled_std = float(np.std(shuffled_scores)) or 1e-6
    gap = real - shuffled_mean
    has_signal = gap > max(3 * shuffled_std, 0.02)

    fig, ax = viz.new_fig(4.6, 3.0)
    ax.bar(["real labels", "shuffled labels"], [real, shuffled_mean], color=[viz.ACCENT, viz.MUTED])
    ax.set_ylabel(scoring)
    chart = viz.save_svg(fig, out_dir / "signal_check.svg")

    if has_signal:
        conclusion = (
            f"real signal present: {scoring} {real:.3f} with real labels vs {shuffled_mean:.3f} "
            f"with shuffled labels (gap {gap:+.3f})"
        )
        confidence = "high" if gap > 0.1 else "medium"
    else:
        conclusion = (
            f"NO detectable signal: {scoring} {real:.3f} with real labels is indistinguishable "
            f"from {shuffled_mean:.3f} with shuffled labels"
        )
        confidence = "high" if gap < shuffled_std else "medium"
    return {
        "conclusion": conclusion,
        "confidence": confidence,
        "details": {
            "score_real": real,
            "score_shuffled_mean": shuffled_mean,
            "score_shuffled_std": shuffled_std,
            "gap": gap,
            "has_signal": bool(has_signal),
        },
        "chart": chart,
    }


def _label_noise(model, X, y, cv, prep, has_signal) -> dict:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    proba_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    issues = find_label_issues(labels=y, pred_probs=proba_oof)
    noise_rate = float(issues.mean())
    quality = get_label_quality_scores(labels=y, pred_probs=proba_oof)
    classes = prep["classes"]
    order = np.argsort(quality)
    suspects = []
    for i in order[:30]:
        if not issues[i]:
            continue
        suspects.append(
            {
                "dataset_row": int(prep["original_row_index"][i]),
                "given_label": classes[y[i]],
                "suggested_label": classes[int(np.argmax(proba_oof[i]))],
                "label_quality": round(float(quality[i]), 4),
            }
        )

    if not has_signal:
        conclusion = (
            f"nominal estimate {noise_rate:.1%}, but UNRELIABLE — without feature signal "
            "(see signal_check) confident-learning noise estimates are not meaningful"
        )
        confidence = "low"
    else:
        conclusion = f"estimated {noise_rate:.1%} of labels are inconsistent with the feature signal"
        confidence = "high" if noise_rate > 2 * NOISE_RATE_ALARM or noise_rate < NOISE_RATE_ALARM / 2 else "medium"
    return {
        "conclusion": conclusion,
        "confidence": confidence,
        "details": {"estimated_noise_rate": noise_rate, "n_flagged": int(issues.sum()), "suspects": suspects},
        "chart": None,
    }


def _conflict_rate(df, target_column, task_type) -> dict:
    feature_cols = [c for c in df.columns if c != target_column]
    duplicated = df[df.duplicated(subset=feature_cols, keep=False)]
    n = len(df)
    if duplicated.empty:
        return {
            "conclusion": "no exact duplicate feature rows — the conflict-rate bound is not informative here",
            "confidence": "high",
            "details": {"duplicate_fraction": 0.0, "irreducible_error_lower_bound": 0.0},
            "chart": None,
        }
    groups = duplicated.groupby(feature_cols, dropna=False)[target_column]
    if task_type == "classification":
        # In each duplicate group, rows not matching the majority label are unavoidable errors.
        conflicting = int(sum(size - counts.max() for size, counts in
                              ((len(g), g.value_counts()) for _, g in groups)))
        bound = conflicting / n
        detail = {"duplicate_fraction": len(duplicated) / n, "irreducible_error_lower_bound": bound}
        conclusion = (
            f"{len(duplicated)/n:.1%} of rows are exact feature duplicates; conflicting labels "
            f"among them put a hard lower bound of {bound:.1%} on the error rate"
            if bound > 0
            else f"{len(duplicated)/n:.1%} of rows are duplicates but their labels agree — no conflict bound"
        )
    else:
        within_std = float(np.mean([g.std(ddof=0) for _, g in groups if len(g) > 1]))
        overall_std = float(df[target_column].std(ddof=0)) or 1e-12
        detail = {
            "duplicate_fraction": len(duplicated) / n,
            "within_duplicate_target_std": within_std,
            "overall_target_std": overall_std,
        }
        conclusion = (
            f"identical feature rows differ in target by std {within_std:.4g} "
            f"({within_std/overall_std:.0%} of overall target std) — irreducible noise floor"
        )
        detail["irreducible_error_lower_bound"] = within_std
    return {"conclusion": conclusion, "confidence": "high", "details": detail, "chart": None}


def _learning_curve(model, X, y, cv, scoring, out_dir, quick, seed) -> dict:
    sizes = np.linspace(0.2, 1.0, 4 if quick else 6)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=sizes, shuffle=True, random_state=seed
    )
    val_mean = val_scores.mean(axis=1)
    total_gain = float(val_mean[-1] - val_mean[0])
    last_gain = float(val_mean[-1] - val_mean[-2])
    plateaued = bool(abs(last_gain) < max(0.2 * abs(total_gain), 0.005))

    fig, ax = viz.new_fig()
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", color=viz.MUTED, label="train")
    ax.plot(train_sizes, val_mean, marker="o", color=viz.ACCENT, label="validation")
    ax.set_xlabel("training rows")
    ax.set_ylabel(scoring)
    ax.legend()
    chart = viz.save_svg(fig, out_dir / "learning_curve.svg")

    conclusion = (
        f"validation {scoring} has plateaued (last-step gain {last_gain:+.4f}) — more data alone is unlikely to help"
        if plateaued
        else f"validation {scoring} is still rising (last-step gain {last_gain:+.4f}) — more data would likely help"
    )
    return {
        "conclusion": conclusion,
        "confidence": "medium",
        "details": {
            "train_sizes": train_sizes.tolist(),
            "val_scores": val_mean.tolist(),
            "plateaued": plateaued,
            "last_step_gain": last_gain,
        },
        "chart": chart,
    }


def _feature_signal(X, y, feature_names, classification, binary, model, cv, out_dir, seed) -> dict:
    sample = np.random.default_rng(seed).choice(len(y), min(len(y), 5000), replace=False)
    mi_fn = mutual_info_classif if classification else mutual_info_regression
    mi = mi_fn(X[sample], y[sample], random_state=seed)
    ranked = sorted(zip(feature_names, mi), key=lambda t: t[1], reverse=True)
    top = [{"feature": name, "mutual_info": round(float(value), 4)} for name, value in ranked[:15]]

    leakage_hint = None
    if binary and top and top[0]["mutual_info"] > 0:
        top_idx = feature_names.index(top[0]["feature"])
        single = float(
            cross_val_score(model, X[:, [top_idx]], y, cv=cv, scoring="roc_auc").mean()
        )
        if single > 0.95:
            leakage_hint = {
                "feature": top[0]["feature"],
                "single_feature_auc": single,
                "warning": "a single feature nearly separates the target — check for leakage",
            }

    chart = None
    if top:
        rows = top[:10][::-1]
        fig, ax = viz.new_fig(6.4, 0.8 + 0.4 * len(rows))
        ax.barh([r["feature"] for r in rows], [r["mutual_info"] for r in rows], color=viz.ACCENT)
        ax.set_xlabel("mutual information with target")
        chart = viz.save_svg(fig, out_dir / "feature_signal.svg")

    informative = sum(1 for r in top if r["mutual_info"] > 0.01)
    conclusion = f"{informative} feature(s) carry measurable signal; strongest: {top[0]['feature']}" if top else "no features to rank"
    if leakage_hint:
        conclusion += f" — WARNING: {leakage_hint['warning']} (single-feature AUC {leakage_hint['single_feature_auc']:.3f})"
    return {
        "conclusion": conclusion,
        "confidence": "medium",
        "details": {"top_features": top, "leakage_hint": leakage_hint},
        "chart": chart,
    }


def _simple_model_bound(X, y, cv, scoring, classification, seed, quick) -> dict:
    k = max(3, min(15, len(y) // 20))
    if classification:
        candidates = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            f"knn_{k}": KNeighborsClassifier(n_neighbors=k),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed, max_iter=60 if quick else 200),
        }
    else:
        candidates = {
            "ridge": Ridge(),
            f"knn_{k}": KNeighborsRegressor(n_neighbors=k),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed, max_iter=60 if quick else 200),
        }
    scores = {}
    for name, estimator in candidates.items():
        try:
            result = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
            scores[name] = {"mean": float(result.mean()), "std": float(result.std())}
        except Exception as exc:
            scores[name] = {"error": str(exc)}
    valid = {k2: v for k2, v in scores.items() if "mean" in v}
    best_name = max(valid, key=lambda k2: valid[k2]["mean"]) if valid else None
    conclusion = (
        f"quick reference band: best simple model ({best_name}) reaches {scoring} "
        f"{valid[best_name]['mean']:.3f} ± {valid[best_name]['std']:.3f}"
        if best_name
        else "no simple model could be scored"
    )
    return {
        "conclusion": conclusion,
        "confidence": "medium",
        "details": {"scores": scores, "best": best_name},
        "chart": None,
    }


def _synthesize(items: dict, task_type: str, scoring: str) -> dict:
    signal = items["signal_check"]["details"]
    findings: list[str] = []
    recommendations: list[str] = []

    if not signal["has_signal"]:
        return {
            "verdict": "no_signal",
            "confidence": "high" if items["signal_check"]["confidence"] == "high" else "medium",
            "headline": (
                "The features carry no detectable signal for the target: a capable model scores "
                f"{signal['score_real']:.3f} {scoring}, statistically indistinguishable from "
                f"{signal['score_shuffled_mean']:.3f} on deliberately shuffled labels. "
                "No modeling effort can fix this — the data itself cannot answer the question."
            ),
            "findings": [items["signal_check"]["conclusion"]],
            "recommendations": [
                "Revisit feature engineering / data collection: the current features do not encode the target.",
                "Verify the labels describe what you think they describe.",
            ],
        }

    noise_rate = items.get("label_noise", {}).get("details", {}).get("estimated_noise_rate", 0.0)
    plateaued = items["learning_curve"]["details"]["plateaued"]
    conflict_bound = items["conflict_rate"]["details"].get("irreducible_error_lower_bound", 0.0)
    label_problem = task_type == "classification" and noise_rate >= NOISE_RATE_ALARM

    if label_problem:
        findings.append(items["label_noise"]["conclusion"])
        recommendations.append(
            "Review the suspect-label list in the verdict report and re-annotate; "
            "label noise this high caps every model."
        )
    if task_type == "classification" and conflict_bound > 0.01:
        findings.append(items["conflict_rate"]["conclusion"])
        recommendations.append(
            "Identical inputs with different labels put a hard floor on the error rate — "
            "collect distinguishing features or accept the ceiling."
        )
    findings.append(items["learning_curve"]["conclusion"])
    if plateaued:
        recommendations.append("More rows alone are unlikely to help; invest in features or labels instead.")
    else:
        recommendations.append("The learning curve is still rising — collecting more data is a credible lever.")

    if label_problem or (plateaued and conflict_bound > 0.01):
        verdict, confidence = "data_limited", "high" if label_problem and plateaued else "medium"
        headline = (
            "The evidence points to a data-limited ceiling: "
            + ("label noise is the dominant problem. " if label_problem else "")
            + ("identical inputs disagree on the answer. " if conflict_bound > 0.01 else "")
        ).strip()
    elif not plateaued:
        verdict, confidence = "more_data_needed", "medium"
        headline = "The data has real signal and the learning curve is still rising — more data is the cheapest next win."
    else:
        verdict, confidence = "model_limited", "medium"
        headline = (
            "The data looks healthy (signal present, low label noise, curve plateaued) — "
            "remaining gains must come from modeling and features."
        )
    return {
        "verdict": verdict,
        "confidence": confidence,
        "headline": headline,
        "findings": findings,
        "recommendations": recommendations,
    }
