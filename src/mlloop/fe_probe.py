"""Feature-engineering opportunity probe: price FE before spending runs on it.

Screens arithmetic combinations (difference / ratio / product) of the top
numeric features, plus stacked-model features (isolation-forest anomaly score,
out-of-fold kNN prediction), for incremental cross-validated signal over the
current representation. The probe generates hypotheses; it never replaces a
proper ledger run — screening gains carry a multiple-testing caveat and the
stacked features carry a mild out-of-fold reuse bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .tabular import prepare

MIN_GAIN = 0.005  # gains below this are never worth a run, whatever the SEM says


def run_fe_probe(
    *,
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    top_k: int = 5,
    quick: bool = False,
    seed: int = 0,
    max_rows: int = 20000,
) -> dict:
    prep = prepare(df, target_column, task_type, max_rows=max_rows, seed=seed)
    X_base, y = prep["X"], prep["y"]
    classification = task_type == "classification"
    binary = classification and len(prep["classes"]) == 2
    scoring = "roc_auc" if binary else ("accuracy" if classification else "r2")
    n_splits = 3 if quick else 5
    if classification:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = HistGradientBoostingClassifier(random_state=seed, max_iter=60 if quick else 200)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = HistGradientBoostingRegressor(random_state=seed, max_iter=60 if quick else 200)

    base_scores = cross_val_score(model, X_base, y, cv=cv, scoring=scoring)
    score_base = float(base_scores.mean())

    def eval_gain(extra: dict[str, np.ndarray]) -> dict:
        """Paired fold-wise comparison against the base representation.

        Base and augmented models share the exact same CV splits, so the gain's
        uncertainty is the SEM of the paired per-fold differences — much tighter
        than the raw fold-score spread. 3x SEM (not 2x) because the sweep tests
        many candidates.
        """
        X_aug = np.hstack(
            [X_base] + [np.asarray(v, dtype=float).reshape(-1, 1) for v in extra.values()]
        )
        aug_scores = cross_val_score(model, X_aug, y, cv=cv, scoring=scoring)
        diff = aug_scores - base_scores
        gain = float(diff.mean())
        sem = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        return {
            "gain": round(gain, 4),
            "sem": round(sem, 4),
            "significant": bool(gain > max(3 * sem, MIN_GAIN)),
        }

    # -- arithmetic combinations of the top-k numeric raw features ---------
    # Ranking uses the base model's permutation importance, not marginal mutual
    # information: features that only matter through interactions (XOR-like) have
    # zero marginal signal but nonzero model importance — exactly the case where
    # an engineered combination pays off.
    raw_numeric = prep["feature_frame"].select_dtypes(include="number")
    raw_numeric = raw_numeric.loc[:, raw_numeric.nunique() > 2]
    mi_fn = mutual_info_classif if classification else mutual_info_regression
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(y), min(len(y), 4000), replace=False)

    combos: dict[str, np.ndarray] = {}
    if raw_numeric.shape[1] >= 2:
        from sklearn.inspection import permutation_importance

        filled = raw_numeric.fillna(raw_numeric.median())
        fitted = model.fit(X_base, y)
        perm_sample = rng.choice(len(y), min(len(y), 3000), replace=False)
        perm = permutation_importance(
            fitted, X_base[perm_sample], y[perm_sample], n_repeats=3, random_state=seed
        )
        importance_of = dict(zip(prep["feature_names"], perm.importances_mean))
        ranked_cols = sorted(
            filled.columns, key=lambda c: importance_of.get(c, 0.0), reverse=True
        )
        if sum(importance_of.get(c, 0.0) > 0 for c in filled.columns) < 2:
            mi = mi_fn(filled.to_numpy()[sample], y[sample], random_state=seed)
            ranked_cols = [c for _, c in sorted(zip(mi, filled.columns),
                                                key=lambda t: t[0], reverse=True)]
        top = list(ranked_cols)[:top_k]
        for i, a in enumerate(top):
            va = filled[a].to_numpy(dtype=float)
            for b in top[i + 1:]:
                vb = filled[b].to_numpy(dtype=float)
                combos[f"diff({a},{b})"] = va - vb
                combos[f"product({a},{b})"] = va * vb
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.abs(vb) > 1e-12, va / vb, np.nan)
                median = float(np.nanmedian(ratio)) if np.isfinite(np.nanmedian(ratio)) else 0.0
                combos[f"ratio({a},{b})"] = np.nan_to_num(ratio, nan=median, posinf=median, neginf=median)

    combined = eval_gain(combos) if combos else {"gain": 0.0, "sem": 0.0, "significant": False}
    top_candidates: list[dict] = []
    if combos and (combined["significant"] or combined["gain"] > MIN_GAIN):
        combo_matrix = np.column_stack(list(combos.values()))
        combo_mi = mi_fn(combo_matrix[sample], y[sample], random_state=seed)
        ranked = sorted(zip(combo_mi, combos.keys()), key=lambda t: t[0], reverse=True)[:6]
        for _, name in ranked:
            top_candidates.append({"feature": name, **eval_gain({name: combos[name]})})
        top_candidates.sort(key=lambda c: c["gain"], reverse=True)

    # -- stacked-model features --------------------------------------------
    stacked: dict[str, dict] = {}
    iso = IsolationForest(n_estimators=100, random_state=seed).fit(X_base)
    stacked["isolation_forest_score"] = eval_gain({"iso": iso.score_samples(X_base)})
    k = int(max(3, min(25, len(y) // 20)))
    if classification:
        knn_oof = cross_val_predict(KNeighborsClassifier(n_neighbors=k), X_base, y, cv=cv, method="predict_proba")
        knn_feature = knn_oof[:, 1] if binary else knn_oof.max(axis=1)
    else:
        knn_feature = cross_val_predict(KNeighborsRegressor(n_neighbors=k), X_base, y, cv=cv)
    stacked["knn_oof_prediction"] = eval_gain({"knn": knn_feature})

    candidates = (
        [("combined arithmetic set", combined)]
        + [(c["feature"], c) for c in top_candidates]
        + list(stacked.items())
    )
    significant = [(name, c) for name, c in candidates if c["significant"]]
    n_swept = len(combos) + len(stacked)

    if significant:
        best_name, best = max(significant, key=lambda t: t[1]["gain"])
        conclusion = (
            f"screening found incremental signal: {best_name} gains +{best['gain']:.4f} "
            f"{scoring} (paired 3x-SEM bar {max(3 * best['sem'], MIN_GAIN):.4f}, "
            f"{n_swept} candidates swept) — register a hypothesis and confirm with a proper run"
        )
    else:
        best = max((c for _, c in candidates), key=lambda c: c["gain"])
        conclusion = (
            f"swept {n_swept} candidates over the top-{top_k} numeric features: best screening "
            f"gain {best['gain']:+.4f} {scoring} fails the paired 3x-SEM significance bar "
            "(multiple-testing adjusted) — feature engineering is unlikely to beat the "
            "current representation"
        )

    return {
        "scoring": scoring,
        "score_base": round(score_base, 4),
        "combined_arithmetic": combined,
        "top_candidates": top_candidates,
        "stacked_feature_gains": stacked,
        "n_candidates_swept": n_swept,
        "n_rows_used": prep["n_rows"],
        "verdict": "fe_worth_testing" if significant else "fe_unlikely_to_help",
        "conclusion": conclusion,
        "note": (
            "Screening estimates only: gains use paired fold-wise comparison with a 3x-SEM "
            "bar (multiple-testing adjusted); stacked features reuse out-of-fold predictions "
            "(mild bias). Confirm any promising candidate with a hypothesis-driven ledger run."
        ),
    }
