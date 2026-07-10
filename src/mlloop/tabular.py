"""Shared featurization for the forensics quick models.

Deliberately simple: numeric columns as-is (median-imputed), low-cardinality
categoricals one-hot encoded, everything else dropped and reported. The quick
models are reference probes, not candidate solutions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ONEHOT = 15
MAX_ENCODED_COLS = 150


def prepare(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    max_rows: int = 20000,
    seed: int = 0,
) -> dict:
    subsampled = False
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
        subsampled = True
    df = df.reset_index(drop=False).rename(columns={"index": "__orig_row__"})

    if task_type == "classification":
        codes, classes = pd.factorize(df[target_column])
        y = codes.astype(int)
        valid = y >= 0  # factorize maps NaN to -1
        classes = [str(c) for c in classes]
    else:
        y = pd.to_numeric(df[target_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        classes = None
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    X = df.drop(columns=[target_column, "__orig_row__"])
    blocks: list[np.ndarray] = []
    names: list[str] = []
    dropped: list[str] = []
    for col in X.columns:
        if len(names) >= MAX_ENCODED_COLS:
            dropped.append(col)
            continue
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            filled = series.fillna(series.median())
            blocks.append(filled.to_numpy(dtype=float).reshape(-1, 1))
            names.append(col)
        else:
            as_str = series.astype(str)
            if as_str.nunique() <= MAX_ONEHOT:
                dummies = pd.get_dummies(as_str, prefix=col)
                blocks.append(dummies.to_numpy(dtype=float))
                names.extend(dummies.columns.tolist())
            else:
                dropped.append(col)
    if not blocks:
        raise ValueError(
            "no usable features after encoding — every column is high-cardinality "
            "text or non-numeric"
        )

    return {
        "X": np.hstack(blocks),
        "y": y,
        "feature_names": names,
        "dropped_columns": dropped,
        "classes": classes,
        "n_rows": len(df),
        "subsampled": subsampled,
        "original_row_index": df["__orig_row__"].to_numpy(),
        "feature_frame": X,
    }
