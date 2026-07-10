"""Artifact contract validation and dataset fingerprinting.

Diagnostics never read user training code; they consume the standardized
artifacts each run must write into its artifact directory (DESIGN.md §6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REQUIRED_PREDICTION_COLUMNS = ("row_id", "y_true", "y_pred")
REQUIRED_META_KEYS: dict[str, type] = {
    "model_desc": str,
    "hyperparams": dict,
    "features": list,
    "seed": int,
}


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a csv/tsv/parquet dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"unsupported dataset format '{suffix}' (use csv, tsv, or parquet)")


def fingerprint_dataset(path: Path | str) -> dict:
    """Fingerprint a csv/tsv/parquet dataset: shape, column dtypes, content hash."""
    path = Path(path)
    df = load_dataset(path)

    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)

    return {
        "path": str(path.resolve()),
        "rows": int(len(df)),
        "columns": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "file_bytes": path.stat().st_size,
        "sha256": sha.hexdigest(),
    }


def validate_run_artifacts(run_dir: Path | str, task_type: str) -> dict:
    """Check a finished run's artifact directory against the contract.

    Returns ``{"valid", "errors", "warnings", "num_prediction_rows", "meta"}``.
    Errors block run_finish; warnings are recorded but do not block.
    """
    run_dir = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    num_rows: int | None = None
    meta: dict | None = None

    preds_path = run_dir / "predictions.parquet"
    if not preds_path.exists():
        errors.append(
            "predictions.parquet is missing — write held-out predictions with columns "
            f"{list(REQUIRED_PREDICTION_COLUMNS)} (plus proba_<class> columns for classification)."
        )
    else:
        try:
            columns = set(pq.read_schema(preds_path).names)
            missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in columns]
            if missing:
                errors.append(f"predictions.parquet is missing required columns: {missing}")
            num_rows = pq.ParquetFile(preds_path).metadata.num_rows
            if num_rows == 0:
                errors.append("predictions.parquet has zero rows")
            if task_type == "classification" and not any(c.startswith("proba_") for c in columns):
                warnings.append(
                    "no proba_<class> columns in predictions.parquet — calibration and "
                    "label-noise diagnostics will be unavailable for this run"
                )
        except Exception as exc:  # unreadable / corrupt parquet
            errors.append(f"predictions.parquet is unreadable: {exc}")

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        errors.append(
            "meta.json is missing — required keys: "
            + ", ".join(f"{k} ({t.__name__})" for k, t in REQUIRED_META_KEYS.items())
        )
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                errors.append("meta.json must contain a JSON object")
                meta = None
            else:
                for key, typ in REQUIRED_META_KEYS.items():
                    if key not in meta:
                        errors.append(f"meta.json is missing required key '{key}'")
                    elif not isinstance(meta[key], typ):
                        errors.append(
                            f"meta.json key '{key}' must be {typ.__name__}, "
                            f"got {type(meta[key]).__name__}"
                        )
                if "train_seconds" not in (meta or {}):
                    warnings.append("meta.json: 'train_seconds' is recommended")
        except json.JSONDecodeError as exc:
            errors.append(f"meta.json is not valid JSON: {exc}")

    if not (run_dir / "cv_predictions.parquet").exists():
        warnings.append(
            "cv_predictions.parquet is missing (recommended) — out-of-fold predictions "
            "enable label-noise and learning-curve diagnostics in Phase 1"
        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_prediction_rows": num_rows,
        "meta": meta,
    }
