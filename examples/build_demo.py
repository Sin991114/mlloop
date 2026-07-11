"""Build a demo ledger with a realistic iteration story for the dashboard.

The __main__ guard is mandatory on Windows: libraries used by forensics may spawn
worker processes, and spawn re-imports this script.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlloop.service import LedgerService


def main() -> None:
    ws = Path(sys.argv[1]).resolve()
    if (ws / ".mlloop").exists():
        shutil.rmtree(ws / ".mlloop")
    ws.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = 2000
    x1, x2, x3 = rng.uniform(size=n), rng.uniform(size=n), rng.normal(size=n)
    y = ((x1 + x2 + 0.2 * rng.normal(size=n)) > 1.0).astype(int)
    flip = rng.choice(n, int(n * 0.2), replace=False)
    y[flip] = 1 - y[flip]
    pd.DataFrame({"signal_a": x1, "signal_b": x2, "noise_feat": x3, "label": y}).to_csv(
        ws / "demo_data.csv", index=False
    )

    svc = LedgerService(ws)
    svc.goal_define(
        task_type="classification", dataset_path=str(ws / "demo_data.csv"),
        target_column="label", primary_metric="auc", metric_direction="maximize",
        target_value=0.85, policy={"max_runs": 30},
    )

    def artifacts(run, seed):
        d = Path(run["artifact_dir"])
        idx = rng.choice(n, 400, replace=False)
        proba = np.clip(y[idx] * 0.6 + rng.uniform(size=400) * 0.4, 0, 1)
        pd.DataFrame({
            "row_id": idx, "y_true": y[idx],
            "y_pred": (proba > 0.5).astype(int), "proba_1": proba,
        }).to_parquet(d / "predictions.parquet", index=False)
        (d / "meta.json").write_text(json.dumps({
            "model_desc": run["run_id"], "hyperparams": {"seed": seed},
            "features": ["signal_a", "signal_b", "noise_feat"], "seed": seed,
            "train_seconds": 3.2,
            "feature_importance": {"signal_a": 0.45, "signal_b": 0.45, "noise_feat": 0.1},
        }))
        (d / "train.py").write_text(f"# demo training stub (seeded)\nSEED = {seed}\n")
        (d / "infer.py").write_text("# demo inference stub: python infer.py in.csv out.csv\n")
        (d / "model.pkl").write_bytes(b"demo-model-stub")

    def run_cycle(intent, auc, kind="experiment", hypothesis_id=None, seed=0):
        run = svc.run_start(intent=intent, kind=kind, hypothesis_id=hypothesis_id)
        artifacts(run, seed)
        svc.run_finish(run_id=run["run_id"], metrics={"auc": auc, "train_auc": auc + 0.04})
        svc.diagnose_run(run_id=run["run_id"])
        return run["run_id"]

    r1 = run_cycle("logistic regression baseline on raw features", 0.612, kind="baseline")

    h1 = svc.hypothesis_register(
        statement="Tree-based models capture the additive interaction the linear baseline misses",
        rationale="error slices show failures concentrated where signal_a and signal_b are mid-range",
        prediction="a gradient-boosted model improves AUC by more than the 0.012 noise floor",
        test_plan="HistGradientBoosting with defaults, same split",
    )["hypothesis"]["id"]
    r2 = run_cycle("gradient boosting, default params", 0.688, hypothesis_id=h1, seed=1)
    svc.hypothesis_resolve(hypothesis_id=h1, resolution="confirmed", evidence_run_ids=[r2],
                           narrative="AUC +0.076 over baseline, far beyond the noise floor")
    svc.decision_record(summary="Adopt gradient boosting as the working model family",
                        evidence={"runs": [r1, r2]}, next_action="tune regularization")

    h2 = svc.hypothesis_register(
        statement="The model is overfitting; stronger regularization recovers validation AUC",
        rationale="train/validation gap of 0.04 in R2 diagnosis",
        prediction="halving learning rate and adding L2 improves validation AUC beyond noise",
        test_plan="same model, lr=0.05, l2_regularization=1.0",
    )["hypothesis"]["id"]
    r3 = run_cycle("gradient boosting + stronger regularization", 0.691, hypothesis_id=h2, seed=2)
    svc.hypothesis_resolve(hypothesis_id=h2, resolution="refuted", evidence_run_ids=[r3],
                           narrative="AUC +0.003, inside the 0.012 noise floor — not real")

    h3 = svc.hypothesis_register(
        statement="Feature engineering on the interaction term lifts the ceiling",
        rationale="both informative features contribute additively; explicit sum feature may help",
        prediction="adding signal_a+signal_b as a feature improves AUC beyond noise",
        test_plan="same model with engineered sum feature",
    )["hypothesis"]["id"]
    r4 = run_cycle("gradient boosting + engineered sum feature", 0.685, hypothesis_id=h3, seed=3)
    svc.hypothesis_resolve(hypothesis_id=h3, resolution="refuted", evidence_run_ids=[r4],
                           narrative="AUC -0.006 vs best; the model already captures the interaction")
    svc.decision_record(
        summary="Two consecutive refuted hypotheses with flat AUC — suspect the data, run forensics",
        evidence={"runs": [r3, r4]}, next_action="forensics_run",
    )

    svc.forensics_run(quick=True)
    print("demo ledger ready:", ws / ".mlloop")
    print(json.dumps(svc.status()["stagnation"]))


if __name__ == "__main__":
    main()
