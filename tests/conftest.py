import json
from pathlib import Path

import pandas as pd
import pytest

from mlloop.service import LedgerService


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def dataset(workspace):
    df = pd.DataFrame(
        {
            "f1": range(100),
            "f2": [i % 7 for i in range(100)],
            "label": [i % 2 for i in range(100)],
        }
    )
    path = workspace / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def svc(workspace):
    return LedgerService(workspace)


@pytest.fixture
def svc_with_goal(svc, dataset):
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
    )
    return svc


@pytest.fixture
def make_artifacts():
    def _make(run_dir, n=50, seed=0, with_proba=True, with_cv=False, meta_overrides=None):
        run_dir = Path(run_dir)
        df = pd.DataFrame(
            {
                "row_id": range(n),
                "y_true": [i % 2 for i in range(n)],
                "y_pred": [(i + seed) % 2 for i in range(n)],
            }
        )
        if with_proba:
            df["proba_1"] = 0.5
        df.to_parquet(run_dir / "predictions.parquet", index=False)
        meta = {
            "model_desc": "logistic regression",
            "hyperparams": {"C": 1.0},
            "features": ["f1", "f2"],
            "seed": seed,
            "train_seconds": 1.2,
        }
        meta.setdefault("feature_importance", {"f1": 0.6, "f2": 0.4})
        meta.update(meta_overrides or {})
        (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (run_dir / "train.py").write_text(f"# seeded training stub\nSEED = {seed}\n", encoding="utf-8")
        (run_dir / "infer.py").write_text("# inference stub: python infer.py in.csv out.csv\n", encoding="utf-8")
        (run_dir / "model.pkl").write_bytes(b"stub-model-bytes")
        if with_cv:
            df.to_parquet(run_dir / "cv_predictions.parquet", index=False)
        return run_dir

    return _make


@pytest.fixture
def svc_ready(svc_with_goal, make_artifacts):
    """Service with a finished AND diagnosed baseline run (R1, auc=0.6)."""
    run = svc_with_goal.run_start(intent="majority-class baseline", kind="baseline")
    make_artifacts(run["artifact_dir"])
    svc_with_goal.run_finish(run_id=run["run_id"], metrics={"auc": 0.6})
    svc_with_goal.diagnose_run(run_id=run["run_id"])
    return svc_with_goal
