"""Custom metric scripts and the metric-choice advisory."""

import numpy as np
import pandas as pd
import pytest

from mlloop.service import GateError, LedgerService

METRIC_SCRIPT = '''
"""Weighted score: correct rows earn their weight; a stand-in for domain metrics."""

def metric(predictions):
    correct = (predictions["y_true"] == predictions["y_pred"])
    weight = predictions["weight"] if "weight" in predictions else 1.0
    return float((correct * weight).sum() / (weight if hasattr(weight, "sum") else len(predictions) * 1.0).sum()
                 if hasattr(weight, "sum") else correct.mean())
'''

SIMPLE_METRIC_SCRIPT = '''
def metric(predictions):
    return float((predictions["y_true"] == predictions["y_pred"]).mean())
'''


def _imbalanced_dataset(tmp_path, majority=0.85):
    rng = np.random.default_rng(2)
    n = 400
    label = (rng.uniform(size=n) > majority).astype(int)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "label": label})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_metric_task_mismatch_refused(tmp_path):
    dataset = _imbalanced_dataset(tmp_path)
    svc = LedgerService(tmp_path)
    with pytest.raises(GateError, match="regression metric"):
        svc.goal_define(
            task_type="classification", dataset_path=str(dataset), target_column="label",
            primary_metric="rmse", metric_direction="minimize",
        )


def test_accuracy_imbalance_advisory(tmp_path):
    dataset = _imbalanced_dataset(tmp_path)
    svc = LedgerService(tmp_path)
    out = svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="accuracy", metric_direction="maximize",
    )
    assert "metric_advisory" in out
    assert "majority class" in out["metric_advisory"]


def test_no_advisory_for_auc(tmp_path):
    dataset = _imbalanced_dataset(tmp_path)
    svc = LedgerService(tmp_path)
    out = svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="auc", metric_direction="maximize",
    )
    assert "metric_advisory" not in out


def test_custom_metric_noise_floor(tmp_path, make_artifacts):
    dataset = _imbalanced_dataset(tmp_path)
    script = tmp_path / "metric.py"
    script.write_text(SIMPLE_METRIC_SCRIPT, encoding="utf-8")
    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="my_domain_score", metric_direction="maximize",
        metric_script=str(script),
    )
    run = svc.run_start(intent="baseline", kind="baseline")
    make_artifacts(run["artifact_dir"], n=100, seed=1)  # seed=1 -> every row wrong? no: mixed below
    # Make errors mixed so the bootstrap has variance.
    d = run["artifact_dir"]
    preds = pd.read_parquet(f"{d}/predictions.parquet")
    preds["y_pred"] = np.where(preds["row_id"] % 3 == 0, 1 - preds["y_true"], preds["y_true"])
    preds.to_parquet(f"{d}/predictions.parquet", index=False)
    svc.run_finish(run_id=run["run_id"], metrics={"my_domain_score": 0.67})

    floor = svc.diagnose_run(run_id=run["run_id"])["results"]["items"]["noise_floor"]
    assert "registered metric script" in floor["conclusion"]
    assert "my_domain_score" in floor["conclusion"]
    assert floor["details"]["metric"] == "my_domain_score"
    assert floor["details"]["std"] > 0


def test_metric_register_after_goal(tmp_path, make_artifacts):
    dataset = _imbalanced_dataset(tmp_path)
    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="my_domain_score", metric_direction="maximize",
    )
    with pytest.raises(GateError, match="metric script"):
        svc.metric_register(script_path=str(tmp_path / "missing.py"))
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(GateError, match="callable"):
        svc.metric_register(script_path=str(bad))

    script = tmp_path / "metric.py"
    script.write_text(SIMPLE_METRIC_SCRIPT, encoding="utf-8")
    out = svc.metric_register(script_path=str(script))
    assert out["ok"]
