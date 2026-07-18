"""The zero-training ensemble probe over stored run predictions."""

import numpy as np
import pandas as pd
import pytest

from mlloop.service import GateError


def _finish_run_with_predictions(svc, make_artifacts, intent, proba, y, hypothesis_id=None, kind=None):
    kind = kind or ("baseline" if hypothesis_id is None else "experiment")
    run = svc.run_start(intent=intent, kind=kind, hypothesis_id=hypothesis_id)
    d = make_artifacts(run["artifact_dir"], n=len(y))
    frame = pd.DataFrame({
        "row_id": np.arange(len(y)),
        "y_true": y,
        "y_pred": (proba >= 0.5).astype(int),
        "proba_1": proba,
    })
    frame.to_parquet(f"{d}/predictions.parquet", index=False)
    from sklearn.metrics import roc_auc_score

    svc.run_finish(run_id=run["run_id"], metrics={"auc": round(float(roc_auc_score(y, proba)), 4)})
    svc.diagnose_run(run_id=run["run_id"])
    return run["run_id"]


def _register(svc):
    return svc.hypothesis_register(
        statement="s", rationale="r", prediction="p", test_plan="t"
    )["hypothesis"]["id"]


def test_ensemble_probe_finds_decorrelated_gain(svc_with_goal, make_artifacts):
    svc = svc_with_goal
    rng = np.random.default_rng(0)
    n = 100  # matches the conftest dataset rows
    y = np.array([i % 2 for i in range(n)])
    noise_a = rng.normal(0, 0.35, n)
    noise_b = rng.normal(0, 0.35, n)
    proba_a = np.clip(0.5 + (y - 0.5) * 0.4 + noise_a, 0, 1)
    proba_b = np.clip(0.5 + (y - 0.5) * 0.4 + noise_b, 0, 1)

    r1 = _finish_run_with_predictions(svc, make_artifacts, "model A", proba_a, y)
    h = _register(svc)
    r2 = _finish_run_with_predictions(svc, make_artifacts, "model B", proba_b, y, hypothesis_id=h)

    result = svc.ensemble_probe()
    assert result["ok"] is True
    assert result["n_common_rows"] == n
    assert set(result["per_run"]) == {r1, r2}
    assert result["gain"] > 0  # independent noise must average out
    assert result["verdict"] in ("ensemble_worth_testing", "ensemble_unlikely_to_help")
    assert result["probe_id"] == "E1"
    stored = svc.ledger_query(view="ensemble_probes")["ensemble_probes"]
    assert stored and stored[0]["id"] == "E1"


def test_ensemble_probe_refuses_disjoint_rows(svc_with_goal, make_artifacts):
    svc = svc_with_goal
    y = np.array([i % 2 for i in range(50)])
    proba = np.clip(0.5 + (y - 0.5) * 0.6, 0, 1)

    run = svc.run_start(intent="baseline", kind="baseline")
    d = make_artifacts(run["artifact_dir"], n=50)
    pd.DataFrame({"row_id": np.arange(0, 50), "y_true": y,
                  "y_pred": (proba >= 0.5).astype(int), "proba_1": proba}
                 ).to_parquet(f"{d}/predictions.parquet", index=False)
    svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.9})
    svc.diagnose_run(run_id=run["run_id"])

    h = _register(svc)
    run2 = svc.run_start(intent="disjoint holdout", hypothesis_id=h)
    d2 = make_artifacts(run2["artifact_dir"], n=50)
    pd.DataFrame({"row_id": np.arange(50, 100), "y_true": y,
                  "y_pred": (proba >= 0.5).astype(int), "proba_1": proba}
                 ).to_parquet(f"{d2}/predictions.parquet", index=False)
    svc.run_finish(run_id=run2["run_id"], metrics={"auc": 0.9})

    result = svc.ensemble_probe()
    assert result["ok"] is False
    assert "common evaluation set" in result["error"]


def test_ensemble_probe_input_validation(svc_ready):
    with pytest.raises(GateError, match="Unknown run_id"):
        svc_ready.ensemble_probe(run_ids=["R99", "R1"])
    with pytest.raises(GateError, match="at least two"):
        svc_ready.ensemble_probe()  # only one finished run exists


def test_compare_runs_paired_significance(svc_with_goal, make_artifacts):
    svc = svc_with_goal
    rng = np.random.default_rng(1)
    n = 100
    y = np.array([i % 2 for i in range(n)])
    base_noise = rng.normal(0, 0.3, n)
    proba_weak = np.clip(0.5 + (y - 0.5) * 0.3 + base_noise, 0, 1)
    proba_strong = np.clip(0.5 + (y - 0.5) * 0.7 + base_noise, 0, 1)  # shared noise, more signal
    proba_twin = np.clip(proba_strong + rng.normal(0, 0.01, n), 0, 1)  # nearly identical to strong

    r1 = _finish_run_with_predictions(svc, make_artifacts, "weak", proba_weak, y)
    h1 = _register(svc)
    r2 = _finish_run_with_predictions(svc, make_artifacts, "strong", proba_strong, y, hypothesis_id=h1)
    h2 = _register(svc)
    r3 = _finish_run_with_predictions(svc, make_artifacts, "twin", proba_twin, y, hypothesis_id=h2)

    clear = svc.compare_runs(run_a=r1, run_b=r2)
    assert clear["ok"] and clear["significant"] is True
    assert clear["improvement_b_over_a"] > 0

    twins = svc.compare_runs(run_a=r2, run_b=r3)
    assert twins["ok"] and twins["significant"] is False
    assert "indistinguishable" in twins["conclusion"]


def test_run_finish_carries_paired_comparison(svc_with_goal, make_artifacts):
    svc = svc_with_goal
    rng = np.random.default_rng(3)
    n = 100
    y = np.array([i % 2 for i in range(n)])
    proba_a = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.2, n), 0, 1)
    proba_b = np.clip(0.5 + (y - 0.5) * 0.8 + rng.normal(0, 0.2, n), 0, 1)

    _finish_run_with_predictions(svc, make_artifacts, "baseline", proba_a, y)
    h = _register(svc)
    run = svc.run_start(intent="better model", hypothesis_id=h)
    d = make_artifacts(run["artifact_dir"], n=n)
    pd.DataFrame({"row_id": np.arange(n), "y_true": y,
                  "y_pred": (proba_b >= 0.5).astype(int), "proba_1": proba_b}
                 ).to_parquet(f"{d}/predictions.parquet", index=False)
    from sklearn.metrics import roc_auc_score

    out = svc.run_finish(run_id=run["run_id"], metrics={"auc": round(float(roc_auc_score(y, proba_b)), 4)})
    paired = out["comparison"]["vs_parent"]["paired"]
    assert paired is not None
    assert "significance_bar" in paired and paired["significance_bar"] > 0


def test_compare_runs_validation(svc_ready):
    with pytest.raises(GateError, match="Unknown run_id"):
        svc_ready.compare_runs(run_a="R1", run_b="R99")
