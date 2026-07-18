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
