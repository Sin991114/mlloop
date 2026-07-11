"""Phase 3: domain-context ledger and the FE-opportunity probe."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlloop.service import GateError, LedgerService


def test_context_register_and_query(svc_with_goal):
    svc = svc_with_goal
    assert svc.status()["feature_context"]["registered"] == 0
    assert "context_hint" in svc.status()

    out = svc.context_register(
        feature="f2",
        meaning="day-of-week the sample was collected (0=Monday)",
        source="dataset documentation",
        details={"units": "weekday index"},
    )
    assert out["ok"] and out["note"] is None
    # Engineered-feature names are allowed but flagged.
    out2 = svc.context_register(feature="f1_f2_ratio", meaning="engineered ratio")
    assert out2["note"] is not None

    status = svc.status()
    assert status["feature_context"]["registered"] == 2
    assert "context_hint" not in status

    rows = svc.ledger_query(view="context")["feature_context"]
    assert {r["feature"] for r in rows} == {"f2", "f1_f2_ratio"}

    # Upsert updates rather than duplicates.
    svc.context_register(feature="f2", meaning="updated meaning")
    rows = svc.ledger_query(view="context")["feature_context"]
    assert len(rows) == 2
    assert next(r for r in rows if r["feature"] == "f2")["meaning"] == "updated meaning"


def test_context_requires_goal_and_fields(svc):
    with pytest.raises(GateError, match="goal_define"):
        svc.context_register(feature="f1", meaning="something")


def test_context_annotates_error_slices(svc_with_goal, make_artifacts):
    svc = svc_with_goal
    svc.context_register(feature="f2", meaning="day-of-week (0=Monday)")
    run = svc.run_start(intent="baseline", kind="baseline")
    d = make_artifacts(run["artifact_dir"], n=100)
    # Concentrate errors on f2 == 0 rows so a slice emerges.
    preds = pd.read_parquet(Path(d) / "predictions.parquet")
    wrong = (preds["row_id"] % 7 == 0).to_numpy()
    preds["y_pred"] = np.where(wrong, 1 - preds["y_true"], preds["y_true"])
    preds.to_parquet(Path(d) / "predictions.parquet", index=False)
    svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.7})

    slices = svc.diagnose_run(run_id=run["run_id"])["results"]["items"]["error_slices"]["details"]["slices"]
    f2_slices = [s for s in slices if s["feature"] == "f2"]
    assert f2_slices and f2_slices[0]["meaning"] == "day-of-week (0=Monday)"


def test_experiment_report_contains_dictionary(svc_ready):
    svc_ready.context_register(feature="f1", meaning="sensor reading in volts", source="user")
    path = svc_ready.report_generate(kind="experiment")["path"]
    content = Path(path).read_text(encoding="utf-8")
    assert "Data dictionary" in content
    assert "sensor reading in volts" in content


def _probe_workspace(tmp_path, kind):
    rng = np.random.default_rng(5)
    if kind == "ratio":
        # Data-starved wide-scale ratio boundary: hard for trees, trivial for FE.
        n = 200
        frame = {f"x{i}": np.exp(rng.uniform(-3, 3, n)) for i in range(1, 11)}
        label = (frame["x1"] / frame["x2"] > 1.0).astype(int)
    else:  # single linear feature carries everything; combos add nothing
        n = 600
        frame = {f"x{i}": rng.uniform(0.5, 2.0, n) for i in range(1, 7)}
        label = (frame["x1"] > 1.25).astype(int)
    frame["label"] = label
    dataset = tmp_path / "data.csv"
    pd.DataFrame(frame).to_csv(dataset, index=False)
    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="auc", metric_direction="maximize",
    )
    return svc


def test_fe_probe_finds_ratio_signal(tmp_path):
    svc = _probe_workspace(tmp_path, "ratio")
    result = svc.fe_probe(quick=True)
    assert result["verdict"] == "fe_worth_testing"
    names = [c["feature"] for c in result["top_candidates"]]
    assert any("x1" in n and "x2" in n for n in names)
    stored = svc.ledger_query(view="fe_probes")["fe_probes"]
    assert stored and stored[0]["id"] == "P1"


def test_fe_probe_negative_when_no_fe_signal(tmp_path):
    svc = _probe_workspace(tmp_path, "single")
    result = svc.fe_probe(quick=True)
    assert result["verdict"] == "fe_unlikely_to_help"
    assert result["n_candidates_swept"] > 0
