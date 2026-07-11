"""Phase 1: diagnostics battery, diagnosis gate, forensics, and reports."""

import numpy as np
import pandas as pd
import pytest

from mlloop.service import GateError, LedgerService


def _register(svc):
    return svc.hypothesis_register(
        statement="s", rationale="r", prediction="p", test_plan="t"
    )["hypothesis"]["id"]


def test_experiment_requires_diagnosis(svc_with_goal, make_artifacts):
    run = svc_with_goal.run_start(intent="baseline", kind="baseline")
    make_artifacts(run["artifact_dir"])
    svc_with_goal.run_finish(run_id=run["run_id"], metrics={"auc": 0.6})
    hypothesis_id = _register(svc_with_goal)

    assert svc_with_goal.status()["state"] == "DIAGNOSE_PENDING"
    with pytest.raises(GateError, match="diagnose_run"):
        svc_with_goal.run_start(intent="experiment", hypothesis_id=hypothesis_id)

    svc_with_goal.diagnose_run(run_id=run["run_id"])
    assert svc_with_goal.status()["state"] == "READY"
    assert svc_with_goal.run_start(intent="experiment", hypothesis_id=hypothesis_id)["ok"]


def test_enforce_diagnosis_can_be_disabled(svc, dataset, make_artifacts):
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
        policy={"enforce_diagnosis": False},
    )
    run = svc.run_start(intent="baseline", kind="baseline")
    make_artifacts(run["artifact_dir"])
    svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.6})
    assert svc.status()["state"] == "READY"
    hypothesis_id = _register(svc)
    assert svc.run_start(intent="experiment", hypothesis_id=hypothesis_id)["ok"]


def test_diagnosis_content_and_idempotency(svc_ready):
    # svc_ready already diagnosed R1; a second call returns the stored results.
    again = svc_ready.diagnose_run(run_id="R1")
    assert again.get("already_diagnosed") is True
    items = again["results"]["items"]
    for expected in ("error_slices", "noise_floor", "confusion", "class_balance", "overfit_gap"):
        assert expected in items
    stored = svc_ready.ledger_query(view="diagnosis", run_id="R1")
    assert stored["results"]["items"].keys() == items.keys()


def _make_signal_dataset(path, n=600, noise_rate=0.0, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    y = ((x1 + x2) > 1.0).astype(int)
    if noise_rate > 0:
        flip = rng.choice(n, int(n * noise_rate), replace=False)
        y[flip] = 1 - y[flip]
    pd.DataFrame({"x1": x1, "x2": x2, "label": y}).to_csv(path, index=False)
    return path


def test_forensics_detects_injected_label_noise(tmp_path):
    """The flagship demo: 20% injected label noise must be caught and quantified."""
    dataset = _make_signal_dataset(tmp_path / "noisy.csv", noise_rate=0.20)
    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
    )
    result = svc.forensics_run(quick=True)
    items = svc.ledger_query(view="forensics")["latest_results"]["items"]

    assert items["signal_check"]["details"]["has_signal"] is True
    estimated = items["label_noise"]["details"]["estimated_noise_rate"]
    assert 0.08 <= estimated <= 0.40, f"injected 20% noise, estimated {estimated:.1%}"
    assert result["verdict"]["verdict"] == "data_limited"
    assert items["label_noise"]["details"]["suspects"], "suspect-label list must not be empty"


def test_forensics_detects_no_signal(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "x1": rng.uniform(size=500),
            "x2": rng.uniform(size=500),
            "label": rng.integers(0, 2, 500),  # independent of the features
        }
    )
    dataset = tmp_path / "random.csv"
    df.to_csv(dataset, index=False)
    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
    )
    result = svc.forensics_run(quick=True)
    assert result["verdict"]["verdict"] == "no_signal"


def test_verdict_report_generation(tmp_path):
    dataset = _make_signal_dataset(tmp_path / "noisy.csv", noise_rate=0.20)
    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
    )
    svc.forensics_run(quick=True)
    result = svc.report_generate(kind="verdict")
    html = (tmp_path / ".mlloop" / "reports").glob("verdict_*.html")
    path = next(iter(html))
    assert str(path) == result["path"]
    content = path.read_text(encoding="utf-8")
    assert "Data Verdict Report" in content
    assert "<svg" in content  # charts inlined
    assert "suspect" in content.lower()


def test_verdict_report_requires_forensics(svc_with_goal):
    with pytest.raises(GateError, match="forensics_run"):
        svc_with_goal.report_generate(kind="verdict")


def test_experiment_report(svc_ready):
    result = svc_ready.report_generate(kind="experiment")
    from pathlib import Path

    content = Path(result["path"]).read_text(encoding="utf-8")
    assert "Experiment Report" in content
    assert "R1" in content


def test_stagnation_triggers_forensics_recommendation(svc_ready, make_artifacts):
    svc = svc_ready  # R1 baseline auc=0.6 finished+diagnosed
    for i in range(3):
        hypothesis_id = _register(svc)
        run = svc.run_start(intent=f"attempt {i}", hypothesis_id=hypothesis_id)
        make_artifacts(run["artifact_dir"], seed=i + 1)
        svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.55})  # never beats 0.6
        svc.diagnose_run(run_id=run["run_id"])
    status = svc.status()
    assert status["stagnation"]["consecutive_non_improving_runs"] == 3
    assert status["stagnation"]["forensics_recommended"] is True


def test_forensics_runs_excluded_from_best(svc_ready, make_artifacts):
    run = svc_ready.run_start(intent="leaky protocol probe", kind="forensics")
    make_artifacts(run["artifact_dir"], seed=9)
    svc_ready.run_finish(run_id=run["run_id"], metrics={"auc": 0.99})
    best = svc_ready.status()["best_run"]
    assert best["id"] == "R1", "a forensics probe must never be crowned best run"
