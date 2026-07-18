"""P2: stopping requires evidence; stagnation suggests pivots."""

import json

from mlloop.service import LedgerService


def test_no_stop_condition_yields_exploration_hint(svc_ready):
    status = svc_ready.status()
    sc = status["stop_conditions"]
    assert sc == {
        "target_met": False,
        "data_limited_verdict": False,
        "budget_exhausted": False,
        "stopping_justified": False,
    }
    assert "exploration_hint" in status
    assert "next hypothesis" in status["exploration_hint"]


def test_target_met_justifies_stopping(svc, dataset, make_artifacts):
    svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="auc", metric_direction="maximize", target_value=0.5,
    )
    run = svc.run_start(intent="baseline", kind="baseline")
    make_artifacts(run["artifact_dir"])
    svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.6})
    svc.diagnose_run(run_id=run["run_id"])
    status = svc.status()
    assert status["stop_conditions"]["target_met"] is True
    assert status["stop_conditions"]["stopping_justified"] is True
    assert "exploration_hint" not in status


def test_budget_exhaustion_justifies_stopping(svc, dataset, make_artifacts):
    svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="auc", metric_direction="maximize", policy={"max_runs": 1},
    )
    run = svc.run_start(intent="baseline", kind="baseline")
    make_artifacts(run["artifact_dir"])
    svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.6})
    svc.diagnose_run(run_id=run["run_id"])
    status = svc.status()
    assert status["stop_conditions"]["budget_exhausted"] is True
    assert status["stop_conditions"]["stopping_justified"] is True


def test_data_limited_verdict_justifies_stopping(svc_ready):
    verdict = {"verdict": "data_limited", "confidence": "high", "headline": "test"}
    with svc_ready.ledger.connect() as con:
        con.execute(
            "INSERT INTO forensics (id, created_at, quick, results, verdict) VALUES (?, ?, 0, ?, ?)",
            ("F1", "2026-07-11T00:00:00+00:00", json.dumps({"items": {}}), json.dumps(verdict)),
        )
    status = svc_ready.status()
    assert status["stop_conditions"]["data_limited_verdict"] is True
    assert status["stop_conditions"]["stopping_justified"] is True


def test_stagnation_suggests_pivots(svc_ready, make_artifacts):
    svc = svc_ready
    for i in range(2):
        hypothesis_id = svc.hypothesis_register(
            statement="s", rationale="r", prediction="p", test_plan="t"
        )["hypothesis"]["id"]
        run = svc.run_start(intent=f"attempt {i}", hypothesis_id=hypothesis_id)
        make_artifacts(run["artifact_dir"], seed=i + 1)
        svc.run_finish(run_id=run["run_id"], metrics={"auc": 0.55})
        svc.diagnose_run(run_id=run["run_id"])
    pivots = svc.status()["stagnation"]["suggested_pivots"]
    assert any("ensemble_probe" in p for p in pivots)
    assert any("model family" in p for p in pivots)
