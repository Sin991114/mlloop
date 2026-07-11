def test_full_loop(svc_with_goal, make_artifacts, workspace):
    svc = svc_with_goal
    assert svc.status()["state"] == "NEED_BASELINE"

    # Baseline
    baseline = svc.run_start(intent="majority-class baseline", kind="baseline")
    assert svc.status()["state"] == "RUN_IN_PROGRESS"
    make_artifacts(baseline["artifact_dir"])
    result = svc.run_finish(run_id=baseline["run_id"], metrics={"auc": 0.61})
    assert result["ok"]
    assert result["comparison"]["vs_parent"] is None

    # Reproducibility deliverables recorded and predictions.csv auto-generated
    from pathlib import Path

    assert (Path(baseline["artifact_dir"]) / "predictions.csv").exists()
    assert result["deliverables"]["training_script"] == "train.py"
    assert result["deliverables"]["inference_script"] == "infer.py"
    assert result["deliverables"]["model_file"] == "model.pkl"

    # Diagnosis gate: no experiment until the finished run is diagnosed
    assert svc.status()["state"] == "DIAGNOSE_PENDING"
    diagnosis = svc.diagnose_run(run_id=baseline["run_id"])
    assert "noise_floor" in diagnosis["results"]["items"]
    assert svc.status()["state"] == "READY"

    # Hypothesis-driven experiment
    hypothesis = svc.hypothesis_register(
        statement="class imbalance dominates the error",
        rationale="baseline errors concentrate in the minority class",
        prediction="class weighting improves auc by more than 0.02",
        test_plan="identical model with class_weight=balanced",
    )["hypothesis"]
    assert hypothesis["id"] == "H1"

    experiment = svc.run_start(
        intent="baseline + class_weight=balanced", hypothesis_id="H1"
    )
    assert experiment["parent_run_id"] == baseline["run_id"]
    make_artifacts(experiment["artifact_dir"], seed=1)
    result = svc.run_finish(run_id=experiment["run_id"], metrics={"auc": 0.68})
    assert result["comparison"]["vs_parent"]["improved"] is True
    assert round(result["comparison"]["vs_parent"]["delta"], 2) == 0.07

    # Resolution and decision
    svc.diagnose_run(run_id=experiment["run_id"])
    resolved = svc.hypothesis_resolve(
        hypothesis_id="H1",
        resolution="confirmed",
        evidence_run_ids=[experiment["run_id"]],
        narrative="auc +0.07 over baseline, above the predicted 0.02 margin",
    )
    assert resolved["ok"]
    decision = svc.decision_record(
        summary="keep class weighting; explore feature interactions next",
        evidence={"runs": [experiment["run_id"]], "hypotheses": ["H1"]},
        next_action="register H2 about f2 signal",
    )
    assert decision["decision_id"] == "D1"

    # Status and queries
    status = svc.status()
    assert status["state"] == "READY"
    assert status["best_run"]["id"] == experiment["run_id"]
    assert status["hypotheses"] == {"confirmed": 1}
    assert status["budget"]["runs_started"] == 2

    summary = svc.ledger_query()
    assert summary["goal"]["primary_metric"] == "auc"
    assert summary["total_runs"] == 2
    assert [h["status"] for h in summary["hypotheses"]] == ["confirmed"]
    assert len(summary["decisions"]) == 1

    run_detail = svc.ledger_query(view="run", run_id=experiment["run_id"])["run"]
    assert run_detail["metrics"] == {"auc": 0.68}
    assert run_detail["meta"]["seed"] == 1

    events = svc.ledger_query(view="events", limit=50)["events"]
    kinds = [event["kind"] for event in events]
    for expected in (
        "goal_defined",
        "run_started",
        "run_finished",
        "run_diagnosed",
        "hypothesis_registered",
        "hypothesis_resolved",
        "decision_recorded",
    ):
        assert expected in kinds

    # JSONL mirror exists and matches event count
    events_file = workspace / ".mlloop" / "events.jsonl"
    assert events_file.exists()
    assert len(events_file.read_text(encoding="utf-8").strip().splitlines()) == len(events)


def test_invalid_artifacts_keep_run_open(svc_with_goal):
    run = svc_with_goal.run_start(intent="baseline", kind="baseline")
    result = svc_with_goal.run_finish(run_id=run["run_id"], metrics={"auc": 0.6})
    assert result["ok"] is False
    assert result["artifact_errors"]
    assert svc_with_goal.status()["state"] == "RUN_IN_PROGRESS"


def test_hypothesis_moves_to_testing_on_run_start(svc_ready):
    hypothesis_id = svc_ready.hypothesis_register(
        statement="s", rationale="r", prediction="p", test_plan="t"
    )["hypothesis"]["id"]
    svc_ready.run_start(intent="test it", hypothesis_id=hypothesis_id)
    board = svc_ready.ledger_query(view="hypotheses")["hypotheses"]
    assert board[0]["status"] == "testing"
