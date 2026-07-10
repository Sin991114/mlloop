import pytest

from mlloop.service import GateError


def register_hypothesis(svc, statement="class imbalance dominates errors"):
    return svc.hypothesis_register(
        statement=statement,
        rationale="errors concentrate in the minority class",
        prediction="class weighting improves auc by more than 0.02",
        test_plan="same model with class_weight=balanced",
    )["hypothesis"]["id"]


def test_run_before_goal_refused(svc):
    with pytest.raises(GateError, match="goal_define"):
        svc.run_start(intent="baseline", kind="baseline")


def test_hypothesis_before_goal_refused(svc):
    with pytest.raises(GateError, match="goal_define"):
        register_hypothesis(svc)


def test_goal_cannot_be_redefined(svc_with_goal, dataset):
    with pytest.raises(GateError, match="already defined"):
        svc_with_goal.goal_define(
            task_type="classification",
            dataset_path=str(dataset),
            target_column="label",
            primary_metric="accuracy",
            metric_direction="maximize",
        )


def test_goal_rejects_unknown_target_column(svc, dataset):
    with pytest.raises(GateError, match="target_column"):
        svc.goal_define(
            task_type="classification",
            dataset_path=str(dataset),
            target_column="does_not_exist",
            primary_metric="auc",
            metric_direction="maximize",
        )


def test_first_run_must_be_baseline(svc_with_goal):
    hypothesis_id = register_hypothesis(svc_with_goal)
    with pytest.raises(GateError, match="baseline"):
        svc_with_goal.run_start(intent="try xgboost", hypothesis_id=hypothesis_id)


def test_baseline_rejects_hypothesis(svc_with_goal):
    hypothesis_id = register_hypothesis(svc_with_goal)
    with pytest.raises(GateError, match="must not reference"):
        svc_with_goal.run_start(intent="baseline", kind="baseline", hypothesis_id=hypothesis_id)


def test_second_baseline_refused(svc_ready):
    with pytest.raises(GateError, match="already exists"):
        svc_ready.run_start(intent="another baseline", kind="baseline")


def test_experiment_without_hypothesis_refused(svc_ready):
    with pytest.raises(GateError, match="hypothesis"):
        svc_ready.run_start(intent="try xgboost")


def test_experiment_with_unknown_hypothesis_refused(svc_ready):
    with pytest.raises(GateError, match="Unknown hypothesis"):
        svc_ready.run_start(intent="try xgboost", hypothesis_id="H99")


def test_experiment_with_resolved_hypothesis_refused(svc_ready, make_artifacts):
    hypothesis_id = register_hypothesis(svc_ready)
    run = svc_ready.run_start(intent="class weighting", hypothesis_id=hypothesis_id)
    make_artifacts(run["artifact_dir"], seed=1)
    svc_ready.run_finish(run_id=run["run_id"], metrics={"auc": 0.65})
    svc_ready.diagnose_run(run_id=run["run_id"])
    svc_ready.hypothesis_resolve(
        hypothesis_id=hypothesis_id,
        resolution="confirmed",
        evidence_run_ids=[run["run_id"]],
        narrative="auc improved beyond the predicted margin",
    )
    with pytest.raises(GateError, match="already resolved"):
        svc_ready.run_start(intent="more weighting", hypothesis_id=hypothesis_id)


def test_one_run_at_a_time(svc_with_goal):
    svc_with_goal.run_start(intent="baseline", kind="baseline")
    with pytest.raises(GateError, match="still running"):
        svc_with_goal.run_start(intent="baseline again", kind="baseline")


def test_run_budget_enforced(svc, dataset):
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
        policy={"max_runs": 1},
    )
    run = svc.run_start(intent="baseline", kind="baseline")
    svc.run_abandon(run_id=run["run_id"], reason="testing budget")
    with pytest.raises(GateError, match="budget"):
        svc.run_start(intent="baseline retry", kind="baseline")


def test_abandoned_baseline_allows_retry(svc_with_goal):
    run = svc_with_goal.run_start(intent="baseline", kind="baseline")
    svc_with_goal.run_abandon(run_id=run["run_id"], reason="crashed")
    retry = svc_with_goal.run_start(intent="baseline retry", kind="baseline")
    assert retry["ok"]


def test_forensics_needs_no_hypothesis(svc_ready):
    run = svc_ready.run_start(intent="shuffled-label probe", kind="forensics")
    assert run["ok"] and run["hypothesis_id"] is None


def test_finish_requires_primary_metric(svc_with_goal, make_artifacts):
    run = svc_with_goal.run_start(intent="baseline", kind="baseline")
    make_artifacts(run["artifact_dir"])
    with pytest.raises(GateError, match="primary metric"):
        svc_with_goal.run_finish(run_id=run["run_id"], metrics={"accuracy": 0.9})


def test_resolve_requires_linked_finished_run(svc_ready, make_artifacts):
    hypothesis_id = register_hypothesis(svc_ready)
    with pytest.raises(GateError, match="does not exist"):
        svc_ready.hypothesis_resolve(
            hypothesis_id=hypothesis_id,
            resolution="confirmed",
            evidence_run_ids=["R99"],
            narrative="made up",
        )
    # R1 (the baseline) exists and is finished but never tested this hypothesis.
    with pytest.raises(GateError, match="tested"):
        svc_ready.hypothesis_resolve(
            hypothesis_id=hypothesis_id,
            resolution="confirmed",
            evidence_run_ids=["R1"],
            narrative="baseline says so",
        )


def test_resolve_requires_narrative_and_evidence(svc_ready):
    hypothesis_id = register_hypothesis(svc_ready)
    with pytest.raises(GateError, match="narrative"):
        svc_ready.hypothesis_resolve(
            hypothesis_id=hypothesis_id,
            resolution="refuted",
            evidence_run_ids=["R1"],
            narrative="  ",
        )
    with pytest.raises(GateError, match="evidence_run_ids"):
        svc_ready.hypothesis_resolve(
            hypothesis_id=hypothesis_id,
            resolution="refuted",
            evidence_run_ids=[],
            narrative="no evidence",
        )
