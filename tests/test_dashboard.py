"""Phase 2: the read-only dashboard API."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from mlloop.dashboard import create_app
from mlloop.service import LedgerService


def test_dashboard_state_and_run_detail(svc_ready, workspace):
    client = TestClient(create_app(str(workspace)))

    page = client.get("/")
    assert page.status_code == 200
    assert "MLLoop" in page.text

    state = client.get("/api/state").json()
    assert state["status"]["state"] == "READY"
    assert len(state["runs"]) == 1
    assert state["runs"][0]["diagnosed"] is True
    assert state["workspace"]

    detail = client.get("/api/run/R1").json()
    assert detail["run"]["id"] == "R1"
    assert detail["diagnosis"] is not None
    assert "noise_floor" in detail["diagnosis"]["items"]
    for chart_url in detail["charts"]:
        assert client.get(chart_url).status_code == 200

    assert client.get("/api/run/R99").status_code == 404
    assert client.get("/api/run/DROP TABLE").status_code == 400
    assert client.get("/charts/runs/R1/nonexistent.svg").status_code == 404
    assert client.get("/charts/runs/R1/..%2Fledger.db").status_code in (400, 404)


def test_dashboard_before_goal(tmp_path):
    LedgerService(tmp_path)  # empty ledger
    client = TestClient(create_app(str(tmp_path)))
    state = client.get("/api/state").json()
    assert state["status"]["state"] == "NEED_GOAL"
    assert state["runs"] == []


def test_dashboard_forensics_and_verdict_page(tmp_path):
    rng = np.random.default_rng(7)
    n = 600
    x1, x2 = rng.uniform(size=n), rng.uniform(size=n)
    y = ((x1 + x2) > 1.0).astype(int)
    flip = rng.choice(n, int(n * 0.2), replace=False)
    y[flip] = 1 - y[flip]
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "label": y}).to_csv(dataset, index=False)

    svc = LedgerService(tmp_path)
    svc.goal_define(
        task_type="classification",
        dataset_path=str(dataset),
        target_column="label",
        primary_metric="auc",
        metric_direction="maximize",
    )
    svc.forensics_run(quick=True)

    client = TestClient(create_app(str(tmp_path)))
    state = client.get("/api/state").json()
    assert len(state["forensics"]) == 1
    forensics_id = state["forensics"][0]["id"]

    detail = client.get(f"/api/forensics/{forensics_id}").json()
    assert detail["verdict"]["verdict"] == "data_limited"
    for chart_url in detail["charts"]:
        assert client.get(chart_url).status_code == 200

    verdict_page = client.get(f"/verdict/{forensics_id}")
    assert verdict_page.status_code == 200
    assert "Data Verdict Report" in verdict_page.text
    assert client.get("/verdict/F99").status_code == 404
