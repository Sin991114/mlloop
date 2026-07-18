"""The goal-time GPU consent gate."""

import pytest

from mlloop.service import GateError, LedgerService

FAKE_GPUS = [{"name": "NVIDIA GeForce RTX 4090", "memory": "24564 MiB"}]


def _goal(svc, dataset, **kw):
    return svc.goal_define(
        task_type="classification", dataset_path=str(dataset), target_column="label",
        primary_metric="auc", metric_direction="maximize", **kw,
    )


def test_gpu_detected_requires_user_decision(monkeypatch, svc, dataset):
    monkeypatch.setattr("mlloop.hardware.detect_gpus", lambda *a, **k: list(FAKE_GPUS))
    monkeypatch.delenv("MLLOOP_GPU_DEFAULT", raising=False)
    with pytest.raises(GateError, match="RTX 4090") as excinfo:
        _goal(svc, dataset)
    assert "ask" in str(excinfo.value).lower()


def test_gpu_decision_recorded_in_policy(monkeypatch, svc, dataset):
    monkeypatch.setattr("mlloop.hardware.detect_gpus", lambda *a, **k: list(FAKE_GPUS))
    out = _goal(svc, dataset, policy={"use_gpu": True})
    assert out["hardware"]["use_gpu"] is True
    assert out["hardware"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"
    policy = svc.status()["goal"]["policy"]
    assert policy["use_gpu"] is True
    assert policy["detected_gpus"] == ["NVIDIA GeForce RTX 4090"]


def test_gpu_denied_is_equally_valid(monkeypatch, svc, dataset):
    monkeypatch.setattr("mlloop.hardware.detect_gpus", lambda *a, **k: list(FAKE_GPUS))
    out = _goal(svc, dataset, policy={"use_gpu": False})
    assert out["hardware"]["use_gpu"] is False


def test_headless_env_default(monkeypatch, svc, dataset):
    monkeypatch.setattr("mlloop.hardware.detect_gpus", lambda *a, **k: list(FAKE_GPUS))
    monkeypatch.setenv("MLLOOP_GPU_DEFAULT", "deny")
    out = _goal(svc, dataset)
    assert out["hardware"]["use_gpu"] is False


def test_no_gpu_no_gate(svc, dataset):
    # The autouse fixture already stubs detection to [] — no gate, no hardware key.
    out = _goal(svc, dataset)
    assert "hardware" not in out
