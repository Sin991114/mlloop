import json

import pandas as pd

from mlloop.artifacts import validate_run_artifacts


def test_missing_files(tmp_path):
    report = validate_run_artifacts(tmp_path, "classification")
    assert not report["valid"]
    joined = " ".join(report["errors"])
    assert "predictions.parquet" in joined
    assert "meta.json" in joined


def test_missing_prediction_columns(tmp_path, make_artifacts):
    make_artifacts(tmp_path)
    pd.DataFrame({"row_id": [1], "y_true": [0]}).to_parquet(
        tmp_path / "predictions.parquet", index=False
    )
    report = validate_run_artifacts(tmp_path, "classification")
    assert not report["valid"]
    assert any("y_pred" in error for error in report["errors"])


def test_meta_missing_key(tmp_path, make_artifacts):
    make_artifacts(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["seed"]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    report = validate_run_artifacts(tmp_path, "classification")
    assert not report["valid"]
    assert any("seed" in error for error in report["errors"])


def test_valid_artifacts(tmp_path, make_artifacts):
    make_artifacts(tmp_path, with_cv=True)
    report = validate_run_artifacts(tmp_path, "classification")
    assert report["valid"]
    assert report["errors"] == []
    assert report["num_prediction_rows"] == 50
    assert report["meta"]["model_desc"] == "logistic regression"


def test_missing_cv_predictions_warns(tmp_path, make_artifacts):
    make_artifacts(tmp_path, with_cv=False)
    report = validate_run_artifacts(tmp_path, "classification")
    assert report["valid"]
    assert any("cv_predictions" in warning for warning in report["warnings"])


def test_classification_without_proba_warns(tmp_path, make_artifacts):
    make_artifacts(tmp_path, with_proba=False)
    report = validate_run_artifacts(tmp_path, "classification")
    assert report["valid"]
    assert any("proba" in warning for warning in report["warnings"])


def test_regression_without_proba_no_warning(tmp_path, make_artifacts):
    make_artifacts(tmp_path, with_proba=False)
    report = validate_run_artifacts(tmp_path, "regression")
    assert not any("proba" in warning for warning in report["warnings"])
