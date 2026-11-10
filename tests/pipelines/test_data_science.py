"""Testy nodów data science."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from asi_project.pipelines.data_science.nodes import (
    evaluate_model,
    prepare_train_test,
    train_model,
)


@pytest.fixture
def dummy_data():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_prepare_train_test_splits_data_correctly(dummy_data, monkeypatch):
    mock_config = MagicMock()
    mock_config.model_features = ["feature1", "feature2"]
    mock_config.test_size = 0.2
    mock_config.random_state = 42

    mock_contract = MagicMock()

    def mock_make_xy(data, contract, model_features):
        X = data[model_features]
        y = data["label"]
        return X, y

    def mock_split_train_test(X, y, contract, test_size, random_state):
        return X.iloc[:8], X.iloc[8:], y.iloc[:8], y.iloc[8:]

    monkeypatch.setattr("asi_project.pipelines.data_science.nodes.make_xy", mock_make_xy)
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.split_train_test", mock_split_train_test
    )

    X_train, X_test, y_train, y_test = prepare_train_test(
        data=dummy_data, contract=mock_contract, config=mock_config
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_train_model_returns_pipeline(dummy_data, monkeypatch):
    X_train = dummy_data[["feature1", "feature2"]]
    y_train = dummy_data["label"]

    mock_config = MagicMock()
    mock_config.numeric_features = ["feature1", "feature2"]
    mock_config.categorical_features = []

    def mock_build_baseline_pipeline(numeric_features, categorical_features):
        return Pipeline([("model", LogisticRegression())])

    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.build_baseline_pipeline",
        mock_build_baseline_pipeline,
    )

    model = train_model(X_train, y_train, config=mock_config)

    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")


def test_evaluate_model_returns_metrics_dict(dummy_data, monkeypatch):
    X_train = dummy_data[["feature1", "feature2"]].iloc[:8]
    y_train = dummy_data["label"].iloc[:8]
    X_test = dummy_data[["feature1", "feature2"]].iloc[:2]
    y_test = dummy_data["label"].iloc[:2]

    mock_model = MagicMock(spec=Pipeline)
    mock_contract = MagicMock()
    mock_config = MagicMock()

    mock_dummy_clf = MagicMock()
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.make_dummy", lambda: mock_dummy_clf
    )

    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.compute_metrics",
        lambda y_true, y_pred: {"accuracy": 0.9, "recall": 0.8},
    )

    expected_metrics = {"accuracy": 0.9, "recall": 0.8, "mode": "full"}

    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.build_metrics_record",
        lambda **kwargs: expected_metrics,
    )

    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.validate_metrics_record",
        lambda metrics, contract: [],
    )

    result = evaluate_model(
        model=mock_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        contract=mock_contract,
        config=mock_config,
        data_mode="full",
    )

    assert isinstance(result, dict)
    assert result == expected_metrics
