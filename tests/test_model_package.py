"""Pakiet modelu: zapis -> odczyt -> predykcja bez W&B i bez treningu AutoGluona."""

import json

import pandas as pd
import pytest

from asi_project.data import filter_training_rows, make_xy, normalize_input
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import (
    METADATA_FILE,
    PREDICTION_COLUMNS,
    REQUIRED_METADATA_KEYS,
    ModelPackageError,
    build_metadata,
    load_model_package,
    predict,
    prediction_problems,
    save_model_package,
)


@pytest.fixture
def trained(project):
    data = filter_training_rows(normalize_input(project.sample), project.contract)
    X, y = make_xy(data, project.contract, project.config.model_features)
    model = build_baseline_pipeline(
        project.config.numeric_features,
        project.config.categorical_features,
    )
    model.fit(X, y)
    metadata = build_metadata(
        model_name="logistic_regression",
        framework="sklearn",
        numeric_features=project.config.numeric_features,
        categorical_features=project.config.categorical_features,
        target=project.contract.target,
        positive_class=project.contract.positive_class,
        metrics={"recall": 0.5},
        git_sha="0" * 40,
        data_fingerprint="f" * 64,
        dataset=project.contract.dataset_id,
    )
    return model, metadata, project.sample.head(5)


def test_roundtrip_gives_same_predictions(trained, tmp_path):
    model, metadata, rows = trained
    path = save_model_package(model, metadata, tmp_path / "package")

    package = load_model_package(path)
    result = predict(package, rows)

    assert prediction_problems(result, n_rows=5) == []
    assert tuple(result.columns) == PREDICTION_COLUMNS
    expected = model.predict_proba(rows.loc[:, metadata["features"]])[:, 1]
    assert result["probability"].tolist() == pytest.approx(expected.tolist())


def test_metadata_file_has_required_keys(trained, tmp_path):
    model, metadata, _ = trained
    path = save_model_package(model, metadata, tmp_path / "package")

    saved = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))

    assert set(REQUIRED_METADATA_KEYS) <= set(saved)
    assert saved["target"] == "Churn"
    assert saved["positive_class"] == "1"
    assert "scikit-learn" in saved["library_versions"]


def test_predict_ignores_extra_columns_and_order(trained, tmp_path):
    model, metadata, rows = trained
    package = load_model_package(save_model_package(model, metadata, tmp_path / "package"))
    shuffled = rows.iloc[:, ::-1].assign(extra="x")

    assert predict(package, shuffled).equals(predict(package, rows))


def test_predict_rejects_missing_feature(trained, tmp_path):
    model, metadata, rows = trained
    package = load_model_package(save_model_package(model, metadata, tmp_path / "package"))

    with pytest.raises(ModelPackageError, match="Complains"):
        predict(package, rows.drop(columns=["Complains"]))


def test_load_rejects_incomplete_metadata(trained, tmp_path):
    model, metadata, _ = trained
    path = save_model_package(model, metadata, tmp_path / "package")
    broken = {key: value for key, value in metadata.items() if key != "git_sha"}
    (path / METADATA_FILE).write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ModelPackageError, match="git_sha"):
        load_model_package(path)


def test_save_refuses_to_overwrite_foreign_directory(trained, tmp_path):
    model, metadata, _ = trained
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "important.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ModelPackageError):
        save_model_package(model, metadata, foreign)
    assert (foreign / "important.txt").is_file()


def test_prediction_problems_detects_bad_schema():
    bad = pd.DataFrame({"prediction": [0, 2], "probability": [0.1, 1.5]})

    problems = prediction_problems(bad, n_rows=3)

    assert len(problems) == 3
