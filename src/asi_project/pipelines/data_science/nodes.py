
import pandas as pd

from asi_project.config import DatasetContract, ProjectConfig
from asi_project.data import make_xy, split_train_test
from sklearn.pipeline import Pipeline
from asi_project.model import build_baseline_pipeline
from typing import Any

from asi_project.model import make_dummy
from asi_project.metrics import (
    compute_metrics,
    build_metrics_record,
    validate_metrics_record,
)


def prepare_train_test(
    data: pd.DataFrame,
    contract: DatasetContract,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Przygotowuje X/y i dzieli dane zgodnie z kontraktem."""

    X, y = make_xy(data, contract, config.model_features)

    return split_train_test(
        X,
        y,
        contract,
        config.test_size,
        config.random_state,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ProjectConfig,
) -> Pipeline:
    """Trenuje model wraz z preprocessingiem."""

    model = build_baseline_pipeline(
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contract: DatasetContract,
    config: ProjectConfig,
    data_mode: str,
) -> dict[str, Any]:
    """Ocenia model i przygotowuje metryki."""

    # Trening modelu porównawczego dummy
    dummy = make_dummy()
    dummy.fit(X_train, y_train)

    # Predykcje na zbiorze testowym
    dummy_predictions = dummy.predict(X_test)
    model_predictions = model.predict(X_test)

    # Obliczenie metryk obu modeli
    results = {
        "dummy": compute_metrics(y_test, dummy_predictions),
        "logistic_regression": compute_metrics(
            y_test, model_predictions
        ),
    }

    # Przygotowanie rekordu metryk
    metrics = build_metrics_record(
        contract=contract,
        config=config,
        features=list(X_train.columns),
        n_train=len(y_train),
        n_test=len(y_test),
        data_mode=data_mode,
        results=results,
    )

    # Sprawdzenie zgodności z kontraktem
    problems = validate_metrics_record(metrics, contract)
    if problems:
        raise ValueError("\n".join(problems))

    return metrics
