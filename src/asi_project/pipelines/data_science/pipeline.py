from kedro.pipeline import Pipeline, node

from .experiments import run_autogluon, track_baseline
from .nodes import (
    evaluate_model,
    prepare_train_test,
    train_model,
)


def create_pipeline() -> Pipeline:
    """Łączy split, trening i ewaluację."""

    return Pipeline(
        [
            node(
                func=prepare_train_test,
                inputs=[
                    "processed_data",
                    "dataset_contract",
                    "project_config",
                ],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                ],
                name="split_train_test_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "project_config",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "trained_model",
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "dataset_contract",
                    "project_config",
                    "params:data_mode",
                ],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )


def create_experiment_pipeline() -> Pipeline:
    """Rozszerza trening i ewaluację o run bazowy oraz warianty AutoGluona."""
    return Pipeline(
        [
            node(
                func=track_baseline,
                inputs=["trained_model", "metrics", "dataset_contract", "project_config"],
                outputs="baseline_experiment",
                name="track_baseline_node",
            ),
            node(
                func=run_autogluon,
                inputs=[
                    "baseline_experiment",
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "project_config",
                    "params:run_autogluon",
                    "params:autogluon",
                ],
                outputs="sprint3_results",
                name="run_autogluon_node",
            ),
        ]
    )
