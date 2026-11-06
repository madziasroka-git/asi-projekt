
from kedro.pipeline import Pipeline, node

from .nodes import (
    prepare_train_test,
    train_model,
    evaluate_model,
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
