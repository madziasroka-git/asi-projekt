"""Pipeline przygotowania danych przed modelowaniem."""

from kedro.pipeline import Pipeline, node

from asi_project.data import filter_training_rows, normalize_input

from .nodes import validate_data


def create_pipeline() -> Pipeline:
    """Łączy walidację, normalizację i filtrowanie danych."""
    return Pipeline(
        [
            node(
                func=validate_data,
                inputs=["raw_data", "dataset_contract"],
                outputs="validated_data",
                name="validate_data_node",
            ),
            node(
                func=normalize_input,
                inputs="validated_data",
                outputs="normalized_data",
                name="normalize_input_node",
            ),
            node(
                func=filter_training_rows,
                inputs=["normalized_data", "dataset_contract"],
                outputs="processed_data",
                name="filter_training_rows_node",
            ),
        ]
    )
