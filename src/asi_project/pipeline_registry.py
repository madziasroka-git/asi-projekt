from dataclasses import replace

from kedro.pipeline import Pipeline, node

from asi_project.config import get_contract, load_project_config
from asi_project.pipelines.data_processing.pipeline import create_pipeline as data_processing
from asi_project.pipelines.data_science.pipeline import create_pipeline as data_science


def load_config(split):
    return replace(
        load_project_config(),
        test_size=split["test_size"],
        random_state=split["random_state"],
    )


def register_pipelines():
    config = Pipeline(
        [
            node(load_config, "params:split", "project_config"),
            node(get_contract, "project_config", "dataset_contract"),
        ]
    )
    return {
        "data_processing": config + data_processing(),
        "data_science": config + data_science(),
        "__default__": config + data_processing() + data_science(),
    }
