"""Testy nodów przygotowania danych."""

import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing import create_pipeline
from asi_project.pipelines.data_processing.nodes import validate_data


def test_validate_data_accepts_and_normalizes_column_names(toy_frame, toy_contract):
    raw = toy_frame.rename(columns={"num": "  num  "})
    original = raw.copy(deep=True)

    result = validate_data(raw, toy_contract)

    pd.testing.assert_frame_equal(result, toy_frame)
    pd.testing.assert_frame_equal(raw, original)


def test_validate_data_rejects_missing_target(toy_frame, toy_contract):
    raw = toy_frame.drop(columns=toy_contract.target)

    with pytest.raises(DataContractError):
        validate_data(raw, toy_contract)


def test_pipeline_normalizes_and_filters_data(toy_frame, toy_contract):
    raw = pd.concat(
        [
            toy_frame,
            toy_frame.iloc[[0]],
            toy_frame.iloc[[1]].assign(label=None),
        ],
        ignore_index=True,
    )
    original = raw.copy(deep=True)

    catalog = DataCatalog(
        {
            "raw_data": MemoryDataset(raw),
            "dataset_contract": MemoryDataset(toy_contract),
        }
    )

    outputs = SequentialRunner().run(create_pipeline(), catalog)
    result = outputs["processed_data"].load()

    expected = toy_frame.copy()
    expected["cat"] = ["a", "b", float("nan"), "c"] * 10

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(raw, original)
