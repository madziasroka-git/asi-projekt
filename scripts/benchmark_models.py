"""Mierzy czas predykcji i pamięć zapisanych modeli Sprintu 3."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from timeit import repeat

import pandas as pd
import psutil

from asi_project.config import get_contract, load_project_config
from asi_project.pipelines.data_science.nodes import prepare_train_test


def main() -> None:
    from autogluon.tabular import TabularPredictor
    from kedro_datasets.pickle import PickleDataset

    report_path = Path("data/08_reporting/sprint3_results.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    context = report["context"]
    paths = {run["name"]: Path(run["model_path"]) for run in report["runs"]}

    parser = argparse.ArgumentParser(description="Pomiar czasu predykcji i pamięci modelu.")
    parser.add_argument("model", choices=paths)
    name = parser.parse_args().model
    model_path = paths[name]

    split = context["split"]
    config = replace(
        load_project_config(),
        test_size=split["test_size"],
        random_state=split["random_state"],
    )
    data = pd.read_csv("data/02_intermediate/processed_data.csv")
    _, x_test, _, _ = prepare_train_test(data, get_contract(config), config)

    row = x_test.iloc[[0]]
    is_autogluon = model_path.is_dir()
    if is_autogluon:
        row = row.astype(dict.fromkeys(context["categorical_features"], "category"))

    process = psutil.Process()
    ram_before = process.memory_info().rss

    if is_autogluon:
        model = TabularPredictor.load(str(model_path))
        if not model.persist():
            raise RuntimeError("AutoGluon nie umieścił modelu w RAM-ie.")
    else:
        model = PickleDataset(filepath=str(model_path)).load()

    ram_after = process.memory_info().rss
    memory_mib = (ram_after - ram_before) / 1024**2

    model.predict(row)  # Rozgrzewka.
    times = repeat(lambda: model.predict(row), number=1, repeat=50)
    prediction_ms = median(times) * 1000

    print(f"{name}: {prediction_ms:.3f} ms, przyrost RAM {memory_mib:.3f} MiB")


if __name__ == "__main__":
    main()
