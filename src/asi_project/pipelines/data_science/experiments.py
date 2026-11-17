"""Tracking baseline'u i eksperymenty AutoGluona na podziale danych Kedro."""

import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import pandas as pd
from sklearn.pipeline import Pipeline

from asi_project.config import DatasetContract, ProjectConfig
from asi_project.data import raw_data_path
from asi_project.download import sha256_hex
from asi_project.metrics import compute_metrics


def _log_run(context: dict[str, Any], result: dict[str, Any], root: Path, job_type: str) -> None:
    mode = os.environ.get("WANDB_MODE", "disabled")
    if mode == "disabled":
        return
    if mode not in {"offline", "online"}:
        raise ValueError("WANDB_MODE musi mieć wartość disabled, offline albo online.")

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if mode == "online" and not (entity and project):
        raise ValueError("Ustaw WANDB_ENTITY i WANDB_PROJECT na dane projektu zespołu.")

    import wandb

    reporting_dir = root / "data/08_reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        entity=entity,
        project=project or "asi-project",
        mode=mode,
        dir=str(reporting_dir),
        name=result["name"],
        group=context["experiment_id"],
        job_type=job_type,
        config={**context, "parameters": result["parameters"]},
        settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
    ) as run:
        run.log(result["metrics"])
        result["wandb_run_id"] = run.id
        if mode == "online":
            result["wandb_url"] = run.url


def track_baseline(
    model: Pipeline,
    metrics: dict[str, Any],
    contract: DatasetContract,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Zapisuje istniejący wynik regresji logistycznej jako pierwszy run grupy."""
    source = (
        config.ci_sample_path
        if metrics["data_mode"] == "sample"
        else raw_data_path(config, contract)
    )
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.root, text=True)
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=config.root)
    context = {
        "experiment_id": uuid4().hex,
        "dataset": metrics["dataset"],
        "data_mode": metrics["data_mode"],
        "data_fingerprint": sha256_hex(source.read_bytes()),
        "git_sha": git_sha.strip(),
        "git_dirty": bool(git_status),
        "target": metrics["target"],
        "positive_class": metrics["positive_class"],
        "split": metrics["split"],
        "numeric_features": list(config.numeric_features),
        "categorical_features": list(config.categorical_features),
    }
    baseline = {
        "name": "logistic_regression",
        "parameters": {
            "classifier": model.named_steps["classifier"].get_params(),
            "scikit_learn_version": version("scikit-learn"),
        },
        "metrics": metrics["models"]["logistic_regression"],
        "model_path": "data/06_models/model.pkl",
    }
    _log_run(context, baseline, config.root, "baseline")
    return {"context": context, "runs": [baseline]}


def run_autogluon(
    baseline_experiment: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ProjectConfig,
    enabled: bool,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Trenuje warianty na train i ocenia je na wspólnym zbiorze testowym."""
    if not enabled:
        return baseline_experiment

    from autogluon.tabular import TabularPredictor

    time_limit = settings["time_limit_seconds"]
    if time_limit <= 0 or len(settings["variants"]) < 2:
        raise ValueError("Ustaw dodatni limit czasu oraz co najmniej dwa warianty AutoGluona.")

    context = baseline_experiment["context"]
    runs = list(baseline_experiment["runs"])
    target = context["target"]
    categorical_dtypes = dict.fromkeys(config.categorical_features, "category")
    train = X_train.astype(categorical_dtypes).assign(**{target: y_train})
    test = X_test.astype(categorical_dtypes)
    for name, hyperparameters in settings["variants"].items():
        model_path = Path("data/06_models/autogluon") / context["experiment_id"] / name
        fit_args = {
            "hyperparameters": hyperparameters,
            "time_limit": time_limit,
            "num_cpus": settings["num_cpus"],
            "num_gpus": 0,
            "fit_weighted_ensemble": False,
            "calibrate_decision_threshold": False,
        }
        predictor = TabularPredictor(
            label=target,
            problem_type="binary",
            positive_class=1,
            eval_metric="recall",
            path=str(config.root / model_path),
        )
        started_at = perf_counter()
        predictor.fit(train, **fit_args)
        elapsed = perf_counter() - started_at
        predictor.save_space()
        result = {
            "name": f"autogluon_{name}",
            "parameters": {
                **fit_args,
                "eval_metric": "recall",
                "autogluon_version": version("autogluon.tabular"),
            },
            "metrics": {
                **compute_metrics(y_test, predictor.predict(test)),
                "fit_seconds": elapsed,
            },
            "model_path": model_path.as_posix(),
        }
        _log_run(context, result, config.root, "autogluon")
        runs.append(result)
    return {"context": context, "runs": runs}
