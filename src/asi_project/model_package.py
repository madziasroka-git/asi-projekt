"""Pakiet modelu: wspólny zapis, odczyt i predykcja dla Kedro, smoke modelu i API.

Pakiet to katalog z podkatalogiem ``model`` (``model.joblib`` dla sklearn albo predyktor
AutoGluon) i plikiem ``metadata.json``.
"""

import json
import platform
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from asi_project.data import normalize_input

SCHEMA_VERSION = 1
MODEL_DIR = "model"
METADATA_FILE = "metadata.json"
SKLEARN_FILE = "model.joblib"
FRAMEWORKS = ("sklearn", "autogluon")
PREDICTION_COLUMNS = ("prediction", "probability")
REQUIRED_METADATA_KEYS = (
    "schema_version",
    "model_name",
    "framework",
    "features",
    "numeric_features",
    "categorical_features",
    "target",
    "positive_class",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
)
TRACKED_LIBRARIES = (
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
    "autogluon.tabular",
    "catboost",
    "lightgbm",
)


class ModelPackageError(ValueError):
    """Pakiet modelu albo dane wejściowe nie spełniają kontraktu pakietu."""


@dataclass(frozen=True)
class ModelPackage:
    """Wczytany model wraz z metadanymi."""

    model: Any
    metadata: dict[str, Any]

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    @property
    def framework(self) -> str:
        return self.metadata["framework"]


def library_versions() -> dict[str, str]:
    """Wersje Pythona i zainstalowanych bibliotek, od których zależy odczyt modelu."""
    versions = {"python": platform.python_version()}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_metadata(
    *,
    model_name: str,
    framework: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    target: str,
    positive_class: str,
    metrics: dict[str, Any],
    git_sha: str,
    data_fingerprint: str,
    **extra: Any,
) -> dict[str, Any]:
    """Buduje ``metadata.json``; ``extra`` trafia do metadanych bez zmian (np. dataset, split)."""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "framework": framework,
        "features": [*numeric_features, *categorical_features],
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "target": target,
        "positive_class": str(positive_class),
        "metrics": dict(metrics),
        "git_sha": git_sha,
        "data_fingerprint": data_fingerprint,
        "library_versions": library_versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Zgłasza ``ModelPackageError``, jeśli metadanym brakuje wymaganych pól."""
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ModelPackageError(f"W metadanych pakietu brakuje pól: {missing}.")
    if metadata["framework"] not in FRAMEWORKS:
        raise ModelPackageError(
            f"Nieznany framework {metadata['framework']!r}; dozwolone: {FRAMEWORKS}."
        )
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ModelPackageError(f"Nieobsługiwana wersja schematu: {metadata['schema_version']}.")


def save_model_package(model: Any, metadata: dict[str, Any], path: Path | str) -> Path:
    """Zapisuje model i ``metadata.json`` do katalogu ``path`` (poprzedni pakiet jest zastępowany)."""
    validate_metadata(metadata)
    path = Path(path)
    if path.exists():
        if any(path.iterdir()) and not (path / METADATA_FILE).is_file():
            raise ModelPackageError(f"Katalog {path} nie jest pakietem modelu — nie nadpisuję go.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    model_dir = path / MODEL_DIR
    if metadata["framework"] == "sklearn":
        model_dir.mkdir()
        joblib.dump(model, model_dir / SKLEARN_FILE)
    else:
        model.clone_for_deployment(str(model_dir))
    (path / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_model_package(path: Path | str) -> ModelPackage:
    """Wczytuje pakiet zapisany przez ``save_model_package`` bez ponownego treningu."""
    path = Path(path)
    metadata_path = path / METADATA_FILE
    if not metadata_path.is_file():
        raise ModelPackageError(f"Brak pliku {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    model_dir = path / MODEL_DIR
    if metadata["framework"] == "sklearn":
        model = joblib.load(model_dir / SKLEARN_FILE)
    else:
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(model_dir), verbosity=0)
    return ModelPackage(model=model, metadata=metadata)


def predict(package: ModelPackage, df: pd.DataFrame) -> pd.DataFrame:
    """Predykcja dla surowych rekordów: kolumny ``prediction`` (0/1) i ``probability``."""
    missing = [column for column in package.features if column not in df.columns]
    if missing:
        raise ModelPackageError(f"Brak cech wejściowych: {missing}.")
    X = normalize_input(df.loc[:, package.features])
    if package.framework == "sklearn":
        positive = list(package.model.classes_).index(1)
        probability = package.model.predict_proba(X)[:, positive]
        prediction = package.model.predict(X)
    else:
        categorical = dict.fromkeys(package.metadata["categorical_features"], "category")
        X = X.astype(categorical)
        probability = package.model.predict_proba(X, as_multiclass=False).to_numpy()
        prediction = package.model.predict(X).to_numpy()
    return pd.DataFrame(
        {"prediction": prediction.astype(int), "probability": probability.astype(float)},
        index=df.index,
    )


def prediction_problems(result: pd.DataFrame, n_rows: int) -> list[str]:
    """Sprawdza schemat odpowiedzi ``predict`` (pusta lista = schemat poprawny)."""
    problems: list[str] = []
    if tuple(result.columns) != PREDICTION_COLUMNS:
        problems.append(f"Kolumny {list(result.columns)} zamiast {list(PREDICTION_COLUMNS)}.")
        return problems
    if len(result) != n_rows:
        problems.append(f"Liczba predykcji {len(result)} zamiast {n_rows}.")
    if not pd.api.types.is_integer_dtype(result["prediction"]):
        problems.append(f"Kolumna prediction ma typ {result['prediction'].dtype}.")
    if not set(result["prediction"]).issubset({0, 1}):
        problems.append(f"Predykcje spoza {{0, 1}}: {sorted(set(result['prediction']))}.")
    if not pd.api.types.is_float_dtype(result["probability"]):
        problems.append(f"Kolumna probability ma typ {result['probability'].dtype}.")
    elif not result["probability"].between(0, 1).all():
        problems.append("Prawdopodobieństwa spoza przedziału [0, 1].")
    return problems
