"""Smoke modelu produkcyjnego: pobranie pakietu -> wczytanie -> predykcja 5 wierszy -> schemat.

Uruchamiany w osobnym procesie, niezależnie od treningu:

    python scripts/smoke_model.py
    python scripts/smoke_model.py --package-dir data/06_models/production

Bez ``--package-dir`` pobiera z W&B artefakt ``<entity>/<project>/<artifact>:production``
(zmienne ``WANDB_ENTITY`` i ``WANDB_PROJECT``).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from asi_project.config import load_project_config
from asi_project.model_package import load_model_package, predict, prediction_problems

ARTIFACT_NAME = "iranian-churn-model"
ALIAS = "production"
N_ROWS = 5


def download_package(artifact_name: str, alias: str, target: Path) -> Path:
    import wandb

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not (entity and project):
        raise SystemExit("Ustaw WANDB_ENTITY i WANDB_PROJECT albo podaj --package-dir.")
    reference = f"{entity}/{project}/{artifact_name}:{alias}"
    print(f"Pobieranie artefaktu: {reference}")
    artifact = wandb.Api().artifact(reference, type="model")
    print(f"Wersja artefaktu: {artifact.version}, aliasy: {artifact.aliases}")
    return Path(artifact.download(root=str(target)))


def run_smoke(package_dir: Path, rows: pd.DataFrame) -> list[str]:
    package = load_model_package(package_dir)
    metadata = package.metadata
    print(f"Model: {metadata['model_name']} ({metadata['framework']})")
    print(f"Target: {metadata['target']}, klasa pozytywna: {metadata['positive_class']}")
    print(f"Recall (test): {metadata['metrics'].get('recall')}")
    print(f"Git SHA: {metadata['git_sha']}, dane: {metadata['data_fingerprint'][:12]}")

    inputs = rows.drop(columns=[metadata["target"]], errors="ignore")
    result = predict(package, inputs)
    print(pd.concat([inputs.loc[:, package.features], result], axis=1).to_string())
    return prediction_problems(result, n_rows=len(inputs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package-dir", type=Path, help="lokalny pakiet zamiast pobierania z W&B")
    parser.add_argument("--artifact", default=ARTIFACT_NAME)
    parser.add_argument("--alias", default=ALIAS)
    args = parser.parse_args()

    config = load_project_config()
    rows = pd.read_csv(config.ci_sample_path).head(N_ROWS)

    with tempfile.TemporaryDirectory() as tmp:
        package_dir = args.package_dir or download_package(args.artifact, args.alias, Path(tmp))
        problems = run_smoke(package_dir, rows)

    if problems:
        print("SMOKE FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"SMOKE OK: {N_ROWS} predykcji, schemat odpowiedzi poprawny.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
