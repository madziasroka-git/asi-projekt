"""Aplikacja API korzystająca z zapisanego pakietu modelu."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI

from asi_project.api.schemas import HealthResponse, PredictRequest, PredictResponse
from asi_project.model_package import load_model_package, predict


def create_app(model_path: Path, persistence: Literal["sql", "disabled"]) -> FastAPI:
    """Tworzy aplikację korzystającą ze wskazanego pakietu modelu."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_package = load_model_package(model_path)
        try:
            yield
        finally:
            del app.state.model_package

    app = FastAPI(title="Iranian Churn API", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        """Zwraca status usługi, wersję modelu i tryb persistence."""
        metadata = app.state.model_package.metadata
        run_id = metadata.get("experiment_id", metadata["git_sha"])

        return HealthResponse(
            model_version=f"{metadata['model_name']}:{run_id}",
            persistence=persistence,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(body: PredictRequest) -> PredictResponse:
        """Wykonuje predykcję dla jednego klienta."""
        package = app.state.model_package

        data = pd.DataFrame([body.model_dump(by_alias=True)])
        result = predict(package, data)
        row = result.iloc[0]

        metadata = package.metadata
        run_id = metadata.get("experiment_id", metadata["git_sha"])
        model_version = f"{metadata['model_name']}:{run_id}"

        return PredictResponse(
            prediction=int(row["prediction"]),
            probability=float(row["probability"]),
            model_version=model_version,
        )

    return app
