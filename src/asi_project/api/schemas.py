"""Schematy żądań i odpowiedzi API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Odpowiedź endpointu GET /healthz."""

    status: Literal["ok"] = "ok"
    model_version: str = Field(min_length=1)
    persistence: Literal["sql", "disabled"]


class PredictResponse(BaseModel):
    """Odpowiedź endpointu POST /predict"""

    prediction: Literal[0, 1]
    probability: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1)


class PredictRequest(BaseModel):
    """Dane jednego klienta przekazane do predykcji."""

    model_config = ConfigDict(extra="forbid", strict=True)

    call_failure: int = Field(ge=0, alias="Call Failure")
    subscription_length: int = Field(ge=0, alias="Subscription Length")
    seconds_of_use: int = Field(ge=0, alias="Seconds of Use")
    frequency_of_use: int = Field(ge=0, alias="Frequency of use")
    frequency_of_sms: int = Field(ge=0, alias="Frequency of SMS")
    distinct_called_numbers: int = Field(ge=0, alias="Distinct Called Numbers")

    complains: int = Field(ge=0, le=1, alias="Complains")
    tariff_plan: int = Field(ge=1, le=2, alias="Tariff Plan")
    status: int = Field(ge=1, le=2, alias="Status")
    age_group: int = Field(ge=1, le=5, alias="Age Group")
    charge_amount: int = Field(ge=0, le=10, alias="Charge Amount")
