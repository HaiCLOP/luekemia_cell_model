from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = "Research demo. Not for clinical use."


class Band(BaseModel):
    t_low: float
    t_high: float


class PredictResponse(BaseModel):
    prediction_id: str
    model_version: str
    prob_raw: float = Field(description="Mean leukemic probability of the ensemble members, before calibration")
    prob_calibrated: float = Field(description="Temperature-scaled probability used for the decision")
    decision: Literal["leukemic", "normal", "refer_to_expert"]
    band: Band
    heatmap_png_base64: str | None = Field(description="Class activation map over the resized cell, PNG, base64")
    latency_ms: int
    image_stored: bool = False
    disclaimer: str = DISCLAIMER


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(pattern=r"^[0-9a-fA-F-]{36}$")
    user_label: Literal["leukemic", "normal", "unsure"]
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    feedback_id: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    database: Literal["connected", "disabled", "error"]
    disclaimer: str = DISCLAIMER


class ModelCard(BaseModel):
    name: str
    architecture: str
    temperature: float
    band: Band
    band_target: str
    metrics: dict
    input: dict
    exported: str
    disclaimer: str = DISCLAIMER
