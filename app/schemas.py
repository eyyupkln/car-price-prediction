from __future__ import annotations

from pydantic import BaseModel, Field


class CarFeatures(BaseModel):
    brand: str = Field(..., examples=["toyota"])
    model: str = Field(..., examples=["corolla"])
    variant: str = Field(..., examples=["1.6 Advance"])
    km: int = Field(..., ge=0, examples=[175000])
    color: str = Field(..., examples=["White"])
    city: str = Field(..., examples=["istanbul"])
    year: int = Field(..., ge=1900, le=2100, examples=[2016])
    engine_cc: float = Field(..., ge=0, examples=[1600])


class PredictRequest(BaseModel):
    features: CarFeatures


class PredictResponse(BaseModel):
    predicted_price: float
    predicted_log_price: float
    predicted_raw: float
    confidence_level: float
    predicted_price_lower: float
    predicted_price_upper: float


class OptionsResponse(BaseModel):
    brands: list[str]
    models: list[str]
    variants: list[str]
    colors: list[str]
    cities: list[str]
    years: list[int]

