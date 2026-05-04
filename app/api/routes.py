from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import OptionsResponse, PredictRequest, PredictResponse
from app.services.predictor import get_options, predict


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@router.get("/debug/versions")
def debug_versions() -> dict[str, str | bool]:
    import sklearn
    from sklearn.impute import SimpleImputer

    si = SimpleImputer()
    return {
        "python_executable": __import__("sys").executable,
        "sklearn_version": sklearn.__version__,
        "sklearn_location": getattr(sklearn, "__file__", ""),
        "simpleimputer_has__fill_dtype": hasattr(si, "_fill_dtype"),
    }


@router.get("/options", response_model=OptionsResponse)
def options() -> OptionsResponse:
    return OptionsResponse(**get_options())


@router.post("/predict", response_model=PredictResponse)
def predict_price(body: PredictRequest) -> PredictResponse:
    try:
        result = predict(body.features.model_dump())
        return PredictResponse(
            predicted_price=result.predicted_price,
            predicted_log_price=result.predicted_log_price,
            predicted_raw=result.predicted_raw,
            confidence_level=result.confidence_level,
            predicted_price_lower=result.predicted_price_lower,
            predicted_price_upper=result.predicted_price_upper,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

