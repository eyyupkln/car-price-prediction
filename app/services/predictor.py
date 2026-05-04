from __future__ import annotations

import pickle
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import re
from datetime import datetime


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "car_price_model.pkl"
DATASET_PATH = ROOT_DIR / "data" / "turkey_used_cars.csv"


@dataclass(frozen=True)
class PredictionResult:
    predicted_log_price: float
    predicted_price: float
    predicted_raw: float
    confidence_level: float
    predicted_log_price_lower: float
    predicted_log_price_upper: float
    predicted_price_lower: float
    predicted_price_upper: float


def _load_pickle_model(path: Path) -> Any:

    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


@lru_cache(maxsize=1)
def get_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return _load_pickle_model(MODEL_PATH)


@lru_cache(maxsize=1)
def get_options() -> dict[str, list[str]]:
    if not DATASET_PATH.exists():
        return {
            "brands": [],
            "models": [],
            "variants": [],
            "colors": [],
            "cities": [],
            "years": [],
        }

    df = pd.read_csv(DATASET_PATH)

    def uniq(col: str) -> list[str]:
        if col not in df.columns:
            return []
        return (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace({"": None})
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    return {
        "brands": uniq("brand"),
        "models": uniq("model"),
        "variants": uniq("variant"),
        "colors": uniq("color"),
        "cities": uniq("city"),
        "years": (
            df["year"].dropna().astype(int).drop_duplicates().sort_values().tolist()
            if "year" in df.columns
            else []
        ),
    }


@lru_cache(maxsize=1)
def get_calibration() -> tuple[float, float, float]:

    if not DATASET_PATH.exists():
        return (1.0, 0.0, 0.9)

    df = pd.read_csv(DATASET_PATH)
    base_required = {"brand", "model", "variant", "km", "color", "city", "price"}
    if not base_required.issubset(set(df.columns)):
        return (1.0, 0.0, 0.9)

    sample = df.sample(n=min(8000, len(df)), random_state=42).copy()


    if "car_age" not in sample.columns:
        if "year" in sample.columns:
            current_year = datetime.now().year
            sample["car_age"] = (current_year - sample["year"].astype(float)).clip(lower=0)
        else:
            sample["car_age"] = np.nan

    if "engine_cc" not in sample.columns:
        def parse_engine_cc(v: Any) -> float | None:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            s = str(v)
            m = re.search(r"(\\d(?:[\\.,]\\d)?)", s)
            if not m:
                return None
            try:
                liters = float(m.group(1).replace(",", "."))
            except Exception:
                return None
            if liters < 20:
                return float(liters * 1000.0)
            return float(liters)

        sample["engine_cc"] = sample["variant"].map(parse_engine_cc)


    sample["engine_cc"] = sample["engine_cc"].astype(float)
    if sample["engine_cc"].isna().any():
        sample["engine_cc"] = sample["engine_cc"].fillna(sample["engine_cc"].median())
    if sample["car_age"].isna().any():
        sample["car_age"] = sample["car_age"].fillna(sample["car_age"].median())

    X = sample[["brand", "model", "variant", "km", "color", "city", "car_age", "engine_cc"]]
    y = np.log1p(sample["price"].astype(float).to_numpy())

    model = get_model()
    raw = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if raw.size < 10 or float(np.std(raw)) < 1e-12:
        return (1.0, 0.0, float(np.std(y)) if y.size else 0.9)

    a, b = np.polyfit(raw, y, deg=1)
    y_hat = a * raw + b
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    rmse = max(rmse, 0.15)
    return (float(a), float(b), rmse)


def _z_value(confidence_level: float) -> float:

    if confidence_level >= 0.99:
        return 2.576
    if confidence_level >= 0.95:
        return 1.96
    if confidence_level >= 0.90:
        return 1.645
    if confidence_level >= 0.80:
        return 1.282
    return 1.0


def predict(features: dict[str, Any]) -> PredictionResult:
    model = get_model()
    current_year = datetime.now().year
    car_age = max(0, int(current_year - int(features["year"])))
    X = pd.DataFrame(
        [
            {
                "brand": features["brand"],
                "model": features["model"],
                "variant": features["variant"],
                "km": int(features["km"]),
                "color": features["color"],
                "city": features["city"],
                "car_age": car_age,
                "engine_cc": float(features["engine_cc"]),
            }
        ]
    )

    raw_pred = float(model.predict(X)[0])
    a, b, rmse = get_calibration()
    predicted_log_price = float(a * raw_pred + b)

    confidence_level = float(features.get("confidence_level", 0.95))

    predicted_real_price = float(math.expm1(predicted_log_price))

    z = _z_value(confidence_level)
    half_width = z * rmse

    predicted_log_price_lower = float(predicted_log_price - half_width)
    predicted_log_price_upper = float(predicted_log_price + half_width)

    dynamic_margin = predicted_real_price * 0.05
    predicted_price_lower = predicted_real_price - dynamic_margin
    predicted_price_upper = predicted_real_price + dynamic_margin

    predicted_price = float(math.expm1(predicted_log_price))
    #predicted_price_lower = float(math.expm1(predicted_log_price_lower))
    #predicted_price_upper = float(math.expm1(predicted_log_price_upper))
    return PredictionResult(
        predicted_log_price=predicted_log_price,
        predicted_price=predicted_price,
        predicted_raw=raw_pred,
        confidence_level=confidence_level,
        predicted_log_price_lower=predicted_log_price_lower,
        predicted_log_price_upper=predicted_log_price_upper,
        predicted_price_lower=predicted_price_lower,
        predicted_price_upper=predicted_price_upper,
    )

