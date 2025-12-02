# ai_models/priceprediction.py

import pathlib
from typing import Any, Dict

import joblib  # type: ignore
import pandas as pd

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "perfume_price_model.pkl"

PRICE_MODEL = None
MODEL_LOAD_ERROR: Exception | None = None

try:
    PRICE_MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    MODEL_LOAD_ERROR = e


def predict_price_from_features(features: Dict[str, Any]) -> float:
    """
    Backend receives:
      product_type, brand, model_name, gender, capacity_ml

    Training pipeline expects columns:
      brand, perfume_name, gender, size_ml
    """
    if PRICE_MODEL is None:
        raise RuntimeError(
            f"Price model not loaded from {MODEL_PATH}. "
            f"Original error: {MODEL_LOAD_ERROR}"
        )

    # Map API payload -> training feature names
    internal = {
        "brand": features["brand"],
        "perfume_name": features["model_name"],
        "gender": features["gender"],
        "size_ml": features["capacity_ml"],
    }
    # product_type is not used in this version of the model

    df = pd.DataFrame([internal])

    y_pred = PRICE_MODEL.predict(df)
    return float(y_pred[0])
