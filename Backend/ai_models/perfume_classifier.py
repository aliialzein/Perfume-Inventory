# ai_models/perfume_classifier.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import io
import json
from typing import Dict, Any

import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from tensorflow import keras  # type: ignore

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR

BRAND_MODEL_PATH = WEIGHTS_DIR / "Perfumes_brand.h5"
BRAND_LABELS_PATH = WEIGHTS_DIR / "brand_classes.json"

NAME_GENDER_MODEL_PATH = WEIGHTS_DIR / "Perfume_NameGender_final.keras"
NAME_GENDER_LABELS_PATH = WEIGHTS_DIR / "name_gender_labels.json"

IMG_SIZE = (224, 224)


# ---------------------------------------------------------------------
# Lazy loaders (models + labels are loaded once and cached)
# ---------------------------------------------------------------------


@lru_cache
def load_brand_model():
    if not BRAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"Brand model not found at {BRAND_MODEL_PATH}")
    return keras.models.load_model(BRAND_MODEL_PATH, compile=False)


@lru_cache
def load_name_gender_model():
    if not NAME_GENDER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Name+Gender model not found at {NAME_GENDER_MODEL_PATH}"
        )
    return keras.models.load_model(NAME_GENDER_MODEL_PATH, compile=False)


@lru_cache
def load_brand_labels() -> list[str]:
    """
    Load brand labels, supporting either:
    - ["Chanel", "Dior", ...]
    - {"brand_classes": ["Chanel", "Dior", ...]}
    """
    if not BRAND_LABELS_PATH.exists():
        raise FileNotFoundError(f"Brand labels not found at {BRAND_LABELS_PATH}")

    with BRAND_LABELS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "brand_classes" in data:
            labels = data["brand_classes"]
        else:
            labels = next(iter(data.values()))
    elif isinstance(data, list):
        labels = data
    else:
        raise ValueError(
            f"Unexpected format in {BRAND_LABELS_PATH}: {type(data)}"
        )

    return [str(x) for x in labels]


@lru_cache
def load_name_gender_labels() -> tuple[list[str], list[str]]:
    """
    Load name + gender labels, supporting either:
    - {"name_classes": [...], "gender_classes": [...]}
    - or {"names": [...], "genders": [...]}
    """
    if not NAME_GENDER_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Name+Gender labels not found at {NAME_GENDER_LABELS_PATH}"
        )

    with NAME_GENDER_LABELS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected format in {NAME_GENDER_LABELS_PATH}: {type(data)}"
        )

    # main expected keys
    name_classes = data.get("name_classes") or data.get("names")
    gender_classes = data.get("gender_classes") or data.get("genders")

    if name_classes is None or gender_classes is None:
        raise ValueError(
            f"name_gender_labels.json missing name/gender arrays. Keys: {list(data.keys())}"
        )

    return [str(x) for x in name_classes], [str(x) for x in gender_classes]


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------


def preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw uploaded bytes to a (1, 224, 224, 3) float32 numpy array.
    """
    if not file_bytes:
        raise ValueError("Empty image bytes")

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------


def predict_brand(arr: np.ndarray) -> str:
    model = load_brand_model()
    classes = load_brand_labels()

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return classes[idx]

def predict_name_and_gender(arr: np.ndarray) -> tuple[str, str]:
    model = load_name_gender_model()
    name_classes, gender_classes = load_name_gender_labels()

    name_probs, gender_probs = model.predict(arr, verbose=0)
    name_idx = int(np.argmax(name_probs[0]))
    gender_idx = int(np.argmax(gender_probs[0]))

    model_name = name_classes[name_idx]
    gender = gender_classes[gender_idx]
    return model_name, gender

# ---------------------------------------------------------------------
# Main API used by routers
# ---------------------------------------------------------------------

def classify_perfume(file_bytes: bytes) -> Dict[str, Any]:
    """
    Given image bytes, return:
    {
      "product_type": "perfume",
      "brand": "...",
      "model_name": "...",
      "gender": "..."
    }
    """
    arr = preprocess_image_bytes(file_bytes)

    brand = predict_brand(arr)
    model_name, gender = predict_name_and_gender(arr)

    return {
        "product_type": "perfume",
        "brand": brand,
        "model_name": model_name,
        "gender": gender,
    }

# ---------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ai_models.perfume_classifier path/to/image.jpg")
        raise SystemExit(1)

    img_path = Path(sys.argv[1])
    with img_path.open("rb") as f:
        b = f.read()

    out = classify_perfume(b)
    print("Classification:", out)
