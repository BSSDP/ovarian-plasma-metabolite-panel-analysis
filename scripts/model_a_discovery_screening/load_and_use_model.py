#!/usr/bin/env python
"""Load and apply the locked ModelA pipeline.

The probability threshold in this package was derived internally from repeated
out-of-fold predictions. It is not an externally validated clinical threshold.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).with_name("final_model.pkl")


def load_model(model_path: Path = MODEL_PATH) -> dict:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def predict(model_package: dict, data: pd.DataFrame) -> pd.DataFrame:
    feature_names = model_package["feature_names"]
    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    probability = model_package["pipeline"].predict_proba(data[feature_names])[:, 1]
    threshold = float(model_package["internally_derived_youden_threshold"])
    prediction = np.where(probability >= threshold, "M", "N")
    return pd.DataFrame(
        {
            "malignant_probability": probability,
            "predicted_label": prediction,
            "internally_derived_threshold": threshold,
        },
        index=data.index,
    )


if __name__ == "__main__":
    package = load_model()
    print(f"Selected model: {package['selected_model_label']}")
    print(f"Required features: {', '.join(package['feature_names'])}")
    print(f"Internally derived threshold: {package['internally_derived_youden_threshold']:.6f}")
