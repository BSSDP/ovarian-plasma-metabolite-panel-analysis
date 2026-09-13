from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
BASE = PROJECT_ROOT / "11_trarget" / "analysis_results_requested" / "11_two_step_clinical_application_Fig5_S5"
TABLE_DIR = BASE / "tables"
LOG_DIR = BASE / "logs"
INPUT = TABLE_DIR / "11_step2_paired_predictions.tsv"
CALIBRATION_OUT = TABLE_DIR / "11_step2_calibration_transport.tsv"
THRESHOLD_OUT = TABLE_DIR / "11_step2_threshold_transport.tsv"
RUN_LOG = LOG_DIR / "11_step2_calibration_transport_run.json"

THRESHOLD = 0.4779920056448391
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260909


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return np.log(probability / (1.0 - probability))


def calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    x = logit(p)
    value = 0.0
    for _ in range(100):
        fitted = expit(value + x)
        gradient = float(np.sum(y - fitted))
        information = float(np.sum(fitted * (1.0 - fitted)))
        if information <= 1e-12:
            break
        update = gradient / information
        value += update
        if abs(update) < 1e-10:
            break
    return float(value)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = logit(p)
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        fitted = expit(design @ coefficients)
        weights = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        gradient = design.T @ (y - fitted)
        information = design.T @ (weights[:, None] * design)
        try:
            update = np.linalg.solve(information + np.eye(2) * 1e-10, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        coefficients += update
        if np.max(np.abs(update)) < 1e-9:
            break
    if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) > 100:
        return np.nan, np.nan
    return float(coefficients[0]), float(coefficients[1])


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    full_intercept, slope = calibration_intercept_slope(y, p)
    return {
        "prevalence": float(np.mean(y)),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_expected_ratio": float(np.sum(y) / np.sum(p)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "calibration_in_the_large": calibration_intercept(y, p),
        "calibration_model_intercept": full_intercept,
        "calibration_slope": slope,
    }


def stratified_bootstrap(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rows = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        rows.append(point_metrics(y[indices], p[indices]))
    return pd.DataFrame(rows)


def interval(values: pd.Series) -> tuple[float, float, int]:
    finite = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.nan, np.nan, 0
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(len(finite))


def threshold_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    call = p >= THRESHOLD
    tp = int(np.sum(call & (y == 1)))
    fn = int(np.sum(~call & (y == 1)))
    tn = int(np.sum(~call & (y == 0)))
    fp = int(np.sum(call & (y == 0)))
    return {
        "n": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "threshold": THRESHOLD,
        "predicted_positive_fraction": float(np.mean(call)),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "accuracy": (tp + tn) / len(y),
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "probability_median": float(np.median(p)),
        "probability_q1": float(np.quantile(p, 0.25)),
        "probability_q3": float(np.quantile(p, 0.75)),
        "fraction_within_0.025_of_threshold": float(np.mean(np.abs(p - THRESHOLD) <= 0.025)),
        "fraction_within_0.05_of_threshold": float(np.mean(np.abs(p - THRESHOLD) <= 0.05)),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT, sep="\t")
    required = {"cohort", "label", "combined_probability", "combined_call"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    calibration_rows = []
    threshold_rows = []
    generation = {
        "Development": "mean of 20 held-out OOF probabilities per participant",
        "Independent validation": "mean probability from 100 fixed development-fold models",
    }

    for cohort in ["Development", "Independent validation"]:
        subset = data.loc[data["cohort"].eq(cohort)].copy()
        y = subset["label"].to_numpy(dtype=int)
        p = subset["combined_probability"].to_numpy(dtype=float)
        if not np.array_equal(p >= THRESHOLD, subset["combined_call"].astype(bool).to_numpy()):
            raise AssertionError(f"{cohort}: stored combined_call does not match fixed threshold")

        point = point_metrics(y, p)
        bootstrap = stratified_bootstrap(y, p)
        row: dict[str, float | int | str] = {
            "cohort": cohort,
            "n": int(len(subset)),
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
            "prediction_generation": generation[cohort],
            **point,
        }
        for metric in point:
            low, high, valid = interval(bootstrap[metric])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_bootstrap_valid"] = valid
        calibration_rows.append(row)

        threshold_rows.append(
            {
                "cohort": cohort,
                "prediction_generation": generation[cohort],
                **threshold_metrics(y, p),
            }
        )

    calibration = pd.DataFrame(calibration_rows)
    threshold = pd.DataFrame(threshold_rows)
    calibration.to_csv(CALIBRATION_OUT, sep="\t", index=False)
    threshold.to_csv(THRESHOLD_OUT, sep="\t", index=False)

    log = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "fixed_threshold": THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "bootstrap_sampling": "stratified by outcome at participant level",
        "outputs": [str(CALIBRATION_OUT), str(THRESHOLD_OUT)],
    }
    RUN_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("Calibration transport audit")
    print(calibration.to_string(index=False))
    print("\nFixed-threshold transport audit")
    print(threshold.to_string(index=False))


if __name__ == "__main__":
    main()
