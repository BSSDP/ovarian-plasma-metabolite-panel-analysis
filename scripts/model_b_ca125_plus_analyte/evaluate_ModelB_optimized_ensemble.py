#!/usr/bin/env python
"""Evaluate an optimized probability ensemble against the locked single ModelB."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path(__file__).resolve().parents[1]
TABLES, MODELS = ANALYSIS / "tables", ANALYSIS / "models"
RUNNER_PATH = Path(__file__).with_name("run_ModelB_CA125_8analyte_nested_cv.py")
SEED, WEIGHT_STEP = 42, 0.05


def import_runner():
    spec = importlib.util.spec_from_file_location("modelb_runner_ensemble", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CA125CompositionScaler(BaseEstimator, TransformerMixin):
    """Scale CA125 separately and convert selected metabolites to compositional values."""

    def fit(self, X, y=None):
        transformed = self._base_transform(X)
        self.scaler_ = StandardScaler().fit(transformed)
        return self

    def transform(self, X):
        return self.scaler_.transform(self._base_transform(X))

    @staticmethod
    def _base_transform(X):
        X = np.asarray(X, dtype=float)
        ca125 = X[:, [0]]
        metabolites = np.log1p(np.clip(X[:, 1:], a_min=0, a_max=None))
        denominator = metabolites.sum(axis=1, keepdims=True)
        denominator[denominator == 0] = 1
        return np.c_[ca125, metabolites / denominator]


def gpc():
    return GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=SEED, max_iter_predict=300)


def models(runner, n_predictors):
    selector = lambda: runner.AnchoredRFE(n_features_to_select=n_predictors)
    return {
        "linear_discriminant": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("selector", selector()),
                ("classifier", LinearDiscriminantAnalysis()),
            ]
        ),
        "gaussian_process": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("selector", selector()),
                ("classifier", gpc()),
            ]
        ),
        "composition_gaussian_process": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", selector()),
                ("composition_scale", CA125CompositionScaler()),
                ("classifier", gpc()),
            ]
        ),
    }


def weight_grid():
    values = np.arange(0, 1 + WEIGHT_STEP / 2, WEIGHT_STEP)
    for a in values:
        for b in values:
            c = 1 - a - b
            if c >= -1e-9:
                yield round(float(a), 10), round(float(b), 10), round(float(max(0, c)), 10)


def choose_weights(y, predictions):
    best = None
    for weights in weight_grid():
        probability = sum(weight * predictions[name] for weight, name in zip(weights, predictions))
        candidate = (roc_auc_score(y, probability), -sum((weight - 1 / 3) ** 2 for weight in weights), weights)
        if best is None or candidate > best:
            best = candidate
    return best[2], best[0]


def repeat_metrics(predictions):
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        y, probability, pred = group["true_label"].to_numpy(), group["probability"].to_numpy(), group["predicted_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": roc_auc_score(y, probability),
                "pr_auc": average_precision_score(y, probability),
                "sensitivity": recall_score(y, pred),
                "specificity": tn / (tn + fp),
                "f1": f1_score(y, pred),
            }
        )
    return pd.DataFrame(rows)


def main():
    runner = import_runner()
    sys.modules["__main__"].AnchoredRFE = runner.AnchoredRFE
    modela = runner.load_modela_module()
    df, X, y, predictors = runner.load_data()
    with open(MODELS / "final_model.pkl", "rb") as handle:
        single_package = pickle.load(handle)
    n_predictors = int(single_package["selected_n_predictors"])
    base = models(runner, n_predictors)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    prediction_rows, weight_rows = [], []
    for outer_index, (train, test) in enumerate(outer.split(X, y)):
        repeat, fold = outer_index // 5 + 1, outer_index % 5 + 1
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + outer_index)
        inner_predictions, test_predictions = {}, {}
        for name, estimator in base.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inner_predictions[name] = cross_val_predict(
                    clone(estimator), X[train], y[train], cv=inner, method="predict_proba", n_jobs=-1
                )[:, 1]
                test_predictions[name] = clone(estimator).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        weights, inner_auc = choose_weights(y[train], inner_predictions)
        inner_probability = sum(weight * inner_predictions[name] for weight, name in zip(weights, base))
        threshold = modela.youden_threshold(y[train], inner_probability)
        probability = sum(weight * test_predictions[name] for weight, name in zip(weights, base))
        weight_rows.append({"repeat": repeat, "fold": fold, "inner_auc": inner_auc, **{f"{name}_weight": weight for name, weight in zip(base, weights)}})
        prediction_rows.extend(
            {
                "repeat": repeat,
                "fold": fold,
                "sample_index": int(sample),
                "true_label": int(y[sample]),
                "probability": float(probability[i]),
                "threshold_from_inner_oof": threshold,
                "predicted_label": int(probability[i] >= threshold),
            }
            for i, sample in enumerate(test)
        )
        print(f"Completed ensemble repeat {repeat:02d}, fold {fold}: weights={weights}")
    predictions, weights = pd.DataFrame(prediction_rows), pd.DataFrame(weight_rows)
    metrics = repeat_metrics(predictions)
    predictions.to_csv(TABLES / "ModelB_optimized_ensemble_outer_predictions.tsv", sep="\t", index=False)
    weights.to_csv(TABLES / "ModelB_optimized_ensemble_weights.tsv", sep="\t", index=False)
    metrics.to_csv(TABLES / "ModelB_optimized_ensemble_repeat_metrics.tsv", sep="\t", index=False)

    single_metrics = pd.read_csv(TABLES / "ModelB_augmented_repeat_metrics.tsv", sep="\t")
    paired = metrics[["repeat", "roc_auc"]].merge(
        single_metrics[["repeat", "roc_auc"]], on="repeat", suffixes=("_ensemble", "_single")
    )
    delta = paired["roc_auc_ensemble"] - paired["roc_auc_single"]
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])
    low, high = np.quantile(boot, [0.025, 0.975])
    select_ensemble = bool(delta.mean() >= 0.01 and low > 0)
    summary = {
        "public_name": "Optimized ensemble model",
        "selected_predictor_count": n_predictors,
        "ensemble_mean_auc": metrics["roc_auc"].mean(),
        "single_model_mean_auc": single_metrics["roc_auc"].mean(),
        "paired_delta_auc": delta.mean(),
        "paired_delta_auc_bootstrap_ci": [low, high],
        "ensemble_selection_rule": "delta AUC >= 0.01 and paired bootstrap 95% CI lower bound > 0",
        "ensemble_selected": select_ensemble,
    }
    (TABLES / "ModelB_optimized_ensemble_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
