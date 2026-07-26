#!/usr/bin/env python
"""Leakage-controlled ModelC development with frozen independent validation."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
TABLES = ROOT / "tables"
MODELS = ROOT / "models"
SOURCE = ROOT.parent / "04_targeted_measurement_clinical_association" / "tables" / "integrated_targeted_area_ratio_long.tsv"
MODELA_SCRIPT = PROJECT / "05_diagnostic_model_discovery" / "scripts" / "train_ModelA_nested_cv.py"

SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS = 5
WEIGHT_STEP = 0.05
PROTOCOL = "20260615_modelc_nested_independent_v1"
ANALYTES = ["3-GPA", "Acetylcarnitine", "Creatine", "DHEA-S", "Arginine", "Carnitine", "Phenylalanine", "Tryptophan"]


def load_modela():
    spec = importlib.util.spec_from_file_location("modela_training", MODELA_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODELA = load_modela()
MODEL_LABELS = {**MODELA.MODEL_LABELS, "three_view_ensemble": "Optimized ensemble model"}


def youden_threshold(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds)
    score = tpr[valid] - fpr[valid]
    candidates = np.flatnonzero(np.isclose(score, score.max()))
    return float(np.min(thresholds[valid][candidates]))


def metric_row(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "accuracy": accuracy_score(y, predicted),
        "sensitivity": recall_score(y, predicted),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted),
        "malignant_class_f1": f1_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_data():
    long = pd.read_csv(SOURCE, sep="\t")
    use = long[
        long["is_primary"].eq(True)
        & long["group_code"].isin(["N", "M"])
        & long["analyte"].isin(ANALYTES)
    ].copy()
    metadata = use[["sample_uid", "batch_display", "group_code"]].drop_duplicates()
    area = use.pivot(index="sample_uid", columns="analyte", values="Area Ratio").reset_index()
    log2_area = use.pivot(index="sample_uid", columns="analyte", values="log2_area_ratio").reset_index()
    meta = metadata.merge(area, on="sample_uid", validate="one_to_one")
    log2_frame = metadata.merge(log2_area, on="sample_uid", validate="one_to_one")
    meta["label"] = meta["group_code"].eq("M").astype(int)
    log2_frame["label"] = log2_frame["group_code"].eq("M").astype(int)
    meta["dataset"] = np.where(meta["batch_display"].eq("Batch1"), "exploratory_cohort", "independent_validation_cohort")
    log2_frame["dataset"] = meta["dataset"]
    assert meta[ANALYTES].notna().all().all() and (meta[ANALYTES] > 0).all().all()
    audit = meta.groupby(["dataset", "group_code"]).size().rename("n").reset_index()
    audit.to_csv(TABLES / "ModelC_input_audit.tsv", sep="\t", index=False)
    return meta, log2_frame


def standard_pipeline(classifier):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def candidate_specs():
    return MODELA.model_specs()


def gpc():
    return GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=SEED, max_iter_predict=300)


def three_view_models():
    return {
        "log2_lda": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("classifier", LinearDiscriminantAnalysis())]),
        "log2_gpc": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("classifier", gpc())]),
        "area_composition_gpc": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("composition", Normalizer(norm="l1")),
                ("scale", StandardScaler()),
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
                yield (round(float(a), 10), round(float(b), 10), round(float(max(c, 0)), 10))


def choose_weights(y, probabilities):
    best = None
    for weights in weight_grid():
        p = sum(w * probabilities[name] for w, name in zip(weights, probabilities))
        candidate = (roc_auc_score(y, p), -sum((w - 1 / 3) ** 2 for w in weights), weights)
        if best is None or candidate > best:
            best = candidate
    return best[2], best[0]


class ThreeViewModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = tuple(weights)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        log2_x, area_x = X
        probability = (
            self.weights[0] * self.models["log2_lda"].predict_proba(log2_x)[:, 1]
            + self.weights[1] * self.models["log2_gpc"].predict_proba(log2_x)[:, 1]
            + self.weights[2] * self.models["area_composition_gpc"].predict_proba(area_x)[:, 1]
        )
        return np.column_stack([1 - probability, probability])


def repeat_summary(predictions):
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        rows.append({"repeat": repeat, **metric_row(group["true_label"].to_numpy(), group["probability"].to_numpy(), 0.5)})
        pred = group["predicted_label"].to_numpy()
        y = group["true_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows[-1].update(
            {
                "accuracy": accuracy_score(y, pred),
                "sensitivity": recall_score(y, pred),
                "specificity": tn / (tn + fp),
                "f1": f1_score(y, pred),
                "malignant_class_f1": f1_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metrics(y, probability, threshold, n_boot=1000):
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append({"bootstrap": i + 1, **metric_row(y[idx], probability[idx], threshold)})
    return pd.DataFrame(rows)


def fit_three_view_fold(log2_train, area_train, y_train, log2_test, area_test, inner):
    models = three_view_models()
    inner_prob = {}
    fitted = {}
    outer_prob = {}
    for name, model in models.items():
        source_train = area_train if name == "area_composition_gpc" else log2_train
        source_test = area_test if name == "area_composition_gpc" else log2_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inner_prob[name] = cross_val_predict(clone(model), source_train, y_train, cv=inner, method="predict_proba", n_jobs=-1)[:, 1]
            fitted[name] = clone(model).fit(source_train, y_train)
            outer_prob[name] = fitted[name].predict_proba(source_test)[:, 1]
    weights, inner_auc = choose_weights(y_train, inner_prob)
    inner_ensemble = sum(w * inner_prob[name] for w, name in zip(weights, inner_prob))
    test_ensemble = sum(w * outer_prob[name] for w, name in zip(weights, outer_prob))
    return fitted, weights, inner_auc, inner_ensemble, test_ensemble


def nested_development(log2_x, area_x, y):
    checkpoint = TABLES / "ModelC_nested_outer_selection_checkpoint.tsv"
    pred_checkpoint = TABLES / "ModelC_nested_outer_predictions_checkpoint.tsv"
    protocol_path = TABLES / "ModelC_nested_protocol.json"
    protocol_path.write_text(json.dumps({"protocol": PROTOCOL, "outer": "5x20", "inner": "5-fold", "seed": SEED, "fixed_analytes": ANALYTES}, indent=2), encoding="utf-8")
    selection_rows, prediction_rows = [], []
    specs = candidate_specs()
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    for outer_index, (train, test) in enumerate(outer.split(log2_x, y)):
        repeat, fold = outer_index // OUTER_SPLITS + 1, outer_index % OUTER_SPLITS + 1
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_index)
        candidates = []
        for key, (classifier, grid) in specs.items():
            pipeline = standard_pipeline(classifier)
            search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True, error_score=np.nan)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(log2_x[train], y[train])
                    inner_probability = cross_val_predict(clone(search.best_estimator_), log2_x[train], y[train], cv=inner, method="predict_proba", n_jobs=-1)[:, 1]
                threshold = youden_threshold(y[train], inner_probability)
                probability = search.best_estimator_.predict_proba(log2_x[test])[:, 1]
                row = {"repeat": repeat, "fold": fold, "model_key": key, "model_label": MODEL_LABELS[key], "inner_auc": float(search.best_score_), "best_params_json": json.dumps(search.best_params_, sort_keys=True), **metric_row(y[test], probability, threshold)}
                selection_rows.append(row)
                candidates.append((key, float(search.best_score_), probability, threshold))
            except Exception as exc:
                selection_rows.append({"repeat": repeat, "fold": fold, "model_key": key, "model_label": MODEL_LABELS[key], "error": str(exc)[:400]})
        _, weights, inner_auc, inner_probability, probability = fit_three_view_fold(log2_x[train], area_x[train], y[train], log2_x[test], area_x[test], inner)
        threshold = youden_threshold(y[train], inner_probability)
        selection_rows.append({"repeat": repeat, "fold": fold, "model_key": "three_view_ensemble", "model_label": MODEL_LABELS["three_view_ensemble"], "inner_auc": inner_auc, "best_params_json": json.dumps({"weights": weights}), **metric_row(y[test], probability, threshold)})
        candidates.append(("three_view_ensemble", inner_auc, probability, threshold))
        winner_key, _, winner_probability, winner_threshold = max(candidates, key=lambda x: x[1])
        for local, sample_index in enumerate(test):
            prediction_rows.append({"repeat": repeat, "fold": fold, "sample_index": int(sample_index), "true_label": int(y[sample_index]), "probability": float(winner_probability[local]), "threshold_from_inner_oof": winner_threshold, "predicted_label": int(winner_probability[local] >= winner_threshold), "selected_model": winner_key})
        pd.DataFrame(selection_rows).to_csv(checkpoint, sep="\t", index=False)
        pd.DataFrame(prediction_rows).to_csv(pred_checkpoint, sep="\t", index=False)
        print(f"Completed repeat {repeat:02d}, fold {fold}: {winner_key}", flush=True)
    return pd.DataFrame(selection_rows), pd.DataFrame(prediction_rows)


def lock_configuration(selection, log2_x, area_x, y):
    valid = selection.dropna(subset=["roc_auc"]).copy()
    summary = valid.groupby(["model_key", "model_label"], as_index=False).agg(mean_auc=("roc_auc", "mean"), sd_auc=("roc_auc", "std"), ci_low=("roc_auc", lambda x: x.quantile(.025)), ci_high=("roc_auc", lambda x: x.quantile(.975)), mean_pr_auc=("pr_auc", "mean"), mean_brier=("brier", "mean"), completed_folds=("roc_auc", "size")).sort_values("mean_auc", ascending=False)
    selected = str(summary.iloc[0]["model_key"])
    summary.to_csv(TABLES / "ModelC_model_screening_summary.tsv", sep="\t", index=False)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    if selected == "three_view_ensemble":
        models, weights, _, oof_probability, _ = fit_three_view_fold(log2_x, area_x, y, log2_x, area_x, inner)
        package = {"model_key": selected, "model_label": MODEL_LABELS[selected], "base_models": models, "weights": weights, "feature_names": ANALYTES, "input_views": ["log2_area_ratio", "raw_area_ratio"]}
    else:
        classifier, grid = candidate_specs()[selected]
        search = GridSearchCV(standard_pipeline(classifier), grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
        search.fit(log2_x, y)
        final_model = search.best_estimator_
        oof_probability = cross_val_predict(clone(final_model), log2_x, y, cv=inner, method="predict_proba", n_jobs=-1)[:, 1]
        final_model.fit(log2_x, y)
        package = {"model_key": selected, "model_label": MODEL_LABELS[selected], "model": final_model, "best_params": search.best_params_, "feature_names": ANALYTES, "input_views": ["log2_area_ratio"]}
    threshold = youden_threshold(y, oof_probability)
    package["locked_threshold"] = threshold
    package["validation_protocol"] = {
        "development": "Repeated nested CV 5 folds x 20 repeats",
        "independent_validation_frozen": True,
        "threshold_source": "provisional 5-fold exploratory-cohort OOF Youden; replaced by repeated outer-OOF Youden before validation",
    }
    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "ModelC_locked_model.pkl", "wb") as handle:
        pickle.dump(package, handle)
    return package, summary


def locked_development_oof(package, log2_x, area_x, y):
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    for outer_index, (train, test) in enumerate(outer.split(log2_x, y)):
        repeat, fold = outer_index // OUTER_SPLITS + 1, outer_index % OUTER_SPLITS + 1
        if package["model_key"] == "three_view_ensemble":
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_index)
            _, weights, _, inner_probability, probability = fit_three_view_fold(log2_x[train], area_x[train], y[train], log2_x[test], area_x[test], inner)
            threshold = youden_threshold(y[train], inner_probability)
        else:
            fitted = clone(package["model"]).fit(log2_x[train], y[train])
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_index)
            inner_probability = cross_val_predict(clone(package["model"]), log2_x[train], y[train], cv=inner, method="predict_proba", n_jobs=-1)[:, 1]
            threshold = youden_threshold(y[train], inner_probability)
            probability = fitted.predict_proba(log2_x[test])[:, 1]
        for local, sample_index in enumerate(test):
            rows.append({"repeat": repeat, "fold": fold, "sample_index": int(sample_index), "true_label": int(y[sample_index]), "probability": float(probability[local]), "threshold_from_inner_oof": threshold, "predicted_label": int(probability[local] >= threshold)})
    return pd.DataFrame(rows)


def predict_locked(package, log2_x, area_x):
    if package["model_key"] == "three_view_ensemble":
        return (
            package["weights"][0] * package["base_models"]["log2_lda"].predict_proba(log2_x)[:, 1]
            + package["weights"][1] * package["base_models"]["log2_gpc"].predict_proba(log2_x)[:, 1]
            + package["weights"][2] * package["base_models"]["area_composition_gpc"].predict_proba(area_x)[:, 1]
        )
    return package["model"].predict_proba(log2_x)[:, 1]


def main():
    TABLES.mkdir(exist_ok=True)
    meta, log2_frame = load_data()
    dev = meta["dataset"].eq("exploratory_cohort").to_numpy()
    val = meta["dataset"].eq("independent_validation_cohort").to_numpy()
    area_dev, area_val = meta.loc[dev, ANALYTES].to_numpy(), meta.loc[val, ANALYTES].to_numpy()
    log2_dev, log2_val = log2_frame.loc[dev, ANALYTES].to_numpy(), log2_frame.loc[val, ANALYTES].to_numpy()
    y_dev, y_val = meta.loc[dev, "label"].to_numpy(), meta.loc[val, "label"].to_numpy()

    selection, procedure_predictions = nested_development(log2_dev, area_dev, y_dev)
    selection.to_csv(TABLES / "ModelC_nested_outer_selection.tsv", sep="\t", index=False)
    procedure_predictions.to_csv(TABLES / "ModelC_nested_selection_procedure_predictions.tsv", sep="\t", index=False)
    package, screening = lock_configuration(selection, log2_dev, area_dev, y_dev)
    dev_predictions = locked_development_oof(package, log2_dev, area_dev, y_dev)
    dev_predictions.to_csv(TABLES / "ModelC_locked_exploratory_outer_predictions.tsv", sep="\t", index=False)
    dev_metrics = repeat_summary(dev_predictions)
    dev_metrics.to_csv(TABLES / "ModelC_locked_exploratory_repeat_metrics.tsv", sep="\t", index=False)
    averaged = dev_predictions.groupby(["sample_index", "true_label"], as_index=False).agg(probability=("probability", "mean"), prediction_count=("probability", "size"))
    averaged.to_csv(TABLES / "ModelC_exploratory_averaged_oof_predictions.tsv", sep="\t", index=False)
    threshold = youden_threshold(averaged["true_label"].to_numpy(), averaged["probability"].to_numpy())
    package["locked_threshold"] = threshold
    package["validation_protocol"]["threshold_source"] = "exploratory-cohort repeated outer-OOF averaged-probability Youden"
    with open(MODELS / "ModelC_locked_model.pkl", "wb") as handle:
        pickle.dump(package, handle)

    val_probability = predict_locked(package, log2_val, area_val)
    val_predictions = meta.loc[val, ["sample_uid", "group_code"]].copy()
    val_predictions["true_label"] = y_val
    val_predictions["probability"] = val_probability
    val_predictions["predicted_label"] = (val_probability >= threshold).astype(int)
    val_predictions.to_csv(TABLES / "ModelC_independent_validation_predictions.tsv", sep="\t", index=False)
    val_boot = bootstrap_metrics(y_val, val_probability, threshold)
    val_boot.to_csv(TABLES / "ModelC_independent_validation_bootstrap_metrics.tsv", sep="\t", index=False)

    summary = {
        "selected_model_key": package["model_key"],
        "selected_model_label": package["model_label"],
        "locked_threshold": threshold,
        "exploratory_n": len(y_dev),
        "independent_validation_n": len(y_val),
        "exploratory_averaged_oof": metric_row(averaged["true_label"].to_numpy(), averaged["probability"].to_numpy(), threshold),
        "exploratory_repeat_mean": dev_metrics.mean(numeric_only=True).to_dict(),
        "independent_validation": metric_row(y_val, val_probability, threshold),
    }
    (TABLES / "ModelC_nested_independent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    qa = [
        "# ModelC nested development and independent validation QA",
        "",
        f"- Selected model: {package['model_label']}.",
        f"- Exploratory cohort: n={len(y_dev)} (N={(y_dev == 0).sum()}, M={(y_dev == 1).sum()}).",
        f"- Independent validation cohort: n={len(y_val)} (N={(y_val == 0).sum()}, M={(y_val == 1).sum()}).",
        f"- Locked exploratory-cohort Youden threshold: {threshold:.6f}.",
        f"- Independent validation ROC AUC: {summary['independent_validation']['roc_auc']:.6f}.",
        "- All eight analytes were fixed before modelling.",
        "- Independent validation data were excluded from model, hyperparameter, weight and threshold selection.",
        "- Every exploratory-cohort sample has 20 outer-fold predictions.",
    ]
    (ROOT / "ModelC_nested_independent_QA.md").write_text("\n".join(qa), encoding="utf-8")


if __name__ == "__main__":
    main()
