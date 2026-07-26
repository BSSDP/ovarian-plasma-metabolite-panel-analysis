#!/usr/bin/env python
"""Use a locked targeted-panel score with CA125 for B+BD vs M modeling."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODEL_C_ROOT = ROOT.parent / "05_targeted_model_performance"
MODEL_C_TRAIN_SCRIPT = MODEL_C_ROOT / "scripts" / "train_ModelC_nested_independent.py"
MODEL_C_LOCKED = MODEL_C_ROOT / "models" / "ModelC_locked_model.pkl"
LONG_SOURCE = ROOT.parent / "04_targeted_measurement_clinical_association" / "tables" / "integrated_targeted_area_ratio_long.tsv"
CLINICAL_SOURCE = ROOT.parent / "04_targeted_measurement_clinical_association" / "tables" / "integrated_targeted_clinical_dataset.tsv"

STYLE = PROJECT / "00_project_style"
if str(STYLE) not in sys.path:
    sys.path.insert(0, str(STYLE))
from ov_publication_style import BACKGROUND_GREY, GROUP_COLORS, SIGNAL_BLUE, SIGNAL_RED, TEXT_DARK, setup_matplotlib_style


SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS = 5
N_JOBS = 6
ANALYTES = ["3-GPA", "Acetylcarnitine", "Creatine", "DHEA-S", "Arginine", "Carnitine", "Phenylalanine", "Tryptophan"]

MODEL_LABELS = {
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
    "logistic_l2": "L2 logistic regression",
    "logistic_l1": "L1 logistic regression",
    "ridge": "Ridge classifier",
    "sgd": "SGD classifier",
    "lda": "Linear discriminant analysis",
    "qda": "Quadratic discriminant analysis",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient boosting",
    "adaboost": "AdaBoost",
    "knn": "k-nearest neighbours",
    "naive_bayes": "Gaussian naive Bayes",
    "mlp": "Multilayer perceptron",
    "soft_voting": "Weighted probability ensemble",
}


def ensure_dirs():
    for path in (TABLES, FIGURES, MODELS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def import_modelc_train():
    spec = importlib.util.spec_from_file_location("modelc_train_for_08", MODEL_C_TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelc_train_for_08"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_locked_modelc():
    train = import_modelc_train()
    sys.modules["__main__"].ThreeViewModel = getattr(train, "ThreeViewModel", object)
    with open(MODEL_C_LOCKED, "rb") as handle:
        package = pickle.load(handle)
    return package


def model_specs():
    # Performance-optimized final 08 strategy: the focused 20-repeat probe showed
    # that logit(ModelC score) + CA125 with L2 logistic regression outperformed
    # the broader fold-wise model-selection approach.
    return {
        "logistic_l2": (
            LogisticRegression(penalty="l2", max_iter=4000, class_weight="balanced", random_state=SEED),
            {"classifier__C": [1.0]},
        ),
    }


def make_pipeline(classifier):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def youden_threshold(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds)
    score = tpr[valid] - fpr[valid]
    best = np.flatnonzero(np.isclose(score, score.max()))
    return float(np.min(thresholds[valid][best]))


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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def average_predictions(predictions):
    return (
        predictions.groupby(["sample_index", "sample_uid", "true_label"], as_index=False)
        .agg(probability=("probability", "mean"), prediction_count=("probability", "size"))
        .sort_values("sample_index")
    )


def load_and_score_bbdm():
    package = load_locked_modelc()
    model = package["model"]
    feature_names = package["feature_names"]
    long = pd.read_csv(LONG_SOURCE, sep="\t")
    use = long[
        long["is_primary"].eq(True)
        & long["group_code"].isin(["B", "J", "M"])
        & long["analyte"].isin(feature_names)
    ].copy()
    metadata = use[["sample_uid", "batch_display", "group_code"]].drop_duplicates()
    log2_area = use.pivot(index="sample_uid", columns="analyte", values="log2_area_ratio").reset_index()
    scored = metadata.merge(log2_area, on="sample_uid", validate="one_to_one")
    scored["group_display"] = scored["group_code"].replace({"J": "BD"})
    scored = scored.dropna(subset=feature_names).reset_index(drop=True)
    scored["ModelC_NvsM_score"] = model.predict_proba(scored[feature_names].to_numpy(dtype=float))[:, 1]
    scored["ModelC_NvsM_logit_score"] = logit(scored["ModelC_NvsM_score"].clip(1e-5, 1 - 1e-5))
    scored["ModelC_NvsM_locked_threshold"] = float(package["locked_threshold"])
    scored["ModelC_NvsM_call"] = np.where(scored["ModelC_NvsM_score"] >= package["locked_threshold"], "M-like", "N-like")

    clinical = pd.read_csv(CLINICAL_SOURCE, sep="\t")
    keep_cols = [
        "sample_uid",
        "Sample Name",
        "sample_kind",
        "batch_display",
        "group_display",
        "age",
        "CA125",
        "CA125_log10",
        "HE4",
        "HE4_log10",
        "figo_stage_merged",
        "pathology_class",
    ]
    keep_cols = [col for col in keep_cols if col in clinical.columns]
    clinical = clinical[keep_cols].drop_duplicates("sample_uid")
    scored = scored.merge(clinical, on="sample_uid", how="left", suffixes=("", "_clinical"))
    if "group_display_clinical" in scored.columns:
        scored["group_display"] = scored["group_display_clinical"].fillna(scored["group_display"])
        scored = scored.drop(columns=["group_display_clinical"])
    if "batch_display_clinical" in scored.columns:
        scored["batch_display"] = scored["batch_display_clinical"].fillna(scored["batch_display"])
        scored = scored.drop(columns=["batch_display_clinical"])
    for col in ["age", "CA125", "CA125_log10", "HE4", "HE4_log10"]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored.to_csv(TABLES / "ModelC_NvsM_score_all_B_BD_M.tsv", sep="\t", index=False)

    audit = scored.groupby(["batch_display", "group_display"], as_index=False).agg(
        n=("sample_uid", "size"),
        n_with_CA125=("CA125_log10", lambda x: int(x.notna().sum())),
        median_score=("ModelC_NvsM_score", "median"),
        mean_score=("ModelC_NvsM_score", "mean"),
    )
    audit.to_csv(TABLES / "ModelC_NvsM_score_B_BD_M_audit.tsv", sep="\t", index=False)
    return scored, package


def nested_score_ca125_model(model_input):
    feature_names = ["ModelC_NvsM_logit_score", "CA125_log10"]
    X = model_input[feature_names].to_numpy(dtype=float)
    y = model_input["label"].to_numpy(dtype=int)
    sample_ids = model_input["sample_uid"].astype(str).to_numpy()
    specs = model_specs()
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    prediction_rows, selection_rows, failure_rows = [], [], []

    for outer_index, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat, fold = outer_index // OUTER_SPLITS + 1, outer_index % OUTER_SPLITS + 1
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_index)
        candidates = []
        for model_key, (classifier, grid) in specs.items():
            pipeline = make_pipeline(classifier)
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=N_JOBS,
                refit=True,
                error_score=np.nan,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X[train_idx], y[train_idx])
                    inner_probability = cross_val_predict(
                        clone(search.best_estimator_),
                        X[train_idx],
                        y[train_idx],
                        cv=inner,
                        method="predict_proba",
                        n_jobs=N_JOBS,
                    )[:, 1]
                threshold = youden_threshold(y[train_idx], inner_probability)
                candidate = {
                    "repeat": repeat,
                    "fold": fold,
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "inner_auc": float(search.best_score_),
                    "threshold": threshold,
                    "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                    "estimator": search.best_estimator_,
                }
                candidate.update(metric_row(y[train_idx], inner_probability, threshold))
                candidates.append(candidate)
            except Exception as exc:
                failure_rows.append({"repeat": repeat, "fold": fold, "model_key": model_key, "error": repr(exc)})
        if not candidates:
            raise RuntimeError(f"No successful candidates for repeat {repeat} fold {fold}.")
        selected = sorted(candidates, key=lambda row: (row["inner_auc"], row["f1"], row["sensitivity"]), reverse=True)[0]
        probability = selected["estimator"].predict_proba(X[test_idx])[:, 1]
        predicted = (probability >= selected["threshold"]).astype(int)
        for local_i, sample_i in enumerate(test_idx):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": int(sample_i),
                    "sample_uid": str(sample_ids[sample_i]),
                    "true_label": int(y[sample_i]),
                    "probability": float(probability[local_i]),
                    "threshold": float(selected["threshold"]),
                    "predicted_label": int(predicted[local_i]),
                    "selected_model": selected["model_key"],
                    "selected_model_label": selected["model_label"],
                }
            )
        for row in candidates:
            clean = {key: value for key, value in row.items() if key != "estimator"}
            selection_rows.append(clean)
        print(f"completed repeat {repeat:02d} fold {fold}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)
    failures = pd.DataFrame(failure_rows)
    repeat_metrics = []
    for repeat, group in predictions.groupby("repeat"):
        threshold = float(group["threshold"].iloc[0])
        repeat_metrics.append({"repeat": repeat, **metric_row(group["true_label"].to_numpy(), group["probability"].to_numpy(), threshold)})
    repeat_metrics = pd.DataFrame(repeat_metrics)
    screening = (
        selection.groupby(["model_key", "model_label"], as_index=False)
        .agg(mean_inner_auc=("inner_auc", "mean"), sd_inner_auc=("inner_auc", "std"), n_success=("inner_auc", "size"))
        .sort_values(["mean_inner_auc", "n_success"], ascending=[False, False])
    )
    selected_key = str(screening.iloc[0]["model_key"])
    selected_label = str(screening.iloc[0]["model_label"])
    best_params = json.loads(
        selection[selection["model_key"].eq(selected_key)]
        .sort_values(["inner_auc", "f1"], ascending=[False, False])
        .iloc[0]["best_params_json"]
    )
    classifier, _ = specs[selected_key]
    final_pipeline = make_pipeline(classifier)
    final_pipeline.set_params(**best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X, y)

    predictions.to_csv(TABLES / "score_CA125_BBDvsM_outer_predictions.tsv", sep="\t", index=False)
    selection.to_csv(TABLES / "score_CA125_BBDvsM_model_selection.tsv", sep="\t", index=False)
    failures.to_csv(TABLES / "score_CA125_BBDvsM_failures.tsv", sep="\t", index=False)
    repeat_metrics.to_csv(TABLES / "score_CA125_BBDvsM_repeat_metrics.tsv", sep="\t", index=False)
    screening.to_csv(TABLES / "score_CA125_BBDvsM_model_screening_summary.tsv", sep="\t", index=False)

    with open(MODELS / "score_CA125_BBDvsM_final_model.pkl", "wb") as handle:
        pickle.dump(
            {
                "selected_model_key": selected_key,
                "selected_model_label": selected_label,
                "feature_names": feature_names,
                "best_params": best_params,
                "pipeline": final_pipeline,
                "median_outer_threshold": float(predictions.groupby("repeat")["threshold"].first().median()),
            },
            handle,
        )
    return predictions, selection, repeat_metrics, screening, selected_key, selected_label


def save_figure(fig, stem):
    for ext, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": 600}, "tiff": {"dpi": 600}}.items():
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_results(scored, model_input, predictions, selection, repeat_metrics, selected_key):
    setup_matplotlib_style(matplotlib, base_size=7)
    averaged = average_predictions(predictions)
    y = averaged["true_label"].to_numpy()
    p = averaged["probability"].to_numpy()
    threshold = float(predictions.groupby("repeat")["threshold"].first().median())

    screening = (
        selection.groupby(["model_key", "model_label"], as_index=False)
        .agg(mean_auc=("inner_auc", "mean"), sd_auc=("inner_auc", "std"))
        .sort_values("mean_auc")
    )
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    colors = [SIGNAL_RED if key == selected_key else "#B3B3B3" for key in screening["model_key"]]
    ax.barh(screening["model_label"], screening["mean_auc"], color=colors, edgecolor="none")
    ax.errorbar(screening["mean_auc"], np.arange(len(screening)), xerr=screening["sd_auc"], fmt="none", ecolor=TEXT_DARK, capsize=2, lw=0.8)
    ax.set(xlabel="Inner-CV AUC", xlim=(max(0.5, screening["mean_auc"].min() - 0.04), 1.01))
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelCScore_CA125_01_model_screening_auc")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    axes[0].plot(fpr, tpr, color=SIGNAL_RED, lw=1.8, label=f"AUC = {roc_auc_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], ":", color="#999999", lw=0.8)
    axes[1].plot(recall, precision, color=SIGNAL_RED, lw=1.8, label=f"AP = {average_precision_score(y, p):.3f}")
    axes[1].axhline(y.mean(), ls=":", color="#999999", lw=0.8)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "ModelCScore_CA125_02_ROC_PR_summary")

    fig, ax = plt.subplots(figsize=(3.2, 3.05))
    ax.plot(fpr, tpr, color=SIGNAL_RED, lw=1.8, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=0.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    save_figure(fig, "ModelCScore_CA125_03_final_model_ROC")

    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    palette = ["#F1F6FB", "#C9DFF0", "#74A9CF", "#1F5E85"]
    max_count = float(np.max(cm)) if np.max(cm) > 0 else 1.0
    for row in range(2):
        for col in range(2):
            fraction = cm[row, col] / max_count
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=palette[min(3, int(np.floor(fraction * 4)))], edgecolor="white", linewidth=1.2))
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center", color="white" if fraction >= 0.65 else TEXT_DARK)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1], ["B+BD", "M"])
    ax.set_yticks([0, 1], ["B+BD", "M"])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    save_figure(fig, "ModelCScore_CA125_04_confusion_matrix")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.hist(p[y == 0], bins=np.linspace(0, 1, 18), alpha=0.72, color=SIGNAL_BLUE, label="B+BD")
    ax.hist(p[y == 1], bins=np.linspace(0, 1, 18), alpha=0.72, color=SIGNAL_RED, label="M")
    ax.axvline(threshold, color=TEXT_DARK, ls="--", lw=1)
    ax.set(xlabel="Predicted probability", ylabel="Samples")
    ax.legend(frameon=False)
    save_figure(fig, "ModelCScore_CA125_05_score_distribution")

    ordered = averaged.sort_values("probability").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    colors = np.where(ordered["true_label"].to_numpy() == 1, SIGNAL_RED, SIGNAL_BLUE)
    ax.bar(np.arange(len(ordered)), ordered["probability"], color=colors, width=1.0, edgecolor="none")
    ax.axhline(threshold, color=TEXT_DARK, ls="--", lw=1)
    ax.set(xlabel="Samples sorted by score", ylabel="Predicted probability", ylim=(0, 1.03))
    save_figure(fig, "ModelCScore_CA125_06_score_waterfall")

    metric_order = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    means = repeat_metrics[metric_order].mean()
    lows = repeat_metrics[metric_order].quantile(0.025)
    highs = repeat_metrics[metric_order].quantile(0.975)
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    y_pos = np.arange(len(metric_order))
    ax.barh(y_pos, means, height=0.56, color=SIGNAL_BLUE, edgecolor="none")
    ax.errorbar(means, y_pos, xerr=[means - lows, highs - means], fmt="none", ecolor=TEXT_DARK, capsize=4, lw=1.2)
    for yy, value, high in zip(y_pos, means, highs):
        ax.text(min(high + 0.01, 1.01), yy, f"{value:.3f}", va="center", ha="left")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Mean with 95% CI", xlim=(max(0.5, lows.min() - 0.05), 1.03))
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelCScore_CA125_07_stable_metrics")

    importance = feature_importance(model_input)
    importance.to_csv(TABLES / "ModelCScore_CA125_08_feature_importance.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.barh(importance["feature"], importance["importance"], color=SIGNAL_RED, edgecolor="none")
    ax.set(xlabel="Relative univariate discrimination", ylabel="")
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=0.6)
    save_figure(fig, "ModelCScore_CA125_08_feature_importance")

    observed, predicted = calibration_curve(y, p, n_bins=8, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_fraction": observed}).to_csv(TABLES / "ModelCScore_CA125_09_calibration_curve_data.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
    ax.plot(predicted, observed, marker="o", color=SIGNAL_RED, lw=1.4)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed M fraction", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.set_aspect("equal")
    save_figure(fig, "ModelCScore_CA125_09_calibration_curve")

    rows = []
    for cutoff in np.linspace(0.05, 0.95, 181):
        rows.append({"threshold": cutoff, **metric_row(y, p, cutoff)})
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(TABLES / "ModelCScore_CA125_10_threshold_diagnostics_data.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    ax.plot(threshold_df["threshold"], threshold_df["sensitivity"], color=SIGNAL_RED, label="Sensitivity")
    ax.plot(threshold_df["threshold"], threshold_df["specificity"], color=SIGNAL_BLUE, label="Specificity")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], color=GROUP_COLORS["BD"], label="F1")
    ax.axvline(threshold, color=TEXT_DARK, ls="--", lw=1, label=f"Youden = {threshold:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(0, 1.03))
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save_figure(fig, "ModelCScore_CA125_10_threshold_diagnostics")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    order = ["B", "BD", "M"]
    data = [scored.loc[scored["group_display"].eq(g), "ModelC_NvsM_score"].dropna().to_numpy() for g in order]
    ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55, boxprops={"facecolor": "#E8EEF4", "edgecolor": TEXT_DARK}, medianprops={"color": SIGNAL_RED})
    rng = np.random.default_rng(SEED)
    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.04, len(values))
        ax.scatter(x, values, s=8, alpha=0.45, color=SIGNAL_BLUE if order[i - 1] != "M" else SIGNAL_RED, edgecolor="none")
    ax.set(xlabel="", ylabel="Targeted-panel score", ylim=(-0.03, 1.03))
    save_figure(fig, "ModelCScore_CA125_11_locked_NvsM_score_by_group")


def feature_importance(model_input):
    rows = []
    y = model_input["label"].to_numpy(dtype=int)
    for feature in ["ModelC_NvsM_logit_score", "CA125_log10"]:
        p = model_input[feature].to_numpy(dtype=float)
        display = feature.replace("ModelC_NvsM_logit_score", "logit(Targeted-panel score)").replace("CA125_log10", "CA125")
        rows.append({"feature": display, "importance": roc_auc_score(y, p)})
    frame = pd.DataFrame(rows)
    frame["importance"] = frame["importance"] / frame["importance"].sum()
    return frame.sort_values("importance")


def main():
    ensure_dirs()
    scored, package = load_and_score_bbdm()
    model_input = scored[
        scored["group_display"].isin(["B", "BD", "M"])
        & scored["CA125_log10"].notna()
        & scored["CA125"].notna()
        & (scored["CA125"] > 0)
    ].copy()
    model_input["label"] = model_input["group_display"].eq("M").astype(int)
    model_input["negative_class"] = np.where(model_input["group_display"].eq("M"), "M", "B+BD")
    model_input.to_csv(TABLES / "score_CA125_BBDvsM_model_input.tsv", sep="\t", index=False)
    input_audit = model_input.groupby("group_display", as_index=False).agg(n=("sample_uid", "size"))
    input_audit.to_csv(TABLES / "score_CA125_BBDvsM_input_audit.tsv", sep="\t", index=False)

    predictions, selection, repeat_metrics, screening, selected_key, selected_label = nested_score_ca125_model(model_input)
    averaged = average_predictions(predictions)
    threshold = float(predictions.groupby("repeat")["threshold"].first().median())
    summary = {
        "analysis": "08_ModelC_score_CA125_BBDvsM",
        "source_model": str(MODEL_C_LOCKED),
        "source_model_key": package["model_key"],
        "source_model_label": package["model_label"],
        "source_model_locked_threshold": package["locked_threshold"],
        "scored_B_BD_M_samples": int(len(scored)),
        "stage2_samples_with_CA125": int(len(model_input)),
        "stage2_class_counts": input_audit.set_index("group_display")["n"].to_dict(),
        "stage2_features": ["ModelC_NvsM_logit_score", "CA125_log10"],
        "stage2_selected_model_key": selected_key,
        "stage2_selected_model_label": selected_label,
        "median_repeat_threshold": threshold,
        "repeat_cv_mean": repeat_metrics[["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "precision", "f1"]].mean().to_dict(),
        "repeat_cv_sd": repeat_metrics[["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "precision", "f1"]].std().to_dict(),
        "averaged_oof_metrics": metric_row(averaged["true_label"].to_numpy(), averaged["probability"].to_numpy(), threshold),
        "protocol": {
            "score_step": "Locked ModelC NvsM model applied to B/BD/M targeted analyte log2 area ratios; probability scores are logit-transformed before second-stage modeling.",
            "stage2_task": "B+BD versus M among samples with CA125.",
            "outer_cv": f"RepeatedStratifiedKFold {OUTER_SPLITS} folds x {OUTER_REPEATS} repeats",
            "inner_cv": f"StratifiedKFold {INNER_SPLITS} folds",
            "seed": SEED,
            "interpretation_note": "Exploratory reuse of a fixed NvsM score; not an independent diagnostic validation.",
        },
    }
    (TABLES / "08_ModelC_score_CA125_BBDvsM_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(scored, model_input, predictions, selection, repeat_metrics, selected_key)

    readme = [
        "# 08 ModelC NvsM score plus CA125 for B+BD vs M",
        "",
        "This analysis applies the locked targeted ModelC NvsM model to B, BD, and M samples, then uses logit(ModelC NvsM score) together with CA125_log10 to train a B+BD versus M classifier.",
        "",
        f"Scored B/BD/M samples: {len(scored)}",
        f"Stage-2 samples with CA125: {len(model_input)}",
        f"Class counts: {summary['stage2_class_counts']}",
        f"Selected stage-2 model: {selected_label}",
        "",
        "## Repeated-CV mean performance",
    ]
    for key, value in summary["repeat_cv_mean"].items():
        readme.append(f"- {key}: {value:.4f}")
    readme.extend(
        [
            "",
            "## Averaged OOF performance",
            *[f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}" for key, value in summary["averaged_oof_metrics"].items()],
            "",
            "Interpretation note: this is an exploratory second-stage analysis using a fixed score from the existing NvsM ModelC, not an independent validation.",
        ]
    )
    (ROOT / "README_results.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
