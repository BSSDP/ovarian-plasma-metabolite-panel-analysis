#!/usr/bin/env python
"""Generate the publication ModelB 01-10 figures from nested-CV outputs."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ANALYSIS = Path(__file__).resolve().parents[1]
PROJECT = ANALYSIS.parents[1]
TABLES, MODELS, FIGURES = ANALYSIS / "tables", ANALYSIS / "models", ANALYSIS / "figures"
RUNNER_PATH = Path(__file__).with_name("run_ModelB_CA125_8analyte_nested_cv.py")
MODELA_PATH = PROJECT / "05_diagnostic_model_discovery" / "scripts" / "train_ModelA_nested_cv.py"

PUBLIC_LABELS = {
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
    "decision_tree": "Decision tree",
    "gradient_boosting": "Gradient boosting",
    "adaboost": "AdaBoost",
    "knn": "k-nearest neighbours",
    "naive_bayes": "Gaussian naive Bayes",
    "mlp": "Multilayer perceptron",
    "deep_mlp": "Deep neural network",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "soft_voting": "Weighted probability ensemble",
    "stacking": "Stacked ensemble",
}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_opaque_confusion_matrix(ax, cm, xlabels, ylabels, text_color):
    """Draw confusion matrix as opaque vector rectangles for stable AI import."""
    palette = ["#F1F6FB", "#C9DFF0", "#74A9CF", "#1F5E85"]
    max_count = float(np.max(cm)) if np.max(cm) > 0 else 1.0
    for row in range(2):
        for column in range(2):
            fraction = cm[row, column] / max_count
            color_index = min(3, int(np.floor(fraction * 4)))
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=palette[color_index],
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=1.0,
                )
            )
            ax.text(
                column,
                row,
                str(int(cm[row, column])),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if fraction >= 0.65 else text_color,
            )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1], xlabels)
    ax.set_yticks([0, 1], ylabels)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor("white")


def averaged(raw):
    return raw.groupby(["sample_index", "true_label"], as_index=False).agg(
        probability=("probability", "mean"), prediction_count=("probability", "size")
    )


def plot_screening(modela, screening, selected_key):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=10)
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    colors = plt.cm.tab20(np.linspace(0, 1, screening["model_key"].nunique()))
    for color, (key, group) in zip(colors, screening.groupby("model_key")):
        group = group.sort_values("n_features")
        selected = key == selected_key
        ax.errorbar(
            group["n_features"],
            group["mean_auc"],
            yerr=group["sd_auc"],
            marker="*" if selected else "o",
            ms=6 if selected else 2.6,
            lw=1.7 if selected else 0.75,
            capsize=1.3,
            color=modela.SIGNAL_RED if selected else color,
            alpha=1 if selected else 0.72,
            label=f"{PUBLIC_LABELS.get(key, key)}{' (selected)' if selected else ''}",
            zorder=4 if selected else 1,
        )
    ax.set(
        xlabel="Total number of predictors (CA125 plus selected analytes)",
        ylabel="Outer nested-CV AUC",
        xlim=(0.7, 9.3),
        ylim=(max(0.55, screening["ci_low"].min() - 0.02), 1.0),
    )
    ax.grid(axis="y", color=modela.BACKGROUND_GREY, lw=0.5)
    ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(bottom=0.56, left=0.11, right=0.98, top=0.98)
    save(fig, "ModelB_CA125_01_feature_selection_auc_by_predictor_count")


def plot_roc_pr(modela, augmented, baseline):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9))
    comparison = json.loads((TABLES / "ModelB_augmented_vs_CA125_comparison.json").read_text(encoding="utf-8"))
    delong_p = comparison["delong_auxiliary"]["two_sided_p"]
    for frame, label, color, lw in [
        (baseline, "CA125 only", "#B3B3B3", 1.3),
        (augmented, "CA125 plus selected analytes", modela.SIGNAL_RED, 1.8),
    ]:
        y, p = frame["true_label"].to_numpy(), frame["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        roc_label = f"{label}: AUC = {roc_auc_score(y, p):.3f}"
        if label == "CA125 plus selected analytes":
            roc_label += f"; DeLong P = {delong_p:.3f}"
        axes[0].plot(fpr, tpr, color=color, lw=lw, label=roc_label)
        axes[1].plot(recall, precision, color=color, lw=lw, label=f"{label}: AP = {average_precision_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=0.7)
    axes[1].axhline(augmented["true_label"].mean(), ls="--", color="#999999", lw=0.7)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    for ax in axes:
        ax.legend(frameon=False, fontsize=5.2, loc="lower right")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(w_pad=1.5)
    save(fig, "ModelB_CA125_02_ROC_PR_summary")


def plot_modelb_classification_panels(modela, averaged_predictions, threshold):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    benign = modela.GROUP_COLORS["B"]
    malignant = modela.GROUP_COLORS["M"]
    y = averaged_predictions["true_label"].to_numpy()
    probability = averaged_predictions["probability"].to_numpy()
    predicted = (probability >= threshold).astype(int)

    cm = confusion_matrix(y, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fig.patch.set_alpha(1.0)
    draw_opaque_confusion_matrix(
        ax,
        cm,
        ["Benign/borderline", "Malignant"],
        ["Benign/borderline", "Malignant"],
        modela.TEXT_DARK,
    )
    ax.set(xlabel="Predicted", ylabel="Observed")
    save(fig, "ModelB_CA125_04_confusion_matrix")

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    data = [probability[y == 0], probability[y == 1]]
    violin = ax.violinplot(data, positions=[0, 1], widths=0.72, showextrema=False)
    for body, color in zip(violin["bodies"], [benign, malignant]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.32)
    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(0, 0.04, len(data[0])), data[0], s=7, color=benign, alpha=0.8, linewidth=0)
    ax.scatter(rng.normal(1, 0.04, len(data[1])), data[1], s=7, color=malignant, alpha=0.8, linewidth=0)
    for position, values in enumerate(data):
        ax.plot([position - 0.18, position + 0.18], [np.median(values)] * 2, color=modela.TEXT_DARK, lw=1.2)
    ax.axhline(threshold, ls="--", color=modela.TEXT_DARK, lw=0.8)
    ax.set_xticks([0, 1], ["Benign/borderline", "Malignant"])
    ax.set(ylabel="Predicted probability of malignancy")
    ax.grid(axis="y", color=modela.BACKGROUND_GREY, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "ModelB_CA125_05_score_distribution")

    order = np.argsort(probability)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    colors = np.where(y[order] == 1, malignant, benign)
    ax.bar(np.arange(len(probability)), probability[order], color=colors, width=0.9, linewidth=0)
    ax.axhline(threshold, ls="--", color=modela.TEXT_DARK, lw=0.8)
    ax.set(xlabel="Samples ranked by model score", ylabel="Predicted probability of malignancy")
    ax.grid(axis="y", color=modela.BACKGROUND_GREY, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "ModelB_CA125_06_score_waterfall")


def plot_stable_metrics(modela, augmented_metrics, baseline_metrics):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    names = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    columns = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    means = augmented_metrics[columns].mean()
    lows = augmented_metrics[columns].quantile(0.025)
    highs = augmented_metrics[columns].quantile(0.975)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    ax.barh(y, means, height=0.56, color=modela.SIGNAL_BLUE, edgecolor="none")
    ax.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="none",
        ecolor=modela.TEXT_DARK,
        elinewidth=1.0,
        capsize=3,
    )
    for index, mean in enumerate(means):
        ax.text(highs.iloc[index] + 0.01, index, f"{mean:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y, names)
    lower_limit = max(0.0, float(lows.min()) - 0.08)
    ax.set(xlabel="Mean with 95% CI", xlim=(lower_limit, 1.03))
    ax.invert_yaxis()
    ax.grid(axis="x", color=modela.BACKGROUND_GREY, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "ModelB_CA125_07_stable_metrics")


def plot_selected_feature_importance(modela, selected_features):
    source = TABLES / "ModelA_NvsM_SHAP_feature_importance.tsv"
    importance = pd.read_csv(source, sep="\t")
    importance = importance.loc[importance["feature"].isin(selected_features)].copy()
    importance = importance.sort_values("mean_abs_shap", ascending=True)
    importance["display_feature"] = importance["feature"].replace({"CA125_log10": "CA125"})
    importance.to_csv(TABLES / "ModelB_CA125_SHAP_feature_importance.tsv", sep="\t", index=False)

    colors = np.where(
        importance["direction_by_value"].eq("higher_value_toward_M"),
        modela.SIGNAL_RED,
        modela.SIGNAL_BLUE,
    )
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    fig_height = max(2.6, 0.42 * len(importance) + 1.0)
    fig, ax = plt.subplots(figsize=(4.8, fig_height))
    ax.barh(
        importance["display_feature"],
        importance["mean_abs_shap"],
        color=colors,
        edgecolor="none",
        height=0.66,
    )
    ax.set(xlabel="Mean absolute SHAP value", ylabel="")
    ax.grid(axis="x", color=modela.BACKGROUND_GREY, lw=0.5)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.xaxis.label.set_size(10)
    fig.subplots_adjust(left=0.43, right=0.97, bottom=0.22, top=0.97)
    save(fig, "ModelB_CA125_08_feature_importance")

    if source.exists():
        source.unlink()


def plot_threshold_diagnostics(modela, averaged_predictions, locked_threshold):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    y = averaged_predictions["true_label"].to_numpy()
    probability = averaged_predictions["probability"].to_numpy()
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": recall_score(y, predicted, zero_division=0),
                "specificity": tn / (tn + fp),
                "f1": f1_score(y, predicted, zero_division=0),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(TABLES / "ModelB_CA125_10_threshold_diagnostics_data.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    ax.plot(diagnostics["threshold"], diagnostics["sensitivity"], color=modela.SIGNAL_RED, label="Sensitivity")
    ax.plot(diagnostics["threshold"], diagnostics["specificity"], color=modela.SIGNAL_BLUE, label="Specificity")
    ax.plot(diagnostics["threshold"], diagnostics["f1"], color=modela.GROUP_COLORS["BD"], label="F1")
    ax.axvline(
        locked_threshold,
        color=modela.TEXT_DARK,
        ls="--",
        lw=1,
        label=f"Internal Youden = {locked_threshold:.3f}",
    )
    ax.axvline(0.5, color="#888888", ls=":", lw=1, label="Reference = 0.5")
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(0, 1.03))
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "ModelB_CA125_10_threshold_diagnostics")


def plot_calibration(modela, augmented, baseline):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
    calibration_rows = []
    for frame, label, color in [
        (baseline, "CA125 only", "#B3B3B3"),
        (augmented, "CA125 plus selected analytes", modela.SIGNAL_RED),
    ]:
        observed, predicted = calibration_curve(frame["true_label"], frame["probability"], n_bins=8, strategy="quantile")
        ax.plot(predicted, observed, marker="o", ms=3.5, lw=1.4, color=color, label=label)
        calibration_rows.extend(
            {
                "model": label,
                "mean_predicted_probability": float(mean_predicted),
                "observed_malignant_fraction": float(observed_fraction),
            }
            for mean_predicted, observed_fraction in zip(predicted, observed)
        )
    ax.set(xlabel="Mean predicted probability", ylabel="Observed malignant fraction", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    ax.legend(frameon=False, fontsize=6)
    pd.DataFrame(calibration_rows).to_csv(
        TABLES / "ModelB_CA125_09_calibration_curve_data.tsv", sep="\t", index=False
    )
    save(fig, "ModelB_CA125_09_calibration_curve")


def main():
    runner = import_file("modelb_runner", RUNNER_PATH)
    sys.modules["__main__"].AnchoredRFE = runner.AnchoredRFE
    modela = import_file("modela_plot", MODELA_PATH)
    modela.MODEL_LABELS.update(PUBLIC_LABELS)
    modela.NEG_LABEL, modela.POS_LABEL = "B_BD", "M"
    modela.OUT = TABLES
    modela.save_figure = lambda fig, stem: save(fig, stem.replace("ModelA_NvsM_", "ModelB_CA125_"))

    df, X, y, predictors = runner.load_data()
    with open(MODELS / "final_model.pkl", "rb") as handle:
        package = pickle.load(handle)
    screening = pd.read_csv(TABLES / "ModelB_model_screening_summary.tsv", sep="\t")
    augmented_raw = pd.read_csv(TABLES / "ModelB_augmented_nested_outer_predictions.tsv", sep="\t")
    augmented_metrics = pd.read_csv(TABLES / "ModelB_augmented_repeat_metrics.tsv", sep="\t")
    baseline_raw = pd.read_csv(TABLES / "ModelB_CA125_only_nested_outer_predictions.tsv", sep="\t")
    baseline_metrics = pd.read_csv(TABLES / "ModelB_CA125_only_repeat_metrics.tsv", sep="\t")
    augmented = averaged(augmented_raw)
    baseline = averaged(baseline_raw)

    modela.plot_all(
        X,
        y,
        predictors,
        screening,
        augmented_raw,
        augmented_metrics,
        float(package["internally_derived_youden_threshold"]),
        package["pipeline"],
        package["selected_model_key"],
    )
    plot_screening(modela, screening, package["selected_model_key"])
    plot_roc_pr(modela, augmented, baseline)
    plot_modelb_classification_panels(modela, augmented, float(package["internally_derived_youden_threshold"]))
    plot_stable_metrics(modela, augmented_metrics, baseline_metrics)
    plot_selected_feature_importance(modela, package["selected_features"])
    plot_calibration(modela, augmented, baseline)
    plot_threshold_diagnostics(modela, augmented, float(package["internally_derived_youden_threshold"]))

    for ext in ("pdf", "svg", "png"):
        obsolete = FIGURES / f"ModelB_CA125_01_model_screening_auc_by_feature_count.{ext}"
        if obsolete.exists():
            obsolete.unlink()
    for name in (
        "ModelA_NvsM_09_calibration_curve_data.tsv",
        "ModelA_NvsM_10_threshold_diagnostics_data.tsv",
        "ModelA_NvsM_SHAP_feature_importance.tsv",
    ):
        obsolete = TABLES / name
        if obsolete.exists():
            obsolete.unlink()
    print(json.dumps({"figures": 10, "selected_features": package["selected_features"]}, indent=2))


if __name__ == "__main__":
    main()
