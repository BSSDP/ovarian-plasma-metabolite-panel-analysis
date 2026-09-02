#!/usr/bin/env python
"""Generate ModelC 01-10 using the ModelA publication visual language."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
PROJECT = ROOT.parents[2]
STYLE = PROJECT / "00_project_style"
AGE_ADJUSTMENT_ROOT = (
    PROJECT
    / "14_teacher_requested_additions_20260625"
    / "10_five_age_adjustment_methods"
)
if str(STYLE) not in sys.path:
    sys.path.insert(0, str(STYLE))
from ov_publication_style import BACKGROUND_GREY, GROUP_COLORS, SIGNAL_BLUE, SIGNAL_RED, TEXT_DARK, setup_matplotlib_style

TRAIN_SCRIPT = Path(__file__).with_name("train_ModelC_nested_independent.py")
spec = importlib.util.spec_from_file_location("modelc_train", TRAIN_SCRIPT)
TRAIN = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(TRAIN)

DEV_LABEL = "Discovery cohort (outer OOF)"
VAL_LABEL = "Temporal same-centre validation cohort"
DEV_COLOR = "#9E9E9E"
VAL_COLOR = SIGNAL_RED


def save(fig, stem):
    for ext, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": 600}}.items():
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw_opaque_confusion_matrix(ax, cm, xlabels, ylabels):
    """Draw confusion matrix as editable opaque vector rectangles."""
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
                fontsize=9.5,
                color="white" if fraction >= 0.65 else TEXT_DARK,
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


def averaged_dev():
    return pd.read_csv(TABLES / "ModelC_exploratory_averaged_oof_predictions.tsv", sep="\t")


def validation():
    return pd.read_csv(TABLES / "ModelC_independent_validation_predictions.tsv", sep="\t")


def curves(ax_roc, ax_pr, y, probability, label, color, linestyle="-"):
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    ax_roc.plot(fpr, tpr, color=color, ls=linestyle, lw=1.7, label=f"{label}: AUC = {roc_auc_score(y, probability):.3f}")
    ax_pr.plot(recall, precision, color=color, ls=linestyle, lw=1.7, label=f"{label}: AP = {average_precision_score(y, probability):.3f}")


def metric_values(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Sensitivity": recall_score(y, pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, pred),
        "Macro F1": f1_score(y, pred, average="macro"),
    }


def plot_01(summary):
    selected_key = summary.iloc[0]["model_key"]
    display = summary.nlargest(12, "mean_auc")
    if selected_key not in display["model_key"].values:
        display = pd.concat([display.iloc[:-1], summary.loc[summary["model_key"].eq(selected_key)]])
    ordered = display.sort_values("mean_auc")
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    colors = [SIGNAL_RED if k == selected_key else "#B3B3B3" for k in ordered["model_key"]]
    ax.barh(ordered["model_label"], ordered["mean_auc"], color=colors, edgecolor="none")
    ax.errorbar(ordered["mean_auc"], np.arange(len(ordered)), xerr=ordered["sd_auc"], fmt="none", ecolor=TEXT_DARK, capsize=2, lw=.8)
    ax.set(xlabel="Outer nested-CV AUC", xlim=(max(.5, ordered["mean_auc"].min() - .04), 1.01))
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=.6)
    ax.set_axisbelow(True)
    font_scale = 2.3805
    ax.tick_params(axis="both", labelsize=6.5 * font_scale)
    ax.xaxis.label.set_size(7 * font_scale)
    save(fig, "ModelC_Targeted_01_optimization_stage_auc")


def plot_02(dev, val):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    curves(axes[0], axes[1], dev.true_label.to_numpy(), dev.probability.to_numpy(), DEV_LABEL, DEV_COLOR, "--")
    curves(axes[0], axes[1], val.true_label.to_numpy(), val.probability.to_numpy(), VAL_LABEL, VAL_COLOR)
    axes[0].plot([0, 1], [0, 1], ":", color="#999999", lw=.8)
    axes[1].axhline(val.true_label.mean(), ls=":", color="#999999", lw=.8)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout(w_pad=1.4)
    save(fig, "ModelC_Targeted_02_ROC_PR_summary")


def plot_03(val):
    fig, ax = plt.subplots(figsize=(3.2, 3.05))
    fpr, tpr, _ = roc_curve(val.true_label, val.probability)
    ax.plot(fpr, tpr, color=VAL_COLOR, lw=1.8, label=f"AUC = {roc_auc_score(val.true_label, val.probability):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    save(fig, "ModelC_Targeted_03_final_model_ROC")


def plot_04(val, threshold):
    cm = confusion_matrix(val.true_label, val.probability >= threshold)
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    fig.patch.set_alpha(1.0)
    draw_opaque_confusion_matrix(ax, cm, ["Normal", "Malignant"], ["Normal", "Malignant"])
    ax.set(xlabel="Predicted", ylabel="Observed")
    save(fig, "ModelC_Targeted_04_confusion_matrix")


def plot_05(val, threshold):
    fig, ax = plt.subplots(figsize=(4.1, 3.2))
    groups = [val.loc[val.true_label.eq(i), "probability"].to_numpy() for i in [0, 1]]
    vp = ax.violinplot(groups, positions=[0, 1], showextrema=False)
    for body, color in zip(vp["bodies"], [GROUP_COLORS["N"], GROUP_COLORS["M"]]):
        body.set_facecolor(color); body.set_alpha(.35); body.set_edgecolor("none")
    rng = np.random.default_rng(42)
    for i, (values, color) in enumerate(zip(groups, [GROUP_COLORS["N"], GROUP_COLORS["M"]])):
        ax.scatter(i + rng.normal(0, .045, len(values)), values, s=6, alpha=.7, color=color, edgecolor="none")
    ax.axhline(threshold, ls="--", color=TEXT_DARK, lw=.8)
    ax.set_xticks([0, 1], ["Normal", "Malignant"])
    ax.set_ylabel("Predicted probability of malignancy")
    save(fig, "ModelC_Targeted_05_score_distribution")


def plot_06(val, threshold):
    d = val.sort_values("probability").reset_index(drop=True)
    colors = np.where(d.true_label.eq(1), GROUP_COLORS["M"], GROUP_COLORS["N"])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(np.arange(len(d)), d.probability, color=colors, width=1.0, edgecolor="none")
    ax.axhline(threshold, ls="--", color=TEXT_DARK, lw=.8)
    ax.set(xlabel="Independent validation samples ranked by score", ylabel="Predicted probability")
    save(fig, "ModelC_Targeted_06_score_waterfall")


def plot_07(dev_metrics, val_boot, val, threshold):
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "Macro F1"]
    columns = ["roc_auc", "accuracy", "sensitivity", "specificity", "macro_f1"]
    dev_mean = [dev_metrics[c].mean() for c in columns]
    dev_low = [dev_metrics[c].quantile(.025) for c in columns]
    dev_high = [dev_metrics[c].quantile(.975) for c in columns]
    val_point = metric_values(val.true_label.to_numpy(), val.probability.to_numpy(), threshold)
    val_mean = [val_point[x] for x in labels]
    val_low = [val_boot[c].quantile(.025) for c in columns]
    val_high = [val_boot[c].quantile(.975) for c in columns]
    y = np.arange(len(labels)); h = .34
    fig, ax = plt.subplots(figsize=(4.9, 3.55))
    ax.barh(y - h/2, dev_mean, height=h, color=DEV_COLOR, label=DEV_LABEL)
    ax.barh(y + h/2, val_mean, height=h, color=SIGNAL_BLUE, label=VAL_LABEL)
    ax.errorbar(dev_mean, y-h/2, xerr=[np.array(dev_mean)-dev_low, np.array(dev_high)-dev_mean], fmt="none", ecolor=TEXT_DARK, capsize=2.2, lw=.8)
    ax.errorbar(val_mean, y+h/2, xerr=[np.array(val_mean)-val_low, np.array(val_high)-val_mean], fmt="none", ecolor=TEXT_DARK, capsize=2.2, lw=.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set(xlabel="Metric value", xlim=(.55, 1.02))
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    save(fig, "ModelC_Targeted_07_stable_metrics")


def predict_package(package, log2_x, area_x):
    return TRAIN.predict_locked(package, log2_x, area_x)


def plot_08(package, log2_val, area_val):
    def predict_log2(arr):
        frame = np.asarray(arr)
        if package["model_key"] == "three_view_ensemble":
            return predict_package(package, frame, np.power(2.0, frame))
        return predict_package(package, frame, area_val[:len(frame)])
    background = shap.kmeans(log2_val, min(30, len(log2_val)))
    explainer = shap.KernelExplainer(predict_log2, background)
    values = np.asarray(explainer.shap_values(log2_val, nsamples=300, silent=True))
    if values.ndim == 3: values = values[:, :, -1]
    mean_abs = np.mean(np.abs(values), axis=0)
    direction = [np.corrcoef(log2_val[:, i], values[:, i])[0, 1] for i in range(len(TRAIN.ANALYTES))]
    d = pd.DataFrame({"feature": TRAIN.ANALYTES, "mean_abs_shap": mean_abs, "direction": direction}).sort_values("mean_abs_shap")
    d.to_csv(TABLES / "ModelC_Targeted_SHAP_feature_importance.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(5.1, 3.85))
    ax.barh(d.feature, d.mean_abs_shap, color=[SIGNAL_RED if x >= 0 else SIGNAL_BLUE for x in d.direction], edgecolor="none")
    ax.set_xlabel("Mean |SHAP|")
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=.6); ax.set_axisbelow(True)
    save(fig, "ModelC_Targeted_08_feature_importance")


def plot_09(dev, val):
    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=.8)
    for d, label, color, ls in [(dev, DEV_LABEL, DEV_COLOR, "--"), (val, VAL_LABEL, VAL_COLOR, "-")]:
        frac, mean = calibration_curve(d.true_label, d.probability, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=3, lw=1.4, ls=ls, color=color, label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction malignant", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.legend(frameon=False, fontsize=5.5)
    save(fig, "ModelC_Targeted_09_calibration_curve")


def plot_10(dev, threshold):
    thresholds = np.linspace(.05, .95, 181)
    rows = []
    for t in thresholds:
        m = metric_values(dev.true_label.to_numpy(), dev.probability.to_numpy(), t)
        rows.append({"threshold": t, **m})
    d = pd.DataFrame(rows); d.to_csv(TABLES / "ModelC_Targeted_10_threshold_diagnostics_data.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.55))
    for metric, color in [("Sensitivity", SIGNAL_RED), ("Specificity", SIGNAL_BLUE), ("F1", "#7F7F7F")]:
        ax.plot(d.threshold, d[metric], lw=1.5, color=color, label=metric)
    ax.axvline(threshold, ls="--", color=TEXT_DARK, lw=.9, label=f"Locked threshold = {threshold:.3f}")
    ax.axvline(.5, ls=":", color="#999999", lw=.9, label="Reference threshold = 0.5")
    ax.set(xlabel="Probability threshold", ylabel="Metric value", xlim=(.05, .95), ylim=(0, 1.02))
    ax.legend(
        frameon=False,
        fontsize=5.5,
        loc="upper center",
        bbox_to_anchor=(.5, -.20),
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
    )
    fig.subplots_adjust(bottom=.27)
    save(fig, "ModelC_Targeted_10_threshold_diagnostics")


def repeat_metrics_from_predictions(predictions):
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        y = group["true_label"].to_numpy(dtype=int)
        probability = group["probability"].to_numpy(dtype=float)
        predicted = group["predicted_label"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": roc_auc_score(y, probability),
                "accuracy": accuracy_score(y, predicted),
                "sensitivity": recall_score(y, predicted),
                "specificity": tn / (tn + fp),
                "macro_f1": f1_score(y, predicted, average="macro"),
            }
        )
    return pd.DataFrame(rows)


def plot_11_age_residualized_performance():
    """Compare original and control-derived age-residualized outer-OOF metrics."""
    metric_columns = {
        "AUC": "roc_auc",
        "Accuracy": "accuracy",
        "Sensitivity": "sensitivity",
        "Specificity": "specificity",
        "Macro F1": "macro_f1",
    }
    original = pd.read_csv(
        TABLES / "ModelC_locked_exploratory_repeat_metrics.tsv",
        sep="\t",
    )
    adjusted_predictions = pd.read_csv(
        AGE_ADJUSTMENT_ROOT
        / "tables"
        / "predictions"
        / "normal_control_linear_residualization_repeated_outer_predictions.csv"
    )
    adjusted = repeat_metrics_from_predictions(adjusted_predictions)
    adjusted.to_csv(
        TABLES / "ModelC_Targeted_11_age_residualized_repeat_metrics.tsv",
        sep="\t",
        index=False,
    )

    rows = []
    for display, column in metric_columns.items():
        for model, values in [
            ("Original model", original[column]),
            ("Normal-control age-residualized", adjusted[column]),
        ]:
            rows.append(
                {
                    "metric": display,
                    "model": model,
                    "mean": values.mean(),
                    "ci_95_low": values.quantile(0.025),
                    "ci_95_high": values.quantile(0.975),
                    "n_repeats": len(values),
                }
            )
    figure_data = pd.DataFrame(rows)
    figure_data.to_csv(
        TABLES / "ModelC_Targeted_11_age_residualized_performance_figure_data.tsv",
        sep="\t",
        index=False,
    )

    labels = list(metric_columns)
    original_rows = (
        figure_data.loc[figure_data["model"].eq("Original model")]
        .set_index("metric")
        .loc[labels]
    )
    adjusted_rows = (
        figure_data.loc[
            figure_data["model"].eq("Normal-control age-residualized")
        ]
        .set_index("metric")
        .loc[labels]
    )
    y = np.arange(len(labels))
    height = 0.34
    fig, ax = plt.subplots(figsize=(4.9, 3.55))
    for data, offset, color, label in [
        (original_rows, -height / 2, DEV_COLOR, "Original model"),
        (
            adjusted_rows,
            height / 2,
            SIGNAL_BLUE,
            "Normal-control age-residualized",
        ),
    ]:
        means = data["mean"].to_numpy()
        lows = data["ci_95_low"].to_numpy()
        highs = data["ci_95_high"].to_numpy()
        ax.barh(
            y + offset,
            means,
            height=height,
            color=color,
            edgecolor="none",
            label=label,
        )
        ax.errorbar(
            means,
            y + offset,
            xerr=[means - lows, highs - means],
            fmt="none",
            ecolor=TEXT_DARK,
            capsize=2.2,
            lw=0.8,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Metric value", xlim=(0.55, 1.02))
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    save(fig, "ModelC_Targeted_11_age_residualized_performance")


def main():
    setup_matplotlib_style(mpl, base_size=7)
    FIGURES.mkdir(exist_ok=True)
    summary = json.loads((TABLES / "ModelC_nested_independent_summary.json").read_text(encoding="utf-8"))
    screening = pd.read_csv(TABLES / "ModelC_model_screening_summary.tsv", sep="\t")
    dev = averaged_dev(); val = validation()
    dev_metrics = pd.read_csv(TABLES / "ModelC_locked_exploratory_repeat_metrics.tsv", sep="\t")
    val_boot = pd.read_csv(TABLES / "ModelC_independent_validation_bootstrap_metrics.tsv", sep="\t")
    with open(MODELS / "ModelC_locked_model.pkl", "rb") as handle: package = pickle.load(handle)
    meta, log2_frame = TRAIN.load_data()
    val_mask = meta.dataset.eq("independent_validation_cohort").to_numpy()
    plot_01(screening); plot_02(dev, val); plot_03(val); plot_04(val, summary["locked_threshold"])
    plot_05(val, summary["locked_threshold"]); plot_06(val, summary["locked_threshold"])
    plot_07(dev_metrics, val_boot, val, summary["locked_threshold"])
    plot_08(package, log2_frame.loc[val_mask, TRAIN.ANALYTES].to_numpy(), meta.loc[val_mask, TRAIN.ANALYTES].to_numpy())
    plot_09(dev, val); plot_10(dev, summary["locked_threshold"])
    plot_11_age_residualized_performance()
    (ROOT / "ModelC_publication_figure_QA.md").write_text(
        "# ModelC publication figure QA\n\n"
        "- Figures 01-10 were regenerated from the nested-development and frozen-validation outputs.\n"
        "- Figure 02 and Figure 07 compare discovery-cohort outer OOF results with the temporal same-centre validation cohort.\n"
        "- Independent validation data were not used for model or threshold selection.\n"
        "- Figure 11 compares original and normal-control age-residualized repeated outer-OOF performance in the discovery cohort.\n"
        "- 98figure was not updated.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
