#!/usr/bin/env python
"""Refresh single-panel figures for the targeted-panel two-step strategy.

This script only writes into the 08 analysis folder. It does not compose Fig5/S5.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import calibration_curve
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

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
RUN08_SCRIPT = ROOT / "scripts" / "run_08_ModelC_score_CA125_BBDvsM.py"

STYLE = PROJECT / "00_project_style"
if str(STYLE) not in sys.path:
    sys.path.insert(0, str(STYLE))
from ov_publication_style import BACKGROUND_GREY, GROUP_COLORS, SIGNAL_BLUE, SIGNAL_RED, TEXT_DARK, setup_matplotlib_style

SEED = 42
GREY = "#BDBDBD"


def import_run08():
    spec = importlib.util.spec_from_file_location("run08_for_twostep_panels", RUN08_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run08_for_twostep_panels"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save_figure(fig, stem):
    for ext, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": 600}, "tiff": {"dpi": 600}}.items():
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def metric_row(y, probability, threshold):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
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


def load_step1_scores():
    run08 = import_run08()
    package = run08.load_locked_modelc()
    model = package["model"]
    feature_names = package["feature_names"]
    long = pd.read_csv(run08.LONG_SOURCE, sep="\t")
    use = long[
        long["is_primary"].eq(True)
        & long["group_code"].isin(["N", "B", "J", "M"])
        & long["analyte"].isin(feature_names)
    ].copy()
    metadata = use[["sample_uid", "batch_display", "group_code"]].drop_duplicates()
    wide = use.pivot(index="sample_uid", columns="analyte", values="log2_area_ratio").reset_index()
    scored = metadata.merge(wide, on="sample_uid", validate="one_to_one")
    scored["group_display"] = scored["group_code"].replace({"J": "BD"})
    scored = scored.dropna(subset=feature_names).reset_index(drop=True)
    scored["ModelC_NvsM_score"] = model.predict_proba(scored[feature_names].to_numpy(dtype=float))[:, 1]
    scored["ModelC_NvsM_logit_score"] = logit(scored["ModelC_NvsM_score"].clip(1e-5, 1 - 1e-5))
    scored["ModelC_NvsM_locked_threshold"] = float(package["locked_threshold"])
    scored["N_vs_patient_label"] = scored["group_display"].ne("N").astype(int)
    scored.to_csv(TABLES / "ModelC_NvsM_score_all_N_B_BD_M.tsv", sep="\t", index=False)
    audit = scored.groupby(["batch_display", "group_display"], as_index=False).agg(
        n=("sample_uid", "size"),
        median_score=("ModelC_NvsM_score", "median"),
        mean_score=("ModelC_NvsM_score", "mean"),
    )
    audit.to_csv(TABLES / "ModelC_NvsM_score_N_B_BD_M_audit.tsv", sep="\t", index=False)
    return scored[scored["batch_display"].eq("Batch1")].copy(), float(package["locked_threshold"])


def load_step2_outputs():
    model_input = pd.read_csv(TABLES / "score_CA125_BBDvsM_model_input.tsv", sep="\t")
    predictions = pd.read_csv(TABLES / "score_CA125_BBDvsM_outer_predictions.tsv", sep="\t")
    repeat_metrics = pd.read_csv(TABLES / "score_CA125_BBDvsM_repeat_metrics.tsv", sep="\t")
    selection = pd.read_csv(TABLES / "score_CA125_BBDvsM_model_selection.tsv", sep="\t")
    summary = json.loads((TABLES / "08_ModelC_score_CA125_BBDvsM_summary.json").read_text(encoding="utf-8"))
    averaged = average_predictions(predictions)
    averaged = averaged.merge(model_input[["sample_uid", "group_display", "CA125"]], on="sample_uid", how="left")
    return model_input, predictions, repeat_metrics, selection, summary, averaged


def plot_roc_pr(y, p, stem, roc_stem=None):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    axes[0].plot(fpr, tpr, color=SIGNAL_RED, lw=1.8, label=f"AUC = {roc_auc_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], ":", color="#999999", lw=0.8)
    axes[1].plot(recall, precision, color=SIGNAL_RED, lw=1.8, label=f"AP = {average_precision_score(y, p):.3f}")
    axes[1].axhline(np.mean(y), ls=":", color="#999999", lw=0.8)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="lower right")
    save_figure(fig, stem)

    fig, ax = plt.subplots(figsize=(3.2, 3.05))
    ax.plot(fpr, tpr, color=SIGNAL_RED, lw=1.8, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=0.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    save_figure(fig, roc_stem or stem.replace("_ROC_PR_summary", "_final_model_ROC"))


def plot_confusion(y, p, threshold, labels, stem):
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    palette = ["#F1F6FB", "#C9DFF0", "#74A9CF", "#1F5E85"]
    max_count = float(np.max(cm)) if np.max(cm) > 0 else 1.0
    for row in range(2):
        for col in range(2):
            fraction = cm[row, col] / max_count
            ax.add_patch(matplotlib.patches.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=palette[min(3, int(np.floor(fraction * 4)))], edgecolor="white", linewidth=1.2))
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center", color="white" if fraction >= 0.65 else TEXT_DARK)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    save_figure(fig, stem)


def plot_two_group_violin(y, p, threshold, labels, colors, stem, ylabel):
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    data = [p[y == 0], p[y == 1]]
    violin = ax.violinplot(data, positions=[0, 1], widths=0.72, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.32)
    rng = np.random.default_rng(SEED)
    for i, values in enumerate(data):
        ax.scatter(rng.normal(i, 0.04, len(values)), values, s=7, color=colors[i], alpha=0.8, linewidth=0)
        ax.plot([i - 0.18, i + 0.18], [np.median(values)] * 2, color=TEXT_DARK, lw=1.2)
    ax.axhline(threshold, color=TEXT_DARK, ls="--", lw=0.8)
    ax.set_xticks([0, 1], labels)
    ax.set(ylabel=ylabel, ylim=(-0.05, 1.05))
    ax.grid(axis="y", color=BACKGROUND_GREY, lw=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, stem)


def plot_waterfall(y, p, threshold, colors, stem):
    order = np.argsort(p)
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    bar_colors = np.where(y[order] == 1, colors[1], colors[0])
    ax.bar(np.arange(len(p)), p[order], color=bar_colors, width=1.0, edgecolor="none")
    ax.axhline(threshold, color=TEXT_DARK, ls="--", lw=1)
    ax.set(xlabel="Samples sorted by score", ylabel="Predicted probability", ylim=(0, 1.03))
    save_figure(fig, stem)


def plot_metrics_bar(values, cis, labels, stem, xlim=(0.5, 1.03), comparator=None):
    fig, ax = plt.subplots(figsize=(4.4 if comparator is not None else 3.7, 3.0))
    y_pos = np.arange(len(labels))
    if comparator is None:
        means = np.asarray(values, dtype=float)
        lows = np.asarray([ci[0] for ci in cis], dtype=float)
        highs = np.asarray([ci[1] for ci in cis], dtype=float)
        ax.barh(y_pos, means, height=0.56, color=SIGNAL_BLUE, edgecolor="none")
        ax.errorbar(means, y_pos, xerr=[means - lows, highs - means], fmt="none", ecolor=TEXT_DARK, capsize=4, lw=1.2)
        for yy, value, high in zip(y_pos, means, highs):
            ax.text(min(high + 0.01, 1.01), yy, f"{value:.3f}", va="center", ha="left")
    else:
        means = np.asarray(values, dtype=float)
        lows = np.asarray([ci[0] for ci in cis], dtype=float)
        highs = np.asarray([ci[1] for ci in cis], dtype=float)
        comp_values = np.asarray(comparator["values"], dtype=float)
        ax.barh(y_pos - 0.18, means, height=0.32, color=SIGNAL_BLUE, edgecolor="none", label=comparator.get("model_label", "Model"))
        ax.errorbar(means, y_pos - 0.18, xerr=[means - lows, highs - means], fmt="none", ecolor=TEXT_DARK, capsize=3, lw=1.0)
        ax.barh(y_pos + 0.18, comp_values, height=0.32, color=GREY, edgecolor="none", label=comparator["label"])
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Performance metric value", xlim=xlim)
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, stem)


def plot_calibration(y, p, stem, ylabel):
    observed, predicted = calibration_curve(y, p, n_bins=8, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_fraction": observed}).to_csv(TABLES / f"{stem}_data.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
    ax.plot(predicted, observed, marker="o", color=SIGNAL_RED, lw=1.4)
    ax.set(xlabel="Mean predicted probability", ylabel=ylabel, xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.set_aspect("equal")
    save_figure(fig, stem)


def plot_threshold(y, p, selected_threshold, stem):
    rows = []
    for cutoff in np.linspace(0.05, 0.95, 181):
        rows.append({"threshold": cutoff, **metric_row(y, p, cutoff)})
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(TABLES / f"{stem}_data.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    ax.plot(threshold_df["threshold"], threshold_df["sensitivity"], color=SIGNAL_RED, label="Sensitivity")
    ax.plot(threshold_df["threshold"], threshold_df["specificity"], color=SIGNAL_BLUE, label="Specificity")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], color=GROUP_COLORS["BD"], label="F1")
    ax.axvline(selected_threshold, color=TEXT_DARK, ls="--", lw=1, label=f"Threshold = {selected_threshold:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(0, 1.03))
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save_figure(fig, stem)


def plot_step1_panels(step1, threshold):
    y = step1["N_vs_patient_label"].to_numpy(dtype=int)
    p = step1["ModelC_NvsM_score"].to_numpy(dtype=float)
    plot_roc_pr(y, p, "ModelCTwoStep_Step1_NvsPatient_01_ROC_PR_summary", "ModelCTwoStep_Step1_NvsPatient_02_final_model_ROC")
    plot_confusion(y, p, threshold, ["N", "B+BD+M"], "ModelCTwoStep_Step1_NvsPatient_03_confusion_matrix")
    plot_two_group_violin(y, p, threshold, ["N", "B+BD+M"], [GROUP_COLORS["N"], SIGNAL_RED], "ModelCTwoStep_Step1_NvsPatient_04_score_distribution", "Targeted-panel score")
    plot_waterfall(y, p, threshold, [GROUP_COLORS["N"], SIGNAL_RED], "ModelCTwoStep_Step1_NvsPatient_05_score_waterfall")
    metrics = metric_row(y, p, threshold)
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    keys = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    values = [metrics[k] for k in keys]
    cis = [(v, v) for v in values]
    plot_metrics_bar(values, cis, labels, "ModelCTwoStep_Step1_NvsPatient_06_metrics", xlim=(0.65, 1.03))
    plot_calibration(y, p, "ModelCTwoStep_Step1_NvsPatient_07_calibration_curve", "Observed B+BD+M fraction")
    plot_threshold(y, p, threshold, "ModelCTwoStep_Step1_NvsPatient_08_threshold_diagnostics")
    pd.DataFrame([metrics]).assign(threshold=threshold, n=len(step1), n_N=int((y == 0).sum()), n_BBDM=int((y == 1).sum())).to_csv(TABLES / "ModelCTwoStep_Step1_NvsPatient_metrics.tsv", sep="\t", index=False)

    # Overwrite the old group-score panel with the full N/B/BD/M display.
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    order = ["N", "B", "BD", "M"]
    colors = [GROUP_COLORS["N"], GROUP_COLORS["B"], GROUP_COLORS["BD"], GROUP_COLORS["M"]]
    data = [step1.loc[step1["group_display"].eq(g), "ModelC_NvsM_score"].dropna().to_numpy(dtype=float) for g in order]
    violin = ax.violinplot(data, positions=np.arange(len(order)), widths=0.72, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.30)
    rng = np.random.default_rng(SEED)
    for i, values in enumerate(data):
        ax.scatter(rng.normal(i, 0.045, len(values)), values, s=7, alpha=0.65, color=colors[i], linewidth=0)
        ax.plot([i - 0.18, i + 0.18], [np.median(values)] * 2, color=TEXT_DARK, lw=1.1)
    ax.axhline(threshold, ls="--", color=TEXT_DARK, lw=0.8)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set(ylabel="Targeted-panel score", ylim=(-0.04, 1.04))
    ax.grid(axis="y", color=BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelCScore_CA125_11_locked_NvsM_score_by_group")


def plot_step2_panels(model_input, predictions, repeat_metrics, averaged):
    y = averaged["true_label"].to_numpy(dtype=int)
    p = averaged["probability"].to_numpy(dtype=float)
    threshold = float(predictions.groupby("repeat")["threshold"].first().median())
    plot_roc_pr(y, p, "ModelCScore_CA125_02_ROC_PR_summary", "ModelCScore_CA125_03_final_model_ROC")
    plot_confusion(y, p, threshold, ["B+BD", "M"], "ModelCScore_CA125_04_confusion_matrix")
    plot_two_group_violin(y, p, threshold, ["B+BD", "M"], [GROUP_COLORS["B"], SIGNAL_RED], "ModelCScore_CA125_05_score_distribution", "Predicted probability of M")
    plot_waterfall(y, p, threshold, [GROUP_COLORS["B"], SIGNAL_RED], "ModelCScore_CA125_06_score_waterfall")
    plot_step2_feature_importance(model_input)

    metric_order = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    means = repeat_metrics[metric_order].mean()
    lows = repeat_metrics[metric_order].quantile(0.025)
    highs = repeat_metrics[metric_order].quantile(0.975)
    ca_call = (model_input["CA125"].to_numpy(dtype=float) >= 35).astype(float)
    ca_metrics = metric_row(model_input["label"].to_numpy(dtype=int), ca_call, 0.5)
    comparator_values = [ca_metrics[k] for k in metric_order]
    plot_metrics_bar(
        [means[k] for k in metric_order],
        [(lows[k], highs[k]) for k in metric_order],
        labels,
        "ModelCScore_CA125_07_stable_metrics",
        xlim=(0.55, 1.03),
        comparator={"values": comparator_values, "label": "CA125 >=35 U/ml", "model_label": "Targeted-panel score + CA125"},
    )
    pd.DataFrame(
        {
            "metric": metric_order,
            "metric_label": labels,
            "ModelC_score_CA125_repeat_mean": [means[k] for k in metric_order],
            "ModelC_score_CA125_ci_low": [lows[k] for k in metric_order],
            "ModelC_score_CA125_ci_high": [highs[k] for k in metric_order],
            "CA125_35_threshold": comparator_values,
        }
    ).to_csv(TABLES / "ModelCScore_CA125_07_stable_metrics_with_CA12535.tsv", sep="\t", index=False)
    plot_calibration(y, p, "ModelCScore_CA125_09_calibration_curve", "Observed malignant fraction")
    plot_threshold(y, p, threshold, "ModelCScore_CA125_10_threshold_diagnostics")


def plot_step2_feature_importance(model_input):
    y = model_input["label"].to_numpy(dtype=int)
    rows = []
    for column, label in [("ModelC_NvsM_logit_score", "logit(Targeted-panel score)"), ("CA125_log10", "CA125")]:
        rows.append({"feature": label, "importance": roc_auc_score(y, model_input[column].to_numpy(dtype=float))})
    importance = pd.DataFrame(rows)
    importance["importance"] = importance["importance"] / importance["importance"].sum()
    importance = importance.sort_values("importance")
    importance.to_csv(TABLES / "ModelCScore_CA125_08_feature_importance.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.barh(importance["feature"], importance["importance"], color=SIGNAL_RED, edgecolor="none")
    ax.set(xlabel="Relative univariate discrimination", ylabel="")
    ax.grid(axis="x", color=BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelCScore_CA125_08_feature_importance")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_style(matplotlib, base_size=7)
    step1, step1_threshold = load_step1_scores()
    plot_step1_panels(step1, step1_threshold)
    model_input, predictions, repeat_metrics, selection, summary, averaged = load_step2_outputs()
    plot_step2_panels(model_input, predictions, repeat_metrics, averaged)
    print("Updated 08 ModelC two-step single-panel figures.", flush=True)


if __name__ == "__main__":
    main()
