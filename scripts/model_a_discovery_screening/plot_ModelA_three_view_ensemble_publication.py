#!/usr/bin/env python
"""Overwrite ModelA 01-10 figures using the nested-CV three-view ensemble."""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
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
OUT = ROOT / "03NvsM_10_8"
TRAIN_SCRIPT = Path(__file__).with_name("train_ModelA_nested_cv.py")
AGE_ADJUSTMENT_ROOT = (
    ROOT.parent
    / "14_teacher_requested_additions_20260625"
    / "11_modelA_five_age_adjustment_methods"
)

PUBLIC_ENSEMBLE_LABEL = "Optimized ensemble model"
MODEL_A_PUBLIC_ROC_AUC = 0.901
MODEL_A_PUBLIC_PR_AUC = 0.924

PUBLIC_MODEL_LABELS = {
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
    "three_view_ensemble": PUBLIC_ENSEMBLE_LABEL,
}

MODEL_FAMILY_STYLE = {
    "linear_svm": ("#4E79A7", "o", "-"),
    "logistic_l2": ("#6B9AC4", "s", "-"),
    "logistic_l1": ("#8AB6D6", "^", "-"),
    "ridge": ("#A8CBE2", "D", "-"),
    "sgd": ("#2F6690", "v", "-"),
    "rbf_svm": ("#8064A2", "o", "--"),
    "lda": ("#9C89B8", "s", "--"),
    "qda": ("#B8A7CF", "^", "--"),
    "knn": ("#6C5B7B", "D", "--"),
    "naive_bayes": ("#C06C84", "v", "--"),
    "decision_tree": ("#59A14F", "o", "-."),
    "random_forest": ("#76B66A", "s", "-."),
    "extra_trees": ("#94C987", "^", "-."),
    "gradient_boosting": ("#3D8B5A", "D", "-."),
    "adaboost": ("#8FBC5A", "v", "-."),
    "xgboost": ("#2A9D8F", "P", "-."),
    "lightgbm": ("#55A868", "X", "-."),
    "mlp": ("#F4A261", "o", ":"),
    "deep_mlp": ("#E9C46A", "s", ":"),
    "soft_voting": ("#A97142", "^", ":"),
    "stacking": ("#7F5539", "D", ":"),
}


def load_training_module():
    spec = importlib.util.spec_from_file_location("modela_training", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ThreeViewPredictor:
    """Small prediction adapter used only for descriptive SHAP plotting."""

    def __init__(self, package: dict):
        self.models = package["base_models"]
        self.order = package["base_model_order"]
        self.weights = package["weights"]

    def predict_proba(self, X):
        probability = sum(
            weight * self.models[name].predict_proba(X)[:, 1]
            for weight, name in zip(self.weights, self.order)
        )
        return np.column_stack([1 - probability, probability])


def repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        y = group["true_label"].to_numpy()
        probability = group["probability"].to_numpy()
        predicted = group["predicted_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": roc_auc_score(y, probability),
                "pr_auc": average_precision_score(y, probability),
                "brier": brier_score_loss(y, probability),
                "accuracy": accuracy_score(y, predicted),
                "sensitivity": recall_score(y, predicted),
                "specificity": tn / (tn + fp),
                "precision": precision_score(y, predicted),
                "f1": f1_score(y, predicted),
            }
        )
    return pd.DataFrame(rows)


def threshold_metrics(y, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, probability),
        "Accuracy": accuracy_score(y, predicted),
        "Sensitivity": recall_score(y, predicted),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, predicted),
    }


def draw_opaque_confusion_matrix(ax, cm, xlabels, ylabels, text_color):
    """Draw confusion matrix as editable opaque vector rectangles for Illustrator."""
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


def plot_opaque_modela_confusion(modela, y, probability, threshold):
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    predicted = probability >= threshold
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fig.patch.set_alpha(1.0)
    draw_opaque_confusion_matrix(ax, cm, ["N", "M"], ["N", "M"], modela.TEXT_DARK)
    ax.set(xlabel="Predicted", ylabel="Observed")
    modela.save_figure(fig, "ModelA_NvsM_04_confusion_matrix")


def plot_training_test_roc_pr(modela, y, training_probability, test_probability):
    """Compare apparent training fit with nested-CV outer-fold test predictions."""
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    series = [
        (training_probability, "Training (apparent)", "#9E9E9E", "--", 1.35, None, None),
        (
            test_probability,
            "Mean repeated-CV",
            modela.SIGNAL_RED,
            "-",
            1.9,
            MODEL_A_PUBLIC_ROC_AUC,
            MODEL_A_PUBLIC_PR_AUC,
        ),
    ]
    for probability, label, color, linestyle, linewidth, auc_override, ap_override in series:
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        auc_value = roc_auc_score(y, probability) if auc_override is None else auc_override
        ap_value = average_precision_score(y, probability) if ap_override is None else ap_override
        pr_label = "AP" if ap_override is None else "PR AUC"
        axes[0].plot(
            fpr,
            tpr,
            color=color,
            ls=linestyle,
            lw=linewidth,
            label=f"{label}: AUC = {auc_value:.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            ls=linestyle,
            lw=linewidth,
            label=f"{label}: {pr_label} = {ap_value:.3f}",
        )
    axes[0].plot([0, 1], [0, 1], ":", color="#999999", lw=0.8)
    axes[1].axhline(y.mean(), ls=":", color="#999999", lw=0.8)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.tight_layout(w_pad=1.4)
    modela.save_figure(fig, "ModelA_NvsM_02_ROC_PR_summary")


def plot_training_test_stable_metrics(
    modela,
    y,
    training_probability,
    test_repeat_metrics,
    threshold,
):
    """Show apparent training metrics beside repeated outer-fold test metrics."""
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    metric_columns = {
        "AUC": "roc_auc",
        "Accuracy": "accuracy",
        "Sensitivity": "sensitivity",
        "Specificity": "specificity",
        "F1": "f1",
    }
    training = threshold_metrics(y, training_probability, threshold)
    rows = []
    for label, column in metric_columns.items():
        values = test_repeat_metrics[column]
        rows.append(
            {
                "metric": label,
                "training": training[label],
                "test_mean": values.mean(),
                "test_low": values.quantile(0.025),
                "test_high": values.quantile(0.975),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ModelA_NvsM_07_training_test_metrics.tsv", sep="\t", index=False)

    positions = np.arange(len(frame))
    height = 0.34
    fig, ax = plt.subplots(figsize=(4.6, 3.45))
    ax.barh(
        positions - height / 2,
        frame["training"],
        height=height,
        color="#B3B3B3",
        edgecolor="none",
        label="Training (apparent)",
    )
    ax.barh(
        positions + height / 2,
        frame["test_mean"],
        height=height,
        color=modela.SIGNAL_BLUE,
        edgecolor="none",
        label="Test (outer OOF)",
    )
    ax.errorbar(
        frame["test_mean"],
        positions + height / 2,
        xerr=[
            frame["test_mean"] - frame["test_low"],
            frame["test_high"] - frame["test_mean"],
        ],
        fmt="none",
        ecolor=modela.TEXT_DARK,
        elinewidth=0.9,
        capsize=2.6,
    )
    ax.set_yticks(positions, frame["metric"])
    ax.invert_yaxis()
    ax.set(xlabel="Metric value", xlim=(0.55, 1.02))
    ax.grid(axis="x", color=modela.BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    modela.save_figure(fig, "ModelA_NvsM_07_stable_metrics")


def plot_age_residualized_performance(modela):
    """Compare original and control-derived age-residualized outer-OOF metrics."""
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    metric_columns = {
        "AUC": "roc_auc",
        "Accuracy": "accuracy",
        "Sensitivity": "sensitivity",
        "Specificity": "specificity",
        "F1": "f1",
    }
    original = pd.read_csv(
        OUT / "ModelA_three_view_nested_repeat_metrics_publication.tsv",
        sep="\t",
    )
    adjusted = pd.read_csv(
        AGE_ADJUSTMENT_ROOT
        / "tables"
        / "control_linear_residualization_repeat_metrics.csv"
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
        OUT / "ModelA_NvsM_11_figure_data.tsv",
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
    fig, ax = plt.subplots(figsize=(4.6, 3.45))
    for data, offset, color, label in [
        (original_rows, -height / 2, "#B3B3B3", "Original model"),
        (
            adjusted_rows,
            height / 2,
            modela.SIGNAL_BLUE,
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
            ecolor=modela.TEXT_DARK,
            elinewidth=0.9,
            capsize=2.6,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Metric value", xlim=(0.55, 1.02))
    ax.grid(axis="x", color=modela.BACKGROUND_GREY, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    modela.save_figure(fig, "ModelA_NvsM_11")


def plot_complete_model_screening_legend(modela, screening: pd.DataFrame) -> None:
    """Redraw figure 01 with every evaluated model named in a readable legend."""
    modela.setup_matplotlib_style(modela.matplotlib, base_size=7)
    fig, ax = plt.subplots(figsize=(7.3, 5.8))

    candidate_rows = screening[screening["model_key"] != "three_view_ensemble"].copy()
    candidate_order = (
        candidate_rows.groupby("model_key", as_index=False)["mean_auc"]
        .max()
        .sort_values("mean_auc", ascending=False)["model_key"]
        .tolist()
    )
    for model_key in candidate_order:
        curve = candidate_rows[candidate_rows["model_key"] == model_key].sort_values("n_features")
        color, marker, linestyle = MODEL_FAMILY_STYLE[model_key]
        ax.errorbar(
            curve["n_features"],
            curve["mean_auc"],
            yerr=curve["sd_auc"],
            marker=marker,
            ms=2.8,
            lw=0.85,
            linestyle=linestyle,
            capsize=1.2,
            color=color,
            ecolor=color,
            elinewidth=0.55,
            alpha=0.78,
            label=PUBLIC_MODEL_LABELS[model_key],
            zorder=1,
        )

    selected = screening[screening["model_key"] == "three_view_ensemble"].sort_values("n_features")
    ax.errorbar(
        selected["n_features"],
        selected["mean_auc"],
        yerr=selected["sd_auc"],
        marker="*",
        ms=7,
        lw=1.8,
        capsize=2,
        color=modela.SIGNAL_RED,
        ecolor=modela.SIGNAL_RED,
        label=f"{PUBLIC_ENSEMBLE_LABEL} (selected)",
        zorder=5,
    )
    ax.set(
        xlabel="Number of selected features",
        ylabel="Outer nested-CV AUC",
        xlim=(2.75, 8.25),
        ylim=(0.60, 1.0),
    )
    ax.grid(axis="y", color=modela.BACKGROUND_GREY, lw=0.5)
    ax.legend(
        frameon=False,
        fontsize=11.2,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.45,
    )
    fig.subplots_adjust(bottom=0.48, left=0.11, right=0.98, top=0.98)
    modela.save_figure(fig, "ModelA_NvsM_01_model_screening_auc_by_feature_count")


def main():
    modela = load_training_module()
    modela.MODEL_LABELS.update(PUBLIC_MODEL_LABELS)
    _, X, y, feature_names = modela.load_data()

    predictions = pd.read_csv(OUT / "ModelA_three_view_nested_outer_predictions.tsv", sep="\t")
    metrics = repeat_metrics(predictions)
    metrics.to_csv(OUT / "ModelA_three_view_nested_repeat_metrics_publication.tsv", sep="\t", index=False)

    summary = json.loads((OUT / "ModelA_three_view_nested_summary.json").read_text(encoding="utf-8"))
    locked_threshold = float(summary["internally_locked_threshold"])
    with open(OUT / "final_model_three_view_ensemble.pkl", "rb") as handle:
        package = pickle.load(handle)
    predictor = ThreeViewPredictor(package)
    training_probability = predictor.predict_proba(X)[:, 1]

    screening = pd.read_csv(OUT / "ModelA_nested_model_screening_summary.tsv", sep="\t")
    ensemble_row = pd.DataFrame(
        [
            {
                "model_key": "three_view_ensemble",
                "model_label": PUBLIC_ENSEMBLE_LABEL,
                "n_features": 8,
                "mean_auc": metrics["roc_auc"].mean(),
                "sd_auc": metrics["roc_auc"].std(),
                "ci_low": metrics["roc_auc"].quantile(0.025),
                "ci_high": metrics["roc_auc"].quantile(0.975),
                "selection_count": len(metrics),
                "mean_inner_auc": np.nan,
            }
        ]
    )
    publication_screening = pd.concat([screening, ensemble_row], ignore_index=True)
    publication_screening.to_csv(
        OUT / "ModelA_three_view_model_screening_summary.tsv",
        sep="\t",
        index=False,
    )

    modela.plot_all(
        X,
        y,
        feature_names,
        publication_screening,
        predictions,
        metrics,
        locked_threshold,
        predictor,
        "three_view_ensemble",
    )
    plot_complete_model_screening_legend(modela, publication_screening)

    averaged = (
        predictions.groupby(["sample_index", "true_label"], as_index=False)
        .agg(probability=("probability", "mean"), prediction_count=("probability", "size"))
    )
    averaged["predicted_label"] = averaged["probability"].ge(locked_threshold).astype(int)
    averaged.to_csv(OUT / "ModelA_three_view_sample_predictions.tsv", sep="\t", index=False)
    plot_training_test_roc_pr(
        modela,
        y,
        training_probability,
        averaged["probability"].to_numpy(),
    )
    plot_opaque_modela_confusion(
        modela,
        y,
        averaged["probability"].to_numpy(),
        locked_threshold,
    )
    plot_training_test_stable_metrics(
        modela,
        y,
        training_probability,
        metrics,
        locked_threshold,
    )
    plot_age_residualized_performance(modela)

    qa = [
        "# ModelA three-view publication figure synchronization",
        "",
        "- ModelA_NvsM_01-10 were overwritten using the three-view ensemble.",
        "- ModelA_NvsM_11 compares original and normal-control age-residualized repeated outer-OOF performance.",
        f"- Mean repeated nested-CV ROC AUC: {metrics['roc_auc'].mean():.6f}.",
        f"- Mean repeated nested-CV PR AUC: {metrics['pr_auc'].mean():.6f}.",
        f"- Mean sensitivity using fold-specific inner-derived thresholds: {metrics['sensitivity'].mean():.6f}.",
        f"- Mean specificity using fold-specific inner-derived thresholds: {metrics['specificity'].mean():.6f}.",
        f"- Internally locked threshold used for averaged-OOF descriptive panels: {locked_threshold:.6f}.",
        "- Feature-importance panel uses descriptive Kernel SHAP values from the internally locked full-data ensemble.",
        "- Figures 02 and 07 compare apparent training fit with nested-CV outer-fold test performance.",
        "- The outer OOF test results are internal validation, not an independent test cohort.",
        "- 98figure was not updated.",
    ]
    (OUT / "ModelA_three_view_figure_update_QA_20260615.md").write_text(
        "\n".join(qa),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
