#!/usr/bin/env python
"""Build ModelB from CA125 plus the eight standard-supported metabolite signals.

The script reuses the ModelA candidate-model and nested-CV implementation while
replacing ordinary RFE with an anchored selector that always retains CA125.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path(__file__).resolve().parents[1]
PROJECT = ANALYSIS.parents[1]
MODEL_A_SCRIPT = PROJECT / "05_diagnostic_model_discovery" / "scripts" / "train_ModelA_nested_cv.py"
RAW8 = PROJECT / "05_diagnostic_model_discovery" / "data_clean" / "OV_4.9_raw_highquality8.csv"
CLINICAL = PROJECT / "01_cohort_and_design" / "data_clean" / "clinical_merged_analysis_ready.csv"
TABLES = ANALYSIS / "tables"
MODELS = ANALYSIS / "models"
FIGURES = ANALYSIS / "figures"
REPORTS = ANALYSIS / "reports"
BACKUP = ANALYSIS.with_name("BvsM_CA125_highquality61_incremental_analysis_B_only_backup_20260615")
SEED = 42


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    output = np.empty(len(values), dtype=float)
    output[order] = ranks
    return output


def fast_delong(predictions_sorted_by_label: np.ndarray, positive_count: int):
    classifiers, total_count = predictions_sorted_by_label.shape
    negative_count = total_count - positive_count
    positive_predictions = predictions_sorted_by_label[:, :positive_count]
    negative_predictions = predictions_sorted_by_label[:, positive_count:]
    tx = np.empty((classifiers, positive_count))
    ty = np.empty((classifiers, negative_count))
    tz = np.empty((classifiers, total_count))
    for row in range(classifiers):
        tx[row] = compute_midrank(positive_predictions[row])
        ty[row] = compute_midrank(negative_predictions[row])
        tz[row] = compute_midrank(predictions_sorted_by_label[row])
    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = np.cov(v01) / positive_count + np.cov(v10) / negative_count
    return aucs, covariance


def delong_auc_comparison(y: np.ndarray, augmented: np.ndarray, baseline: np.ndarray) -> dict:
    order = np.argsort(-y)
    positive_count = int(y.sum())
    aucs, covariance = fast_delong(np.vstack([augmented, baseline])[:, order], positive_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    z_score = float((aucs[0] - aucs[1]) / np.sqrt(max(variance, np.finfo(float).eps)))
    return {
        "augmented_auc": float(aucs[0]),
        "ca125_only_auc": float(aucs[1]),
        "delta_auc": float(aucs[0] - aucs[1]),
        "z_score": z_score,
        "two_sided_p": float(2 * norm.sf(abs(z_score))),
    }


def compare_augmented_vs_ca125(augmented_raw: pd.DataFrame, baseline_raw: pd.DataFrame) -> dict:
    augmented = augmented_raw.groupby(["sample_index", "true_label"], as_index=False).agg(
        augmented_probability=("probability", "mean")
    )
    baseline = baseline_raw.groupby(["sample_index", "true_label"], as_index=False).agg(
        ca125_probability=("probability", "mean")
    )
    paired = augmented.merge(baseline, on=["sample_index", "true_label"], validate="one_to_one")
    y = paired["true_label"].to_numpy(dtype=int)
    augmented_probability = paired["augmented_probability"].to_numpy(dtype=float)
    ca125_probability = paired["ca125_probability"].to_numpy(dtype=float)

    rng = np.random.default_rng(SEED)
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    bootstrap_deltas = []
    for _ in range(5000):
        indices = np.concatenate(
            [
                rng.choice(negative_indices, len(negative_indices), replace=True),
                rng.choice(positive_indices, len(positive_indices), replace=True),
            ]
        )
        bootstrap_deltas.append(
            roc_auc_score(y[indices], augmented_probability[indices])
            - roc_auc_score(y[indices], ca125_probability[indices])
        )
    bootstrap_deltas = np.asarray(bootstrap_deltas)
    comparison = {
        "comparison_basis": "paired averaged out-of-fold predictions on identical samples",
        "paired_stratified_bootstrap_iterations": 5000,
        "paired_delta_auc": float(
            roc_auc_score(y, augmented_probability) - roc_auc_score(y, ca125_probability)
        ),
        "paired_delta_auc_bootstrap_ci": [
            float(np.quantile(bootstrap_deltas, 0.025)),
            float(np.quantile(bootstrap_deltas, 0.975)),
        ],
        "paired_bootstrap_two_sided_p": float(
            2
            * min(
                np.mean(bootstrap_deltas <= 0),
                np.mean(bootstrap_deltas >= 0),
            )
        ),
        "delong_auxiliary": delong_auc_comparison(y, augmented_probability, ca125_probability),
    }
    paired.to_csv(TABLES / "ModelB_augmented_vs_CA125_paired_averaged_OOF.tsv", sep="\t", index=False)
    (TABLES / "ModelB_augmented_vs_CA125_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return comparison


def normalize_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


class AnchoredRFE(BaseEstimator, TransformerMixin):
    """Always keep column 0 (CA125) and select a requested number of metabolites."""

    def __init__(self, n_features_to_select: int = 9, step: int = 1, random_state: int = SEED):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        total = int(self.n_features_to_select)
        if total < 2 or total > X.shape[1]:
            raise ValueError(f"Requested {total} predictors for matrix with {X.shape[1]} columns.")
        estimator = LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=3000,
            random_state=self.random_state,
        )
        self.metabolite_selector_ = RFE(
            estimator,
            n_features_to_select=total - 1,
            step=self.step,
        ).fit(X[:, 1:], y)
        self.support_ = np.r_[True, self.metabolite_selector_.support_]
        self.ranking_ = np.r_[1, self.metabolite_selector_.ranking_]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.flatnonzero(self.support_) if indices else self.support_.copy()


def load_modela_module():
    spec = importlib.util.spec_from_file_location("modela_for_modelb", MODEL_A_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    raw = pd.read_csv(RAW8)
    analytes = raw.iloc[:, 0].astype(str).tolist()
    matrix = raw.set_index(raw.columns[0]).T
    matrix.index = matrix.index.map(normalize_id)
    matrix.index.name = "sample_id_norm"
    matrix = matrix.reset_index()

    clinical = pd.read_csv(CLINICAL)
    clinical["sample_id_norm"] = clinical["sample_id"].map(normalize_id)
    clinical["CA125"] = pd.to_numeric(clinical["CA125"], errors="coerce")
    clinical = clinical[
        clinical["malignancy_group"].isin(["B", "BD", "M"]) & clinical["CA125"].gt(0)
    ].copy()
    clinical["CA125_log10"] = np.log10(clinical["CA125"])
    merged = clinical.merge(matrix, on="sample_id_norm", how="inner", validate="one_to_one")
    predictors = ["CA125_log10", *analytes]
    for column in predictors:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["label"] = np.where(merged["malignancy_group"].eq("M"), "M", "B_BD")
    merged["Alignment ID"] = merged["sample_id_norm"]
    merged = merged.reset_index(drop=True)
    expected_counts = {"B": 146, "M": 105, "BD": 22}
    if merged["malignancy_group"].value_counts().to_dict() != expected_counts:
        raise RuntimeError(
            f"Unexpected B/BD/M input counts: {merged['malignancy_group'].value_counts().to_dict()}"
        )
    if merged[predictors].isna().sum().sum():
        raise RuntimeError("ModelB input contains missing CA125 or analyte values.")
    audit_columns = [
        "sample_id",
        "sample_id_norm",
        "malignancy_group",
        "CA125",
        *predictors,
    ]
    merged[audit_columns].to_csv(TABLES / "ModelB_CA125_8analyte_input_audit.tsv", sep="\t", index=False)
    X = merged[predictors].to_numpy(dtype=float)
    y = merged["malignancy_group"].eq("M").astype(int).to_numpy()
    return merged, X, y, predictors


def make_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", AnchoredRFE(n_features_to_select=9)),
            ("classifier", classifier),
        ]
    )


def choose_locked_configuration(selection: pd.DataFrame):
    summary = (
        selection.groupby(["model_key", "model_label", "n_features"], as_index=False)
        .agg(
            mean_auc=("roc_auc", "mean"),
            sd_auc=("roc_auc", "std"),
            ci_low=("roc_auc", lambda x: np.quantile(x, 0.025)),
            ci_high=("roc_auc", lambda x: np.quantile(x, 0.975)),
            selection_count=("roc_auc", "size"),
            mean_inner_auc=("inner_auc", "mean"),
        )
        .sort_values(["mean_auc", "sd_auc"], ascending=[False, True])
    )
    best = summary.iloc[0]
    return str(best["model_key"]), int(best["n_features"]), summary


def save_figure_to_figures(modela, fig, stem: str) -> None:
    stem = stem.replace("ModelA_NvsM_", "ModelB_CA125_")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=modela.FIG_DPI, bbox_inches="tight")
    modela.plt.close(fig)


def evaluate_ca125_only(modela, df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, float, Pipeline]:
    X = df[["CA125_log10"]].to_numpy(dtype=float)
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
            ),
        ]
    )
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    for outer_index, (train, test) in enumerate(outer.split(X, y)):
        repeat, fold = outer_index // 5 + 1, outer_index % 5 + 1
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + outer_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inner_prob = cross_val_predict(clone(estimator), X[train], y[train], cv=inner, method="predict_proba")[:, 1]
            fitted = clone(estimator).fit(X[train], y[train])
        threshold = modela.youden_threshold(y[train], inner_prob)
        prob = fitted.predict_proba(X[test])[:, 1]
        rows.extend(
            {
                "repeat": repeat,
                "fold": fold,
                "sample_index": int(sample),
                "true_label": int(y[sample]),
                "probability": float(prob[i]),
                "threshold_from_inner_oof": threshold,
                "predicted_label": int(prob[i] >= threshold),
            }
            for i, sample in enumerate(test)
        )
    raw = pd.DataFrame(rows)
    metrics = repeat_metrics(raw)
    avg = raw.groupby(["sample_index", "true_label"], as_index=False).agg(
        probability=("probability", "mean"), prediction_count=("probability", "size")
    )
    threshold = modela.youden_threshold(avg["true_label"].to_numpy(), avg["probability"].to_numpy())
    return raw, metrics, threshold, estimator.fit(X, y)


def repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        y = group["true_label"].to_numpy()
        probability = group["probability"].to_numpy()
        pred = group["predicted_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": roc_auc_score(y, probability),
                "pr_auc": average_precision_score(y, probability),
                "brier": brier_score_loss(y, probability),
                "accuracy": accuracy_score(y, pred),
                "sensitivity": recall_score(y, pred),
                "specificity": tn / (tn + fp),
                "f1": f1_score(y, pred),
            }
        )
    return pd.DataFrame(rows)


def organize_outputs() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    for path in TABLES.glob("*.pkl"):
        shutil.move(str(path), MODELS / path.name)
    for name in (
        "ModelA_nested_checkpoint_protocol.json",
        "ModelA_nested_failures_checkpoint.tsv",
        "ModelA_nested_outer_predictions_checkpoint.tsv",
        "ModelA_nested_selection_checkpoint.tsv",
    ):
        path = TABLES / name
        if path.exists():
            path.unlink()
    qa_path = ANALYSIS / "ModelB_CA125_8analyte_nested_cv_QA.md"
    if qa_path.exists():
        shutil.move(str(qa_path), REPORTS / qa_path.name)


def write_old_new_comparison(summary: dict) -> None:
    old_summary_path = BACKUP / "tables" / "ModelB_nested_cv_summary.json"
    if not old_summary_path.exists():
        return
    old = json.loads(old_summary_path.read_text(encoding="utf-8"))
    comparison = {
        "comparison_note": (
            "Descriptive comparison only: B-vs-M and B+BD-vs-M use different study populations "
            "and estimands, so their AUCs are not paired."
        ),
        "B_vs_M_backup": old,
        "B_plus_BD_vs_M_active": summary,
    }
    (REPORTS / "ModelB_B_only_vs_B_plus_BD_descriptive_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    old_aug = old["augmented_metrics_mean"]
    new_aug = summary["augmented_metrics_mean"]
    old_base = old["ca125_only_metrics_mean"]
    new_base = summary["ca125_only_metrics_mean"]
    lines = [
        "# ModelB descriptive comparison: B-vs-M versus B+BD-vs-M",
        "",
        "The two models use different populations and answer different clinical questions; no paired "
        "statistical comparison between their AUCs is appropriate.",
        "",
        "| Analysis | Samples | Selected model | Selected predictors | Augmented AUC | CA125-only AUC | Internal threshold |",
        "|---|---:|---|---|---:|---:|---:|",
        (
            f"| B vs M backup | {old['validation_protocol']['samples']} | {old['selected_model']} | "
            f"{', '.join(old['selected_predictors'])} | {old_aug['roc_auc']:.4f} | "
            f"{old_base['roc_auc']:.4f} | {old['internally_derived_youden_threshold']:.4f} |"
        ),
        (
            f"| B+BD vs M active | {summary['validation_protocol']['samples']} | {summary['selected_model']} | "
            f"{', '.join(summary['selected_predictors'])} | {new_aug['roc_auc']:.4f} | "
            f"{new_base['roc_auc']:.4f} | {summary['internally_derived_youden_threshold']:.4f} |"
        ),
    ]
    (REPORTS / "ModelB_B_only_vs_B_plus_BD_descriptive_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    for directory in (TABLES, MODELS, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    modela = load_modela_module()
    modela.OUT = TABLES
    modela.PROTOCOL_VERSION = "20260615_modelb_ca125_anchored_8analyte_B_plus_BD_v2"
    modela.FEATURE_COUNTS = list(range(2, 10))
    modela.NEG_LABEL, modela.POS_LABEL = "B_BD", "M"
    modela.load_data = load_data
    modela.make_pipeline = make_pipeline
    modela.choose_locked_configuration = choose_locked_configuration
    modela.feature_counts_for_model = lambda model_key: [9] if model_key in {"deep_mlp", "soft_voting", "stacking"} else list(range(2, 10))
    modela.save_figure = lambda fig, stem: save_figure_to_figures(modela, fig, stem)

    df, X, y, predictors = load_data()
    predictions, selection, failures = modela.nested_cv(X, y, predictors)
    predictions.to_csv(TABLES / "ModelB_nested_selection_procedure_predictions.tsv", sep="\t", index=False)
    selection.to_csv(TABLES / "ModelB_nested_outer_selection.tsv", sep="\t", index=False)
    failures.to_csv(TABLES / "ModelB_nested_failures.tsv", sep="\t", index=False)

    model_key, n_predictors, screening = choose_locked_configuration(selection)
    screening.to_csv(TABLES / "ModelB_model_screening_summary.tsv", sep="\t", index=False)
    final_pipeline, best_params = modela.build_locked_configuration(selection, model_key, n_predictors)
    locked_raw, locked_threshold, locked_metrics = modela.locked_oof_predictions(X, y, final_pipeline)
    locked_raw.to_csv(TABLES / "ModelB_augmented_nested_outer_predictions.tsv", sep="\t", index=False)
    locked_metrics.to_csv(TABLES / "ModelB_augmented_repeat_metrics.tsv", sep="\t", index=False)

    ca125_raw, ca125_metrics, ca125_threshold, ca125_model = evaluate_ca125_only(modela, df, y)
    ca125_raw.to_csv(TABLES / "ModelB_CA125_only_nested_outer_predictions.tsv", sep="\t", index=False)
    ca125_metrics.to_csv(TABLES / "ModelB_CA125_only_repeat_metrics.tsv", sep="\t", index=False)
    comparison = compare_augmented_vs_ca125(locked_raw, ca125_raw)

    final_pipeline.fit(X, y)
    support = final_pipeline.named_steps["selector"].support_
    selected_predictors = [name for name, keep in zip(predictors, support) if keep]
    package = {
        "pipeline": final_pipeline,
        "feature_names": predictors,
        "selected_features": selected_predictors,
        "selected_model_key": model_key,
        "selected_model_label": modela.MODEL_LABELS[model_key],
        "selected_n_predictors": n_predictors,
        "internally_derived_youden_threshold": locked_threshold,
        "validation_protocol": {
            "samples": 273,
            "benign": 146,
            "borderline": 22,
            "negative_class": "benign and borderline lesions",
            "negative_class_samples": 168,
            "malignant": 105,
            "outer": "RepeatedStratifiedKFold 5 folds x 20 repeats",
            "inner": "StratifiedKFold 5 folds",
            "seed": SEED,
            "CA125_anchored": True,
            "external_validation": False,
        },
    }
    with open(MODELS / "final_model.pkl", "wb") as handle:
        pickle.dump(package, handle)
    with open(MODELS / "CA125_only_model.pkl", "wb") as handle:
        pickle.dump(
            {
                "pipeline": ca125_model,
                "feature_names": ["CA125_log10"],
                "internally_derived_youden_threshold": ca125_threshold,
            },
            handle,
        )

    summary = {
        "selected_model": package["selected_model_label"],
        "selected_predictors": selected_predictors,
        "internally_derived_youden_threshold": locked_threshold,
        "augmented_metrics_mean": locked_metrics.mean(numeric_only=True).to_dict(),
        "ca125_only_metrics_mean": ca125_metrics.mean(numeric_only=True).to_dict(),
        "delta_auc_mean": float(locked_metrics["roc_auc"].mean() - ca125_metrics["roc_auc"].mean()),
        "paired_averaged_oof_comparison": comparison,
        "best_params": modela.clean_params(best_params),
        "validation_protocol": package["validation_protocol"],
    }
    (TABLES / "ModelB_nested_cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_old_new_comparison(summary)
    qa = [
        "# ModelB CA125-anchored nested-CV QA",
        "",
        "- Input: 273 CA125-available samples (146 benign, 22 borderline, 105 malignant).",
        "- Benign and borderline lesions form the negative class; malignant tumours form the positive class.",
        "- Predictor 0 is CA125_log10 and is always retained.",
        "- Only the eight standard-supported metabolite signals are eligible for RFE.",
        "- Imputation, scaling, RFE, GridSearch and threshold derivation occur inside training folds.",
        f"- Selected model: {package['selected_model_label']}.",
        f"- Selected predictors: {', '.join(selected_predictors)}.",
        f"- Mean augmented ROC AUC: {locked_metrics['roc_auc'].mean():.6f}.",
        f"- Mean CA125-only ROC AUC: {ca125_metrics['roc_auc'].mean():.6f}.",
        f"- Mean paired delta AUC: {summary['delta_auc_mean']:.6f}.",
        "- No external validation; threshold is internally derived.",
        "- The B/BD/M subgroup score review uses averaged outer out-of-fold predictions.",
    ]
    (ANALYSIS / "ModelB_CA125_8analyte_nested_cv_QA.md").write_text("\n".join(qa), encoding="utf-8")
    organize_outputs()
    print("\n".join(qa))


if __name__ == "__main__":
    main()
