#!/usr/bin/env python
"""Rebuild the two-step clinical application analysis for Fig. 5 and Fig. S5.

The finalised analysis-05 targeted-panel score and fixed threshold are reused unchanged.
Step 1 is a paired, same-participant comparison of the panel score and CA125.
Step 2 treats borderline and malignant lesions as the oncology-oriented triage
endpoint and compares fixed-threshold CA125 with an L2 model that combines
CA125 with six targeted analytes. Fold-specific preprocessing and
model fitting generate averaged development OOF predictions and a fixed
100-model prediction ensemble. Its operating threshold is selected in Batch1
and carried unchanged to Batch2. Free-text preoperative diagnosis and age are
not used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import binomtest, norm
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Mandatory publication/export settings: keep SVG text editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7
plt.rcParams["xtick.color"] = "#000000"
plt.rcParams["ytick.color"] = "#000000"
plt.rcParams["axes.labelcolor"] = "#000000"
plt.rcParams["axes.edgecolor"] = "#000000"
plt.rcParams["savefig.facecolor"] = "white"


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT11 = PROJECT_ROOT / "11_trarget" / "analysis_results_requested" / "11_two_step_clinical_application_Fig5_S5"
REQUESTED = ROOT11.parent
TARGET_ROOT = REQUESTED.parent
PROJECT = TARGET_ROOT.parent
TABLES = ROOT11 / "tables"
SOURCE = TABLES / "source_data"
FIGURES = ROOT11 / "figures"
MODELS = ROOT11 / "models"
LOGS = ROOT11 / "logs"

MODEL_ROOT = REQUESTED / "05_targeted_model_performance"
MODEL_TRAIN_SCRIPT = MODEL_ROOT / "scripts" / "train_ModelC_nested_independent.py"
MODEL_FILE = MODEL_ROOT / "models" / "ModelC_locked_model.pkl"
MODEL_VALIDATION = MODEL_ROOT / "tables" / "ModelC_independent_validation_predictions.tsv"
LONG_SOURCE = REQUESTED / "04_targeted_measurement_clinical_association" / "tables" / "integrated_targeted_area_ratio_long.tsv"
CLINICAL_BATCH1 = TARGET_ROOT / "clinical_merged_resolved.xlsx"
CLINICAL_BATCH2 = TARGET_ROOT / "第二批靶向测量临床meta_干净版.xlsx"

SEED = 42
N_BOOT = 2000
PANEL_THRESHOLD = 0.4878243040342936
CA125_THRESHOLD = 35.0
STEP2_ANALYTES = [
    "3-GPA",
    "Acetylcarnitine",
    "Creatine",
    "Arginine",
    "Carnitine",
    "Tryptophan",
]
STEP2_C = 0.01
STEP2_N_SPLITS = 5
STEP2_N_REPEATS = 20

COLORS = {
    "ca125": "#B9B9B9",
    "panel": "#F17C6B",
    "combined": "#83AEC9",
    "clinical": "#FDB462",
    "development": "#7FC8BC",
    "validation": "#B2ADD8",
    "N": "#80B1D3",
    "B": "#8DD3C7",
    "BD": "#FDB462",
    "M": "#FB8072",
    "ink": "#2B2B2B",
    "muted": "#929292",
    "grid": "#E7E7E7",
    "pale_blue": "#EDF4FA",
    "pale_coral": "#FBEDEA",
}

GROUP_ORDER = ["N", "B", "BD", "M"]
COHORT_ORDER = ["Development", "Independent validation"]
METRICS = ["roc_auc", "average_precision", "sensitivity", "specificity", "f1"]


def ensure_dirs() -> None:
    for folder in [TABLES, SOURCE, FIGURES, MODELS, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)
    for folder in [TABLES, SOURCE, FIGURES, MODELS, LOGS]:
        for path in folder.iterdir():
            if path.is_file() and (path.name.startswith("11_") or path.name.startswith("Fig5") or path.name.startswith("FigS5") or path.name.startswith("SourceData_")):
                path.unlink()


def import_model_training_module():
    spec = importlib.util.spec_from_file_location("modelc_train_for_11", MODEL_TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelc_train_for_11"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_locked_panel():
    module = import_model_training_module()
    sys.modules["__main__"].ThreeViewModel = getattr(module, "ThreeViewModel", object)
    with open(MODEL_FILE, "rb") as handle:
        package = pickle.load(handle)
    classifier = package["model"].named_steps.get("classifier")
    for estimator in getattr(classifier, "estimators_", []):
        if isinstance(estimator, LogisticRegression) and not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"
    return package


def normalise_stage(value, batch: str) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().upper()
    if batch == "Batch1":
        if text in {"1", "I"}:
            return "I"
        if text in {"2", "II"}:
            return "II"
        if text in {"3", "III"}:
            return "III"
        if text in {"4", "IV"}:
            return "IV"
        return "Recurrent"
    if text in {"I", "II", "III", "IV"}:
        return text
    return "Missing"


def load_scored_samples():
    package = load_locked_panel()
    feature_names = list(package["feature_names"])
    long = pd.read_csv(LONG_SOURCE, sep="\t", low_memory=False)
    use = long[
        long["is_primary"].eq(True)
        & long["analyte"].isin(feature_names)
        & long["group_code"].isin(["N", "B", "J", "M"])
    ].copy()
    metadata = use[["sample_uid", "batch_display", "group_code", "sample_id_norm"]].drop_duplicates()
    wide = use.pivot(index="sample_uid", columns="analyte", values="log2_area_ratio").reset_index()
    scored = metadata.merge(wide, on="sample_uid", validate="one_to_one")
    scored["group"] = scored["group_code"].replace({"J": "BD"})
    scored["panel_complete"] = scored[feature_names].notna().all(axis=1)
    scored = scored.loc[scored["panel_complete"]].copy()
    scored["panel_score"] = package["model"].predict_proba(scored[feature_names].to_numpy(float))[:, 1]
    scored["panel_call"] = scored["panel_score"].ge(PANEL_THRESHOLD)

    archived = pd.read_csv(MODEL_VALIDATION, sep="\t")
    regenerated = scored[
        scored["batch_display"].eq("Batch2") & scored["group"].isin(["N", "M"])
    ][["sample_uid", "panel_score"]]
    check = archived[["sample_uid", "probability"]].merge(regenerated, on="sample_uid", validate="one_to_one")
    max_difference = float(np.max(np.abs(check["probability"] - check["panel_score"])))
    if len(check) != 209 or max_difference > 1e-12:
        raise RuntimeError(f"Frozen-score reproduction failed: n={len(check)}, max difference={max_difference}")

    batch1 = pd.read_excel(CLINICAL_BATCH1)
    clinical1 = pd.DataFrame(
        {
            "sample_uid": "Batch1_" + batch1.iloc[:, 0].astype(int).astype(str),
            "batch_display": "Batch1",
            "group": batch1.iloc[:, 9].astype(str).replace({"J": "BD"}),
            "CA125": pd.to_numeric(batch1.iloc[:, 17], errors="coerce"),
            "age": pd.to_numeric(batch1["年龄"], errors="coerce"),
            "FIGO_stage": [normalise_stage(v, "Batch1") for v in batch1.iloc[:, 10]],
        }
    )
    batch2 = pd.read_excel(CLINICAL_BATCH2, sheet_name="clinical_meta")
    clinical2 = pd.DataFrame(
        {
            "sample_uid": "Batch2_" + batch2["sample_id"].astype(str),
            "batch_display": "Batch2",
            "group": batch2["group"].astype(str).replace({"J": "BD"}),
            "CA125": pd.to_numeric(batch2["ca125_u_ml"], errors="coerce"),
            "age": pd.to_numeric(batch2["age_years"], errors="coerce"),
            "FIGO_stage": [normalise_stage(v, "Batch2") for v in batch2["figo_stage"]],
        }
    )
    clinical = pd.concat([clinical1, clinical2], ignore_index=True)
    clinical["CA125_log10"] = np.log10(clinical["CA125"].where(clinical["CA125"] > 0))
    scored = scored.merge(
        clinical[["sample_uid", "CA125", "CA125_log10", "age", "FIGO_stage"]],
        on="sample_uid",
        how="left",
        validate="one_to_one",
    )
    scored["cohort"] = scored["batch_display"].map(
        {"Batch1": "Development", "Batch2": "Independent validation"}
    )
    return scored, clinical, feature_names, max_difference


def youden_threshold(y, probability) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds)
    score = tpr[valid] - fpr[valid]
    best = np.flatnonzero(np.isclose(score, score.max()))
    return float(np.min(thresholds[valid][best]))


def strict_increment_threshold(y, probability, reference_call) -> float:
    """Select a development-OOF threshold that adds at least one TP and TN.

    Eligible thresholds must correctly classify at least one additional positive
    and one additional negative relative to the fixed CA125 clinical threshold.
    Among eligible thresholds, accuracy, F1, specificity and then the higher
    threshold are used as deterministic tie-breakers.
    """
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    reference_call = np.asarray(reference_call, dtype=bool)
    ref_tn, _, _, ref_tp = confusion_matrix(y, reference_call, labels=[0, 1]).ravel()
    values = np.sort(np.unique(probability))
    candidates = np.r_[
        values[0] - 1e-12,
        (values[:-1] + values[1:]) / 2,
        values[-1] + 1e-12,
    ]
    eligible = []
    for threshold in candidates:
        row = metric_row(y, probability, threshold)
        if row["tp"] < ref_tp + 1 or row["tn"] < ref_tn + 1:
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError(
            "No development-OOF threshold improved both sensitivity and specificity "
            "by at least one correctly classified participant versus CA125 >=35 U/mL."
        )
    selected = pd.DataFrame(eligible).sort_values(
        ["accuracy", "f1", "specificity", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(selected["threshold"])


def metric_row(y, probability, threshold) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    is_probability = np.all((probability >= 0) & (probability <= 1))
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)) if is_probability else np.nan,
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity": float(recall_score(y, predicted)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    z = norm.ppf(1 - alpha / 2)
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def fixed_metric_from_counts(row: pd.Series, metric: str) -> tuple[float, float, float]:
    """Return a fixed-threshold metric and Wilson 95% CI from confusion counts."""
    tn, fp, fn, tp = (int(row[key]) for key in ["tn", "fp", "fn", "tp"])
    specifications = {
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "accuracy": (tp + tn, tp + tn + fp + fn),
        "npv": (tn, tn + fn),
        "ppv": (tp, tp + fp),
    }
    successes, total = specifications[metric]
    low, high = wilson_interval(successes, total)
    return successes / total, low, high


def derive_stage_sensitivity_and_rescue(step2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive paired fixed-threshold FIGO sensitivity and CA125-negative rescue summaries."""
    malignant = step2[step2["group"].eq("M")].copy()
    malignant["stage_group"] = np.select(
        [malignant["FIGO_stage"].isin(["I", "II"]), malignant["FIGO_stage"].isin(["III", "IV"])],
        ["FIGO I–II", "FIGO III–IV"],
        default="Other",
    )
    stage_rows = []
    rescue_rows = []
    for cohort in COHORT_ORDER:
        cohort_block = malignant[malignant["cohort"].eq(cohort)]
        for stage_group in ["FIGO I–II", "FIGO III–IV"]:
            block = cohort_block[cohort_block["stage_group"].eq(stage_group)]
            for model, call_col in [("CA125", "CA125_call"), ("Targeted-panel score", "panel_call")]:
                detected = int(block[call_col].astype(bool).sum())
                n = int(len(block))
                low, high = wilson_interval(detected, n)
                stage_rows.append(
                    {
                        "cohort": cohort,
                        "stage_group": stage_group,
                        "model": model,
                        "n": n,
                        "detected": detected,
                        "sensitivity": detected / n if n else np.nan,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        for stage_group in ["All malignant", "FIGO I–II", "FIGO III–IV"]:
            block = cohort_block if stage_group == "All malignant" else cohort_block[cohort_block["stage_group"].eq(stage_group)]
            negative = block[~block["CA125_call"].astype(bool)]
            detected = int(negative["panel_call"].astype(bool).sum())
            n = int(len(negative))
            low, high = wilson_interval(detected, n)
            rescue_rows.append(
                {
                    "cohort": cohort,
                    "stage_group": stage_group,
                    "CA125_negative_n": n,
                    "panel_positive_n": detected,
                    "panel_negative_n": n - detected,
                    "panel_detection_rate": detected / n if n else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(stage_rows), pd.DataFrame(rescue_rows)


def stratified_bootstrap(y, probability, threshold, seed) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(N_BOOT):
        index = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        row = metric_row(y[index], probability[index], threshold)
        records.append({metric: row[metric] for metric in METRICS})
    frame = pd.DataFrame(records)
    return {
        metric: (float(frame[metric].quantile(0.025)), float(frame[metric].quantile(0.975)))
        for metric in METRICS
    }


def add_result(records, analysis, cohort, model, y, values, threshold, seed):
    row = metric_row(y, values, threshold)
    cis = stratified_bootstrap(y, values, threshold, seed)
    row.update({"analysis": analysis, "cohort": cohort, "model": model})
    for metric, (low, high) in cis.items():
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    records.append(row)
    return row


def compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midrank = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank[i:j] = 0.5 * (i + j - 1)
        i = j
    result = np.empty(n, dtype=float)
    result[order] = midrank + 1
    return result


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = compute_midrank(positive[r])
        ty[r] = compute_midrank(negative[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, covariance


def delong_test(y, first, second) -> dict:
    y = np.asarray(y, dtype=int)
    order = np.argsort(-y)
    predictions = np.vstack([np.asarray(first, float), np.asarray(second, float)])[:, order]
    aucs, covariance = fast_delong(predictions, int(y.sum()))
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0:
        z_value, p_value = np.nan, np.nan
    else:
        z_value = float((aucs[0] - aucs[1]) / np.sqrt(variance))
        p_value = float(2 * norm.sf(abs(z_value)))
    return {
        "auc_first": float(aucs[0]),
        "auc_second": float(aucs[1]),
        "delta_auc": float(aucs[0] - aucs[1]),
        "z": z_value,
        "p_value": p_value,
    }


def exact_paired_change(reference_correct, new_correct) -> dict:
    reference_correct = np.asarray(reference_correct, bool)
    new_correct = np.asarray(new_correct, bool)
    ref_only = int(np.sum(reference_correct & ~new_correct))
    new_only = int(np.sum(~reference_correct & new_correct))
    discordant = ref_only + new_only
    p_value = float(binomtest(min(ref_only, new_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "reference_correct_only": ref_only,
        "new_correct_only": new_only,
        "discordant_total": discordant,
        "p_value": p_value,
    }


def calibration_rows(y, probability, cohort, model, n_bins=8):
    observed, predicted = calibration_curve(y, probability, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame(
        {
            "cohort": cohort,
            "model": model,
            "mean_predicted_probability": predicted,
            "observed_fraction": observed,
        }
    )


def step2_feature_matrix(frame):
    return np.column_stack(
        [
            frame[STEP2_ANALYTES].to_numpy(float),
            np.log10(frame["CA125"]),
        ]
    )


def fit_step2_model(training):
    features = step2_feature_matrix(training)
    y = training["label"].to_numpy(int)
    splitter = RepeatedStratifiedKFold(
        n_splits=STEP2_N_SPLITS,
        n_repeats=STEP2_N_REPEATS,
        random_state=SEED,
    )
    total = np.zeros(len(training), dtype=float)
    count = np.zeros(len(training), dtype=int)
    repeat_rows = []
    fold_models = []
    for split_index, (train_index, test_index) in enumerate(splitter.split(features, y)):
        repeat = split_index // STEP2_N_SPLITS + 1
        fold = split_index % STEP2_N_SPLITS + 1
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=STEP2_C, class_weight="balanced", max_iter=5000, random_state=SEED)),
            ]
        )
        model.fit(features[train_index], y[train_index])
        fold_models.append(model)
        probability = model.predict_proba(features[test_index])[:, 1]
        total[test_index] += probability
        count[test_index] += 1
        for idx, value in zip(test_index, probability):
            repeat_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_uid": training.iloc[idx]["sample_uid"],
                    "label": int(y[idx]),
                    "probability": float(value),
                }
            )
    oof_probability = total / count
    threshold = strict_increment_threshold(
        y,
        oof_probability,
        training["CA125"].ge(CA125_THRESHOLD),
    )
    final_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=STEP2_C, class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]
    )
    final_model.fit(features, y)
    repeated = pd.DataFrame(repeat_rows)
    repeated["model"] = "Combined model"
    return final_model, fold_models, oof_probability, threshold, repeated


def cohort_name(batch_display: str) -> str:
    return "Development" if batch_display == "Batch1" else "Independent validation"


def build_analysis(scored, clinical):
    sample_flow_rows = []
    for batch in ["Batch1", "Batch2"]:
        c = clinical[clinical["batch_display"].eq(batch)]
        s = scored[scored["batch_display"].eq(batch)]
        for group in GROUP_ORDER:
            cg = c[c["group"].eq(group)]
            sg = s[s["group"].eq(group)]
            sample_flow_rows.append(
                {
                    "cohort": cohort_name(batch),
                    "group": group,
                    "clinical_total": int(len(cg)),
                    "panel_available": int(len(sg)),
                    "CA125_available": int(cg["CA125"].notna().sum()),
                    "panel_and_CA125": int(sg["CA125"].notna().sum()),
                }
            )
    sample_flow = pd.DataFrame(sample_flow_rows)

    step1 = scored[scored["CA125"].notna()].copy()
    step1["label"] = step1["group"].ne("N").astype(int)
    step1["CA125_call"] = step1["CA125"].ge(CA125_THRESHOLD)
    step1["panel_call"] = step1["panel_score"].ge(PANEL_THRESHOLD)
    step1 = step1[
        [
            "sample_uid",
            "cohort",
            "group",
            "label",
            "CA125",
            "panel_score",
            "CA125_call",
            "panel_call",
            "FIGO_stage",
        ]
        + STEP2_ANALYTES
    ].copy()

    step1_performance_records = []
    step1_delong_records = []
    step1_change_records = []
    seed = 100
    for cohort in COHORT_ORDER:
        block = step1[step1["cohort"].eq(cohort)]
        y = block["label"].to_numpy(int)
        add_result(step1_performance_records, "Step 1 N vs B+BD+M", cohort, "CA125", y, block["CA125"], CA125_THRESHOLD, seed)
        seed += 1
        add_result(step1_performance_records, "Step 1 N vs B+BD+M", cohort, "Targeted-panel score", y, block["panel_score"], PANEL_THRESHOLD, seed)
        seed += 1
        d = delong_test(y, block["panel_score"], block["CA125"])
        d.update({"cohort": cohort, "first_model": "Targeted-panel score", "second_model": "CA125", "n": len(block)})
        step1_delong_records.append(d)
        for label_value, endpoint in [(0, "Specificity among N"), (1, "Sensitivity among lesions")]:
            sub = block[block["label"].eq(label_value)]
            ca_correct = sub["CA125_call"].eq(bool(label_value))
            panel_correct = sub["panel_call"].eq(bool(label_value))
            change = exact_paired_change(ca_correct, panel_correct)
            change.update(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "n": len(sub),
                    "CA125_correct": int(ca_correct.sum()),
                    "panel_correct": int(panel_correct.sum()),
                    "CA125_rate": float(ca_correct.mean()),
                    "panel_rate": float(panel_correct.mean()),
                    "reference_model": "CA125",
                    "new_model": "Targeted-panel score",
                }
            )
            step1_change_records.append(change)

    step1_performance = pd.DataFrame(step1_performance_records)
    step1_delong = pd.DataFrame(step1_delong_records)
    step1_changes = pd.DataFrame(step1_change_records)

    lesions = step1[step1["group"].isin(["B", "BD", "M"])].copy()
    lesions["label"] = lesions["group"].isin(["BD", "M"]).astype(int)
    training = lesions[lesions["cohort"].eq("Development")].reset_index(drop=True)
    validation = lesions[lesions["cohort"].eq("Independent validation")].reset_index(drop=True)
    combined_model, combined_fold_models, combined_oof, combined_threshold, combined_repeated = fit_step2_model(training)
    training["combined_probability"] = combined_oof
    validation_features = step2_feature_matrix(validation)
    validation["combined_probability"] = np.mean(
        [model.predict_proba(validation_features)[:, 1] for model in combined_fold_models],
        axis=0,
    )
    step2 = pd.concat([training, validation], ignore_index=True)
    step2["combined_call"] = step2["combined_probability"].ge(combined_threshold)

    step2_performance_records = []
    step2_delong_records = []
    step2_reclassification_rows = []
    model_specs = [
        ("CA125", "CA125", CA125_THRESHOLD),
        ("Combined model", "combined_probability", combined_threshold),
    ]
    comparison_specs = [
        ("Combined model", "combined_probability", "CA125", "CA125"),
    ]
    seed = 300
    for cohort in COHORT_ORDER + ["Pooled audit"]:
        block = step2 if cohort == "Pooled audit" else step2[step2["cohort"].eq(cohort)]
        y = block["label"].to_numpy(int)
        for model_name, column, threshold in model_specs:
            add_result(step2_performance_records, "Step 2 B vs BD+M", cohort, model_name, y, block[column], threshold, seed)
            seed += 1
        for first_name, first_col, second_name, second_col in comparison_specs:
            d = delong_test(y, block[first_col], block[second_col])
            d.update({"cohort": cohort, "first_model": first_name, "second_model": second_name, "n": len(block)})
            step2_delong_records.append(d)
        for reference_model, reference_call, new_model, new_call in [
            ("CA125", "CA125_call", "Combined model", "combined_call"),
        ]:
            for label_value, endpoint in [(0, "Correct B / specificity"), (1, "Detected BD+M / sensitivity")]:
                sub = block[block["label"].eq(label_value)]
                reference_correct = sub[reference_call].eq(bool(label_value))
                new_correct = sub[new_call].eq(bool(label_value))
                change = exact_paired_change(reference_correct, new_correct)
                change.update(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "n": len(sub),
                        "reference_model": reference_model,
                        "new_model": new_model,
                        "reference_correct": int(reference_correct.sum()),
                        "new_correct": int(new_correct.sum()),
                        "reference_rate": float(reference_correct.mean()),
                        "new_rate": float(new_correct.mean()),
                    }
                )
                step2_reclassification_rows.append(change)

    step2_performance = pd.DataFrame(step2_performance_records)
    step2_delong = pd.DataFrame(step2_delong_records)
    step2_reclassification = pd.DataFrame(step2_reclassification_rows)
    stage_sensitivity, ca125_negative_rescue = derive_stage_sensitivity_and_rescue(step2)

    calibration_frames = []
    for cohort in COHORT_ORDER:
        s1 = step1[step1["cohort"].eq(cohort)]
        calibration_frames.append(calibration_rows(s1["label"], s1["panel_score"], cohort, "Step 1 targeted-panel score"))
        s2 = step2[step2["cohort"].eq(cohort)]
        calibration_frames.append(calibration_rows(s2["label"], s2["combined_probability"], cohort, "Step 2 combined model"))
    calibration = pd.concat(calibration_frames, ignore_index=True)

    stage_rows = []
    for cohort in COHORT_ORDER:
        all_m = scored[(scored["cohort"].eq(cohort)) & scored["group"].eq("M")]
        ca_m = all_m[all_m["CA125"].notna()]
        for subset_name, block in [("All panel-scored M", all_m), ("CA125-available / Step 2 M", ca_m)]:
            counts = block["FIGO_stage"].value_counts()
            for stage in ["I", "II", "III", "IV", "Recurrent", "Missing"]:
                stage_rows.append({"cohort": cohort, "subset": subset_name, "stage": stage, "n": int(counts.get(stage, 0))})
    stage_composition = pd.DataFrame(stage_rows)

    coefficients = pd.DataFrame(
        {
            "model": "Combined model",
            "feature": STEP2_ANALYTES + ["log10(CA125)"],
            "standardized_coefficient": combined_model.named_steps["classifier"].coef_[0],
        }
    )
    repeated_predictions = combined_repeated

    return {
        "sample_flow": sample_flow,
        "step1": step1,
        "step1_performance": step1_performance,
        "step1_delong": step1_delong,
        "step1_changes": step1_changes,
        "step2": step2,
        "step2_performance": step2_performance,
        "step2_delong": step2_delong,
        "step2_reclassification": step2_reclassification,
        "stage_sensitivity": stage_sensitivity,
        "ca125_negative_rescue": ca125_negative_rescue,
        "calibration": calibration,
        "stage_composition": stage_composition,
        "coefficients": coefficients,
        "combined_model": combined_model,
        "combined_fold_models": combined_fold_models,
        "combined_threshold": combined_threshold,
        "repeated_predictions": repeated_predictions,
    }


def export_tables(results, feature_names, max_difference):
    file_map = {
        "sample_flow": "11_sample_flow.tsv",
        "step1": "11_step1_paired_predictions.tsv",
        "step1_performance": "11_step1_performance_by_cohort.tsv",
        "step1_delong": "11_step1_paired_DeLong.tsv",
        "step1_changes": "11_step1_fixed_threshold_paired_changes.tsv",
        "step2": "11_step2_paired_predictions.tsv",
        "step2_performance": "11_step2_performance_by_cohort.tsv",
        "step2_delong": "11_step2_paired_DeLong.tsv",
        "step2_reclassification": "11_step2_fixed_threshold_reclassification.tsv",
        "stage_sensitivity": "11_fixed_threshold_sensitivity_by_stage.tsv",
        "ca125_negative_rescue": "11_CA125_negative_panel_detection.tsv",
        "calibration": "11_calibration_source.tsv",
        "stage_composition": "11_malignant_stage_composition.tsv",
        "coefficients": "11_combined_model_coefficients.tsv",
        "repeated_predictions": "11_combined_model_repeated_outer_predictions.tsv",
    }
    for key, name in file_map.items():
        results[key].to_csv(TABLES / name, sep="\t", index=False)

    results["step1"][["sample_uid", "cohort", "group", "panel_score"]].to_csv(
        SOURCE / "SourceData_Fig5b_score_spectrum.tsv", sep="\t", index=False
    )
    results["step1"][["sample_uid", "cohort", "group", "label", "CA125", "panel_score"]].to_csv(
        SOURCE / "SourceData_Fig5c_step1_paired_ROC.tsv", sep="\t", index=False
    )
    results["step1_changes"].to_csv(SOURCE / "SourceData_Fig5de_step1_operating_points.tsv", sep="\t", index=False)
    results["step2_reclassification"].to_csv(SOURCE / "SourceData_Fig5fg_step2_reclassification.tsv", sep="\t", index=False)
    results["step2_performance"].to_csv(SOURCE / "SourceData_Fig5fg_step2_operating_points.tsv", sep="\t", index=False)
    results["sample_flow"].to_csv(SOURCE / "SourceData_FigS5a_sample_availability.tsv", sep="\t", index=False)
    results["calibration"].to_csv(SOURCE / "SourceData_FigS5_calibration.tsv", sep="\t", index=False)
    results["stage_sensitivity"].to_csv(SOURCE / "SourceData_FigS5b_stage_sensitivity.tsv", sep="\t", index=False)
    results["ca125_negative_rescue"].to_csv(SOURCE / "SourceData_FigS5e_CA125_negative_panel_detection.tsv", sep="\t", index=False)

    with open(MODELS / "11_combined_model.pkl", "wb") as handle:
        pickle.dump(
            {
                "prediction_ensemble": results["combined_fold_models"],
                "reference_full_development_fit": results["combined_model"],
                "features": STEP2_ANALYTES + ["log10(CA125)"],
                "threshold": results["combined_threshold"],
                "threshold_rule": "Development OOF only: at least one additional TP and TN versus CA125 >=35 U/mL, then maximise accuracy",
                "training_cohort": "Batch1 B versus BD+M with paired six-analyte measurements and CA125",
                "validation_cohort": "Batch2 B versus BD+M with paired six-analyte measurements and CA125",
                "prediction_rule": f"Mean probability from {STEP2_N_SPLITS * STEP2_N_REPEATS} repeated-fold L2 models",
                "age_used": False,
                "preoperative_diagnosis_used": False,
            },
            handle,
        )

    summary = {
        "analysis": "Two-step clinical application; Step 2 compares CA125 with a six-analyte-plus-CA125 L2 model for B versus BD+M",
        "feature_names": feature_names,
        "step2_analytes": STEP2_ANALYTES,
        "locked_panel_threshold": PANEL_THRESHOLD,
        "CA125_threshold_U_per_mL": CA125_THRESHOLD,
        "combined_threshold_from_development_OOF": results["combined_threshold"],
        "step2_threshold_selection": "Development OOF only: at least one additional TP and TN versus CA125 >=35 U/mL, then maximise accuracy",
        "step2_prediction_rule": f"Mean probability from {STEP2_N_SPLITS * STEP2_N_REPEATS} repeated-fold L2 models",
        "frozen_score_reproduction_max_abs_difference": max_difference,
        "step1_n": results["step1"].groupby("cohort").size().to_dict(),
        "step2_n": results["step2"].groupby("cohort").size().to_dict(),
        "step1_delong": results["step1_delong"].to_dict(orient="records"),
        "step2_delong": results["step2_delong"].to_dict(orient="records"),
        "interpretation_boundary": "Step 2 uses an oncology-oriented triage endpoint in which borderline and malignant lesions form the positive class; age and free-text preoperative diagnosis are excluded.",
    }
    (TABLES / "11_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def add_panel_label(ax, label, x=-0.12, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def format_p(p):
    if p < 0.001:
        return f"P={p:.1e}"
    return f"P={p:.3f}"


def save_figure(fig, stem, dpi=600):
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": dpi},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)


def draw_schematic(ax):
    ax.set_axis_off()
    boxes = [
        (0.01, 0.58, 0.22, 0.27, "Plasma\nsample", "#F7F7F7", COLORS["ink"]),
        (0.38, 0.58, 0.24, 0.27, "Targeted-panel\nscore", "#E7F1F8", COLORS["combined"]),
        (0.76, 0.58, 0.22, 0.27, "Step 1\nN vs B+BD+M", "#FBEDEA", COLORS["M"]),
        (0.76, 0.12, 0.22, 0.27, "CA125\nmeasurement", "#FFF4D9", "#DB9B21"),
        (0.38, 0.12, 0.24, 0.27, "Step 2\nsix analytes + CA125", "#E7F1F8", COLORS["combined"]),
        (0.01, 0.12, 0.22, 0.27, "B vs BD+M\ntriage", "#FBEDEA", COLORS["M"]),
    ]
    for x, y, w, h, text, face, edge in boxes:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=face, edgecolor=edge, linewidth=0.9
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.7, color=COLORS["ink"])
    arrow = dict(arrowstyle="-|>", color=COLORS["ink"], lw=1.0, shrinkA=1, shrinkB=1)
    ax.annotate("", xy=(0.38, 0.715), xytext=(0.23, 0.715), arrowprops=arrow)
    ax.annotate("", xy=(0.76, 0.715), xytext=(0.62, 0.715), arrowprops=arrow)
    ax.annotate("", xy=(0.87, 0.39), xytext=(0.87, 0.58), arrowprops=arrow)
    ax.annotate("", xy=(0.62, 0.255), xytext=(0.76, 0.255), arrowprops=arrow)
    ax.annotate("", xy=(0.23, 0.255), xytext=(0.38, 0.255), arrowprops=arrow)


def draw_score_spectrum(ax, scored):
    rng = np.random.default_rng(SEED)
    for ci, cohort in enumerate(COHORT_ORDER):
        for gi, group in enumerate(GROUP_ORDER):
            values = scored[(scored["group"].eq(group)) & scored["cohort"].eq(cohort)]["panel_score"].to_numpy()
            pos = gi + ci * 4.7
            bp = ax.boxplot(
                [values], positions=[pos], widths=0.48, patch_artist=True, showfliers=False,
                medianprops=dict(color=COLORS["ink"], lw=1.0),
                whiskerprops=dict(color=COLORS[group], lw=0.8),
                capprops=dict(color=COLORS[group], lw=0.8),
                boxprops=dict(facecolor=COLORS[group], edgecolor=COLORS[group], alpha=0.28, lw=0.8),
            )
            del bp
            jitter = rng.normal(pos, 0.08, len(values))
            # Keep individual samples crisp at journal scale and editable in
            # vector exports. A fine white edge separates overlapping points.
            ax.scatter(
                jitter,
                values,
                s=6.0,
                color=COLORS[group],
                alpha=0.68,
                edgecolor="white",
                linewidth=0.18,
                rasterized=False,
                zorder=3,
            )
    ax.axhline(PANEL_THRESHOLD, ls="--", lw=0.8, color=COLORS["muted"])
    ax.axvline(4.15, color="#E5E5E5", lw=0.7)
    ax.set_xticks(list(range(4)) + [4.7 + i for i in range(4)], GROUP_ORDER * 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.65, 8.35)
    ax.set_ylabel("Score", labelpad=1)
    ax.set_title("Score across the lesion spectrum", fontsize=7.2, fontweight="bold", pad=16)
    ax.text(1.5, 1.015, "Discovery cohort", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.7)
    ax.text(6.2, 1.015, "Temporal validation", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.7)


def draw_step1_roc(ax, step1, step1_delong, cohort, short_title):
    block = step1[step1["cohort"].eq(cohort)]
    y = block["label"].to_numpy(int)
    fpr, tpr, _ = roc_curve(y, block["panel_score"])
    auc = roc_auc_score(y, block["panel_score"])
    ax.plot(fpr, tpr, color=COLORS["panel"], lw=1.45, label=f"Panel score: AUC = {auc:.3f}")
    for model, column, color, threshold, marker in [
        ("CA125", "CA125", COLORS["ca125"], CA125_THRESHOLD, "s"),
        ("Panel", "panel_score", COLORS["panel"], PANEL_THRESHOLD, "o"),
    ]:
        call = block[column].to_numpy(float) >= threshold
        tn, fp, fn, tp = confusion_matrix(y, call, labels=[0, 1]).ravel()
        ax.scatter(
            fp / (tn + fp), tp / (tp + fn), s=23, marker=marker, color=color,
            edgecolor="black", linewidth=0.45, zorder=5,
            label=(f"CA125 ≥ {threshold:.0f} U/mL" if model == "CA125" else f"Panel ≥ {threshold:.3f}"),
        )
    ax.plot([0, 1], [0, 1], ":", color="#A8A8A8", lw=0.8)
    ax.set(xlim=(-0.03, 1), ylim=(0, 1.02), xlabel="1 − specificity", ylabel="Sensitivity")
    ax.set_title(f"{short_title}, n = {len(block)}", fontsize=7.2, fontweight="bold")
    ax.legend(loc="lower right", fontsize=5.15)


def draw_metric_bars(ax, performance, cohort, first_model, second_model, first_color, second_color, title, delong=None, show_legend=True):
    step1 = second_model == "Targeted-panel score"
    metrics = (
        [("Sensitivity", "sensitivity"), ("NPV", "npv"), ("Accuracy", "accuracy"), ("Specificity", "specificity")]
        if step1
        else [("Specificity", "specificity"), ("PPV", "ppv"), ("Accuracy", "accuracy"), ("Sensitivity", "sensitivity")]
    )
    y = np.arange(len(metrics))[::-1]
    height = 0.25
    model_specs = [(first_model, first_color, 0.14), (second_model, second_color, -0.14)]
    for model, color, offset in model_specs:
        row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(model)].iloc[0]
        vals, low, high = [], [], []
        for _, metric in metrics:
            value, lo, hi = fixed_metric_from_counts(row, metric)
            vals.append(value)
            low.append(lo)
            high.append(hi)
        vals = np.asarray(vals, float)
        low = np.asarray(low, float)
        high = np.asarray(high, float)
        bars = ax.barh(
            y + offset, vals, height=height, color=color, edgecolor="white", linewidth=0.45,
            label=model.replace("Targeted-panel score", "Panel"), zorder=2,
        )
        ax.errorbar(
            vals, y + offset, xerr=np.maximum(0.0, np.vstack([vals - low, high - vals])),
            fmt="none", ecolor="#616161", elinewidth=0.75, capsize=1.8, zorder=3,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                max(0.055, value - 0.018), bar.get_y() + bar.get_height() / 2,
                f"{100 * value:.0f}%", ha="right", va="center", fontsize=5.0,
                color=COLORS["ink"], zorder=4,
            )
    ax.set_yticks(y, [m[0] for m in metrics])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fixed-threshold rate")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    first_row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(first_model)].iloc[0]
    second_row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(second_model)].iloc[0]
    if step1:
        note = f"Missed lesions: {int(first_row['fn'])} → {int(second_row['fn'])}"
    else:
        note = f"Detected BD+M: {int(first_row['tp'])} → {int(second_row['tp'])}"
    ax.set_title(f"{title}\n{note}", fontsize=6.8, fontweight="bold", linespacing=1.25)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=5.1)


def draw_step2_metric_bars(ax, performance, cohort, title, show_legend=True):
    metrics = [("Sensitivity", "sensitivity"), ("Specificity", "specificity"), ("Accuracy", "accuracy"), ("PPV", "ppv")]
    y = np.arange(len(metrics))[::-1]
    height = 0.28
    model_specs = [
        ("CA125", COLORS["ca125"], 0.16, "CA125 ≥ 35 U/mL"),
        ("Combined model", COLORS["panel"], -0.16, "CA125 + 6-analyte panel"),
    ]
    for model, color, offset, label in model_specs:
        row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(model)].iloc[0]
        vals, low, high = [], [], []
        for _, metric in metrics:
            value, lo, hi = fixed_metric_from_counts(row, metric)
            vals.append(value); low.append(lo); high.append(hi)
        vals = np.asarray(vals, float); low = np.asarray(low, float); high = np.asarray(high, float)
        bars = ax.barh(y + offset, vals, height=height, color=color, edgecolor="white", linewidth=0.45, label=label, zorder=2)
        ax.errorbar(vals, y + offset, xerr=np.maximum(0.0, np.vstack([vals - low, high - vals])), fmt="none", ecolor="#616161", elinewidth=0.7, capsize=1.6, zorder=3)
        for bar, value in zip(bars, vals):
            ax.text(max(0.055, value - 0.016), bar.get_y() + bar.get_height() / 2, f"{100 * value:.0f}%", ha="right", va="center", fontsize=4.7, color=COLORS["ink"], zorder=4)
    ax.set_yticks(y, [m[0] for m in metrics])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fixed-threshold rate")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    counts = []
    for model, _, _, _ in model_specs:
        row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(model)].iloc[0]
        counts.append(int(row["tp"]))
    ax.set_title(f"{title}\nDetected BD+M: {counts[0]} → {counts[1]}", fontsize=6.8, fontweight="bold", linespacing=1.25)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=5.1, columnspacing=0.9, handlelength=1.2)


def draw_step1_metrics_grouped(ax, performance, delong):
    """Show the paired Step-1 comparison separately in discovery and temporal validation.

    This is a figure-only restyling of frozen analysis-11 tables.  Each cohort
    gets the same three operating metrics so that the visual conclusion matches
    the paired ROC panels rather than implying a pooled result.
    """
    metrics = [("AUC", "roc_auc"), ("Sensitivity", "sensitivity"), ("Specificity", "specificity")]
    cohorts = [("Development", "Discovery cohort"), ("Independent validation", "Temporal validation")]
    rows = []
    for cohort, display in cohorts:
        for label, metric in metrics:
            rows.append((cohort, display, label, metric))
    y = np.arange(len(rows))[::-1]
    height = 0.22
    for model, color, offset, label in [
        ("CA125", COLORS["ca125"], 0.12, "CA125"),
        ("Targeted-panel score", COLORS["panel"], -0.12, "Panel"),
    ]:
        values, lowers, uppers = [], [], []
        for cohort, _, _, metric in rows:
            row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(model)].iloc[0]
            values.append(float(row[metric]))
            lowers.append(float(row[f"{metric}_ci_low"]))
            uppers.append(float(row[f"{metric}_ci_high"]))
        values = np.asarray(values)
        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)
        ax.barh(y + offset, values, height=height, color=color, edgecolor="none", label=label)
        ax.errorbar(values, y + offset, xerr=np.vstack([values - lowers, uppers - values]), fmt="none", ecolor="#666666", elinewidth=0.60, capsize=1.6)
    ax.axhline(2.5, color="#D7D7D7", lw=0.7)
    ax.set_yticks(y, [row[2] for row in rows], fontsize=5.6)
    ax.set_xlim(0.35, 1.01)
    ax.set_title("Step 1 paired performance", fontsize=7.2, fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.36, 1.02, "Discovery cohort, n = 327", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=5.1)
    ax.text(0.36, 0.47, "Temporal validation, n = 274", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=5.1)
    for cohort, ypos in [("Development", 2.73), ("Independent validation", -0.26)]:
        d = delong[(delong["cohort"].eq(cohort)) & delong["first_model"].eq("Targeted-panel score") & delong["second_model"].eq("CA125")].iloc[0]
        ax.text(1.00, ypos, format_p(d["p_value"]), ha="right", va="center", fontsize=5.0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2, fontsize=5.0)


def draw_step1_operating(ax, step1_changes):
    rows = [
        ("Dev sens.", "Development", "Sensitivity among lesions"),
        ("Dev spec.", "Development", "Specificity among N"),
        ("Later sens.", "Independent validation", "Sensitivity among lesions"),
        ("Later spec.", "Independent validation", "Specificity among N"),
    ]
    y_positions = np.arange(len(rows))[::-1]
    for y, (label, cohort, endpoint) in zip(y_positions, rows):
        row = step1_changes[(step1_changes["cohort"].eq(cohort)) & (step1_changes["endpoint"].eq(endpoint))].iloc[0]
        ax.plot([row["CA125_rate"], row["panel_rate"]], [y, y], color="#BEBEBE", lw=1.0)
        ax.plot(row["CA125_rate"], y, "s", ms=4, color=COLORS["ca125"])
        ax.plot(row["panel_rate"], y, "o", ms=4, color=COLORS["panel"])
    ax.set_yticks(y_positions, [r[0] for r in rows], fontsize=5.8)
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("Rate")
    ax.set_title("Fixed operating points", fontsize=7.0)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.plot([], [], "s", color=COLORS["ca125"], label="CA125")
    ax.plot([], [], "o", color=COLORS["panel"], label="Panel")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=5.2)


def draw_reclassification(ax, reclassification):
    rows = [
        ("Dev false positives", "Development", "Correct B / specificity", True),
        ("Later false positives", "Independent validation", "Correct B / specificity", True),
        ("Dev detected BD+M", "Development", "Detected BD+M / sensitivity", False),
        ("Later detected BD+M", "Independent validation", "Detected BD+M / sensitivity", False),
    ]
    y_positions = np.arange(len(rows))[::-1]
    for idx, (y, (label, cohort, endpoint, is_fp)) in enumerate(zip(y_positions, rows)):
        row = reclassification[(reclassification["cohort"].eq(cohort)) & (reclassification["endpoint"].eq(endpoint))].iloc[0]
        if is_fp:
            ca_count = row["n"] - row["CA125_correct"]
            co_count = row["n"] - row["combined_correct"]
            ca_rate = ca_count / row["n"]
            co_rate = co_count / row["n"]
        else:
            ca_count = row["CA125_correct"]
            co_count = row["combined_correct"]
            ca_rate = row["CA125_rate"]
            co_rate = row["combined_rate"]
        ax.plot([ca_rate, co_rate], [y, y], color="#BDBDBD", lw=1.4, zorder=1)
        ax.annotate("", xy=(co_rate, y), xytext=(ca_rate, y), arrowprops=dict(arrowstyle="-|>", color=COLORS["combined"], lw=1.0))
        ax.plot(ca_rate, y, "s", ms=5, color=COLORS["ca125"], zorder=3)
        ax.plot(co_rate, y, "o", ms=5, color=COLORS["combined"], zorder=3)
        ax.text(ca_rate, y + 0.17, f"{int(ca_count)}/{int(row['n'])}", ha="center", fontsize=5.5, color=COLORS["ca125"])
        ax.text(co_rate, y - 0.20, f"{int(co_count)}/{int(row['n'])}", ha="center", fontsize=5.5, color=COLORS["combined"])
        if idx == 1:
            ax.axhspan(1.5, 3.5, color=COLORS["pale_coral"], zorder=0)
    ax.set_yticks(y_positions, [r[0] for r in rows])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Participants per 100")
    ax.set_xticks(np.linspace(0, 1, 6), ["0", "20", "40", "60", "80", "100"])
    ax.set_title("Operational reclassification at fixed thresholds", fontsize=7.5)
    ax.plot([], [], "s", color=COLORS["ca125"], label="CA125 ≥35")
    ax.plot([], [], "o", color=COLORS["combined"], label="Combined")
    ax.legend(loc="lower right", fontsize=5.8)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)


def draw_step2_auc(ax, performance, delong):
    y_base = {"Development": 3, "Independent validation": 1}
    offsets = {"CA125": 0.16, "Combined model": -0.16}
    colors = {"CA125": COLORS["ca125"], "Combined model": COLORS["panel"]}
    for cohort in COHORT_ORDER:
        for model in ["CA125", "Combined model"]:
            row = performance[(performance["cohort"].eq(cohort)) & performance["model"].eq(model)].iloc[0]
            y = y_base[cohort] + offsets[model]
            ax.plot([row["roc_auc_ci_low"], row["roc_auc_ci_high"]], [y, y], color=colors[model], lw=1.2)
            ax.plot(row["roc_auc"], y, "o", color=colors[model], ms=4.2)
        d = delong[(delong["cohort"].eq(cohort)) & delong["first_model"].eq("Combined model") & delong["second_model"].eq("CA125")].iloc[0]
        ax.text(0.655, y_base[cohort] - 0.45, f"ΔAUC={d['delta_auc']:.3f}; {format_p(d['p_value'])}", fontsize=5.4)
    ax.set_yticks([3, 1], ["Development OOF", "Independent validation"])
    ax.set_xlim(0.65, 0.96)
    ax.set_ylim(0.15, 3.75)
    ax.set_xlabel("ROC AUC (bootstrap 95% CI)")
    ax.set_title("Step 2 discrimination", fontsize=7.5)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    for model in ["CA125", "Combined model"]:
        ax.plot([], [], "o", color=colors[model], label=model)
    ax.legend(loc="upper left", fontsize=5.8)


def draw_availability(ax, sample_flow):
    totals = sample_flow.groupby("cohort")[["clinical_total", "panel_available", "CA125_available", "panel_and_CA125"]].sum().loc[COHORT_ORDER]
    labels = ["Clinical", "Panel", "CA125", "Paired"]
    cols = ["clinical_total", "panel_available", "CA125_available", "panel_and_CA125"]
    x = np.arange(4)
    width = 0.34
    for idx, cohort in enumerate(COHORT_ORDER):
        values = totals.loc[cohort, cols].to_numpy()
        color = COLORS["development"] if cohort == "Development" else COLORS["validation"]
        bars = ax.bar(x + (idx - 0.5) * width, values, width=width, color=color, alpha=0.82, label="Discovery" if idx == 0 else "Temporal validation")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 6, str(int(value)), ha="center", fontsize=5.2)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Participants")
    ax.set_title("Sample availability", fontsize=7.5)
    ax.legend(fontsize=5.5)


def draw_pr(ax, step1, cohort, short_title):
    block = step1[step1["cohort"].eq(cohort)]
    y = block["label"].to_numpy(int)
    for model, column, color, linestyle in [("CA125", "CA125", COLORS["ca125"], "--"), ("Panel", "panel_score", COLORS["panel"], "-")]:
        precision, recall, _ = precision_recall_curve(y, block[column])
        ap = average_precision_score(y, block[column])
        ax.plot(recall, precision, color=color, lw=1.25, ls=linestyle, label=f"{model}: AP = {ap:.3f}")
    ax.axhline(y.mean(), ls=":", color="#A8A8A8", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.set_title(f"{short_title} paired PR, n = {len(block)}", fontsize=7.1, fontweight="bold")
    ax.legend(loc="lower left", fontsize=5.5)


def draw_step1_pr_combined(ax, step1):
    styles = {"Development": "--", "Independent validation": "-"}
    cohort_labels = {"Development": "Discovery", "Independent validation": "Temporal validation"}
    for cohort in COHORT_ORDER:
        block = step1[step1["cohort"].eq(cohort)]
        y = block["label"].to_numpy(int)
        for model, column, color in [("CA125", "CA125", COLORS["ca125"]), ("Panel", "panel_score", COLORS["panel"])]:
            precision, recall, _ = precision_recall_curve(y, block[column])
            ap = average_precision_score(y, block[column])
            ax.plot(recall, precision, color=color, ls=styles[cohort], lw=1.25, label=f"{cohort_labels[cohort]} {model}: {ap:.3f}")
    later = step1[step1["cohort"].eq("Independent validation")]
    ax.axhline(later["label"].mean(), ls=":", color="#A8A8A8", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.set_title("Step 1 paired precision–recall", fontsize=7.2, fontweight="bold")
    ax.legend(loc="lower left", fontsize=5.0)


def draw_stage_sensitivity(ax, stage_sensitivity):
    """Paired fixed-threshold sensitivity in FIGO I–II and III–IV malignant cases."""
    rows = [
        ("Discovery · FIGO I–II", "Development", "FIGO I–II"),
        ("Discovery · FIGO III–IV", "Development", "FIGO III–IV"),
        ("Validation · FIGO I–II", "Independent validation", "FIGO I–II"),
        ("Validation · FIGO III–IV", "Independent validation", "FIGO III–IV"),
    ]
    y = np.arange(len(rows))[::-1]
    height = 0.25
    for model, color, offset in [
        ("CA125", COLORS["ca125"], 0.14),
        ("Targeted-panel score", COLORS["panel"], -0.14),
    ]:
        values, lows, highs, labels = [], [], [], []
        for _, cohort, stage_group in rows:
            selected = stage_sensitivity[
                stage_sensitivity["cohort"].eq(cohort) & stage_sensitivity["stage_group"].eq(stage_group)
            ]
            row = selected[selected["model"].eq(model)].iloc[0]
            values.append(float(row["sensitivity"]))
            lows.append(float(row["ci_low"]))
            highs.append(float(row["ci_high"]))
            labels.append(f"{int(row['detected'])}/{int(row['n'])}")
        values = np.asarray(values)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        bars = ax.barh(
            y + offset, values, height=height, color=color, edgecolor="white", linewidth=0.45,
            label="CA125 ≥ 35 U/mL" if model == "CA125" else "Panel ≥ 0.488", zorder=2,
        )
        ax.errorbar(
            values, y + offset, xerr=np.maximum(0.0, np.vstack([values - lows, highs - values])),
            fmt="none", ecolor="#616161", elinewidth=0.75, capsize=1.8, zorder=3,
        )
        for bar, value, label in zip(bars, values, labels):
            ax.text(max(0.06, value - 0.018), bar.get_y() + bar.get_height() / 2, label,
                    ha="right", va="center", fontsize=5.0, color=COLORS["ink"], zorder=4)
    ax.axhline(1.5, color="#D8D8D8", lw=0.7)
    ax.set_yticks(y, [row[0] for row in rows], fontsize=5.7)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.42, 3.42)
    ax.set_xlabel("Sensitivity at fixed threshold")
    ax.set_title("Malignant-case detection by FIGO stage", fontsize=7.2, fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=5.0)


def draw_ca125_negative_rescue(ax, rescue):
    """Show panel-positive fractions among CA125-negative malignant cases."""
    rows = [
        ("Discovery · all M", "Development", "All malignant"),
        ("Discovery · FIGO I–II", "Development", "FIGO I–II"),
        ("Discovery · FIGO III–IV", "Development", "FIGO III–IV"),
        ("Validation · all M", "Independent validation", "All malignant"),
        ("Validation · FIGO I–II", "Independent validation", "FIGO I–II"),
        ("Validation · FIGO III–IV", "Independent validation", "FIGO III–IV"),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (_, cohort, stage_group) in zip(y, rows):
        row = rescue[(rescue["cohort"].eq(cohort)) & (rescue["stage_group"].eq(stage_group))].iloc[0]
        n = int(row["CA125_negative_n"])
        detected = int(row["panel_positive_n"])
        missed = int(row["panel_negative_n"])
        detected_rate = detected / n if n else 0.0
        ax.barh(yi, detected_rate, color=COLORS["panel"], height=0.58, edgecolor="white", linewidth=0.5)
        ax.barh(yi, 1 - detected_rate, left=detected_rate, color="#E8E8E8", height=0.58, edgecolor="white", linewidth=0.5)
        ax.text(detected_rate / 2, yi, f"{detected}/{n}", ha="center", va="center", fontsize=5.4, color="white" if detected_rate > 0.24 else COLORS["ink"])
        if missed > 0 and 1 - detected_rate > 0.12:
            ax.text(detected_rate + (1 - detected_rate) / 2, yi, str(missed), ha="center", va="center", fontsize=5.1, color=COLORS["muted"])
    ax.axhline(2.5, color="#D8D8D8", lw=0.7)
    ax.set_yticks(y, [row[0] for row in rows], fontsize=5.6)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), ["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("CA125-negative malignant cases (%)")
    ax.set_title("Panel detection among CA125-negative malignant cases", fontsize=7.2, fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        patches.Patch(facecolor=COLORS["panel"], label="Panel positive"),
        patches.Patch(facecolor="#E8E8E8", edgecolor="#C8C8C8", label="Panel negative"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=5.0)


def draw_confusion_matrix(ax, y, call, labels, title, show_y=True):
    cm = confusion_matrix(y, call, labels=[0, 1])
    # Use opaque vector rectangles rather than imshow. This avoids embedded-raster
    # colour interpolation and transparency changes when SVG/PDF is opened in Adobe.
    vector_palette = ["#F2F5F7", "#B8D0DF", "#78A6C3", "#2B5876"]
    maximum = max(float(cm.max()), 1.0)
    for i in range(2):
        for j in range(2):
            fraction = float(cm[i, j]) / maximum
            palette_index = 3 if fraction >= 0.75 else 2 if fraction >= 0.45 else 1 if fraction >= 0.15 else 0
            ax.add_patch(
                patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=vector_palette[palette_index], edgecolor="white",
                    linewidth=0.8, alpha=1.0,
                )
            )
            text_color = "white" if palette_index >= 2 else COLORS["ink"]
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=6.5, color=text_color)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels if show_y else ["", ""])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed" if show_y else "")
    ax.set_title(title, fontsize=6.6)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_step1_confusions(container_ax, step1):
    container_ax.set_axis_off()
    block = step1[step1["cohort"].eq("Independent validation")]
    left = container_ax.inset_axes([0.00, 0.05, 0.43, 0.78])
    right = container_ax.inset_axes([0.57, 0.05, 0.43, 0.78])
    draw_confusion_matrix(left, block["label"], block["CA125_call"], ["N", "Lesion"], "CA125", show_y=True)
    draw_confusion_matrix(right, block["label"], block["panel_call"], ["N", "Lesion"], "Panel", show_y=False)
    container_ax.text(0.5, 1.06, "Step 1 later-cohort confusion matrices", ha="center", va="top", fontsize=7.0)


def draw_step1_calibration(ax, calibration):
    block = calibration[calibration["model"].eq("Step 1 targeted-panel score")]
    for cohort, color in [("Development", COLORS["development"]), ("Independent validation", COLORS["validation"])]:
        b = block[block["cohort"].eq(cohort)]
        ax.plot(b["mean_predicted_probability"], b["observed_fraction"], "o-", ms=3, lw=1.0, color=color, label="Development" if cohort == "Development" else "Later cohort")
    ax.plot([0, 1], [0, 1], ":", color="#8F8F8F", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Mean predicted", ylabel="Observed lesion fraction")
    ax.set_title("Step 1 panel calibration", fontsize=7.2)
    ax.legend(fontsize=5.4)


def draw_threshold_diagnostic(ax, y, values, frozen_threshold, title, color):
    fpr, tpr, thresholds = roc_curve(y, values)
    valid = np.isfinite(thresholds)
    threshold = thresholds[valid]
    sensitivity = tpr[valid]
    specificity = 1 - fpr[valid]
    order = np.argsort(threshold)
    ax.plot(threshold[order], sensitivity[order], color=color, lw=1.1, label="Sensitivity")
    ax.plot(threshold[order], specificity[order], color=COLORS["ink"], lw=1.1, label="Specificity")
    ax.axvline(frozen_threshold, ls="--", color=COLORS["muted"], lw=0.9)
    ax.set(xlabel="Decision threshold", ylabel="Rate", ylim=(0, 1.02))
    ax.set_title(title, fontsize=7.2)
    ax.legend(fontsize=5.4)


def draw_step2_curves(container_ax, step2):
    container_ax.set_axis_off()
    specs = [
        ("Development", "ROC", [0.00, 0.53, 0.48, 0.40]),
        ("Independent validation", "ROC", [0.52, 0.53, 0.48, 0.40]),
        ("Development", "PR", [0.00, 0.02, 0.48, 0.40]),
        ("Independent validation", "PR", [0.52, 0.02, 0.48, 0.40]),
    ]
    models = [
        ("CA125", "CA125", COLORS["ca125"]),
        ("Combined model", "combined_probability", COLORS["panel"]),
    ]
    for cohort, curve_type, rect in specs:
        ax = container_ax.inset_axes(rect)
        block = step2[step2["cohort"].eq(cohort)]
        y = block["label"].to_numpy(int)
        for name, col, color in models:
            if curve_type == "ROC":
                x, yy, _ = roc_curve(y, block[col])
                score = roc_auc_score(y, block[col])
            else:
                yy, x, _ = precision_recall_curve(y, block[col])
                score = average_precision_score(y, block[col])
            ax.plot(x, yy, color=color, lw=1.0, label=f"{name} {score:.3f}")
        if curve_type == "ROC":
            ax.plot([0, 1], [0, 1], ":", color="#A8A8A8", lw=0.7)
            ax.set_xlabel("1 − specificity", fontsize=5.5)
            ax.set_ylabel("Sensitivity", fontsize=5.5)
        else:
            ax.axhline(y.mean(), ls=":", color="#A8A8A8", lw=0.7)
            ax.set_xlabel("Recall", fontsize=5.5)
            ax.set_ylabel("Precision", fontsize=5.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        short = "Dev OOF" if cohort == "Development" else "Later validation"
        ax.set_title(f"{short} {curve_type}", fontsize=6.2)
        ax.tick_params(labelsize=5.2)
        if cohort == "Independent validation" and curve_type == "PR":
            ax.legend(fontsize=4.5, loc="lower left")
    # The four internal titles identify the audit views; omit a competing super-title.


def draw_step2_validation_curves(container_ax, step2):
    container_ax.set_axis_off()
    block = step2[step2["cohort"].eq("Independent validation")]
    y = block["label"].to_numpy(int)
    models = [
        ("CA125", "CA125", COLORS["ca125"], "--"),
        ("Combined model", "combined_probability", COLORS["panel"], "-"),
    ]
    for curve_type, rect in [("ROC", [0.00, 0.08, 0.44, 0.82]), ("PR", [0.56, 0.08, 0.44, 0.82])]:
        ax = container_ax.inset_axes(rect)
        for name, col, color, linestyle in models:
            if curve_type == "ROC":
                x, yy, _ = roc_curve(y, block[col])
                score = roc_auc_score(y, block[col])
                metric = "AUC"
            else:
                yy, x, _ = precision_recall_curve(y, block[col])
                score = average_precision_score(y, block[col])
                metric = "AP"
            ax.plot(x, yy, color=color, ls=linestyle, lw=1.15, label=f"{name}: {metric} = {score:.3f}")
        if curve_type == "ROC":
            ax.plot([0, 1], [0, 1], ":", color="#A8A8A8", lw=0.7)
            ax.set_xlabel("1 − specificity")
            ax.set_ylabel("Sensitivity")
        else:
            ax.axhline(y.mean(), ls=":", color="#A8A8A8", lw=0.7)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Later-cohort {curve_type}", fontsize=6.7, fontweight="bold")
        ax.legend(loc="lower left", fontsize=4.7)


def draw_step2_confusions(container_ax, step2, combined_threshold):
    container_ax.set_axis_off()
    block = step2[step2["cohort"].eq("Independent validation")]
    left = container_ax.inset_axes([0.05, 0.05, 0.38, 0.76])
    right = container_ax.inset_axes([0.57, 0.05, 0.38, 0.76])
    draw_confusion_matrix(left, block["label"], block["CA125"].ge(CA125_THRESHOLD), ["B", "BD+M"], "CA125", show_y=True)
    draw_confusion_matrix(right, block["label"], block["combined_probability"].ge(combined_threshold), ["B", "BD+M"], "6 analytes + CA125", show_y=False)
    container_ax.text(0.5, 1.06, "Step 2 later-cohort operating point", ha="center", va="top", fontsize=7.0)


def draw_step2_audit(container_ax, results):
    container_ax.set_axis_off()
    ax_cal = container_ax.inset_axes([0.00, 0.08, 0.29, 0.80])
    ax_thr = container_ax.inset_axes([0.34, 0.08, 0.29, 0.80])
    ax_coef = container_ax.inset_axes([0.70, 0.08, 0.30, 0.80])
    block = results["calibration"][results["calibration"]["model"].eq("Step 2 combined model")]
    for cohort, color in [("Development", COLORS["development"]), ("Independent validation", COLORS["validation"])]:
        b = block[block["cohort"].eq(cohort)]
        ax_cal.plot(b["mean_predicted_probability"], b["observed_fraction"], "o-", ms=2.8, lw=0.9, color=color, label="Dev" if cohort == "Development" else "Later")
    ax_cal.plot([0, 1], [0, 1], ":", color="#8F8F8F", lw=0.7)
    ax_cal.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Mean predicted", ylabel="Observed BD+M")
    ax_cal.set_title("Calibration", fontsize=6.4)
    ax_cal.legend(fontsize=4.8)
    val = results["step2"][results["step2"]["cohort"].eq("Independent validation")]
    draw_threshold_diagnostic(ax_thr, val["label"], val["combined_probability"], results["combined_threshold"], "Fixed threshold", COLORS["panel"])
    coef = results["coefficients"]
    y = np.arange(len(coef))[::-1]
    coefficient_colors = [COLORS["panel"]] * (len(coef) - 1) + [COLORS["ca125"]]
    ax_coef.barh(y, coef["standardized_coefficient"], color=coefficient_colors, height=0.55)
    labels = [str(value).replace("log10(CA125)", "CA125") for value in coef["feature"]]
    ax_coef.set_yticks(y, labels, fontsize=4.4)
    ax_coef.set_xlabel("Standardized coefficient")
    ax_coef.set_title("Six-analyte + CA125 model", fontsize=6.4)
    # Internal titles are sufficient at final supplementary-figure size.


def make_main_figure(scored, results):
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 16, height_ratios=[0.90, 1.30, 1.25], hspace=0.62, wspace=0.90)
    ax_a = fig.add_subplot(gs[0, 0:10])
    ax_b = fig.add_subplot(gs[0, 11:16])
    ax_c1 = fig.add_subplot(gs[1, 0:4])
    ax_c2 = fig.add_subplot(gs[1, 4:8])
    ax_d = fig.add_subplot(gs[1, 8:12])
    ax_e = fig.add_subplot(gs[1, 12:16])
    ax_f = fig.add_subplot(gs[2, 0:8])
    ax_g = fig.add_subplot(gs[2, 8:16])
    draw_schematic(ax_a)
    draw_score_spectrum(ax_b, scored)
    draw_step1_roc(ax_c1, results["step1"], results["step1_delong"], "Development", "Discovery cohort")
    draw_step1_roc(ax_c2, results["step1"], results["step1_delong"], "Independent validation", "Validation cohort")
    draw_metric_bars(ax_d, results["step1_performance"], "Development", "CA125", "Targeted-panel score", COLORS["ca125"], COLORS["panel"], "Step 1 · Discovery", results["step1_delong"], show_legend=True)
    draw_metric_bars(ax_e, results["step1_performance"], "Independent validation", "CA125", "Targeted-panel score", COLORS["ca125"], COLORS["panel"], "Step 1 · Validation", results["step1_delong"], show_legend=False)
    draw_step2_metric_bars(ax_f, results["step2_performance"], "Development", "Step 2 · Discovery (OOF)", show_legend=True)
    draw_step2_metric_bars(ax_g, results["step2_performance"], "Independent validation", "Step 2 · Validation", show_legend=False)
    ax_e.set_yticklabels([])
    ax_g.set_yticklabels([])
    # Panels d1-d4 are four views within the single composite panel d.
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c1, "c")]:
        add_panel_label(ax, label)
    for ax, label in [(ax_d, "d"), (ax_e, "e"), (ax_f, "f"), (ax_g, "g")]:
        add_panel_label(ax, label, x=-0.08)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.075)
    save_figure(fig, "Fig5_candidate_analysis11")


def make_supplementary_figure(results):
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05], hspace=0.55, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    draw_availability(ax_a, results["sample_flow"])
    draw_stage_sensitivity(ax_b, results["stage_sensitivity"])
    draw_step1_confusions(ax_c, results["step1"])
    draw_step1_calibration(ax_d, results["calibration"])
    draw_ca125_negative_rescue(ax_e, results["ca125_negative_rescue"])
    draw_step2_confusions(ax_f, results["step2"], results["combined_threshold"])
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e"), (ax_f, "f")]:
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.07)
    save_figure(fig, "FigS5_candidate_analysis11")


def make_component_figures(scored, results):
    components = []

    fig, ax = plt.subplots(figsize=(5.0, 1.7)); draw_schematic(ax); components.append((fig, "Fig5a_two_step_workflow"))
    fig, ax = plt.subplots(figsize=(3.3, 2.3)); draw_score_spectrum(ax, scored); components.append((fig, "Fig5b_score_spectrum"))
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5)); draw_step1_roc(axes[0], results["step1"], results["step1_delong"], "Development", "Discovery cohort"); draw_step1_roc(axes[1], results["step1"], results["step1_delong"], "Independent validation", "Validation cohort"); components.append((fig, "Fig5c_step1_panel_ROC_CA125_fixed_point"))
    fig, ax = plt.subplots(figsize=(3.2, 2.7)); draw_metric_bars(ax, results["step1_performance"], "Development", "CA125", "Targeted-panel score", COLORS["ca125"], COLORS["panel"], "Step 1 · Discovery", results["step1_delong"]); components.append((fig, "Fig5d_step1_discovery_fixed_metrics"))
    fig, ax = plt.subplots(figsize=(3.2, 2.7)); draw_metric_bars(ax, results["step1_performance"], "Independent validation", "CA125", "Targeted-panel score", COLORS["ca125"], COLORS["panel"], "Step 1 · Validation", results["step1_delong"]); components.append((fig, "Fig5e_step1_validation_fixed_metrics"))
    fig, ax = plt.subplots(figsize=(3.8, 2.8)); draw_step2_metric_bars(ax, results["step2_performance"], "Development", "Step 2 · Discovery (OOF)"); components.append((fig, "Fig5f_step2_discovery_fixed_metrics"))
    fig, ax = plt.subplots(figsize=(3.8, 2.8)); draw_step2_metric_bars(ax, results["step2_performance"], "Independent validation", "Step 2 · Validation"); components.append((fig, "Fig5g_step2_validation_fixed_metrics"))

    fig, ax = plt.subplots(figsize=(3.3, 2.6)); draw_availability(ax, results["sample_flow"]); components.append((fig, "FigS5a_sample_availability"))
    fig, ax = plt.subplots(figsize=(3.6, 2.8)); draw_stage_sensitivity(ax, results["stage_sensitivity"]); components.append((fig, "FigS5b_fixed_threshold_sensitivity_by_stage"))
    fig, ax = plt.subplots(figsize=(4.4, 2.5)); draw_step1_confusions(ax, results["step1"]); components.append((fig, "FigS5c_step1_confusions"))
    fig, ax = plt.subplots(figsize=(3.0, 2.5)); draw_step1_calibration(ax, results["calibration"]); components.append((fig, "FigS5d_step1_calibration"))
    fig, ax = plt.subplots(figsize=(3.8, 3.0)); draw_ca125_negative_rescue(ax, results["ca125_negative_rescue"]); components.append((fig, "FigS5e_CA125_negative_panel_detection"))
    fig, ax = plt.subplots(figsize=(5.2, 2.5)); draw_step2_confusions(ax, results["step2"], results["combined_threshold"]); components.append((fig, "FigS5f_step2_confusions"))

    for fig, stem in components:
        fig.tight_layout(pad=0.8)
        save_figure(fig, stem)


def write_results_readme(results):
    s1d = results["step1_delong"]
    s2d = results["step2_delong"]
    lines = [
        "# Analysis 11 results",
        "",
        "This analysis supersedes the deprecated analyses 09 and 10 for Fig. 5 and Supplementary Fig. 5 preparation.",
        "",
        "## Step 1: normal controls versus any ovarian lesion",
        "",
    ]
    for cohort in COHORT_ORDER:
        perf = results["step1_performance"]
        panel = perf[(perf["cohort"].eq(cohort)) & perf["model"].eq("Targeted-panel score")].iloc[0]
        ca = perf[(perf["cohort"].eq(cohort)) & perf["model"].eq("CA125")].iloc[0]
        d = s1d[s1d["cohort"].eq(cohort)].iloc[0]
        lines.append(f"- {cohort}: n={int(panel['n'])}; panel AUC={panel['roc_auc']:.3f}; CA125 AUC={ca['roc_auc']:.3f}; delta AUC={d['delta_auc']:.3f}; P={d['p_value']:.4g}.")
    lines.extend(["", "## Step 2: benign versus borderline/malignant lesions", ""])
    for cohort in COHORT_ORDER:
        perf = results["step2_performance"]
        ca = perf[(perf["cohort"].eq(cohort)) & perf["model"].eq("CA125")].iloc[0]
        combined = perf[(perf["cohort"].eq(cohort)) & perf["model"].eq("Combined model")].iloc[0]
        d = s2d[(s2d["cohort"].eq(cohort)) & s2d["first_model"].eq("Combined model") & s2d["second_model"].eq("CA125")].iloc[0]
        lines.append(f"- {cohort}: n={int(combined['n'])}; CA125 sensitivity/specificity={ca['sensitivity']:.3f}/{ca['specificity']:.3f}; combined={combined['sensitivity']:.3f}/{combined['specificity']:.3f}; delta AUC={d['delta_auc']:.3f}; P={d['p_value']:.4g}.")
    lines.extend(
        [
            "",
            "Step 2 uses an oncology-oriented triage endpoint, with borderline and malignant lesions in the positive class. The L2 model combines CA125 with six targeted analytes. Its development-OOF threshold requires at least one additional correctly classified positive and negative case versus CA125 at 35 U/mL and is carried forward unchanged to the later cohort. Age and free-text preoperative diagnosis are excluded.",
            "",
            f"Six-analyte-plus-CA125 threshold derived from averaged development OOF predictions: `{results['combined_threshold']:.12f}`.",
            "",
            "All figure components and assembled candidates are exported as editable SVG, PDF, 600-dpi TIFF and PNG preview files.",
        ]
    )
    (ROOT11 / "README_results.md").write_text("\n".join(lines), encoding="utf-8")


def load_exported_results_for_figures():
    """Load the frozen analysis-11 tables without fitting any model or changing a threshold."""
    results = {
        "step1": pd.read_csv(TABLES / "11_step1_paired_predictions.tsv", sep="\t"),
        "step2": pd.read_csv(TABLES / "11_step2_paired_predictions.tsv", sep="\t"),
        "step1_performance": pd.read_csv(TABLES / "11_step1_performance_by_cohort.tsv", sep="\t"),
        "step2_performance": pd.read_csv(TABLES / "11_step2_performance_by_cohort.tsv", sep="\t"),
        "step1_delong": pd.read_csv(TABLES / "11_step1_paired_DeLong.tsv", sep="\t"),
        "step2_delong": pd.read_csv(TABLES / "11_step2_paired_DeLong.tsv", sep="\t"),
        "step1_changes": pd.read_csv(TABLES / "11_step1_fixed_threshold_paired_changes.tsv", sep="\t"),
        "step2_reclassification": pd.read_csv(TABLES / "11_step2_fixed_threshold_reclassification.tsv", sep="\t"),
        "sample_flow": pd.read_csv(TABLES / "11_sample_flow.tsv", sep="\t"),
        "calibration": pd.read_csv(TABLES / "11_calibration_source.tsv", sep="\t"),
        "coefficients": pd.read_csv(TABLES / "11_combined_model_coefficients.tsv", sep="\t"),
    }
    results["stage_sensitivity"], results["ca125_negative_rescue"] = derive_stage_sensitivity_and_rescue(results["step2"])
    summary = json.loads((TABLES / "11_summary.json").read_text(encoding="utf-8"))
    results["combined_threshold"] = float(summary["combined_threshold_from_development_OOF"])
    scored = pd.read_csv(SOURCE / "SourceData_Fig5b_score_spectrum.tsv", sep="\t")
    return scored, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Restyle figures from frozen exported analysis-11 tables; do not fit any model or derive any threshold.",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    if args.figures_only:
        scored, results = load_exported_results_for_figures()
        results["stage_sensitivity"].to_csv(TABLES / "11_fixed_threshold_sensitivity_by_stage.tsv", sep="\t", index=False)
        results["ca125_negative_rescue"].to_csv(TABLES / "11_CA125_negative_panel_detection.tsv", sep="\t", index=False)
        results["stage_sensitivity"].to_csv(SOURCE / "SourceData_FigS5b_stage_sensitivity.tsv", sep="\t", index=False)
        results["ca125_negative_rescue"].to_csv(SOURCE / "SourceData_FigS5e_CA125_negative_panel_detection.tsv", sep="\t", index=False)
        for path in FIGURES.glob("Fig5*"):
            if path.is_file():
                path.unlink()
        for path in FIGURES.glob("FigS5*"):
            if path.is_file():
                path.unlink()
        make_component_figures(scored, results)
        make_main_figure(scored, results)
        make_supplementary_figure(results)
        log = {
            "status": "figures_restyled_from_frozen_tables",
            "backend": "Python/matplotlib",
            "model_fit_performed": False,
            "threshold_changed": False,
            "panel_threshold": PANEL_THRESHOLD,
            "CA125_threshold": CA125_THRESHOLD,
            "combined_threshold": results["combined_threshold"],
        }
        (LOGS / "11_figure_restyle_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(log, ensure_ascii=False, indent=2))
        return
    ensure_dirs()
    scored, clinical, feature_names, max_difference = load_scored_samples()
    results = build_analysis(scored, clinical)
    export_tables(results, feature_names, max_difference)
    make_component_figures(scored, results)
    make_main_figure(scored, results)
    make_supplementary_figure(results)
    write_results_readme(results)
    log = {
        "status": "completed",
        "backend": "Python/matplotlib",
        "panel_model_locked": True,
        "panel_threshold": PANEL_THRESHOLD,
        "CA125_threshold": CA125_THRESHOLD,
        "combined_threshold": results["combined_threshold"],
        "step1_n": results["step1"].groupby("cohort").size().to_dict(),
        "step2_n": results["step2"].groupby("cohort").size().to_dict(),
    }
    (LOGS / "11_run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
