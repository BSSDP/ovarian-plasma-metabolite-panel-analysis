#!/usr/bin/env python
"""AI-assisted, leakage-controlled repeated nested-CV training for ModelA."""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import RFE
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
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUT = ROOT / "03NvsM_10_8"
LABEL_INPUT = OUT / "_prepared_model_input.csv"
HIGHQUALITY8_INPUT = ROOT / "data_clean" / "OV_4.9_raw_highquality8.csv"
STYLE_DIR = PROJECT_ROOT / "00_project_style"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
from ov_publication_style import (  # noqa: E402
    BACKGROUND_GREY,
    GROUP_COLORS,
    SIGNAL_BLUE,
    SIGNAL_RED,
    TEXT_DARK,
    setup_matplotlib_style,
)

SEED = 42
PROTOCOL_VERSION = "20260614_ai_neural_ensemble_v1"
OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS = 5
FEATURE_COUNTS = [3, 4, 5, 6, 7, 8]
AI_MODEL_KEYS = {"mlp", "deep_mlp", "soft_voting", "stacking", "xgboost", "lightgbm"}
POS_LABEL = "M"
NEG_LABEL = "N"
N_JOBS = -1
FIG_DPI = 600

MODEL_LABELS = {
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
    "logistic_l2": "Logistic L2",
    "logistic_l1": "Logistic L1",
    "ridge": "Ridge",
    "sgd": "SGD",
    "lda": "LDA",
    "qda": "QDA",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "decision_tree": "Decision tree",
    "gradient_boosting": "Gradient boosting",
    "adaboost": "AdaBoost",
    "knn": "KNN",
    "naive_bayes": "Gaussian NB",
    "mlp": "MLP",
    "deep_mlp": "Deep neural network",
    "soft_voting": "Weighted probability ensemble",
    "stacking": "OOF stacked ensemble",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def model_specs() -> dict[str, tuple[object, dict[str, list]]]:
    specs: dict[str, tuple[object, dict[str, list]]] = {
        "linear_svm": (
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
            {"classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        ),
        "rbf_svm": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale", "auto"]},
        ),
        "logistic_l2": (
            LogisticRegression(penalty="l2", max_iter=4000, class_weight="balanced", random_state=SEED),
            {"classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        ),
        "logistic_l1": (
            LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=4000, class_weight="balanced", random_state=SEED
            ),
            {"classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        ),
        "ridge": (
            CalibratedClassifierCV(RidgeClassifier(class_weight="balanced", random_state=SEED), method="sigmoid", cv=3),
            {"classifier__estimator__alpha": [0.1, 1.0, 10.0]},
        ),
        "sgd": (
            SGDClassifier(loss="log_loss", class_weight="balanced", random_state=SEED, max_iter=4000),
            {"classifier__alpha": [0.0001, 0.001, 0.01]},
        ),
        "lda": (LinearDiscriminantAnalysis(), {}),
        "qda": (QuadraticDiscriminantAnalysis(), {"classifier__reg_param": [0.0, 0.1, 0.3]}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
            {"classifier__n_estimators": [100, 200, 300], "classifier__max_depth": [3, 5, 7]},
        ),
        "extra_trees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
            {"classifier__n_estimators": [100, 200], "classifier__max_depth": [3, 5]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"classifier__max_depth": [3, 5, 7, 10]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
            },
        ),
        "adaboost": (
            AdaBoostClassifier(random_state=SEED),
            {"classifier__n_estimators": [50, 100, 200], "classifier__learning_rate": [0.5, 1.0]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [3, 5, 7, 9], "classifier__weights": ["uniform", "distance"]},
        ),
        "naive_bayes": (GaussianNB(), {}),
        "mlp": (
            MLPClassifier(max_iter=2000, random_state=SEED, early_stopping=True),
            {"classifier__hidden_layer_sizes": [(50,), (100,), (50, 25)], "classifier__alpha": [0.0001, 0.001]},
        ),
        "deep_mlp": (
            MLPClassifier(
                max_iter=4000,
                random_state=SEED,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
            ),
            {
                "classifier__hidden_layer_sizes": [(64, 32, 16), (128, 64, 32)],
                "classifier__alpha": [0.001, 0.01],
                "classifier__learning_rate_init": [0.0005, 0.001],
            },
        ),
        "soft_voting": (
            VotingClassifier(
                estimators=[
                    ("lda", LinearDiscriminantAnalysis()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=0.1,
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=SEED,
                        ),
                    ),
                    (
                        "rbf_svm",
                        SVC(
                            C=1.0,
                            kernel="rbf",
                            gamma="scale",
                            probability=True,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                    (
                        "extra_trees",
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_features="sqrt",
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=1,
                        ),
                    ),
                ],
                voting="soft",
                n_jobs=1,
            ),
            {
                "classifier__weights": [
                    (1, 1, 1, 1),
                    (2, 2, 1, 1),
                    (3, 2, 1, 1),
                ]
            },
        ),
        "stacking": (
            StackingClassifier(
                estimators=[
                    ("lda", LinearDiscriminantAnalysis()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=0.1,
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=SEED,
                        ),
                    ),
                    (
                        "rbf_svm",
                        SVC(
                            C=1.0,
                            kernel="rbf",
                            gamma="scale",
                            probability=True,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                    (
                        "extra_trees",
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_features="sqrt",
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=1,
                        ),
                    ),
                ],
                final_estimator=LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=SEED,
                ),
                cv=5,
                stack_method="predict_proba",
                n_jobs=1,
            ),
            {
                "classifier__final_estimator__C": [0.01, 0.1, 1.0],
                "classifier__passthrough": [False, True],
            },
        ),
    }
    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = (
            XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
            },
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = (
            LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=1, verbosity=-1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
            },
        )
    except ImportError:
        pass
    return specs


def feature_counts_for_model(model_key: str) -> list[int]:
    """Use all validated inputs for the formal neural-network and ensemble candidates."""
    if model_key in {"deep_mlp", "soft_voting", "stacking"}:
        return [8]
    return FEATURE_COUNTS


def make_pipeline(classifier: object) -> Pipeline:
    selector = RFE(
        LogisticRegression(penalty="l2", class_weight="balanced", max_iter=3000, random_state=SEED),
        n_features_to_select=8,
        step=1,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("classifier", classifier),
        ]
    )


def clean_params(params: dict) -> dict:
    result = {}
    for key, value in params.items():
        key = key.replace("classifier__", "")
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def json_params(params: dict) -> str:
    return json.dumps(
        {key: (value.item() if isinstance(value, np.generic) else value) for key, value in params.items()},
        sort_keys=True,
    )


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, prob)
    valid = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    score = tpr - fpr
    best = np.flatnonzero(np.isclose(score, score.max()))
    # Lower threshold wins a tie, which prioritises sensitivity.
    return float(np.min(thresholds[best]))


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
        "accuracy": accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_opaque_confusion_matrix(ax, cm, xlabels, ylabels):
    """Draw editable opaque vector cells instead of imshow raster patches."""
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


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    labels = pd.read_csv(LABEL_INPUT).iloc[:, :2].copy()
    labels.columns = ["Alignment ID", "label"]
    labels["Alignment ID"] = labels["Alignment ID"].astype(str)

    highquality8 = pd.read_csv(HIGHQUALITY8_INPUT)
    feature_names = highquality8.iloc[:, 0].astype(str).tolist()
    if len(feature_names) != 8 or len(set(feature_names)) != 8:
        raise ValueError("OV_4.9_raw_highquality8.csv must contain exactly eight unique features.")

    feature_matrix = highquality8.set_index(highquality8.columns[0]).T
    feature_matrix.index = feature_matrix.index.astype(str)
    feature_matrix.index.name = "Alignment ID"
    df = labels.merge(
        feature_matrix.reset_index(),
        on="Alignment ID",
        how="inner",
        validate="one_to_one",
    )
    df = df[df["label"].isin([NEG_LABEL, POS_LABEL])].reset_index(drop=True)
    X = df[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = (df["label"] == POS_LABEL).astype(int).to_numpy()
    assert len(df) == 222 and int(y.sum()) == 122 and len(feature_names) == 8
    return df, X, y, feature_names


def nested_cv(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = OUT / "ModelA_nested_outer_predictions_checkpoint.tsv"
    selection_checkpoint = OUT / "ModelA_nested_selection_checkpoint.tsv"
    failures_checkpoint = OUT / "ModelA_nested_failures_checkpoint.tsv"
    protocol_checkpoint = OUT / "ModelA_nested_checkpoint_protocol.json"
    checkpoint_files = [checkpoint, selection_checkpoint, failures_checkpoint]
    if any(path.exists() for path in checkpoint_files):
        if not protocol_checkpoint.exists():
            raise RuntimeError(
                "Existing ModelA checkpoints predate protocol tracking. Remove the three "
                "checkpoint TSV files before running the current AI-assisted protocol."
            )
        saved_protocol = json.loads(protocol_checkpoint.read_text(encoding="utf-8"))
        if saved_protocol.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                "Existing ModelA checkpoints use a different modelling protocol. Remove the "
                "checkpoint TSV files before rerunning."
            )
    else:
        protocol_checkpoint.write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "outer_splits": OUTER_SPLITS,
                    "outer_repeats": OUTER_REPEATS,
                    "inner_splits": INNER_SPLITS,
                    "seed": SEED,
                    "candidate_models": list(model_specs()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    existing = pd.read_csv(checkpoint, sep="\t") if checkpoint.exists() else pd.DataFrame()
    existing_selection = pd.read_csv(selection_checkpoint, sep="\t") if selection_checkpoint.exists() else pd.DataFrame()
    existing_failures = pd.read_csv(failures_checkpoint, sep="\t") if failures_checkpoint.exists() else pd.DataFrame()
    done = set(zip(existing.get("repeat", []), existing.get("fold", [])))

    prediction_rows = existing.to_dict("records")
    selection_rows = existing_selection.to_dict("records")
    failure_rows = existing_failures.to_dict("records")
    specs = model_specs()
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)

    for outer_index, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat = outer_index // OUTER_SPLITS + 1
        fold = outer_index % OUTER_SPLITS + 1
        if (repeat, fold) in done:
            continue
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_index)
        candidates = []
        for model_key, (classifier, grid) in specs.items():
            for requested_n_features in feature_counts_for_model(model_key):
                pipeline = make_pipeline(classifier)
                param_grid = {"selector__n_features_to_select": [requested_n_features], **grid}
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=inner,
                    n_jobs=N_JOBS,
                    refit=True,
                    error_score=np.nan,
                    return_train_score=False,
                )
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        search.fit(X[train_idx], y[train_idx])
                    if not np.isfinite(search.best_score_):
                        raise RuntimeError("No finite inner-CV score.")
                    estimator = search.best_estimator_
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        inner_prob = cross_val_predict(
                            clone(estimator),
                            X[train_idx],
                            y[train_idx],
                            cv=inner,
                            method="predict_proba",
                            n_jobs=N_JOBS,
                        )[:, 1]
                    threshold = youden_threshold(y[train_idx], inner_prob)
                    test_prob = estimator.predict_proba(X[test_idx])[:, 1]
                    selected_mask = estimator.named_steps["selector"].support_
                    selected_features = [feature_names[i] for i, keep in enumerate(selected_mask) if keep]
                    outer_metric = metrics(y[test_idx], test_prob, threshold)
                    candidate = {
                        "repeat": repeat,
                        "fold": fold,
                        "model_key": model_key,
                        "model_label": MODEL_LABELS[model_key],
                        "inner_auc": float(search.best_score_),
                        "n_features": len(selected_features),
                        "selected_features": "|".join(selected_features),
                        "threshold_from_inner_oof": threshold,
                        "best_params_json": json_params(search.best_params_),
                        **outer_metric,
                    }
                    selection_rows.append(candidate)
                    candidates.append(
                        (model_key, float(search.best_score_), estimator, test_prob, threshold, selected_features)
                    )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "model": model_key,
                            "n_features": requested_n_features,
                            "error": str(exc)[:500],
                        }
                    )

        if not candidates:
            raise RuntimeError(f"No model succeeded for repeat {repeat}, fold {fold}.")
        model_key, inner_auc, estimator, test_prob, threshold, selected_features = max(
            candidates, key=lambda item: item[1]
        )
        for local_i, sample_i in enumerate(test_idx):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": int(sample_i),
                    "sample_id": str(sample_i),
                    "true_label": int(y[sample_i]),
                    "probability": float(test_prob[local_i]),
                    "threshold_from_inner_oof": threshold,
                    "predicted_label": int(test_prob[local_i] >= threshold),
                    "selected_model": model_key,
                    "selected_n_features": len(selected_features),
                }
            )
        pd.DataFrame(prediction_rows).to_csv(checkpoint, sep="\t", index=False)
        pd.DataFrame(selection_rows).to_csv(selection_checkpoint, sep="\t", index=False)
        pd.DataFrame(failure_rows).to_csv(failures_checkpoint, sep="\t", index=False)
        print(f"Completed outer repeat {repeat:02d}/{OUTER_REPEATS}, fold {fold}/{OUTER_SPLITS}: {model_key}")

    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows), pd.DataFrame(failure_rows)


def choose_locked_configuration(selection: pd.DataFrame) -> tuple[str, int, pd.DataFrame]:
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
        .sort_values("mean_auc", ascending=False)
    )
    best = summary.iloc[0]
    eight = summary[summary["n_features"] == 8].sort_values("mean_auc", ascending=False)
    if eight.empty:
        raise RuntimeError("No eight-feature candidate completed; the final ModelA must use all eight validated inputs.")
    # Fewer-feature candidates remain in the screening figure, but the locked ModelA
    # is deliberately constrained to the complete standard-supported analyte set.
    best = eight.iloc[0]
    return str(best["model_key"]), int(best["n_features"]), summary


def build_locked_configuration(selection: pd.DataFrame, model_key: str, n_features: int) -> tuple[Pipeline, dict]:
    classifier, _ = model_specs()[model_key]
    pipeline = make_pipeline(classifier)
    subset = selection[(selection["model_key"] == model_key) & (selection["n_features"] == n_features)]
    if subset.empty:
        raise RuntimeError("No outer-fold selections available for the locked configuration.")
    chosen_json = subset["best_params_json"].value_counts().index[0]
    chosen_params = json.loads(chosen_json)
    if "classifier__hidden_layer_sizes" in chosen_params:
        chosen_params["classifier__hidden_layer_sizes"] = tuple(chosen_params["classifier__hidden_layer_sizes"])
    pipeline.set_params(**chosen_params)
    return pipeline, chosen_params


def locked_oof_predictions(X: np.ndarray, y: np.ndarray, estimator: Pipeline) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    rows = []
    repeat_metrics = []
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    for outer_index, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat = outer_index // OUTER_SPLITS + 1
        fold = outer_index % OUTER_SPLITS + 1
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inner_prob = cross_val_predict(
                clone(estimator), X[train_idx], y[train_idx], cv=inner, method="predict_proba", n_jobs=N_JOBS
            )[:, 1]
            fitted = clone(estimator).fit(X[train_idx], y[train_idx])
        threshold = youden_threshold(y[train_idx], inner_prob)
        prob = fitted.predict_proba(X[test_idx])[:, 1]
        for i, sample_idx in enumerate(test_idx):
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": int(sample_idx),
                    "true_label": int(y[sample_idx]),
                    "probability": float(prob[i]),
                    "threshold_from_inner_oof": threshold,
                    "predicted_label": int(prob[i] >= threshold),
                }
            )
    raw = pd.DataFrame(rows)
    averaged = (
        raw.groupby(["sample_index", "true_label"], as_index=False)
        .agg(probability=("probability", "mean"), prediction_count=("probability", "size"))
    )
    locked_threshold = youden_threshold(averaged["true_label"].to_numpy(), averaged["probability"].to_numpy())
    for repeat, group in raw.groupby("repeat"):
        # Each repeat contains one prediction for every sample; use fold-specific thresholds.
        y_r = group["true_label"].to_numpy()
        p_r = group["probability"].to_numpy()
        pred_r = group["predicted_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_r, pred_r, labels=[0, 1]).ravel()
        repeat_metrics.append(
            {
                "repeat": repeat,
                "roc_auc": roc_auc_score(y_r, p_r),
                "pr_auc": average_precision_score(y_r, p_r),
                "brier": brier_score_loss(y_r, p_r),
                "accuracy": accuracy_score(y_r, pred_r),
                "sensitivity": recall_score(y_r, pred_r),
                "specificity": tn / (tn + fp),
                "f1": f1_score(y_r, pred_r),
            }
        )
    return raw, locked_threshold, pd.DataFrame(repeat_metrics)


def selection_procedure_repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the fully nested procedure selected only by inner-CV evidence."""
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        y = group["true_label"].to_numpy()
        prob = group["probability"].to_numpy()
        pred = group["predicted_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": roc_auc_score(y, prob),
                "pr_auc": average_precision_score(y, prob),
                "brier": brier_score_loss(y, prob),
                "accuracy": accuracy_score(y, pred),
                "sensitivity": recall_score(y, pred),
                "specificity": tn / (tn + fp),
                "f1": f1_score(y, pred),
            }
        )
    return pd.DataFrame(rows)


def plot_all(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    screening: pd.DataFrame,
    locked_predictions: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    locked_threshold: float,
    final_pipeline: Pipeline,
    selected_model_key: str,
) -> None:
    setup_matplotlib_style(matplotlib, base_size=7)
    neg, pos = GROUP_COLORS["N"], GROUP_COLORS["M"]
    avg = (
        locked_predictions.groupby(["sample_index", "true_label"], as_index=False)
        .agg(probability=("probability", "mean"), prediction_count=("probability", "size"))
        .sort_values("sample_index")
    )
    yy, pp = avg["true_label"].to_numpy(), avg["probability"].to_numpy()

    # 01 screening
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    top_models = (
        screening.groupby("model_key")["mean_auc"].max().sort_values(ascending=False).head(8).index.tolist()
    )
    for model_key in [key for key in top_models if key != selected_model_key]:
        d = screening[screening["model_key"] == model_key].sort_values("n_features")
        ax.errorbar(
            d["n_features"],
            d["mean_auc"],
            yerr=d["sd_auc"],
            marker="o",
            ms=2.4,
            lw=0.7,
            capsize=1.5,
            color="#B8B8B8",
            ecolor="#D2D2D2",
            label=MODEL_LABELS[model_key],
            alpha=0.75,
            zorder=1,
        )
    selected_curve = screening[screening["model_key"] == selected_model_key].sort_values("n_features")
    ax.errorbar(
        selected_curve["n_features"],
        selected_curve["mean_auc"],
        yerr=selected_curve["sd_auc"],
        marker="o",
        ms=3.5,
        lw=1.5,
        capsize=2,
        color=SIGNAL_RED,
        ecolor=SIGNAL_RED,
        label=f"Selected: {MODEL_LABELS[selected_model_key]}",
        zorder=3,
    )
    selected_row = selected_curve.loc[selected_curve["mean_auc"].idxmax()]
    ax.scatter(
        [selected_row["n_features"]],
        [selected_row["mean_auc"]],
        marker="*",
        s=42,
        color=SIGNAL_RED,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.set(xlabel="Number of selected features", ylabel="Outer nested-CV AUC", ylim=(0.65, 1.0))
    ax.legend(frameon=False, fontsize=5.8, ncol=2)
    ax.grid(axis="y", color=BACKGROUND_GREY, lw=0.5)
    save_figure(fig, "ModelA_NvsM_01_model_screening_auc_by_feature_count")

    # 02 ROC/PR
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9))
    fpr, tpr, _ = roc_curve(yy, pp)
    axes[0].plot(fpr, tpr, color=pos, lw=1.8, label=f"Averaged OOF AUC = {roc_auc_score(yy, pp):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=0.7)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    precision, recall, _ = precision_recall_curve(yy, pp)
    axes[1].plot(recall, precision, color=pos, lw=1.8, label=f"Averaged OOF AP = {average_precision_score(yy, pp):.3f}")
    axes[1].axhline(yy.mean(), ls="--", color="#999999", lw=0.7)
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    for ax in axes:
        ax.legend(frameon=False)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "ModelA_NvsM_02_ROC_PR_summary")

    # 03 final ROC
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(fpr, tpr, color=pos, lw=1.8, label=f"Averaged OOF AUC = {roc_auc_score(yy, pp):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.7)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", aspect="equal")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "ModelA_NvsM_03_final_model_ROC")

    # 04 confusion
    pred = (pp >= locked_threshold).astype(int)
    cm = confusion_matrix(yy, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.2, 3))
    fig.patch.set_alpha(1.0)
    draw_opaque_confusion_matrix(ax, cm, ["N", "M"], ["N", "M"])
    ax.set(xlabel="Predicted", ylabel="Observed")
    save_figure(fig, "ModelA_NvsM_04_confusion_matrix")

    # 05 distribution
    fig, ax = plt.subplots(figsize=(4.1, 3))
    data = [pp[yy == 0], pp[yy == 1]]
    vp = ax.violinplot(data, positions=[0, 1], widths=.72, showextrema=False)
    for body, color in zip(vp["bodies"], [neg, pos]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.45)
    rng = np.random.default_rng(SEED)
    ax.scatter(rng.normal(0, .04, len(data[0])), data[0], s=7, color=neg, alpha=.75, linewidth=0)
    ax.scatter(rng.normal(1, .04, len(data[1])), data[1], s=7, color=pos, alpha=.75, linewidth=0)
    for position, values in enumerate(data):
        ax.plot(
            [position - .18, position + .18],
            [np.median(values), np.median(values)],
            color=TEXT_DARK,
            lw=1.2,
        )
    ax.axhline(locked_threshold, ls="--", color=TEXT_DARK, lw=0.8)
    ax.set_xticks([0, 1], ["N", "M"])
    ax.set(ylabel="Predicted probability of M")
    ax.grid(axis="y", color="#E4E9ED", lw=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelA_NvsM_05_score_distribution")

    # 06 waterfall
    order = np.argsort(pp)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = np.where(yy[order] == 1, pos, neg)
    ax.bar(np.arange(len(pp)), pp[order], color=colors, width=.9, linewidth=0)
    ax.axhline(locked_threshold, ls="--", color=TEXT_DARK, lw=.8)
    ax.set(xlabel="Samples ranked by model score", ylabel="Predicted probability of M")
    ax.grid(axis="y", color="#E4E9ED", lw=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelA_NvsM_06_score_waterfall")

    # 07 stable metrics
    metric_order = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    means = repeat_metrics[metric_order].mean()
    lows = repeat_metrics[metric_order].quantile(.025)
    highs = repeat_metrics[metric_order].quantile(.975)
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    y_pos = np.arange(len(metric_order))
    ax.barh(y_pos, means, height=.56, color=SIGNAL_BLUE, edgecolor="none")
    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lows, highs - means],
        fmt="none",
        ecolor=TEXT_DARK,
        capsize=4,
        lw=1.2,
    )
    for yy_pos, value, high in zip(y_pos, means, highs):
        ax.text(min(high + .01, 1.01), yy_pos, f"{value:.3f}", va="center", ha="left")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlim(.55, 1.03)
    ax.set_xlabel("Mean with 95% CI")
    ax.grid(axis="x", color="#E4E9ED", lw=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, "ModelA_NvsM_07_stable_metrics")

    # 08 SHAP explanation of the locked full-data pipeline; descriptive, not validation evidence.
    import shap

    X_frame = pd.DataFrame(X, columns=feature_names)

    def predict_prob(array: np.ndarray) -> np.ndarray:
        return final_pipeline.predict_proba(np.asarray(array, dtype=float))[:, 1]

    background = shap.kmeans(X_frame, min(30, len(X_frame)))
    explainer = shap.KernelExplainer(predict_prob, background)
    shap_values = explainer.shap_values(X_frame, nsamples=240, silent=True)
    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[-1])
    else:
        shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    rows = []
    for index, feature in enumerate(feature_names):
        values = X_frame[feature].to_numpy(dtype=float)
        feature_shap = shap_values[:, index]
        ok = np.isfinite(values) & np.isfinite(feature_shap)
        corr = np.corrcoef(values[ok], feature_shap[ok])[0, 1] if ok.sum() >= 3 else np.nan
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.nanmean(np.abs(feature_shap))),
                "mean_shap": float(np.nanmean(feature_shap)),
                "value_shap_corr": float(corr) if np.isfinite(corr) else np.nan,
                "direction_by_value": "higher_value_toward_M" if corr >= 0 else "higher_value_toward_N",
            }
        )
    importance = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=True)
    importance.to_csv(OUT / "ModelA_NvsM_SHAP_feature_importance.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(5.1, 3.85))
    ax.barh(
        importance["feature"],
        importance["mean_abs_shap"],
        color=np.where(importance["direction_by_value"].eq("higher_value_toward_M"), pos, neg),
        edgecolor="none",
        height=.72,
    )
    ax.set_xlabel("Mean |SHAP|", fontsize=9.2)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E4E9ED", lw=0.7)
    ax.tick_params(axis="both", labelsize=8.7, length=3, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C0C8")
    ax.spines["bottom"].set_color("#B8C0C8")
    ax.set_axisbelow(True)
    save_figure(fig, "ModelA_NvsM_08_feature_importance")

    # 09 calibration
    frac, mean = calibration_curve(yy, pp, n_bins=8, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": mean, "observed_fraction": frac}).to_csv(
        OUT / "ModelA_NvsM_09_calibration_curve_data.tsv", sep="\t", index=False
    )
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=.7)
    ax.plot(mean, frac, marker="o", color=pos, lw=1.4)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed malignant fraction", aspect="equal")
    save_figure(fig, "ModelA_NvsM_09_calibration_curve")

    # 10 threshold diagnostics
    rows = []
    for threshold in np.linspace(.05, .95, 181):
        rows.append({"threshold": threshold, **metrics(yy, pp, threshold)})
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(OUT / "ModelA_NvsM_10_threshold_diagnostics_data.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4.1, 3))
    ax.plot(threshold_df["threshold"], threshold_df["sensitivity"], color=pos, label="Sensitivity")
    ax.plot(threshold_df["threshold"], threshold_df["specificity"], color=neg, label="Specificity")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], color=GROUP_COLORS["BD"], label="F1")
    ax.axvline(locked_threshold, color=TEXT_DARK, ls="--", lw=1, label=f"Internal Youden = {locked_threshold:.3f}")
    ax.axvline(.5, color="#888888", ls=":", lw=1, label="Reference = 0.5")
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(0, 1.03))
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save_figure(fig, "ModelA_NvsM_10_threshold_diagnostics")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, X, y, feature_names = load_data()
    predictions, selection, failures = nested_cv(X, y, feature_names)
    predictions.to_csv(OUT / "ModelA_nested_outer_predictions.tsv", sep="\t", index=False)
    selection.to_csv(OUT / "ModelA_nested_outer_selection.tsv", sep="\t", index=False)
    failures.to_csv(OUT / "ModelA_nested_failures.tsv", sep="\t", index=False)
    procedure_metrics = selection_procedure_repeat_metrics(predictions)
    procedure_metrics.to_csv(
        OUT / "ModelA_fully_nested_selection_procedure_repeat_metrics.tsv", sep="\t", index=False
    )
    (
        predictions.drop_duplicates(["repeat", "fold"]).groupby(
            ["selected_model", "selected_n_features"], as_index=False
        )
        .agg(outer_fold_wins=("fold", "size"))
        .sort_values("outer_fold_wins", ascending=False)
        .to_csv(OUT / "ModelA_nested_outer_winner_frequency.tsv", sep="\t", index=False)
    )
    (
        selection.groupby(
            ["model_key", "model_label", "n_features", "best_params_json"], as_index=False
        )
        .agg(outer_fold_frequency=("fold", "size"))
        .sort_values(["model_key", "n_features", "outer_fold_frequency"], ascending=[True, True, False])
        .to_csv(OUT / "ModelA_nested_best_params_frequency.tsv", sep="\t", index=False)
    )
    grid_manifest = {
        model_key: {"parameter_grid": grid, "feature_counts": feature_counts_for_model(model_key)}
        for model_key, (_, grid) in model_specs().items()
    }
    (OUT / "ModelA_grid_search_manifest.json").write_text(
        json.dumps(grid_manifest, indent=2), encoding="utf-8"
    )

    model_key, n_features, screening = choose_locked_configuration(selection)
    screening.to_csv(OUT / "ModelA_nested_model_screening_summary.tsv", sep="\t", index=False)
    final_pipeline, best_params = build_locked_configuration(selection, model_key, n_features)
    locked_raw, locked_threshold, repeat_metrics = locked_oof_predictions(X, y, final_pipeline)
    locked_raw.to_csv(OUT / "ModelA_locked_configuration_outer_predictions.tsv", sep="\t", index=False)
    repeat_metrics.to_csv(OUT / "ModelA_nested_repeat_metrics.tsv", sep="\t", index=False)

    avg = (
        locked_raw.groupby(["sample_index", "true_label"], as_index=False)
        .agg(probability=("probability", "mean"), prediction_count=("probability", "size"))
    )
    avg["prediction_at_locked_threshold"] = (avg["probability"] >= locked_threshold).astype(int)
    avg.to_csv(OUT / "sample_predictions_nested_cv.tsv", sep="\t", index=False)
    compatibility_predictions = avg.copy()
    compatibility_predictions["Alignment ID"] = compatibility_predictions["sample_index"].map(
        df["Alignment ID"].to_dict()
    )
    compatibility_predictions["true_group"] = np.where(
        compatibility_predictions["true_label"].eq(1), "M", "N"
    )
    compatibility_predictions["predicted_group"] = np.where(
        compatibility_predictions["probability"].ge(locked_threshold), "M", "N"
    )
    compatibility_predictions["threshold"] = locked_threshold
    compatibility_columns = [
        "Alignment ID",
        "true_group",
        "probability",
        "predicted_group",
        "threshold",
        "prediction_count",
    ]
    compatibility_predictions[compatibility_columns].to_csv(OUT / "sample_predictions.csv", index=False)
    compatibility_predictions.loc[
        compatibility_predictions["true_group"].ne(compatibility_predictions["predicted_group"])
    ].to_csv(OUT / "error_predictions.csv", index=False)

    final_pipeline.fit(X, y)
    selected = [
        feature_names[i] for i, keep in enumerate(final_pipeline.named_steps["selector"].support_) if keep
    ]
    package = {
        "pipeline": final_pipeline,
        "feature_names": feature_names,
        "selected_features": selected,
        "selected_model_key": model_key,
        "selected_model_label": MODEL_LABELS[model_key],
        "selected_n_features": n_features,
        "final_feature_constraint": "All eight standard-supported inputs are required",
        "best_params": clean_params(best_params),
        "internally_derived_youden_threshold": locked_threshold,
        "reference_threshold": 0.5,
        "validation_protocol": {
            "samples": 222,
            "normal": 100,
            "malignant": 122,
            "outer": "RepeatedStratifiedKFold, 5 folds x 20 repeats",
            "inner": "StratifiedKFold, 5 folds",
            "seed": SEED,
            "protocol_version": PROTOCOL_VERSION,
            "threshold": "Youden threshold derived from inner OOF predictions only",
            "external_validation": False,
            "ai_assisted_screening": (
                "Classical statistical, kernel, tree-based, boosting, neural-network "
                "and ensemble learners compared under the same nested-CV protocol"
            ),
        },
    }
    with open(OUT / "final_model.pkl", "wb") as handle:
        pickle.dump(package, handle)
    with open(OUT / "ModelA_nested_cv_summary.json", "w", encoding="utf-8") as handle:
        nested_summary = {
            **{k: v for k, v in package.items() if k != "pipeline"},
            "nested_cv_metrics_mean": repeat_metrics.mean(numeric_only=True).to_dict(),
            "nested_cv_metrics_ci_low": repeat_metrics.quantile(.025, numeric_only=True).to_dict(),
            "nested_cv_metrics_ci_high": repeat_metrics.quantile(.975, numeric_only=True).to_dict(),
            "fully_nested_selection_procedure_metrics_mean": procedure_metrics.mean(
                numeric_only=True
            ).to_dict(),
            "fully_nested_selection_procedure_metrics_ci_low": procedure_metrics.quantile(
                .025, numeric_only=True
            ).to_dict(),
            "fully_nested_selection_procedure_metrics_ci_high": procedure_metrics.quantile(
                .975, numeric_only=True
            ).to_dict(),
            "runtime_minutes": (time.time() - started) / 60,
        }
        json.dump(nested_summary, handle, indent=2)
    compatibility_summary = {
        "analysis_status": "current leakage-controlled repeated nested internal cross-validation",
        "selected_model": package["selected_model_label"],
        "selected_feature_count": package["selected_n_features"],
        "selected_features": package["selected_features"],
        "internally_derived_youden_threshold": package["internally_derived_youden_threshold"],
        "reference_threshold": package["reference_threshold"],
        "validation_protocol": package["validation_protocol"],
        "nested_cv_metrics_mean": nested_summary["nested_cv_metrics_mean"],
        "nested_cv_metrics_ci_low": nested_summary["nested_cv_metrics_ci_low"],
        "nested_cv_metrics_ci_high": nested_summary["nested_cv_metrics_ci_high"],
        "warning": "No external validation; do not use legacy holdout-test wording.",
    }
    (OUT / "integrated_results.json").write_text(
        json.dumps(compatibility_summary, indent=2), encoding="utf-8"
    )
    legacy_shap = OUT / "14_shap_direction_summary.csv"
    if legacy_shap.exists():
        legacy_shap.unlink()

    plot_all(
        X,
        y,
        feature_names,
        screening,
        locked_raw,
        repeat_metrics,
        locked_threshold,
        final_pipeline,
        model_key,
    )

    sync_lines = [
        "# ModelA manuscript numbers requiring synchronization",
        "",
        "- Validation wording: repeated nested internal cross-validation; no external validation.",
        f"- Selected model: {MODEL_LABELS[model_key]}.",
        f"- Selected feature count: {n_features}.",
        f"- Selected features: {', '.join(selected)}.",
        f"- Internally derived Youden threshold: {locked_threshold:.6f}.",
        f"- Mean repeated nested-CV AUC: {repeat_metrics['roc_auc'].mean():.3f}.",
        f"- Mean repeated nested-CV PR AUC: {repeat_metrics['pr_auc'].mean():.3f}.",
        f"- Fully nested model-selection procedure AUC: {procedure_metrics['roc_auc'].mean():.3f}.",
        f"- Fully nested model-selection procedure PR AUC: {procedure_metrics['pr_auc'].mean():.3f}.",
        f"- Mean sensitivity: {repeat_metrics['sensitivity'].mean():.3f}.",
        f"- Mean specificity: {repeat_metrics['specificity'].mean():.3f}.",
        "- The selected-configuration estimate may retain modest configuration-selection optimism because the final configuration was chosen after outer-CV comparison.",
        "- Threshold 0.5 is retained only as a reference sensitivity analysis.",
    ]
    (OUT / "ModelA_manuscript_numbers_to_update.md").write_text("\n".join(sync_lines), encoding="utf-8")
    expected_evaluations = OUTER_SPLITS * OUTER_REPEATS * sum(
        len(feature_counts_for_model(model_key)) for model_key in model_specs()
    )
    evaluated_ai_models = sorted(
        set(selection.loc[selection["model_key"].isin(AI_MODEL_KEYS), "model_label"])
    )
    qa = [
        "# ModelA nested-CV QA",
        "",
        f"- Outer prediction rows: {len(locked_raw)} (expected 222 x 20 = 4440).",
        f"- Predictions per sample: min={avg.prediction_count.min()}, max={avg.prediction_count.max()}.",
        f"- Completed outer folds: {selection[['repeat', 'fold']].drop_duplicates().shape[0]} (expected 100).",
        f"- Successful candidate model/feature-count evaluations: {len(selection)} (expected {expected_evaluations:,} before fit failures).",
        f"- Failed model/fold fits recorded: {len(failures)}.",
        "- Pipeline order: median imputation -> StandardScaler -> RFE -> classifier.",
        "- RFE, scaling, imputation, hyperparameter tuning and threshold derivation are restricted to training folds.",
        f"- Formal AI/ensemble candidates evaluated: {', '.join(evaluated_ai_models) if evaluated_ai_models else 'none completed'}.",
        f"- Fully nested model-selection procedure mean AUC: {procedure_metrics['roc_auc'].mean():.6f}.",
        f"- Selected locked-configuration repeated-CV mean AUC: {repeat_metrics['roc_auc'].mean():.6f}.",
        "- The fully nested procedure estimate is the least biased estimate of the full model-selection workflow.",
        "- The locked-configuration estimate may retain modest configuration-selection optimism.",
        "- The internally derived threshold is not externally validated.",
    ]
    (OUT / "ModelA_nested_cv_QA.md").write_text("\n".join(qa), encoding="utf-8")
    print("\n".join(sync_lines))


if __name__ == "__main__":
    main()
