from __future__ import annotations

import math
import re
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent
ANALYSIS_REQUESTED = OUT.parent
TARGET_ROOT = ANALYSIS_REQUESTED.parent
ROOT = TARGET_ROOT.parent
STYLE_DIR = ROOT / "00_project_style"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
from ov_publication_style import (
    COHORT_COLORS,
    GROUP_COLORS,
    HEATMAP_BLUE,
    HEATMAP_RED,
    HEATMAP_WHITE,
    SIGNAL_BLUE,
    SIGNAL_RED,
    setup_matplotlib_style,
)
TABLES = OUT / "tables"
FIG_ROOT = OUT / "figures"
FIG_PDF = FIG_ROOT / "pdf"
FIG_SVG = FIG_ROOT / "svg"
FIG_PNG = FIG_ROOT / "png_preview"
REPORTS = OUT / "reports"

REAL_ANALYSIS = ANALYSIS_REQUESTED / "real_sample_area_ratio_differential_analysis"
REAL_TABLES = REAL_ANALYSIS / "tables"
REAL_FIGURES = REAL_ANALYSIS / "figures"
SCORE_TSV = ANALYSIS_REQUESTED / "NvsM_batch1_model_score_N_B_BD_M" / "tables" / "all_sample_NBBDM_model_scores.tsv"
CLINICAL_XLSX = TARGET_ROOT / "clinical_merged_resolved.xlsx"
FULL_CLINICAL_XLSX = CLINICAL_XLSX

ANALYTES = ["3-GPA", "Acetylcarnitine", "Creatine", "DHEA-S", "Arginine", "Carnitine", "Phenylalanine", "Tryptophan"]
GROUP4_ORDER = ["N", "B", "BD", "M"]
GROUP4_LABELS = {"N": "Normal", "B": "Benign", "BD": "Borderline", "M": "Malignant"}
GROUP4_PALETTE = {key: GROUP_COLORS[key] for key in GROUP4_ORDER}
GROUP3_ORDER = ["Normal controls", "Benign/borderline lesions", "Malignant tumours"]
GROUP3_PALETTE = {
    "Normal controls": GROUP_COLORS["N"],
    "Benign/borderline lesions": "#FDB462",
    "Malignant tumours": GROUP_COLORS["M"],
}
COHORT_DISPLAY_LABELS = {
    "Batch1": "exploratory cohort",
    "Batch2": "independent validation cohort",
}


def setup_style() -> None:
    setup_matplotlib_style(mpl, sns=sns, base_size=7)


def prepare_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if FIG_ROOT.exists():
        for child in FIG_ROOT.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    FIG_PDF.mkdir(parents=True, exist_ok=True)
    FIG_SVG.mkdir(parents=True, exist_ok=True)
    FIG_PNG.mkdir(parents=True, exist_ok=True)
    gatm = OUT / "GATM_deep_dive"
    if gatm.exists():
        shutil.rmtree(gatm)
    for stale in [
        "targeted_score_PFS_cutpoint_scan.tsv",
        "targeted_score_PFS_exploratory.tsv",
        "targeted_score_clinical_correlations.tsv",
        "targeted_all_sample_heatmap_clinical_annotations.tsv",
        "targeted_all_sample_heatmap_zscore_matrix.tsv",
    ]:
        p = TABLES / stale
        if p.exists():
            p.unlink()


def save_fig(fig: plt.Figure, stem: str, width: float | None = None, height: float | None = None) -> None:
    if width and height:
        fig.set_size_inches(width, height)
    fig.savefig(FIG_PDF / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_SVG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_PNG / f"{stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def norm_id(x) -> str | float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def safe_numeric(x, index=None) -> pd.Series:
    if x is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(x, errors="coerce")


def clean_event(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "/", "否", "无", "未复发", "未转移", "未复发/转移", "0", "0.0"}:
        return 0.0
    if text in {"是", "转移", "复发", "复发/转移", "死亡", "1", "1.0"}:
        return 1.0
    return 1.0


def clean_stage_group(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    roman = {"1": "I", "2": "II", "3": "III", "4": "IV", "I": "I", "II": "II", "III": "III", "IV": "IV"}
    if text in roman:
        return roman[text]
    if "复发" in text or "Recurrence" in text:
        return "Recurrence"
    if text in {"", "NA", "NaN", "nan", "/"}:
        return np.nan
    return "Other/Unclear"


def stage_binary(stage_group):
    if stage_group in {"I", "II"}:
        return "Early stage"
    if stage_group in {"III", "IV"}:
        return "Advanced stage"
    return np.nan


def group_label_from_code(value):
    text = str(value).strip() if not pd.isna(value) else ""
    return {"N": "Normal", "B": "Benign", "BD": "Borderline", "M": "Malignant"}.get(text, np.nan)


def standardize_targeted_clinical_table(clinical: pd.DataFrame) -> pd.DataFrame:
    """Convert the targeted clinical workbook into the analysis field schema."""
    out = pd.DataFrame()
    out["sample_id_norm"] = clinical["Alignment ID"].map(norm_id)
    idx = clinical.index
    out["age"] = safe_numeric(clinical.get("年龄"), idx)
    out["CA125"] = safe_numeric(clinical.get("CA125_value"), idx)
    out["HE4"] = safe_numeric(clinical.get("HE4_value"), idx)
    out["CA125_log10"] = np.nan
    out.loc[out["CA125"] > 0, "CA125_log10"] = np.log10(out.loc[out["CA125"] > 0, "CA125"])
    out["HE4_log10"] = np.nan
    out.loc[out["HE4"] > 0, "HE4_log10"] = np.log10(out.loc[out["HE4"] > 0, "HE4"])
    out["group_label"] = clinical.get("良恶性").map(group_label_from_code)
    out["figo_stage_merged"] = clinical.get("分期")
    out["stage_group"] = clinical.get("分期").map(clean_stage_group)
    out["stage_binary"] = out["stage_group"].map(stage_binary)
    out["pathology_subtype"] = clinical.get("病理类型（细分）")
    out["pathology_class"] = clinical.get("病理类型")
    out["pathology_subtype_abbr"] = clinical.get("病理类型（细分）")
    out["pfs_event"] = clinical.get("是否复发/转移").map(clean_event)
    out["pfs_time"] = safe_numeric(clinical.get("PFS（复发时间-手术时间）_num"), idx)
    out["os_event"] = clinical.get("是否死亡").map(clean_event)
    out["os_time"] = safe_numeric(clinical.get("OS（死亡时间-手术时间）_num"), idx)
    return out.drop_duplicates("sample_id_norm")


def bh_fdr(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    mask = pvals.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvals.loc[mask].astype(float), method="fdr_bh")[1]
    return out


def classify_variable(s: pd.Series) -> str:
    vals = s.dropna()
    if vals.empty:
        return "empty"
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().mean() >= 0.8 and numeric.nunique() > 6:
        return "continuous"
    if vals.nunique() == 2:
        return "binary_categorical"
    return "categorical"


def km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"time": time, "event": event}).dropna().sort_values("time")
    at_risk = len(df)
    surv = 1.0
    xs, ys = [0.0], [1.0]
    for t, block in df.groupby("time", sort=True):
        d = int(block["event"].sum())
        if at_risk > 0 and d > 0:
            xs.extend([float(t), float(t)])
            ys.extend([surv, surv * (1 - d / at_risk)])
            surv *= 1 - d / at_risk
        at_risk -= len(block)
    return np.array(xs), np.array(ys)


def logrank_two_group(time: pd.Series, event: pd.Series, group_high: pd.Series) -> tuple[float, float]:
    df = pd.DataFrame({"time": time, "event": event, "high": group_high}).dropna()
    if df["high"].nunique() != 2 or df["event"].sum() == 0:
        return np.nan, np.nan
    observed = expected = variance = 0.0
    for t in sorted(df.loc[df["event"] == 1, "time"].unique()):
        risk = df[df["time"] >= t]
        events = df[(df["time"] == t) & (df["event"] == 1)]
        n, n1, d, d1 = len(risk), int(risk["high"].sum()), len(events), int(events["high"].sum())
        if n <= 1:
            continue
        observed += d1
        expected += d * n1 / n
        variance += (n1 / n) * (1 - n1 / n) * d * (n - d) / (n - 1)
    if variance <= 0:
        return np.nan, np.nan
    chi2 = (observed - expected) ** 2 / variance
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def cox_univariable(time: pd.Series, event: pd.Series, x: pd.Series) -> dict:
    df = pd.DataFrame({"time": time, "event": event, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 15 or df["event"].sum() < 5 or df["x"].nunique() < 3:
        return {"n": len(df), "events": int(df["event"].sum()) if len(df) else 0, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    z = (df["x"] - df["x"].mean()) / df["x"].std(ddof=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PHReg(df["time"], z.to_frame("x"), status=df["event"]).fit(disp=False)
        beta = float(fit.params[0])
        se = float(fit.bse[0])
        return {"n": len(df), "events": int(df["event"].sum()), "hr": math.exp(beta), "ci_low": math.exp(beta - 1.96 * se), "ci_high": math.exp(beta + 1.96 * se), "p_value": float(fit.pvalues[0])}
    except Exception:
        return {"n": len(df), "events": int(df["event"].sum()), "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_path = REAL_TABLES / "real_sample_long_clean.tsv"
    if not long_path.exists():
        long_path = TABLES / "integrated_targeted_area_ratio_long.tsv"
    long = pd.read_csv(long_path, sep="\t")
    if SCORE_TSV.exists():
        score = pd.read_csv(SCORE_TSV, sep="\t")
    else:
        existing_integrated = TABLES / "integrated_targeted_clinical_dataset.tsv"
        if not existing_integrated.exists():
            raise FileNotFoundError(f"Neither score source exists: {SCORE_TSV} or {existing_integrated}")
        score = pd.read_csv(existing_integrated, sep="\t")
    clinical = pd.read_excel(CLINICAL_XLSX)
    return long, score, clinical


def build_integrated_dataset(long: pd.DataFrame, score: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    score = score.copy()
    score["sample_id_norm"] = score["Sample Name"].map(norm_id)
    score["cohort_role"] = np.where(score["batch_display"].eq("Batch1"), "exploratory cohort", "independent validation cohort")
    score["three_group_display"] = score["group_display"].map({"N": "Normal controls", "B": "Benign/borderline lesions", "BD": "Benign/borderline lesions", "M": "Malignant tumours"})

    clinical = standardize_targeted_clinical_table(clinical)
    keep = ["sample_id_norm", "age", "CA125", "CA125_log10", "HE4", "HE4_log10", "group_label", "figo_stage_merged", "pathology_subtype", "pathology_class", "pathology_subtype_abbr", "stage_group", "stage_binary", "pfs_event", "pfs_time", "os_event", "os_time"]
    clinical = clinical[[c for c in keep if c in clinical.columns]].drop_duplicates("sample_id_norm")
    duplicate_clinical_cols = [c for c in clinical.columns if c != "sample_id_norm" and c in score.columns]
    if duplicate_clinical_cols:
        score = score.drop(columns=duplicate_clinical_cols)
    merged = score.merge(clinical, on="sample_id_norm", how="left", suffixes=("", "_clinical"))
    clinical_cols = [c for c in clinical.columns if c != "sample_id_norm"]
    merged.loc[~merged["batch_display"].eq("Batch1"), clinical_cols] = np.nan

    long = long.rename(columns={"Component Label": "analyte"}).copy()
    long["sample_id_norm"] = long["Sample Name"].map(norm_id)
    long.to_csv(TABLES / "integrated_targeted_area_ratio_long.tsv", sep="\t", index=False)
    merged.to_csv(TABLES / "integrated_targeted_clinical_dataset.tsv", sep="\t", index=False)
    return merged


def plot_01_effect_size() -> None:
    effects_path = REAL_TABLES / "figure_data_07_NvsM_effect_size_by_batch.tsv"
    if not effects_path.exists():
        effects_path = TABLES / "targeted_NvsM_effect_size_by_batch.tsv"
    effects = pd.read_csv(effects_path, sep="\t")
    effects = effects[effects["Component Label"].isin(ANALYTES)].copy()
    order = effects.groupby("Component Label")["log2FC_M_minus_N"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    offsets = {"Batch1": -0.16, "Batch2": 0.16}
    colors = {"Batch1": COHORT_COLORS["Exploratory cohort"], "Batch2": COHORT_COLORS["Independent validation cohort"]}
    ybase = np.arange(len(order))
    for batch in ["Batch1", "Batch2"]:
        sub = effects[effects["batch_display"] == batch].set_index("Component Label").reindex(order).reset_index()
        y = ybase + offsets[batch]
        ax.errorbar(
            sub["log2FC_M_minus_N"],
            y,
            xerr=[sub["log2FC_M_minus_N"] - sub["ci95_low"], sub["ci95_high"] - sub["log2FC_M_minus_N"]],
            fmt="o",
            color=colors[batch],
            ecolor=colors[batch],
            markersize=7.8,
            elinewidth=1.25,
            capthick=1.15,
            capsize=2,
            label=COHORT_DISPLAY_LABELS[batch],
        )
    ax.axvline(0, color="#111827", lw=0.9, ls="--")
    ax.set_yticks(ybase)
    ax.set_yticklabels(order, fontsize=11.4)
    ax.set_xlabel("M - N mean difference in log2(area ratio)")
    ax.set_title("N vs M effect size by cohort", loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelsize=10.6)
    ax.xaxis.label.set_size(11.8)
    ax.title.set_size(13.2)
    ax.legend(frameon=False, fontsize=10.6, handlelength=1.4, handletextpad=0.45)
    ax.grid(axis="x", color="#E5E7EB")
    save_fig(fig, "01_targeted_NvsM_effect_size_by_cohort", 8.2, 5.4)
    effects.to_csv(TABLES / "targeted_NvsM_effect_size_by_batch.tsv", sep="\t", index=False)


def plot_03_plsda() -> None:
    pls_path = REAL_TABLES / "figure_data_10_exploratory_PLSDA.tsv"
    if not pls_path.exists():
        pls_path = TABLES / "targeted_exploratory_PLSDA_NvsM.tsv"
    pls = pd.read_csv(pls_path, sep="\t")
    subset = pls[pls["subset"].astype(str).eq("N_vs_M")].copy()
    if subset.empty:
        subset = pls.copy()
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    markers = {"Batch1": "o", "Batch2": "^"}
    colors = {"N": GROUP_COLORS["N"], "M": GROUP_COLORS["M"]}
    for batch in ["Batch1", "Batch2"]:
        for group in ["N", "M"]:
            g = subset[(subset["batch_display"] == batch) & (subset["group_display"] == group)]
            if g.empty:
                continue
            ax.scatter(g["PLS1"], g["PLS2"], s=30, marker=markers[batch], color=colors[group], edgecolor="white", linewidth=0.4, alpha=0.78)
    group_handles = [mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[g], markeredgecolor="white", markersize=7, label=g) for g in ["N", "M"]]
    batch_handles = [mpl.lines.Line2D([0], [0], marker=markers[b], color="#374151", linestyle="none", markersize=7, label=COHORT_DISPLAY_LABELS[b]) for b in ["Batch1", "Batch2"]]
    ax.legend(handles=group_handles + batch_handles, frameon=False, ncol=2, fontsize=8, loc="best")
    ax.set_xlabel("PLS-DA component 1")
    ax.set_ylabel("PLS-DA component 2")
    ax.set_title("Combined-cohort PLS-DA: N vs M", loc="left", fontweight="bold")
    ax.grid(color="#E5E7EB")
    save_fig(fig, "02_targeted_exploratory_PLSDA_NvsM", 5.8, 5.3)
    subset.to_csv(TABLES / "targeted_exploratory_PLSDA_NvsM.tsv", sep="\t", index=False)


def plot_05_clinical_correlation(df: pd.DataFrame) -> None:
    sub = df[df["cohort_role"].eq("exploratory cohort")].copy()
    rows = []
    targets = ANALYTES + ["NvsM_model_score"]
    continuous_vars = ["CA125", "HE4"]
    categorical_vars = ["group_label", "stage_binary", "stage_group", "pathology_class", "pathology_subtype_abbr"]
    clinical_labels = {
        "CA125": "CA125",
        "HE4": "HE4",
        "group_label": "Disease group",
        "stage_binary": "Early vs advanced",
        "stage_group": "FIGO stage",
        "pathology_class": "Pathology class",
        "pathology_subtype_abbr": "Pathology subtype",
    }
    for c in continuous_vars:
        if c not in sub.columns:
            continue
        for y in targets:
            tmp = sub[[c, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) >= 8 and tmp[c].nunique() >= 3 and tmp[y].nunique() >= 3:
                rho, p = stats.spearmanr(tmp[c], tmp[y])
            else:
                rho, p = np.nan, np.nan
            rows.append({"clinical_variable": c, "clinical_label": clinical_labels[c], "analyte": y, "n": len(tmp), "association": rho, "p_value": p, "test": "Spearman"})
    for c in categorical_vars:
        if c not in sub.columns:
            continue
        for y in targets:
            tmp = sub[[c, y]].replace([np.inf, -np.inf], np.nan).dropna()
            counts = tmp[c].value_counts()
            keep = counts[counts >= 3].index
            tmp = tmp[tmp[c].isin(keep)]
            levels = list(tmp[c].dropna().unique())
            assoc = p = np.nan
            test = "not_tested"
            if len(levels) == 2:
                levels = sorted(levels, key=lambda v: str(v))
                a = tmp.loc[tmp[c].eq(levels[0]), y].astype(float)
                b = tmp.loc[tmp[c].eq(levels[1]), y].astype(float)
                if len(a) >= 3 and len(b) >= 3:
                    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    auc = u / (len(a) * len(b))
                    assoc = 2 * auc - 1
                    test = f"Mann-Whitney; positive={levels[0]}"
            elif len(levels) > 2:
                groups = [tmp.loc[tmp[c].eq(level), y].astype(float) for level in levels]
                groups = [g for g in groups if len(g) >= 3]
                if len(groups) > 2:
                    h, p = stats.kruskal(*groups)
                    assoc = h / max(len(tmp) - 1, 1)
                    test = "Kruskal-Wallis epsilon2"
            rows.append({"clinical_variable": c, "clinical_label": clinical_labels[c], "analyte": y, "n": len(tmp), "association": assoc, "p_value": p, "test": test})
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_fdr(out["p_value"])
    out.to_csv(TABLES / "targeted_analyte_clinical_correlations.tsv", sep="\t", index=False)
    ordered_labels = [clinical_labels[c] for c in continuous_vars + categorical_vars if c in sub.columns]
    heat = out.pivot(index="analyte", columns="clinical_label", values="association").reindex(index=targets, columns=ordered_labels)
    fdr = out.pivot(index="analyte", columns="clinical_label", values="fdr_bh").reindex(index=targets, columns=ordered_labels)
    annot = heat.copy().astype(object)
    for r in annot.index:
        for c in annot.columns:
            val = heat.loc[r, c]
            if pd.isna(val):
                annot.loc[r, c] = ""
                continue
            q = fdr.loc[r, c]
            if pd.notna(q) and q < 0.001:
                star = "***"
            elif pd.notna(q) and q < 0.01:
                star = "**"
            elif pd.notna(q) and q < 0.05:
                star = "*"
            else:
                star = ""
            annot.loc[r, c] = f"{val:.2f}{star}"
    deep_red_blue = LinearSegmentedColormap.from_list("deep_red_blue", [HEATMAP_BLUE, HEATMAP_WHITE, HEATMAP_RED])
    fig, ax = plt.subplots(figsize=(6.2, 4.35))
    sns.heatmap(heat, cmap=deep_red_blue, center=0, vmin=-1, vmax=1, annot=annot, fmt="", linewidths=0.35, cbar_kws={"label": "Association statistic"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([t.get_text().replace("NvsM_model_score", "Model score") for t in ax.get_yticklabels()], rotation=0)
    ax.tick_params(axis="x", labelrotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
        tick_label.set_va("top")
        tick_label.set_rotation_mode("anchor")
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, "03_targeted_analyte_clinical_correlation_heatmap", 6.2, 4.35)


def audit_full_batch1_clinical_fields(df: pd.DataFrame) -> None:
    if not FULL_CLINICAL_XLSX.exists():
        return
    batch1_ids = set(df.loc[df["batch_display"].eq("Batch1"), "Sample Name"].map(norm_id).dropna().astype(str))
    full = pd.read_excel(FULL_CLINICAL_XLSX)
    if "Alignment ID" not in full.columns:
        return
    full["sample_id_norm"] = full["Alignment ID"].map(norm_id).astype(str)
    full = full[full["sample_id_norm"].isin(batch1_ids)].copy()
    pii_like = {"姓名", "电话", "登记号", "登记号_raw", "reg_norm", "reg_id"}
    rows = []
    for c in full.columns:
        if c in pii_like or c.endswith("_raw") or c in ["sample_id_norm"]:
            continue
        vals = full[c]
        rows.append(
            {
                "field": c,
                "available_n": int(vals.notna().sum()),
                "total_n": len(full),
                "available_pct": float(vals.notna().mean() * 100) if len(full) else np.nan,
                "unique_n": int(vals.dropna().nunique()),
                "in_current_integrated_figure04": c in ["age", "CA125", "HE4", "figo_stage_merged", "stage_group", "pathology_subtype", "pfs_time", "os_time"],
                "variable_type_guess": classify_variable(vals),
                "example_values": "; ".join(vals.dropna().astype(str).unique()[:5]),
            }
        )
    pd.DataFrame(rows).sort_values(["available_n", "field"], ascending=[False, True]).to_csv(TABLES / "batch1_full_clinical_field_availability_audit.tsv", sep="\t", index=False)


def categorical_clinical_tests(df: pd.DataFrame) -> None:
    sub = df[df["cohort_role"].eq("exploratory cohort")].copy()
    targets = ANALYTES + ["NvsM_model_score"]
    factors = ["group_display", "stage_binary", "stage_group", "pathology_class", "pathology_subtype_abbr"]
    rows = []
    for factor in factors:
        if factor not in sub.columns:
            continue
        for y in targets:
            tmp = sub[[factor, y]].replace([np.inf, -np.inf], np.nan).dropna()
            counts = tmp[factor].value_counts()
            keep_levels = counts[counts >= 3].index
            tmp = tmp[tmp[factor].isin(keep_levels)]
            levels = list(tmp[factor].dropna().unique())
            groups = [tmp.loc[tmp[factor].eq(level), y].astype(float) for level in levels]
            test = "not_tested"
            stat = p = np.nan
            if len(groups) == 2:
                test = "Mann-Whitney U"
                stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            elif len(groups) > 2:
                test = "Kruskal-Wallis"
                stat, p = stats.kruskal(*groups)
            rows.append(
                {
                    "clinical_factor": factor,
                    "target_variable": y,
                    "test": test,
                    "n": len(tmp),
                    "levels_tested": "; ".join([f"{level}(n={len(group)})" for level, group in zip(levels, groups)]),
                    "statistic": stat,
                    "p_value": p,
                    "interpretation": "Exploratory categorical clinical association; not adjusted for clinical confounding.",
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_fdr(out["p_value"])
    out.to_csv(TABLES / "targeted_analyte_model_score_categorical_clinical_tests.tsv", sep="\t", index=False)


def plot_06_analyte_pfs(df: pd.DataFrame) -> None:
    sub = df[(df["cohort_role"].eq("exploratory cohort")) & (df["group_display"].eq("M"))].copy()
    rows = []
    for y in ANALYTES:
        cox = cox_univariable(sub["pfs_time"], sub["pfs_event"], sub[y])
        tmp = sub[["pfs_time", "pfs_event", y]].dropna()
        if len(tmp) >= 15 and tmp[y].nunique() >= 3:
            high = tmp[y] >= tmp[y].median()
            _, lr_p = logrank_two_group(tmp["pfs_time"], tmp["pfs_event"], high)
        else:
            lr_p = np.nan
        rows.append({"variable": y, **cox, "median_split_logrank_p": lr_p, "interpretation": "Exploratory malignant-only PFS context."})
    out = pd.DataFrame(rows)
    out["cox_fdr_bh"] = bh_fdr(out["p_value"])
    out["logrank_fdr_bh"] = bh_fdr(out["median_split_logrank_p"])
    out.to_csv(TABLES / "targeted_analyte_PFS_exploratory.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    plot_df = out.sort_values("hr")
    xerr_low = plot_df["hr"] - plot_df["ci_low"]
    xerr_high = plot_df["ci_high"] - plot_df["hr"]
    ax.errorbar(plot_df["hr"], range(len(plot_df)), xerr=[xerr_low, xerr_high], fmt="o", color="#555555", ecolor="#999999", markersize=3, linewidth=0.8)
    ax.axvline(1, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["variable"])
    ax.set_xlabel("Exploratory Cox HR per SD")
    save_fig(fig, "04_targeted_analyte_PFS_exploratory", 4.4, 2.8)


def plot_07_score_gradient(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, batch in zip(axes, ["Batch1", "Batch2"]):
        sub = df[df["batch_display"].eq(batch)]
        sns.violinplot(data=sub, x="three_group_display", y="NvsM_model_score", order=GROUP3_ORDER, hue="three_group_display", hue_order=GROUP3_ORDER, palette=GROUP3_PALETTE, inner=None, linewidth=0.7, cut=0, legend=False, ax=ax)
        sns.stripplot(data=sub, x="three_group_display", y="NvsM_model_score", order=GROUP3_ORDER, color="#2b2b2b", alpha=0.45, size=1.8, jitter=0.22, ax=ax)
        ax.set_title(COHORT_DISPLAY_LABELS[batch])
        ax.set_xlabel("")
        ax.set_ylabel("Model score" if ax is axes[0] else "")
        ax.set_xticks(range(len(GROUP3_ORDER)))
        ax.set_xticklabels(["Normal", "Benign/\nborderline", "Malignant"])
        ax.set_ylim(-0.03, 1.03)
    save_fig(fig, "05_model_score_three_group_gradient_B_BD_merged", 7.2, 3.0)


def plot_06_score_pfs(df: pd.DataFrame) -> None:
    sub = df[(df["cohort_role"].eq("exploratory cohort")) & (df["group_display"].eq("M"))].copy()
    cox = cox_univariable(sub["pfs_time"], sub["pfs_event"], sub["NvsM_model_score"])
    tmp = sub[["pfs_time", "pfs_event", "NvsM_model_score"]].dropna()
    lr_p = np.nan
    best_cut = np.nan
    best_label = "not_available"
    cut_rows = []
    if len(tmp) >= 15 and tmp["pfs_event"].sum() >= 5:
        candidates = np.unique(np.quantile(tmp["NvsM_model_score"], np.linspace(0.25, 0.75, 51)))
        min_group_n = max(10, int(len(tmp) * 0.15))
        for cut in candidates:
            high = tmp["NvsM_model_score"] >= cut
            if high.sum() < min_group_n or (~high).sum() < min_group_n:
                continue
            chi2, p = logrank_two_group(tmp["pfs_time"], tmp["pfs_event"], high)
            cut_rows.append({"cutpoint": cut, "n_low": int((~high).sum()), "n_high": int(high.sum()), "logrank_chi2": chi2, "nominal_p": p})
        cut_df = pd.DataFrame(cut_rows)
        if not cut_df.empty and cut_df["nominal_p"].notna().any():
            cut_df["fdr_bh_across_candidate_cutpoints"] = bh_fdr(cut_df["nominal_p"])
            best = cut_df.sort_values(["nominal_p", "cutpoint"]).iloc[0]
            best_cut = float(best["cutpoint"])
            lr_p = float(best["nominal_p"])
            best_label = "maximally_selected_nominal_logrank"
        else:
            best_cut = float(tmp["NvsM_model_score"].median())
            best_label = "median_fallback"
        tmp["score_group"] = np.where(tmp["NvsM_model_score"] >= best_cut, "High score", "Low score")
        if not cut_rows:
            _, lr_p = logrank_two_group(tmp["pfs_time"], tmp["pfs_event"], tmp["score_group"].eq("High score"))
            cut_df = pd.DataFrame([{"cutpoint": best_cut, "n_low": int((tmp["score_group"] == "Low score").sum()), "n_high": int((tmp["score_group"] == "High score").sum()), "logrank_chi2": np.nan, "nominal_p": lr_p, "fdr_bh_across_candidate_cutpoints": np.nan}])
        cut_df.to_csv(TABLES / "model_score_PFS_cutpoint_scan.tsv", sep="\t", index=False)
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        for label, color in [("Low score", SIGNAL_BLUE), ("High score", SIGNAL_RED)]:
            part = tmp[tmp["score_group"].eq(label)]
            x, y = km_curve(part["pfs_time"].to_numpy(float), part["pfs_event"].to_numpy(float))
            ax.step(x, y, where="post", label=f"{label.replace('score', 'model score')} (n={len(part)})", color=color, linewidth=1.3)
        ax.text(0.05, 0.12, f"Cut={best_cut:.3f}\nNominal p={lr_p:.3g}" if pd.notna(lr_p) else "Exploratory PFS", transform=ax.transAxes, fontsize=7.0)
        ax.set_xlabel("PFS time")
        ax.set_ylabel("Progression-free fraction")
        ax.tick_params(labelsize=7.0)
        ax.xaxis.label.set_size(7.8)
        ax.yaxis.label.set_size(7.8)
        ax.legend(frameon=False, loc="upper right", fontsize=6.8)
        save_fig(fig, "06_model_score_PFS_exploratory_optimized_cutpoint", 3.6, 2.8)
    pd.DataFrame([{**cox, "variable": "NvsM_model_score", "display_variable": "Model score", "selected_cutpoint": best_cut, "cutpoint_rule": best_label, "selected_cutpoint_logrank_p": lr_p, "interpretation": "Exploratory malignant-only PFS context; cutpoint is data-selected and not validated."}]).to_csv(TABLES / "model_score_PFS_exploratory.tsv", sep="\t", index=False)


def write_manifest_and_qa(df: pd.DataFrame) -> None:
    rows = []
    for fmt_dir, fmt in [(FIG_PDF, "pdf"), (FIG_SVG, "svg"), (FIG_PNG, "png_preview")]:
        for p in sorted(fmt_dir.glob("*")):
            rows.append({"figure_file": p.name, "format": fmt, "path": str(p), "bytes": p.stat().st_size})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(TABLES / "targeted_measurement_and_clinical_analysis_manifest.tsv", sep="\t", index=False)

    pdf_lines = []
    for p in sorted(FIG_PDF.glob("*.pdf")):
        b = p.read_bytes()
        pdf_lines.append(f"- `{p.name}`: /Font={b.count(b'/Font')}, /Type3={b.count(b'/Type3')}")
    lines = [
        "# Targeted measurement integrated clean rebuild QA",
        "",
        "- `GATM_deep_dive` was removed from this integrated targeted-measurement directory.",
        "- Figures are organised as `figures/pdf`, `figures/svg`, and `figures/png_preview`.",
        "- Figure order: targeted N-vs-M effect and group-pattern summary, clinical association heatmap, then model-score analyses.",
        f"- Clinical source: `{CLINICAL_XLSX}`.",
        "- PFS/OS events were parsed from the targeted clinical workbook: recurrence/metastasis/death-positive entries = 1; negative, none, and slash entries = 0.",
        f"- Integrated dataset rows x columns: {df.shape[0]} x {df.shape[1]}",
        f"- Batch1 rows with any clinical variable merged: {int((df[df['batch_display'].eq('Batch1')][['CA125','HE4','group_label','stage_group','pathology_subtype','pfs_event','pfs_time']].notna().any(axis=1)).sum())}",
        "- Validation cohort clinical variables are intentionally not merged by numeric IDs.",
        "- TNM was not recovered from current tables; FIGO/stage fields are used.",
        "- Clinical associations and PFS analyses are exploratory; the model-score PFS cutpoint is data-selected and not validated.",
        "",
        "## Group counts",
        "",
        df.groupby(["batch_display", "group_display"]).size().unstack(fill_value=0).to_markdown(),
        "",
        "## PDF font screen",
        "",
        *pdf_lines,
    ]
    (REPORTS / "targeted_integrated_clean_rebuild_QA.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    prepare_dirs()
    long, score, clinical = read_sources()
    df = build_integrated_dataset(long, score, clinical)
    plot_01_effect_size()
    plot_03_plsda()
    plot_05_clinical_correlation(df)
    audit_full_batch1_clinical_fields(df)
    categorical_clinical_tests(df)
    plot_06_analyte_pfs(df)
    plot_07_score_gradient(df)
    plot_06_score_pfs(df)
    write_manifest_and_qa(df)


if __name__ == "__main__":
    main()
