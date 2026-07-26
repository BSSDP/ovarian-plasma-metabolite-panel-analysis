from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy import sparse
from scipy.stats import mannwhitneyu, wilcoxon


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "12_scFEA"
INPUT = ROOT / "02_inputs"
RESULTS = ROOT / "03_results"
OUT = ROOT / "04_downstream_analysis"
FIG = ROOT / "05_figures" / "focused_display"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "00_project_style"))

try:
    from ov_publication_style import (
        HEATMAP_BLUE,
        HEATMAP_RED,
        HEATMAP_WHITE,
        SIGNAL_BLUE,
        SIGNAL_RED,
        TEXT_DARK,
        save_publication_figure,
        setup_matplotlib_style,
    )
except Exception:
    HEATMAP_BLUE = "#2166AC"
    HEATMAP_WHITE = "#F7F7F7"
    HEATMAP_RED = "#B2182B"
    SIGNAL_BLUE = "#80B1D3"
    SIGNAL_RED = "#FB8072"
    TEXT_DARK = "#2F2F2F"

    def setup_matplotlib_style(mpl, sns=None, base_size: float = 7.5) -> None:
        mpl.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": base_size})

    def save_publication_figure(fig, out_stem: Path, width: float, height: float, dpi: int = 600) -> None:
        fig.set_size_inches(width, height)
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in [".pdf", ".svg", ".png"]:
            fig.savefig(out_stem.with_suffix(ext), bbox_inches="tight", dpi=dpi)


setup_matplotlib_style(matplotlib, base_size=7.5)
UMAP_FIGSIZE = (5.4, 5.0)
UMAP_ADJUST = {"left": 0.09, "right": 0.84, "bottom": 0.09, "top": 0.88}
UMAP_AXIS_PADDING = 0.02
UMAP_POINT_SIZE = 4.2
UMAP_ALPHA = 0.74
UMAP_CMAP = LinearSegmentedColormap.from_list("ov_score", [HEATMAP_BLUE, HEATMAP_WHITE, HEATMAP_RED])

DATASETS = {
    "GSE217517": {
        "h5ad": PROJECT_ROOT
        / "08_multiomics_validation_scRNA_TCGA"
        / "08A_scRNA_preprocess"
        / "GSE217517"
        / "10_kegg_enrichment"
        / "adata_kegg_scored.h5ad",
        "input_dir": INPUT / "GSE217517_epithelial",
        "result_dir": RESULTS / "GSE217517_epithelial" / "full",
        "metadata_file": "GSE217517_epithelial_cell_metadata.tsv",
        "flux_file": "GSE217517_epithelial_flux_m168.csv",
        "balance_file": "GSE217517_epithelial_metabolite_balance_c70.csv",
        "normal_n": 2702,
        "tumour_n": 6939,
    },
    "GSE184880": {
        "h5ad": PROJECT_ROOT
        / "08_multiomics_validation_scRNA_TCGA"
        / "08A_scRNA_preprocess"
        / "scRNAGSE184880"
        / "10_kegg_enrichment"
        / "adata_kegg_scored.h5ad",
        "input_dir": INPUT / "GSE184880_epithelial",
        "result_dir": RESULTS / "GSE184880_epithelial" / "full",
        "metadata_file": "GSE184880_epithelial_cell_metadata.tsv",
        "flux_file": "GSE184880_epithelial_flux_m168.csv",
        "balance_file": "GSE184880_epithelial_metabolite_balance_c70.csv",
        "reference_umap_table": PROJECT_ROOT
        / "08_multiomics_validation_scRNA_TCGA"
        / "08A_scRNA_preprocess"
        / "scRNAGSE184880"
        / "25_epithelial_four_pathway_activity"
        / "tables"
        / "epithelial_reference_umap_cnv_four_pathway_scores.csv",
        "normal_n": 2021,
        "tumour_n": 3969,
    },
}

ARG_CREATINE_AXIS_MODULES = ["M_19", "M_39", "M_62", "M_64", "M_65", "M_66", "M_67", "M_68"]

DISPLAY_ITEMS = {
    "Arginine balance": {
        "source": "balance",
        "column": "Arginine",
        "ylabel": "Arginine balance",
        "note": "metabolite-balance node",
    },
    "M19 GATM/GAMT module": {
        "source": "flux",
        "column": "M_19",
        "ylabel": "M19 inferred flux",
        "note": "arginine/proline/creatine flux module",
    },
}


def parse_tumour_state(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    tumour_tokens = {"tumor", "tumour", "tumor-like", "tumour-like", "malignant", "true", "1"}
    normal_tokens = {"normal", "normal-like", "false", "0"}
    unknown = sorted(set(text) - tumour_tokens - normal_tokens)
    if unknown:
        raise ValueError(f"Unexpected is_tumor_cell values: {unknown}")
    return text.isin(tumour_tokens)


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "P = NA"
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


def _decode(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def read_h5ad_index(handle: h5py.File, group: str) -> list[str]:
    node = handle[group]
    if "_index" in node:
        return _decode(node["_index"][:])
    if "index" in node:
        return _decode(node["index"][:])
    raise KeyError(f"No index field found in {group}")


def read_gene_vector_from_h5ad(h5ad_path: Path, gene: str, cell_ids: pd.Index) -> pd.Series:
    """Read one gene vector from the log1p layer when available and align it to metadata."""
    with h5py.File(h5ad_path, "r") as handle:
        obs_names = pd.Index(read_h5ad_index(handle, "obs"))
        var_names = pd.Index(read_h5ad_index(handle, "var"))
        matches = np.where(var_names.str.upper() == gene.upper())[0]
        if len(matches) != 1:
            raise ValueError(f"Expected one match for {gene}, found {len(matches)} in {h5ad_path}")
        gene_idx = int(matches[0])

        if "layers" in handle and "log1p" in handle["layers"]:
            matrix = handle["layers"]["log1p"]
        else:
            matrix = handle["X"]

        if isinstance(matrix, h5py.Group):
            data = matrix["data"][:]
            indices = matrix["indices"][:]
            indptr = matrix["indptr"][:]
            shape = tuple(matrix["shape"][:])
            mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
            values = np.asarray(mat[:, gene_idx].todense()).ravel()
        else:
            values = np.asarray(matrix[:, gene_idx]).ravel()

    series = pd.Series(values, index=obs_names, name=gene)
    missing = cell_ids.difference(series.index)
    if len(missing):
        raise ValueError(f"{len(missing)} metadata cells missing from h5ad expression vector")
    return series.loc[cell_ids]


def load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = DATASETS[dataset]
    meta = pd.read_csv(cfg["input_dir"] / cfg["metadata_file"], sep="\t", index_col=0)
    meta["is_tumor_cell"] = parse_tumour_state(meta["is_tumor_cell"])
    meta["state"] = np.where(meta["is_tumor_cell"], "Tumour-like", "Normal-like")
    if cfg.get("reference_umap_table"):
        ref = pd.read_csv(cfg["reference_umap_table"], index_col=0)
        missing = meta.index.difference(ref.index)
        if len(missing):
            raise ValueError(f"{dataset}: {len(missing)} epithelial cells missing from reference UMAP table")
        meta["UMAP_1"] = ref.loc[meta.index, "umap_1"].astype(float).to_numpy()
        meta["UMAP_2"] = ref.loc[meta.index, "umap_2"].astype(float).to_numpy()
    flux = pd.read_csv(cfg["result_dir"] / cfg["flux_file"], index_col=0).loc[meta.index]
    balance = pd.read_csv(cfg["result_dir"] / cfg["balance_file"], index_col=0).loc[meta.index]
    return meta, flux, balance


def sample_state_values(meta: pd.DataFrame, values: pd.Series, require_paired: bool = True) -> pd.DataFrame:
    tmp = meta[["sample", "state"]].copy()
    tmp["value"] = values.loc[meta.index].astype(float)
    wide = tmp.groupby(["sample", "state"], observed=True)["value"].mean().unstack("state")
    if require_paired:
        return wide.dropna(subset=["Normal-like", "Tumour-like"])
    return wide


def compare_sample_states(sample_wide: pd.DataFrame, paired: bool) -> tuple[dict, np.ndarray, np.ndarray, pd.DataFrame]:
    normal = sample_wide["Normal-like"].dropna().to_numpy(dtype=float)
    tumour = sample_wide["Tumour-like"].dropna().to_numpy(dtype=float)
    complete = sample_wide.dropna(subset=["Normal-like", "Tumour-like"])
    if paired:
        stat, p = wilcoxon(
            complete["Tumour-like"].to_numpy(dtype=float),
            complete["Normal-like"].to_numpy(dtype=float),
            alternative="two-sided",
            zero_method="wilcox",
        )
        delta_values = complete["Tumour-like"].to_numpy(dtype=float) - complete["Normal-like"].to_numpy(dtype=float)
        test = "paired Wilcoxon signed-rank"
    else:
        stat, p = mannwhitneyu(tumour, normal, alternative="two-sided")
        delta_values = tumour[:, None] - normal[None, :]
        test = "Mann-Whitney U on sample-state means"
    stats = {
        "n_normal_like_samples": int(len(normal)),
        "n_tumour_like_samples": int(len(tumour)),
        "n_paired_samples": int(len(complete)),
        "normal_like_median": float(np.median(normal)) if len(normal) else np.nan,
        "tumour_like_median": float(np.median(tumour)) if len(tumour) else np.nan,
        "median_delta": float(np.median(delta_values)) if np.size(delta_values) else np.nan,
        "test": test,
        "statistic": float(stat),
        "p": float(p),
    }
    return stats, normal, tumour, complete


def cell_state_values(meta: pd.DataFrame, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    aligned = values.loc[meta.index].astype(float)
    normal = aligned.loc[~meta["is_tumor_cell"]].dropna().to_numpy(dtype=float)
    tumour = aligned.loc[meta["is_tumor_cell"]].dropna().to_numpy(dtype=float)
    return normal, tumour


def compare_cell_states(meta: pd.DataFrame, values: pd.Series) -> dict:
    normal, tumour = cell_state_values(meta, values)
    stat, p = mannwhitneyu(tumour, normal, alternative="two-sided")
    delta_values = tumour[:, None] - normal[None, :]
    return {
        "comparison_level": "cell-level",
        "test": "two-sided Mann-Whitney U test on epithelial cells",
        "n_normal_like_cells": int(len(normal)),
        "n_tumour_like_cells": int(len(tumour)),
        "normal_like_median": float(np.median(normal)),
        "tumour_like_median": float(np.median(tumour)),
        "median_delta": float(np.median(delta_values)),
        "statistic": float(stat),
        "p_value": float(p),
    }


def arginine_creatine_axis_score(flux: pd.DataFrame) -> pd.Series:
    missing = [module for module in ARG_CREATINE_AXIS_MODULES if module not in flux.columns]
    if missing:
        raise ValueError(f"Missing arginine/proline/creatine axis modules in flux matrix: {missing}")
    values = flux[ARG_CREATINE_AXIS_MODULES].astype(float)
    sd = values.std(axis=0, ddof=0)
    z = values.sub(values.mean(axis=0), axis=1)
    z = z.div(sd.replace(0, np.nan), axis=1)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.mean(axis=1).rename("arginine_creatine_axis_score")


def paired_box(ax, sample_wide: pd.DataFrame, ylabel: str) -> dict:
    normal = sample_wide["Normal-like"].to_numpy(dtype=float)
    tumour = sample_wide["Tumour-like"].to_numpy(dtype=float)
    stat, p = wilcoxon(tumour, normal, alternative="two-sided", zero_method="wilcox")
    delta = float(np.median(tumour - normal))
    positions = [0, 1]
    bp = ax.boxplot(
        [normal, tumour],
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_DARK, "linewidth": 0.8},
        boxprops={"linewidth": 0.65, "color": TEXT_DARK},
        whiskerprops={"linewidth": 0.55, "color": TEXT_DARK},
        capprops={"linewidth": 0.55, "color": TEXT_DARK},
    )
    for patch, color in zip(bp["boxes"], [SIGNAL_BLUE, SIGNAL_RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for _, row in sample_wide.iterrows():
        ax.plot(positions, [row["Normal-like"], row["Tumour-like"]], color="#9E9E9E", linewidth=0.45, alpha=0.65, zorder=1)
    rng = np.random.default_rng(20260616)
    ax.scatter(np.full(len(normal), 0) + rng.normal(0, 0.035, len(normal)), normal, s=12, color=SIGNAL_BLUE, edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter(np.full(len(tumour), 1) + rng.normal(0, 0.035, len(tumour)), tumour, s=12, color=SIGNAL_RED, edgecolor="white", linewidth=0.25, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Normal-like", "Tumour-like"], rotation=0)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.98, f"paired n={len(sample_wide)}\n{p_label(float(p))}", transform=ax.transAxes, va="top", ha="left", fontsize=6.8)
    return {
        "n_paired_samples": int(len(sample_wide)),
        "normal_like_median": float(np.median(normal)),
        "tumour_like_median": float(np.median(tumour)),
        "median_delta": delta,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p": float(p),
    }


def gatm_expression_violin(ax, meta: pd.DataFrame, expression: pd.Series, dataset: str) -> dict:
    table = meta[["state", "sample", "is_tumor_cell"]].copy()
    table["raw_value"] = expression.loc[meta.index].astype(float)
    raw_normal = table.loc[~table["is_tumor_cell"], "raw_value"].to_numpy(dtype=float)
    raw_tumour = table.loc[table["is_tumor_cell"], "raw_value"].to_numpy(dtype=float)
    stat, p = mannwhitneyu(raw_tumour, raw_normal, alternative="two-sided")

    sd = float(np.nanstd(table["raw_value"].to_numpy(dtype=float)))
    if sd == 0 or not np.isfinite(sd):
        table["display_value"] = 0.0
    else:
        table["display_value"] = (table["raw_value"] - float(np.nanmean(table["raw_value"]))) / sd
    low_clip, high_clip = np.nanquantile(table["display_value"].to_numpy(dtype=float), [0.005, 0.975])
    if not np.isfinite(low_clip) or not np.isfinite(high_clip) or low_clip >= high_clip:
        low_clip, high_clip = float(np.nanmin(table["display_value"])), float(np.nanmax(table["display_value"]))
    table["plot_value"] = table["display_value"].clip(lower=low_clip, upper=high_clip)
    normal = table.loc[~table["is_tumor_cell"], "plot_value"].to_numpy(dtype=float)
    tumour = table.loc[table["is_tumor_cell"], "plot_value"].to_numpy(dtype=float)

    parts = ax.violinplot(
        [normal, tumour],
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], [SIGNAL_BLUE, SIGNAL_RED]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.36)

    bp = ax.boxplot(
        [normal, tumour],
        positions=[0, 1],
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_DARK, "linewidth": 0.8},
        boxprops={"linewidth": 0.6, "color": TEXT_DARK},
        whiskerprops={"linewidth": 0.55, "color": TEXT_DARK},
        capprops={"linewidth": 0.55, "color": TEXT_DARK},
    )
    for patch, color in zip(bp["boxes"], [SIGNAL_BLUE, SIGNAL_RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)

    sample_wide = sample_state_values(meta, expression)
    sample_normal = sample_wide["Normal-like"].to_numpy(dtype=float)
    sample_tumour = sample_wide["Tumour-like"].to_numpy(dtype=float)
    sample_stat, sample_p = wilcoxon(sample_tumour, sample_normal, alternative="two-sided", zero_method="wilcox")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal-like", "Tumour-like"])
    ax.set_ylabel("GATM expression (Z-score)")
    ax.set_title("GATM expression", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    y_min, y_max = ax.get_ylim()
    y_bar = y_max - (y_max - y_min) * 0.10
    y_tick = (y_max - y_min) * 0.025
    ax.plot([0, 0, 1, 1], [y_bar - y_tick, y_bar, y_bar, y_bar - y_tick], color=TEXT_DARK, linewidth=0.7)
    ax.text(0.5, y_bar + y_tick * 0.4, f"cell-level {p_label(float(p))}", ha="center", va="bottom", fontsize=6.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sample_interpretation = (
        "cell-level visualisation; sample-level paired test not significant, supportive context only"
        if not np.isfinite(sample_p) or sample_p >= 0.05
        else "cell-level visualisation with sample-level paired support"
    )
    return {
        "dataset": dataset,
        "n_normal_like_cells": int(len(normal)),
        "n_tumour_like_cells": int(len(tumour)),
        "cell_level_normal_median": float(np.median(raw_normal)),
        "cell_level_tumour_median": float(np.median(raw_tumour)),
        "cell_level_delta": float(np.median(raw_tumour) - np.median(raw_normal)),
        "mannwhitneyu_statistic": float(stat),
        "mannwhitneyu_p": float(p),
        "display_scale": "cell-level Z-score of log-normalized GATM expression, winsorized at the 0.5th and 97.5th percentiles for plotting only",
        "sample_level_n_paired": int(len(sample_wide)),
        "sample_level_normal_median": float(np.median(sample_normal)),
        "sample_level_tumour_median": float(np.median(sample_tumour)),
        "sample_level_delta": float(np.median(sample_tumour - sample_normal)),
        "sample_level_wilcoxon_statistic": float(sample_stat),
        "sample_level_wilcoxon_p": float(sample_p),
        "interpretation": sample_interpretation,
    }


def gatm_positive_fraction_boxplot(ax, meta: pd.DataFrame, expression: pd.Series, dataset: str) -> dict:
    table = meta[["sample", "state", "is_tumor_cell"]].copy()
    table["positive"] = expression.loc[meta.index].astype(float) > 0
    wide = table.groupby(["sample", "state"], observed=True)["positive"].mean().unstack("state")
    wide = wide.dropna(subset=["Normal-like", "Tumour-like"])

    normal = wide["Normal-like"].to_numpy(dtype=float)
    tumour = wide["Tumour-like"].to_numpy(dtype=float)
    stat, p = wilcoxon(tumour, normal, alternative="two-sided", zero_method="wilcox")
    delta = float(np.median(tumour - normal))

    positions = [0, 1]
    bp = ax.boxplot(
        [normal * 100, tumour * 100],
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_DARK, "linewidth": 0.9},
        boxprops={"linewidth": 0.7, "color": TEXT_DARK},
        whiskerprops={"linewidth": 0.6, "color": TEXT_DARK},
        capprops={"linewidth": 0.6, "color": TEXT_DARK},
    )
    for patch, color in zip(bp["boxes"], [SIGNAL_BLUE, SIGNAL_RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(20260616)
    for _, row in wide.iterrows():
        ax.plot(positions, [row["Normal-like"] * 100, row["Tumour-like"] * 100], color="#9E9E9E", linewidth=0.55, alpha=0.70, zorder=1)
    ax.scatter(np.full(len(normal), 0) + rng.normal(0, 0.035, len(normal)), normal * 100, s=12, color=SIGNAL_BLUE, edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter(np.full(len(tumour), 1) + rng.normal(0, 0.035, len(tumour)), tumour * 100, s=12, color=SIGNAL_RED, edgecolor="white", linewidth=0.25, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Normal-like", "Tumour-like"])
    ax.set_ylabel("GATM+ epithelial cells (%)")
    ax.set_title("GATM detection fraction", loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax.text(0.02, 0.98, f"paired n={len(wide)}\n{p_label(float(p))}", transform=ax.transAxes, va="top", ha="left", fontsize=6.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return {
        "dataset": dataset,
        "n_paired_samples": int(len(wide)),
        "normal_like_positive_fraction_median": float(np.median(normal)),
        "tumour_like_positive_fraction_median": float(np.median(tumour)),
        "median_delta": delta,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p": float(p),
        "file_stem": f"{dataset}_GATM_positive_fraction_paired_boxplot",
        "interpretation": "sample-level paired GATM-positive epithelial-cell fraction",
    }


def umap_panel(ax, meta: pd.DataFrame, values: pd.Series, label: str) -> None:
    x = meta["UMAP_1"].to_numpy(dtype=float)
    y = meta["UMAP_2"].to_numpy(dtype=float)
    v = values.loc[meta.index].to_numpy(dtype=float)
    sd = np.nanstd(v)
    if sd == 0 or not np.isfinite(sd):
        z = np.zeros_like(v)
    else:
        z = (v - np.nanmean(v)) / sd
    order = np.argsort(z)
    sc = ax.scatter(
        x[order],
        y[order],
        c=z[order],
        s=UMAP_POINT_SIZE,
        cmap=UMAP_CMAP,
        alpha=UMAP_ALPHA,
        edgecolors="none",
        vmin=-2,
        vmax=2,
        rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.78, aspect=22, pad=0.018)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=8.5, width=0.8, length=3)
    cbar.set_label(f"{label}\n(Z-score)", fontsize=9.5, rotation=270, labelpad=17)

    ax.set_title(f"{label}\n(Z-score normalized)", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=9.5)
    ax.set_ylabel("UMAP 2", fontsize=9.5)
    ax.tick_params(axis="both", which="major", labelsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    x_range = float(np.nanmax(x) - np.nanmin(x))
    y_range = float(np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(float(np.nanmin(x)) - x_range * UMAP_AXIS_PADDING, float(np.nanmax(x)) + x_range * UMAP_AXIS_PADDING)
    ax.set_ylim(float(np.nanmin(y)) - y_range * UMAP_AXIS_PADDING, float(np.nanmax(y)) + y_range * UMAP_AXIS_PADDING)
    ax.margins(0)
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def tumour_sample_umap_inputs(dataset: str, meta: pd.DataFrame, values: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if dataset != "GSE184880":
        return meta, values
    if "group" not in meta.columns:
        raise KeyError("GSE184880 metadata does not contain a group column for tumour-sample UMAP filtering")
    filtered_meta = meta.loc[meta["group"].astype(str).eq("Tumor")].copy()
    return filtered_meta, values.loc[filtered_meta.index]


def save_umap_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 900
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)


def make_three_metric_split_violin(dataset: str) -> pd.DataFrame:
    meta, flux, balance = load_dataset(dataset)
    metric_values = {
        "Arginine\nbalance": balance["Arginine"],
        "Arginine-creatine\naxis score": arginine_creatine_axis_score(flux),
        "M19 GATM/GAMT\nmodule": flux["M_19"],
    }

    rows = []
    plot_rows = []
    for label, values in metric_values.items():
        sample_wide = sample_state_values(meta, values, require_paired=True)
        compare_stats, _, _, complete = compare_sample_states(sample_wide, paired=True)
        stacked = sample_wide[["Normal-like", "Tumour-like"]].stack().dropna()
        sd = float(stacked.std(ddof=0))
        if sd == 0 or not np.isfinite(sd):
            z_wide = sample_wide.copy() * 0.0
        else:
            z_wide = (sample_wide - float(stacked.mean())) / sd
        rows.append(
            {
                "dataset": dataset,
                "metric": label.replace("\n", " "),
                "comparison_level": "sample-level paired",
                "test": compare_stats["test"],
                "display_scale": "sample-state Z-score",
                "n_normal_like_samples": compare_stats["n_normal_like_samples"],
                "n_tumour_like_samples": compare_stats["n_tumour_like_samples"],
                "n_paired_samples": compare_stats["n_paired_samples"],
                "normal_like_raw_median": compare_stats["normal_like_median"],
                "tumour_like_raw_median": compare_stats["tumour_like_median"],
                "raw_median_delta": compare_stats["median_delta"],
                "statistic": compare_stats["statistic"],
                "p_value": compare_stats["p"],
            }
        )
        for sample, row in z_wide.iterrows():
            plot_rows.append({"sample": sample, "metric": label, "state": "Normal-like", "z": float(row["Normal-like"])})
            plot_rows.append({"sample": sample, "metric": label, "state": "Tumour-like", "z": float(row["Tumour-like"])})

    plot_df = pd.DataFrame(plot_rows)
    stats_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    categories = list(metric_values.keys())
    positions = np.arange(len(categories), dtype=float)
    rng = np.random.default_rng(20260616)
    for x, label in zip(positions, categories):
        normal = plot_df.loc[(plot_df["metric"] == label) & (plot_df["state"] == "Normal-like"), "z"].to_numpy(dtype=float)
        tumour = plot_df.loc[(plot_df["metric"] == label) & (plot_df["state"] == "Tumour-like"), "z"].to_numpy(dtype=float)
        for values, side, color in [(normal, "left", SIGNAL_BLUE), (tumour, "right", SIGNAL_RED)]:
            parts = ax.violinplot(
                values,
                positions=[x],
                widths=0.82,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = parts["bodies"][0]
            verts = body.get_paths()[0].vertices
            if side == "left":
                verts[:, 0] = np.minimum(verts[:, 0], x)
            else:
                verts[:, 0] = np.maximum(verts[:, 0], x)
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.42)

        ax.boxplot(
            normal,
            positions=[x - 0.12],
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": TEXT_DARK, "linewidth": 0.8},
            boxprops={"facecolor": SIGNAL_BLUE, "alpha": 0.28, "linewidth": 0.55, "color": TEXT_DARK},
            whiskerprops={"linewidth": 0.5, "color": TEXT_DARK},
            capprops={"linewidth": 0.5, "color": TEXT_DARK},
        )
        ax.boxplot(
            tumour,
            positions=[x + 0.12],
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": TEXT_DARK, "linewidth": 0.8},
            boxprops={"facecolor": SIGNAL_RED, "alpha": 0.28, "linewidth": 0.55, "color": TEXT_DARK},
            whiskerprops={"linewidth": 0.5, "color": TEXT_DARK},
            capprops={"linewidth": 0.5, "color": TEXT_DARK},
        )
        for sample in plot_df.loc[plot_df["metric"] == label, "sample"].unique():
            n_value = plot_df.loc[(plot_df["metric"] == label) & (plot_df["sample"] == sample) & (plot_df["state"] == "Normal-like"), "z"].iloc[0]
            t_value = plot_df.loc[(plot_df["metric"] == label) & (plot_df["sample"] == sample) & (plot_df["state"] == "Tumour-like"), "z"].iloc[0]
            ax.plot([x - 0.12, x + 0.12], [n_value, t_value], color="#9E9E9E", linewidth=0.38, alpha=0.55, zorder=1)
        ax.scatter(np.full(len(normal), x - 0.12) + rng.normal(0, 0.018, len(normal)), normal, s=10, color=SIGNAL_BLUE, edgecolor="white", linewidth=0.22, zorder=3)
        ax.scatter(np.full(len(tumour), x + 0.12) + rng.normal(0, 0.018, len(tumour)), tumour, s=10, color=SIGNAL_RED, edgecolor="white", linewidth=0.22, zorder=3)

        p = stats_df.loc[stats_df["metric"] == label.replace("\n", " "), "p_value"].iloc[0]
        y_max = max(float(np.max(normal)), float(np.max(tumour)))
        y_bar = y_max + 0.18
        ax.plot([x - 0.18, x - 0.18, x + 0.18, x + 0.18], [y_bar - 0.06, y_bar, y_bar, y_bar - 0.06], color=TEXT_DARK, linewidth=0.55)
        ax.text(x, y_bar + 0.04, p_label(float(p)), ha="center", va="bottom", fontsize=6.4)

    ax.axhline(0, color="#BDBDBD", linewidth=0.55, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Sample-level Z-score")
    comparison_label = "paired sample-state comparison (n=8)"
    ax.set_title(f"{dataset} scFEA focused metrics", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    ax.text(0.99, 1.02, comparison_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color="#666666")
    ax.scatter([], [], s=18, color=SIGNAL_BLUE, label="Normal-like")
    ax.scatter([], [], s=18, color=SIGNAL_RED, label="Tumour-like")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2, fontsize=6.8, handletextpad=0.4, columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_publication_figure(fig, FIG / f"{dataset}_scFEA_three_metric_split_violin", width=6.2, height=3.1)
    plt.close(fig)

    return stats_df


def safe_stem(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("|", "")
        .replace("-", "_")
        .replace("__", "_")
    )


def make_dataset_figures(dataset: str) -> tuple[list[dict], list[dict]]:
    meta, flux, balance = load_dataset(dataset)
    stats_rows: list[dict] = []
    axis_rows: list[dict] = []

    for display_name, cfg in DISPLAY_ITEMS.items():
        values = balance[cfg["column"]] if cfg["source"] == "balance" else flux[cfg["column"]]
        compare_stats = compare_cell_states(meta, values)
        stats = {
            "comparison_level": "cell-level",
            "test": compare_stats["test"],
            "n_normal_like_cells": compare_stats["n_normal_like_cells"],
            "n_tumour_like_cells": compare_stats["n_tumour_like_cells"],
            "normal_like_median": compare_stats["normal_like_median"],
            "tumour_like_median": compare_stats["tumour_like_median"],
            "median_delta": compare_stats["median_delta"],
            "statistic": compare_stats["statistic"],
            "p_value": compare_stats["p_value"],
        }

        fig_umap, ax_umap = plt.subplots(figsize=UMAP_FIGSIZE)
        fig_umap.subplots_adjust(**UMAP_ADJUST)
        umap_meta, umap_values = tumour_sample_umap_inputs(dataset, meta, values)
        umap_panel(ax_umap, umap_meta, umap_values, display_name)
        save_umap_figure(fig_umap, FIG / f"{dataset}_{safe_stem(display_name)}_UMAP")
        plt.close(fig_umap)

        stats.update(
            {
                "dataset": dataset,
                "display_item": display_name,
                "source": cfg["source"],
                "column": cfg["column"],
                "normal_like_cells": int((~meta["is_tumor_cell"]).sum()),
                "tumour_like_cells": int(meta["is_tumor_cell"].sum()),
                "umap_cells_plotted": int(len(umap_meta)),
                "umap_filter": "group == Tumor" if dataset == "GSE184880" else "all epithelial cells",
                "umap_file_stem": f"{dataset}_{safe_stem(display_name)}_UMAP",
            }
        )
        stats_rows.append(stats)

    axis_score = arginine_creatine_axis_score(flux)
    axis_compare = compare_cell_states(meta, axis_score)
    axis_stats = {
        "comparison_level": "cell-level",
        "test": axis_compare["test"],
        "n_normal_like_cells": axis_compare["n_normal_like_cells"],
        "n_tumour_like_cells": axis_compare["n_tumour_like_cells"],
        "normal_like_median": axis_compare["normal_like_median"],
        "tumour_like_median": axis_compare["tumour_like_median"],
        "median_delta": axis_compare["median_delta"],
        "statistic": axis_compare["statistic"],
        "p_value": axis_compare["p_value"],
    }
    axis_stats.update(
        {
            "dataset": dataset,
            "axis": "arginine/proline/creatine",
            "modules": ";".join(ARG_CREATINE_AXIS_MODULES),
            "score_definition": "mean of per-module Z-scored scFEA flux values across the predefined modules",
            "file_stem": f"{dataset}_scFEA_three_metric_split_violin",
        }
    )
    axis_rows.append(axis_stats)

    return stats_rows, axis_rows


def make_overview_figure() -> pd.DataFrame:
    module_map = pd.read_csv(OUT / "targeted_scFEA_axis_module_map.tsv", sep="\t")
    balance_overlap = pd.read_csv(OUT / "target_related_scFEA_metabolite_balance_overlap.tsv", sep="\t")
    module_count = 168
    balance_count = 70
    direct_balance = "; ".join(balance_overlap.loc[balance_overlap["overlap_type"] == "direct/exact", "balance_metabolite"].tolist())
    related_balance = "; ".join(balance_overlap.loc[balance_overlap["overlap_type"] != "direct/exact", "balance_metabolite"].head(8).tolist())
    selected_modules = module_map.loc[module_map["recommended_display"].astype(str).str.lower() == "yes", ["module", "axis", "display_label", "hit_genes", "plasma_analytes"]]
    overview = pd.DataFrame(
        [
            {
                "scFEA_output": "Flux matrix",
                "content": f"{module_count} reaction modules",
                "targeted_display": "M_19 GATM/GAMT module",
                "interpretation": "transcriptome-inferred module flux, not direct flux measurement",
            },
            {
                "scFEA_output": "Metabolite-balance matrix",
                "content": f"{balance_count} metabolite balance nodes",
                "targeted_display": "Arginine balance",
                "interpretation": "inferred local balance signal, not metabolite abundance",
            },
            {
                "scFEA_output": "Target module map",
                "content": f"{len(module_map)} target-related module hits; {len(selected_modules)} previously marked display modules",
                "targeted_display": "Focus narrowed to M_19",
                "interpretation": "broad map retained in tables; focused figure uses only M_19",
            },
            {
                "scFEA_output": "Target balance overlap",
                "content": f"Direct overlaps include {direct_balance}",
                "targeted_display": "Arginine",
                "interpretation": f"Related nodes include {related_balance}",
            },
        ]
    )
    overview_path = OUT / "focused_scFEA_output_overview.tsv"
    overview.to_csv(overview_path, sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.axis("off")
    ax.text(0.0, 1.02, "scFEA output overview", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.0,
        0.86,
        "The scFEA run generated two result layers for each epithelial-cell dataset: module fluxes and metabolite-balance nodes.",
        fontsize=7.5,
        transform=ax.transAxes,
    )
    y_positions = [0.62, 0.43, 0.24, 0.05]
    for y, (_, row) in zip(y_positions, overview.iterrows()):
        ax.text(0.02, y, row["scFEA_output"], fontsize=7.6, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
        ax.text(0.31, y, row["content"], fontsize=7.0, color=TEXT_DARK, transform=ax.transAxes)
        ax.text(0.70, y, row["targeted_display"], fontsize=7.0, color=SIGNAL_RED, transform=ax.transAxes)
    ax.text(0.02, 0.75, "Output", fontsize=6.7, color="#666666", transform=ax.transAxes)
    ax.text(0.31, 0.75, "What it contains", fontsize=6.7, color="#666666", transform=ax.transAxes)
    ax.text(0.70, 0.75, "Focused display", fontsize=6.7, color="#666666", transform=ax.transAxes)
    ax.plot([0, 1], [0.72, 0.72], color="#BDBDBD", linewidth=0.6, transform=ax.transAxes)
    save_publication_figure(fig, FIG / "scFEA_output_overview", width=7.4, height=2.8)
    plt.close(fig)
    return overview


def main() -> None:
    rows = []
    axis_rows = []
    split_violin_rows = []
    for dataset in DATASETS:
        dataset_rows, dataset_axis_rows = make_dataset_figures(dataset)
        rows.extend(dataset_rows)
        axis_rows.extend(dataset_axis_rows)
        split_violin_rows.append(make_three_metric_split_violin(dataset))
    split_violin_stats = pd.concat(split_violin_rows, ignore_index=True)
    split_violin_stats_path = OUT / "focused_scFEA_three_metric_split_violin_statistics.tsv"
    split_violin_stats.to_csv(split_violin_stats_path, sep="\t", index=False)
    stats = pd.DataFrame(rows)
    stats_path = OUT / "focused_scFEA_Arginine_M19_display_statistics.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)
    axis_stats = pd.DataFrame(axis_rows)
    axis_stats_path = OUT / "focused_scFEA_axis_score_statistics.tsv"
    axis_stats.to_csv(axis_stats_path, sep="\t", index=False)
    manifest = pd.DataFrame(
        [
            {"file_stem": "GSE217517_Arginine_balance_UMAP", "purpose": "Arginine balance epithelial-cell UMAP overlay"},
            {"file_stem": "GSE217517_M19_GATM_GAMT_module_UMAP", "purpose": "M19 GATM/GAMT epithelial-cell UMAP overlay"},
            {"file_stem": "GSE217517_scFEA_three_metric_split_violin", "purpose": "combined split half-violin view of three GSE217517 sample-level scFEA metrics"},
            {"file_stem": "GSE184880_Arginine_balance_UMAP", "purpose": "Arginine balance epithelial-cell UMAP overlay"},
            {"file_stem": "GSE184880_M19_GATM_GAMT_module_UMAP", "purpose": "M19 GATM/GAMT epithelial-cell UMAP overlay"},
            {"file_stem": "GSE184880_scFEA_three_metric_split_violin", "purpose": "combined split half-violin view of three GSE184880 sample-level scFEA metrics"},
        ]
    )
    manifest.to_csv(OUT / "focused_scFEA_display_manifest.tsv", sep="\t", index=False)

    qa = [
        "# Focused scFEA axis-score and GATM-expression update",
        "",
        "No scFEA neural-network inference was rerun. This update reuses existing epithelial-cell flux matrices, metabolite-balance matrices, metadata and source h5ad expression matrices.",
        "",
        "## Arginine/proline/creatine axis score",
        "",
        f"Predefined modules: {', '.join(ARG_CREATINE_AXIS_MODULES)}.",
        "For each dataset, each module was Z-scored across all epithelial cells, and the axis score was calculated as the mean Z-score across the predefined modules. Statistical testing used paired sample-level normal-like versus tumour-like mean scores.",
        "",
        "## GATM expression",
        "",
        "GATM single-gene expression plots are intentionally not generated in the focused display because tumour-like versus normal-like epithelial expression direction is not consistent across the two datasets. GSE217517 has higher cell-level rank/detection in tumour-like cells, whereas GSE184880 does not support tumour-like upregulation at the cell-level expression distribution. Keep GATM as part of the M19 GATM/GAMT module only if discussing the predefined arginine-creatine axis.",
        "",
        "## Output tables",
        "",
        f"- {axis_stats_path.name}",
        f"- {split_violin_stats_path.name}",
        "- focused_scFEA_display_manifest.tsv",
    ]
    (OUT / "focused_scFEA_axis_score_and_GATM_violin_QA.md").write_text("\n".join(qa), encoding="utf-8")

    print(f"Wrote statistics: {stats_path}")
    print(f"Wrote axis statistics: {axis_stats_path}")
    print(f"Wrote split violin rows: {len(split_violin_stats)}")
    print(f"Wrote figures to: {FIG}")


if __name__ == "__main__":
    main()
