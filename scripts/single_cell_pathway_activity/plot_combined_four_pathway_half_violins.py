#!/usr/bin/env python
"""Plot both epithelial scRNA-seq datasets as paired split half violins.

The violins display the unchanged cell-level pathway-score distributions. The
inferential annotations are read from the fixed sample-level analysis, which
summarises each pathway by the within-sample median and uses a two-sided exact
paired Wilcoxon signed-rank test. Each pathway contains one split violin per
dataset: normal epithelial cells on the left and tumour-like epithelial cells
on the right.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "08_multiomics_validation_scRNA_TCGA" / "08A_scRNA_preprocess" / "25_epithelial_four_pathway_activity_combined"
PREPROCESS = ROOT.parent
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
DATASETS = {
    "GSE217517": PREPROCESS / "GSE217517" / "25_epithelial_four_pathway_activity",
    "GSE184880": PREPROCESS / "scRNAGSE184880" / "25_epithelial_four_pathway_activity",
}
PATHWAY_ORDER = [
    "Arginine/proline",
    "Phenylalanine",
    "Tryptophan",
    "Steroid hormone",
]
STATUS_COLORS = {"Normal": "#7BAFD4", "Tumor": "#F27D72"}
DATASET_ALPHA = {"GSE217517": 0.62, "GSE184880": 0.94}
DATASET_CENTERS = {"GSE217517": -0.22, "GSE184880": 0.22}
HALF_WIDTH = 0.17


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    for extension, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(FIGURES / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frames = []
    for dataset, directory in DATASETS.items():
        long_data = pd.read_csv(directory / "tables" / "epithelial_four_pathway_long_data.csv")
        long_data["dataset"] = dataset
        long_frames.append(long_data)
    combined_long = pd.concat(long_frames, ignore_index=True)
    combined_stats = pd.read_csv(TABLES / "sample_level_four_pathway_paired_statistics.tsv", sep="\t")
    combined_long["pathway_label"] = pd.Categorical(
        combined_long["pathway_label"], categories=PATHWAY_ORDER, ordered=True
    )
    combined_stats["pathway_label"] = pd.Categorical(
        combined_stats["pathway_label"], categories=PATHWAY_ORDER, ordered=True
    )
    return combined_long, combined_stats


def fdr_label(fdr: float) -> str:
    if fdr < 0.0001:
        stars = "****"
    elif fdr < 0.001:
        stars = "***"
    elif fdr < 0.01:
        stars = "**"
    elif fdr < 0.05:
        stars = "*"
    else:
        stars = "ns"
    value = f"{fdr:.4f}" if fdr < 0.1 else f"{fdr:.3f}"
    return f"{stars}  FDR = {value}"


def draw_half_violin(ax: plt.Axes, values: np.ndarray, position: float, side: str, color: str, alpha: float) -> None:
    violin = ax.violinplot([values], positions=[position], widths=HALF_WIDTH * 2, showmeans=False, showmedians=False, showextrema=False, bw_method=0.22)
    body = violin["bodies"][0]
    vertices = body.get_paths()[0].vertices
    if side == "left":
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
        box_x, box_width = position - HALF_WIDTH * 0.78, HALF_WIDTH * 0.78
        median_start, median_end = position - HALF_WIDTH * 0.78, position
        whisker_x = position - HALF_WIDTH * 0.39
    else:
        vertices[:, 0] = np.maximum(vertices[:, 0], position)
        box_x, box_width = position, HALF_WIDTH * 0.78
        median_start, median_end = position, position + HALF_WIDTH * 0.78
        whisker_x = position + HALF_WIDTH * 0.39
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(alpha)
    body.set_linewidth(0.55)
    q05, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    ax.plot([whisker_x, whisker_x], [q05, q95], color="#4E5A61", lw=0.55, zorder=4)
    ax.plot([whisker_x - 0.035, whisker_x + 0.035], [q05, q05], color="#4E5A61", lw=0.55, zorder=4)
    ax.plot([whisker_x - 0.035, whisker_x + 0.035], [q95, q95], color="#4E5A61", lw=0.55, zorder=4)
    ax.add_patch(
        Rectangle((box_x, q25), box_width, q75 - q25, facecolor="white", edgecolor=color, alpha=0.72, linewidth=0.7, zorder=5)
    )
    ax.plot([median_start, median_end], [median, median], color="#222222", lw=1.0, zorder=6)


def draw_bracket(ax: plt.Axes, x_left: float, x_right: float, y: float, height: float, label: str) -> None:
    ax.plot([x_left, x_left, x_right, x_right], [y, y + height, y + height, y], color="#202428", lw=0.7, clip_on=False)
    ax.text((x_left + x_right) / 2, y + height, label, ha="center", va="bottom", fontsize=5.7, color="#202428")


def plot_combined(long_data: pd.DataFrame, stats: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(7.15, 3.95))
    y_min = float(long_data["pathway_score"].min())
    y_max = float(long_data["pathway_score"].max())
    y_range = y_max - y_min
    for pathway_index, pathway in enumerate(PATHWAY_ORDER):
        for dataset in DATASETS:
            center = pathway_index + DATASET_CENTERS[dataset]
            for status, side in [("Normal", "left"), ("Tumor", "right")]:
                values = long_data.loc[
                    (long_data["dataset"].eq(dataset))
                    & (long_data["pathway_label"].eq(pathway))
                    & (long_data["is_tumor_cell"].eq(status)),
                    "pathway_score",
                ].to_numpy(dtype=float)
                if len(values) == 0:
                    raise ValueError(f"Missing {dataset} {pathway} {status} values.")
                draw_half_violin(ax, values, center, side, STATUS_COLORS[status], DATASET_ALPHA[dataset])
            stat = stats.loc[(stats["dataset"].eq(dataset)) & (stats["pathway_label"].eq(pathway))].iloc[0]
            pair_max = float(
                long_data.loc[
                    (long_data["dataset"].eq(dataset)) & (long_data["pathway_label"].eq(pathway)), "pathway_score"
                ].max()
            )
            baseline = pair_max + y_range * (0.035 if dataset == "GSE217517" else 0.105)
            draw_bracket(
                ax,
                center - HALF_WIDTH * 0.78,
                center + HALF_WIDTH * 0.78,
                baseline,
                y_range * 0.018,
                fdr_label(float(stat["bh_fdr_within_dataset"])),
            )
        ax.text(
            pathway_index + DATASET_CENTERS["GSE217517"],
            -0.152,
            "GSE217517",
            ha="center",
            va="top",
            fontsize=5.4,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
        ax.text(
            pathway_index + DATASET_CENTERS["GSE184880"],
            -0.152,
            "GSE184880",
            ha="center",
            va="top",
            fontsize=5.4,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    ax.set_xticks(np.arange(len(PATHWAY_ORDER)), PATHWAY_ORDER)
    ax.set_xlim(-0.6, len(PATHWAY_ORDER) - 0.4)
    ax.set_ylim(y_min - y_range * 0.10, y_max + y_range * 0.24)
    ax.set_ylabel("ssGSEA pathway score")
    ax.set_title("Epithelial pathway activity across two scRNA-seq datasets", fontsize=8.5, pad=7)
    ax.grid(axis="y", color="#E2E6E8", lw=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS["Normal"], markeredgecolor="none", markersize=6, label="Normal epithelial"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS["Tumor"], markeredgecolor="none", markersize=6, label="Tumour-like epithelial"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.205), ncol=2, fontsize=6.2, columnspacing=1.8)
    figure.subplots_adjust(bottom=0.28, left=0.105, right=0.995, top=0.90)
    save(figure, "combined_four_pathway_score_split_half_violins_by_cnv_tumor_status")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    setup_style()
    long_data, stats = read_inputs()
    long_data.to_csv(TABLES / "combined_epithelial_four_pathway_long_data.csv", index=False)
    stats.sort_values(["pathway_label", "dataset"]).to_csv(
        TABLES / "combined_epithelial_four_pathway_statistics.csv", index=False
    )
    plot_combined(long_data, stats)


if __name__ == "__main__":
    main()
