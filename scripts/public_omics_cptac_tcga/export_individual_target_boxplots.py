#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "08_multiomics_validation_scRNA_TCGA" / "08D_CPTAC_OV"
ANALYSIS_DIR = ROOT / "09_proteomics_tumor_vs_normal_kegg"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIGURE_DIR = ANALYSIS_DIR / "figures"
INDIVIDUAL_DIR = FIGURE_DIR / "06_target_kegg_pathway_boxplots_individual"
SAMPLE_STATUS_FILE = ROOT / "summary" / "ov_proteomics_sample_status.csv"

TARGET_KEGG_PATHWAYS = {
    "KEGG_ARGININE_AND_PROLINE_METABOLISM": "Arginine and proline metabolism",
    "KEGG_PHENYLALANINE_TYROSINE_AND_TRYPTOPHAN_BIOSYNTHESIS": "Phenylalanine, tyrosine and tryptophan biosynthesis",
    "KEGG_PHENYLALANINE_METABOLISM": "Phenylalanine metabolism",
    "KEGG_ARGININE_BIOSYNTHESIS": "Arginine biosynthesis",
    "KEGG_LYSINE_DEGRADATION": "Lysine degradation",
    "KEGG_GLYCINE_SERINE_AND_THREONINE_METABOLISM": "Glycine, serine and threonine metabolism",
    "KEGG_TRYPTOPHAN_METABOLISM": "Tryptophan metabolism",
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS": "Steroid hormone biosynthesis",
    "KEGG_TYROSINE_METABOLISM": "Tyrosine metabolism",
    "KEGG_FATTY_ACID_DEGRADATION": "Fatty acid degradation",
}

ORDER = ["normal", "tumor"]
PALETTE = {"tumor": "#FB8072", "normal": "#80B1D3"}

FOCUS_PATHWAYS = {
    "KEGG_ARGININE_AND_PROLINE_METABOLISM": "#D89C2B",
    "KEGG_PHENYLALANINE_METABOLISM": "#FB8072",
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS": "#80B1D3",
    "KEGG_TRYPTOPHAN_METABOLISM": "#7B4EA3",
}

FOCUS_PATHWAY_ORDER = [
    "KEGG_ARGININE_AND_PROLINE_METABOLISM",
    "KEGG_PHENYLALANINE_METABOLISM",
    "KEGG_TRYPTOPHAN_METABOLISM",
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS",
]

FOCUS_SHORT_LABELS = {
    "KEGG_ARGININE_AND_PROLINE_METABOLISM": "Arginine/proline",
    "KEGG_PHENYLALANINE_METABOLISM": "Phenylalanine",
    "KEGG_TRYPTOPHAN_METABOLISM": "Tryptophan",
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS": "Steroid hormone",
}


def slugify(term: str) -> str:
    return term.lower().replace("kegg_", "")


def format_fdr(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def draw_target_pathway_barplot(comp: pd.DataFrame) -> None:
    plot_df = comp[comp["term"].isin(TARGET_KEGG_PATHWAYS)].copy()
    plot_df["label"] = plot_df["term"].map(TARGET_KEGG_PATHWAYS)
    plot_df["is_focus"] = plot_df["term"].isin(FOCUS_PATHWAYS)
    plot_df = plot_df.sort_values("score_diff_tumor_minus_normal", ascending=True)
    plot_df["figure_label"] = plot_df["label"]
    plot_df["log2FC_tumor_minus_normal"] = plot_df["score_diff_tumor_minus_normal"]
    plot_df["annotation"] = plot_df.apply(
        lambda row: f"log2FC={row['log2FC_tumor_minus_normal']:.3f}; FDR={format_fdr(row['fdr'])}",
        axis=1,
    )
    plot_df.to_csv(INDIVIDUAL_DIR / "06_target_kegg_pathway_focus_bar_figure_data.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    y = range(len(plot_df))
    colors = [FOCUS_PATHWAYS.get(term, "#C9D6DD") for term in plot_df["term"]]
    edgecolors = ["#1A1A1A" if is_focus else "#FFFFFF" for is_focus in plot_df["is_focus"]]
    linewidths = [1.05 if is_focus else 0.35 for is_focus in plot_df["is_focus"]]

    ax.barh(
        y,
        plot_df["log2FC_tumor_minus_normal"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
        height=0.72,
    )
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["figure_label"], fontsize=9)
    ax.invert_yaxis()

    for tick, is_focus in zip(ax.get_yticklabels(), plot_df["is_focus"]):
        tick.set_fontweight("bold" if is_focus else "normal")
        tick.set_color("#111111" if is_focus else "#4F5B62")

    min_x = min(-0.62, plot_df["log2FC_tumor_minus_normal"].min() - 0.05)
    max_x = max(0.38, plot_df["log2FC_tumor_minus_normal"].max() + 0.18)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel("Tumor - Normal pathway score difference (log2FC)", fontsize=10)
    ax.set_title("CPTAC-OV target KEGG pathway score changes", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax.grid(axis="x", color="#E7EBEE", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B8C2C8")
    ax.tick_params(axis="y", length=0)

    for i, row in enumerate(plot_df.itertuples()):
        x = 0.035 if row.log2FC_tumor_minus_normal <= 0 else row.log2FC_tumor_minus_normal + 0.025
        ax.text(
            x,
            i,
            row.annotation,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold" if row.is_focus else "normal",
            color="#111111" if row.is_focus else "#4F5B62",
        )

    fig.subplots_adjust(left=0.39, right=0.97, top=0.91, bottom=0.13)
    fig.savefig(INDIVIDUAL_DIR / "06_target_kegg_pathway_focus_bar.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(INDIVIDUAL_DIR / "06_target_kegg_pathway_focus_bar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_grouped_focus_boxplot(long_df: pd.DataFrame, comp: pd.DataFrame) -> None:
    """Draw the four focused CPTAC pathway scores on one shared coordinate axis."""
    plot_df = long_df[long_df["term"].isin(FOCUS_PATHWAY_ORDER)].copy()
    plot_df["pathway_label"] = pd.Categorical(
        plot_df["term"].map(FOCUS_SHORT_LABELS),
        categories=[FOCUS_SHORT_LABELS[term] for term in FOCUS_PATHWAY_ORDER],
        ordered=True,
    )
    plot_df["sample_class"] = pd.Categorical(
        plot_df["sample_class"],
        categories=ORDER,
        ordered=True,
    )
    comp_map = comp.set_index("term")[["fdr", "pvalue"]].to_dict("index")

    fig, ax = plt.subplots(figsize=(6.4, 3.25))
    sns.boxplot(
        data=plot_df,
        x="pathway_label",
        y="score",
        hue="sample_class",
        order=[FOCUS_SHORT_LABELS[term] for term in FOCUS_PATHWAY_ORDER],
        hue_order=ORDER,
        palette=PALETTE,
        width=0.62,
        fliersize=0,
        linewidth=1.0,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="pathway_label",
        y="score",
        hue="sample_class",
        order=[FOCUS_SHORT_LABELS[term] for term in FOCUS_PATHWAY_ORDER],
        hue_order=ORDER,
        dodge=True,
        palette=PALETTE,
        alpha=0.45,
        size=2.2,
        linewidth=0,
        ax=ax,
    )

    # Remove duplicate legends caused by boxplot + stripplot.
    handles, labels = ax.get_legend_handles_labels()
    label_map = {"normal": "Normal", "tumor": "Tumor"}
    uniq = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in ORDER and label not in seen:
            uniq.append((handle, label_map[label]))
            seen.add(label)
    ax.legend(
        [item[0] for item in uniq],
        [item[1] for item in uniq],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8,
        handletextpad=0.45,
        columnspacing=1.3,
    )

    y_min = float(plot_df["score"].min())
    y_max = float(plot_df["score"].max())
    y_range = y_max - y_min if y_max > y_min else 1.0
    for idx, term in enumerate(FOCUS_PATHWAY_ORDER):
        meta = comp_map.get(term, {})
        if not meta:
            continue
        fdr = float(meta["fdr"])
        label = f"FDR={format_fdr(fdr)}"
        group_max = float(plot_df.loc[plot_df["term"] == term, "score"].max())
        x1, x2 = idx - 0.20, idx + 0.20
        y = group_max + y_range * 0.035
        h = y_range * 0.025
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#202428", lw=0.9)
        ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=7.0, color="#202428")

    ax.set_title("CPTAC-OV four-pathway proteomic scores", fontsize=9.5, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Pathway score", fontsize=8.5)
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.20)
    ax.grid(True, axis="y", color="#E6ECEF", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", labelsize=7.5, colors="#2E3439")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB4BA")
    ax.spines["bottom"].set_color("#AAB4BA")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(INDIVIDUAL_DIR / "06_target_kegg_four_pathway_grouped_boxplot.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(INDIVIDUAL_DIR / "06_target_kegg_four_pathway_grouped_boxplot.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(INDIVIDUAL_DIR / "06_target_kegg_four_pathway_grouped_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(TABLE_DIR / "target_pathway_scores.csv")
    status = pd.read_csv(SAMPLE_STATUS_FILE)[["Patient_ID", "sample_class"]]
    status["Patient_ID"] = status["Patient_ID"].astype(str)
    status = status[status["sample_class"].isin(ORDER)]

    comp = pd.read_csv(TABLE_DIR / "target_pathway_score_comparison.csv")
    comp_map = comp.set_index("term")[["fdr"]].to_dict("index")
    draw_target_pathway_barplot(comp)

    long_df = scores.melt(id_vars="Patient_ID", var_name="term", value_name="score")
    long_df["Patient_ID"] = long_df["Patient_ID"].astype(str)
    long_df = long_df.merge(status, on="Patient_ID", how="inner")
    draw_grouped_focus_boxplot(long_df, comp)

    for term, label in TARGET_KEGG_PATHWAYS.items():
        sub = long_df[long_df["term"] == term].copy()
        if sub.empty:
            continue

        meta = comp_map.get(term, {})
        fig, ax = plt.subplots(figsize=(2.1, 4.8))
        sns.boxplot(
            data=sub,
            x="sample_class",
            y="score",
            hue="sample_class",
            order=ORDER,
            palette=PALETTE,
            width=0.55,
            fliersize=0,
            ax=ax,
            legend=False,
        )
        sns.stripplot(
            data=sub,
            x="sample_class",
            y="score",
            order=ORDER,
            color="black",
            alpha=0.5,
            size=2.5,
            ax=ax,
        )
        if meta:
            ax.text(
                0.96,
                0.96,
                f"FDR={meta['fdr']:.3g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
            )
        ax.set_xlabel("")
        ax.set_ylabel(f"{label} score", fontsize=9)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(["Normal", "Tumor"])
        fig.tight_layout()

        stem = f"06_{slugify(term)}_boxplot"
        fig.savefig(INDIVIDUAL_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(INDIVIDUAL_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
