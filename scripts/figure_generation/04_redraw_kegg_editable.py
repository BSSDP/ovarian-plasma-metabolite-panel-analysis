#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw KEGG/ORA bubble plot as editable PDF/SVG."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figure_final"
RESULTS_TABLE_DIR = ROOT / "data" / "results_tables"
INPUT_CSV = RESULTS_TABLE_DIR / "pathway_results.csv"
LEGACY_INPUT_CSV = FIGURE_DIR / "pathway_results.csv"
PROJECT_ROOT = ROOT.parent
STYLE_DIR = PROJECT_ROOT / "00_project_style"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
from ov_publication_style import SIGNAL_RED, setup_matplotlib_style

OUT_STUB = FIGURE_DIR / "kegg_editable"
OUT_PDF = OUT_STUB.with_suffix(".pdf")
OUT_SVG = OUT_STUB.with_suffix(".svg")
OUT_PNG = OUT_STUB.with_suffix(".png")

# Also provide a simple canonical PDF name beside the historical PNG.
OUT_CANONICAL_PDF = FIGURE_DIR / "kegg.pdf"

setup_matplotlib_style(mpl, base_size=8)


def truncate_label(label: str, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3].rstrip() + "..."


def load_data() -> pd.DataFrame:
    input_csv = INPUT_CSV if INPUT_CSV.exists() else LEGACY_INPUT_CSV
    df = pd.read_csv(input_csv, index_col=0)
    df = df.rename_axis("pathway").reset_index()
    df["enrichment_ratio"] = df["Hits"] / df["Expected"]
    df["plot_label"] = df["pathway"].map(truncate_label)
    df = df.sort_values("-log10(p)", ascending=False).head(25).copy()
    # Reverse order for matplotlib's bottom-to-top y coordinates.
    df["plot_label"] = pd.Categorical(
        df["plot_label"],
        categories=list(reversed(df["plot_label"].tolist())),
        ordered=True,
    )
    return df


def size_from_ratio(ratio: np.ndarray) -> np.ndarray:
    # Match the visual range of the reference PNG: small ratios remain visible;
    # high ratios become large but not oversized.
    return 24 + ratio * 12


def draw(df: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "ov_kegg_pvalue",
        [SIGNAL_RED, "#FDB462", "#FFFFB3"],
    )
    norm = Normalize(vmin=0.0, vmax=0.72)

    fig = plt.figure(figsize=(9.0, 8.2))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.6, 1.45],
        left=0.33,
        right=0.94,
        top=0.91,
        bottom=0.11,
        wspace=0.10,
    )
    ax = fig.add_subplot(gs[0, 0])
    leg_ax = fig.add_subplot(gs[0, 1])
    leg_ax.axis("off")

    y = np.arange(len(df))
    colors = cmap(norm(df["Raw p"].to_numpy()))
    sizes = size_from_ratio(df["enrichment_ratio"].to_numpy())

    ax.scatter(
        df["-log10(p)"],
        y,
        s=sizes,
        c=colors,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["plot_label"], fontsize=8.6, color="#303030")
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 4.07)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("-log10 (p-value)", fontsize=9.4)
    ax.set_title("Overview of enriched metabolite sets", fontsize=10.2, pad=8)

    ax.grid(True, which="major", color="#E9E9E9", linewidth=0.55)
    ax.grid(True, which="minor", axis="x", color="#F1F1F1", linewidth=0.35)
    ax.set_xticks(np.arange(0, 4.5, 0.5), minor=True)
    ax.tick_params(axis="both", colors="#303030", width=0.6, length=2.5, labelsize=8.4)
    ax.tick_params(axis="x", which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.7)

    # Bubble-size legend.
    leg_ax.text(
        0.02,
        0.70,
        "Enrichment Ratio",
        fontsize=9.2,
        ha="left",
        va="bottom",
        transform=leg_ax.transAxes,
    )
    legend_ratios = [5, 10, 15, 20, 25]
    y0 = 0.675
    for i, ratio in enumerate(legend_ratios):
        yy = y0 - i * 0.035
        leg_ax.scatter(
            [0.13],
            [yy],
            s=size_from_ratio(np.array([ratio]))[0],
            c="#303030",
            edgecolors="#303030",
            transform=leg_ax.transAxes,
            clip_on=False,
        )
        leg_ax.text(
            0.31,
            yy,
            str(ratio),
            fontsize=8.4,
            ha="left",
            va="center",
            transform=leg_ax.transAxes,
        )

    # P-value colorbar.
    leg_ax.text(
        0.02,
        0.43,
        "P-value",
        fontsize=9.2,
        ha="left",
        va="bottom",
        transform=leg_ax.transAxes,
    )
    cax = leg_ax.inset_axes([0.02, 0.27, 0.17, 0.15])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6])
    cbar.ax.tick_params(labelsize=8.4, length=0, pad=4)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    for tick in cbar.ax.yaxis.get_ticklines():
        tick.set_color("white")

    save_kwargs = dict(facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_PDF, **save_kwargs)
    fig.savefig(OUT_SVG, **save_kwargs)
    fig.savefig(OUT_PNG, dpi=600, **save_kwargs)
    fig.savefig(OUT_CANONICAL_PDF, **save_kwargs)
    plt.close(fig)


def main() -> None:
    df = load_data()
    draw(df)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_SVG}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CANONICAL_PDF}")


if __name__ == "__main__":
    main()
