#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot targeted QC stability after outlier removal only.

This panel intentionally does not show before-removal results, so it can be used in
the main targeted-validation figure to emphasize post-cleaning assay stability.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

RUNTIME_PKGS = Path(os.environ.get("OV_LOCAL_PACKAGES", ""))
if RUNTIME_PKGS.exists():
    sys.path.append(str(RUNTIME_PKGS))

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR.parent
ANALYSIS_REQUESTED = OUT_ROOT.parent
ROOT = ANALYSIS_REQUESTED.parent
PROJECT_ROOT = ROOT.parent
STYLE_DIR = PROJECT_ROOT / "00_project_style"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
from ov_publication_style import BACKGROUND_GREY, SIGNAL_BLUE, SIGNAL_RED, setup_matplotlib_style

setup_matplotlib_style(matplotlib, base_size=7)
SOURCE_XLSX = ROOT / r"readable_raw\final_area_ratio_semiquant\qc_level_report\outlier_sensitivity\qc_outlier_sensitivity.xlsx"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

MODEL_METABOLITES = [
    "3-Guanidinopropionic acid",
    "Acetylcarnitine 1",
    "CREATINE 1",
    "Dehydroisoandrosterone sulfate",
    "L-ARGININE",
    "L-CARNITINE 1",
    "L-Phenylalanine 1",
    "Tryptophan 1",
]

DISPLAY = {
    "3-Guanidinopropionic acid": "3-GPA",
    "Acetylcarnitine 1": "Acetylcarnitine",
    "CREATINE 1": "Creatine",
    "Dehydroisoandrosterone sulfate": "DHEAS",
    "L-ARGININE": "Arginine",
    "L-CARNITINE 1": "Carnitine",
    "L-Phenylalanine 1": "Phenylalanine",
    "Tryptophan 1": "Tryptophan",
}


def savefig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SOURCE_XLSX, sheet_name="before_after_comparison")
    df = df[df["Component Name"].isin(MODEL_METABOLITES)].copy()
    df["display_name"] = df["Component Name"].map(DISPLAY)
    df["filtered_max_cv_pct"] = pd.to_numeric(df["filtered_max_cv_pct"], errors="coerce")
    df["filtered_QH_over_QL_median_ratio"] = pd.to_numeric(df["filtered_QH_over_QL_median_ratio"], errors="coerce")
    df["stable_call"] = pd.cut(
        df["filtered_max_cv_pct"],
        bins=[-float("inf"), 15, 20, 30, float("inf")],
        labels=["Good", "Acceptable", "Caution", "Poor"],
    )
    df = df.sort_values("filtered_max_cv_pct", ascending=True)
    df.to_csv(TABLE_DIR / "after_outlier_qc_stability_summary.tsv", sep="\t", index=False, encoding="utf-8-sig")

    colors = df["filtered_max_cv_pct"].map(lambda x: SIGNAL_BLUE if x <= 15 else ("#FDB462" if x <= 20 else SIGNAL_RED))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = range(len(df))
    ax.barh(y, df["filtered_max_cv_pct"], color=colors, edgecolor="none", height=0.62)
    ax.axvline(15, color="#30343B", linewidth=0.9, linestyle="--")
    ax.axvline(20, color=BACKGROUND_GREY, linewidth=0.8, linestyle=":")
    for yy, cv, ratio in zip(y, df["filtered_max_cv_pct"], df["filtered_QH_over_QL_median_ratio"]):
        ax.text(cv + 1.0, yy, f"{cv:.1f}% | {ratio:.1f}x", va="center", ha="left", fontsize=6.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["display_name"], fontsize=7.2)
    ax.set_xlabel("Post-cleaning maximum QC CV (%)", fontsize=7.5)
    ax.set_xlim(0, max(32, float(df["filtered_max_cv_pct"].max()) + 8))
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E7ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    savefig(fig, "Fig5B_after_outlier_QC_stability")


if __name__ == "__main__":
    main()
