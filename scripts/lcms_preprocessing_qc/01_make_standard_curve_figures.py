from __future__ import annotations

from pathlib import Path
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BASE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
STYLE_DIR = PROJECT_ROOT / "00_project_style"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
from ov_publication_style import BACKGROUND_GREY, SIGNAL_BLUE, SIGNAL_RED, setup_matplotlib_style

TABLE_DIR = BASE / "tables"
FIG_DIR = BASE / "figures"
INDIVIDUAL_DIR = FIG_DIR / "individual_pdfs"

POINTS_PATH = TABLE_DIR / "standard_curve_points_with_true_concentration.csv"
FIT_PATH = TABLE_DIR / "standard_curve_fit_summary.csv"

FIT_RANGE_TO_SHOW = "S1-S6"
STD_CURVE_BASE_SIZE = 10

setup_matplotlib_style(matplotlib, base_size=STD_CURVE_BASE_SIZE)
plt.rcParams["axes.unicode_minus"] = False


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def plot_component(ax: plt.Axes, component: str, points: pd.DataFrame, fit_row: pd.Series) -> None:
    sub = points[points["Component Name"].eq(component)].copy()
    sub = sub[sub["Sample Name"].astype(str).isin(["S1", "S2", "S3", "S4", "S5", "S6"])]
    sub = sub.sort_values("true_final_concentration")

    x = pd.to_numeric(sub["true_final_concentration"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(sub["Area Ratio"], errors="coerce").to_numpy(float)
    labels = sub["Sample Name"].astype(str).to_numpy()
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    labels = labels[keep]

    slope = float(fit_row["slope"])
    intercept = float(fit_row["intercept"])
    r2 = float(fit_row["R2"])

    x_min = 0.0
    x_max = max(float(np.nanmax(x)) * 1.08, 1.0)
    x_line = np.linspace(x_min, x_max, 200)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, color=SIGNAL_BLUE, linewidth=1.6, label="Linear fit (S1-S6)")
    ax.scatter(x, y, s=34, color=SIGNAL_RED, edgecolor="white", linewidth=0.55, zorder=3, label="Standards S1-S6")

    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-9)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-9)
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi + x_span * 0.015, yi + y_span * 0.025, label, fontsize=8.2, color="#2F2F2F")

    abbr = str(fit_row.get("abbr", ""))
    title = component if not abbr or abbr == "nan" else f"{component} ({abbr})"
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("True final concentration")
    ax.set_ylabel("Area Ratio")
    ax.grid(color=BACKGROUND_GREY, linewidth=0.55, alpha=0.65)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    annotation = f"y = {slope:.4g}x + {intercept:.4g}\nR2 = {r2:.4f}"
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox=dict(facecolor="white", edgecolor=BACKGROUND_GREY, boxstyle="round,pad=0.2"),
    )
    ax.legend(loc="lower right", fontsize=8.4, frameon=False)


def main() -> None:
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(POINTS_PATH)
    fits = pd.read_csv(FIT_PATH)
    fits = fits[fits["fit_range"].eq(FIT_RANGE_TO_SHOW)].copy()
    if fits.empty:
        raise ValueError(f"No rows found for fit_range={FIT_RANGE_TO_SHOW!r} in {FIT_PATH}")

    components = fits["Component Name"].drop_duplicates().tolist()
    all_pdf_path = FIG_DIR / "standard_curve_plots_all_analytes.pdf"

    with PdfPages(all_pdf_path) as pdf:
        for component in components:
            fit_row = fits[fits["Component Name"].eq(component)].iloc[0]
            fig, ax = plt.subplots(figsize=(5.7, 4.6))
            plot_component(ax, component, points, fit_row)
            fig.tight_layout()

            individual_path = INDIVIDUAL_DIR / f"{safe_name(component)}_standard_curve.pdf"
            fig.savefig(individual_path, bbox_inches="tight")
            fig.savefig(individual_path.with_suffix(".svg"), bbox_inches="tight")
            fig.savefig(individual_path.with_suffix(".png"), dpi=450, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {individual_path}")
            print(f"Wrote {individual_path.with_suffix('.png')}")

    print(f"Wrote {all_pdf_path}")

    n_cols = 2
    n_rows = int(np.ceil(len(components) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.4, 4.5 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, component in zip(axes, components):
        fit_row = fits[fits["Component Name"].eq(component)].iloc[0]
        plot_component(ax, component, points, fit_row)
    for ax in axes[len(components) :]:
        ax.axis("off")
    fig.tight_layout()
    all_png_path = FIG_DIR / "standard_curve_plots_all_analytes.png"
    fig.savefig(all_png_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {all_png_path}")


if __name__ == "__main__":
    main()
