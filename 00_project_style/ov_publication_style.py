from __future__ import annotations

from pathlib import Path


GROUP_ORDER_FULL = ["Normal", "Benign", "Borderline", "Malignant"]
GROUP_ORDER_SHORT = ["N", "B", "BD", "M"]

GROUP_COLORS = {
    "Normal": "#80B1D3", "Benign": "#8DD3C7", "Borderline": "#FDB462", "Malignant": "#FB8072",
    "N": "#80B1D3", "B": "#8DD3C7", "BD": "#FDB462", "M": "#FB8072",
    "Normal controls": "#80B1D3", "Benign lesions": "#8DD3C7",
    "Borderline tumours": "#FDB462", "Malignant tumours": "#FB8072",
}

COHORT_COLORS = {
    "Discovery cohort": "#8DD3C7",
    "Temporal same-centre validation cohort": "#BEBADA",
    "Training": "#8DD3C7",
    "Validation": "#BEBADA",
}

MODEL_COLORS = {
    "Baseline": "#B3B3B3", "CA125-only": "#B3B3B3", "Augmented": "#FB8072",
    "Targeted model": "#FB8072", "Discovery cohort": "#8DD3C7",
    "Temporal same-centre validation cohort": "#BEBADA",
}

DIRECTION_COLORS = {
    "up": "#FB8072", "down": "#80B1D3", "positive": "#FB8072", "negative": "#80B1D3",
    "high": "#FB8072", "low": "#80B1D3", "background": "#D9D9D9",
    "missing": "#B3B3B3", "other": "#B3B3B3",
}

PATHWAY_COLORS = {
    "Arginine/proline": "#8DD3C7", "Phenylalanine": "#FFFFB3",
    "Tryptophan": "#BEBADA", "Steroid hormone": "#FB8072",
}

HEATMAP_BLUE = "#2166AC"
HEATMAP_WHITE = "#F7F7F7"
HEATMAP_RED = "#B2182B"
SIGNAL_RED = "#FB8072"
SIGNAL_BLUE = "#80B1D3"
BACKGROUND_GREY = "#D9D9D9"
TEXT_DARK = "#000000"
AXIS_GREY = "#000000"


def add_style_path() -> None:
    return None


def setup_matplotlib_style(mpl, sns=None, base_size: float = 7.0) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": base_size, "axes.labelsize": base_size, "axes.titlesize": base_size + 0.5,
        "xtick.labelsize": base_size - 0.5, "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.7, "legend.title_fontsize": base_size - 0.5,
        "axes.linewidth": 0.55, "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150, "savefig.dpi": 600,
        "text.color": TEXT_DARK, "axes.labelcolor": TEXT_DARK, "axes.edgecolor": AXIS_GREY,
        "xtick.color": TEXT_DARK, "ytick.color": TEXT_DARK,
    })
    if sns is not None:
        sns.set_style("ticks")


def save_publication_figure(fig, out_stem: Path, width: float, height: float, dpi: int = 600) -> None:
    fig.set_size_inches(width, height)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".svg", ".png"]:
        fig.savefig(out_stem.with_suffix(ext), bbox_inches="tight", dpi=dpi)
