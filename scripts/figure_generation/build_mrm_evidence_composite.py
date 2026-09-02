from __future__ import annotations

import re
import os
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
LOCAL_PACKAGES = PROJECT / ".local_python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
sys.path.append(str(PROJECT / "11_trarget" / "readable_raw" / "python_pkgs"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from pyteomics import mzml


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 6.4,
        "axes.labelsize": 6.4,
        "axes.titlesize": 7.1,
        "axes.linewidth": 0.55,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

TARGET_ROOT = PROJECT / "11_trarget"
ROOT = TARGET_ROOT / "analysis_results_requested" / "06_MRM_transition_standard_evidence"
TABLE = ROOT / "tables" / "eight_analyte_transition_level_evidence.tsv"
AUDIT = ROOT / "tables" / "eight_analyte_spectral_evidence_audit.tsv"
CURVE = TARGET_ROOT / "readable_raw" / "curve_chromatograms_long.tsv"
PLASMA_DIR = (
    TARGET_ROOT
    / "readable_raw"
    / "final_area_ratio_semiquant"
    / "raw_trace_spotcheck"
    / "mzml"
)
OUT = ROOT / "figures" / "composite"
OUT.mkdir(parents=True, exist_ok=True)

PLASMA = "#D8665B"
STANDARD = "#376D9C"
PLASMA_LIGHT = "#F4D5D1"
STANDARD_LIGHT = "#D6E4EF"
INK = "#262626"
MID = "#747474"
LIGHT = "#E7E7E7"
PALE = "#F6F7F8"
MULTI = "#4D789D"
SINGLE = "#A8A8A8"

MULTI_ANALYTES = [
    "Acetylcarnitine",
    "Creatine",
    "Carnitine",
    "Phenylalanine",
    "Tryptophan",
]
ALL_ANALYTES = [
    "3-Guanidinopropionic acid",
    "Acetylcarnitine",
    "Creatine",
    "DHEA-S",
    "Arginine",
    "Carnitine",
    "Phenylalanine",
    "Tryptophan",
]


def component_name(chrom_id: str) -> str:
    match = re.search(r"name=(.*)$", chrom_id or "")
    return match.group(1).strip() if match else chrom_id


def find_plasma_file(sample: str) -> Path:
    candidates = sorted(
        p
        for p in PLASMA_DIR.glob("*.mzML")
        if p.stem.lower().startswith(f"20260415data-{sample.lower()}")
    )
    if not candidates:
        raise FileNotFoundError(f"No plasma mzML found for {sample}")
    return candidates[-1]


def load_plasma_traces(sample: str, components: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with mzml.MzML(str(find_plasma_file(sample))) as reader:
        for chrom in reader.iterfind("chromatogram"):
            if "selected reaction monitoring chromatogram" not in chrom:
                continue
            name = component_name(chrom.get("id", ""))
            if name not in components:
                continue
            traces[name] = (
                np.asarray(chrom.get("time array", []), dtype=float),
                np.asarray(chrom.get("intensity array", []), dtype=float),
            )
    return traces


def load_standard_traces(components: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curve = pd.read_csv(CURVE, sep="\t", usecols=["Sample Name", "component_from_mzML", "RT_min", "intensity"])
    curve = curve[curve["Sample Name"].eq("S6") & curve["component_from_mzML"].isin(components)]
    return {
        name: (sub["RT_min"].to_numpy(float), sub["intensity"].to_numpy(float))
        for name, sub in curve.groupby("component_from_mzML", sort=False)
    }


def add_panel_label(ax, label: str, x: float = -0.04, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom", ha="left")


def style_axis(ax) -> None:
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(colors=INK)


def draw_workflow(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", x=-0.015, y=0.96)

    boxes = [
        (0.02, 0.22, 0.18, 0.55, "Authentic standard\nS6 mixture", STANDARD_LIGHT, STANDARD),
        (0.235, 0.22, 0.18, 0.55, "Representative\nplasma", PLASMA_LIGHT, PLASMA),
        (0.48, 0.16, 0.22, 0.67, "Targeted MRM\nQ1 → Q3\nquantifier + qualifier", "#EDF1F4", INK),
        (0.765, 0.16, 0.215, 0.67, "Identity evidence\nmethod-relative RT\n+ transition profile", "#E9EEF2", MULTI),
    ]
    for x, y, w, h, text, face, edge in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face, edgecolor=edge, linewidth=0.85,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.3)
    for x0, x1 in [(0.20, 0.48), (0.415, 0.48), (0.70, 0.765)]:
        ax.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>", mutation_scale=8, lw=0.75, color=MID))
    ax.text(0.59, 0.05, "Evidence is stratified by the number of monitored transitions", ha="center", va="bottom", fontsize=5.8, color=MID)


def plot_mirror(ax, analyte: str, evidence: pd.DataFrame, show_xlabel: bool) -> None:
    sub = evidence[evidence["analyte"].eq(analyte)].sort_values("method_q3")
    x = np.arange(len(sub), dtype=float)
    plasma = sub["plasma_relative_intensity_pct"].to_numpy(float)
    standard = sub["standard_relative_intensity_pct"].to_numpy(float)
    roles = sub["method_role"].fillna("").str.lower().tolist()
    widths = [2.5 if role == "quantifier" else 1.45 for role in roles]
    for xi, yp, ys, lw in zip(x, plasma, standard, widths):
        ax.vlines(xi, 0, yp, color=PLASMA, linewidth=lw, zorder=3)
        ax.vlines(xi, 0, -ys, color=STANDARD, linewidth=lw, zorder=3)
        ax.plot(xi, yp, "o", ms=2.2, color=PLASMA, zorder=4)
        ax.plot(xi, -ys, "o", ms=2.2, color=STANDARD, zorder=4)
    ax.axhline(0, color=MID, lw=0.5)
    ax.set_ylim(-112, 112)
    ax.set_xlim(-0.45, len(sub) - 0.55 + 0.1)
    ax.set_yticks([-100, 0, 100])
    ax.set_yticklabels(["100", "0", "100"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.3f}" for v in sub["method_q3"]])
    ax.set_xlabel("Product ion m/z" if show_xlabel else "")
    ax.set_title(analyte, loc="left", pad=2.5, fontweight="bold", color=INK)
    q1 = sub["method_q1"].dropna().iloc[0]
    ax.text(0.995, 0.97, f"Q1 {q1:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=MID)
    ax.tick_params(axis="x", length=0, pad=1.5)
    style_axis(ax)


def normalized_window(times: np.ndarray, values: np.ndarray, expected: float, half_width: float = 0.42):
    mask = np.isfinite(times) & np.isfinite(values) & (times >= expected - half_width) & (times <= expected + half_width)
    x = times[mask] - expected
    y = values[mask]
    peak = np.nanmax(y) if y.size else np.nan
    if not y.size or not np.isfinite(peak) or peak <= 0:
        return np.array([]), np.array([])
    return x, y / peak


def plot_chromatograms(
    ax,
    analyte: str,
    evidence: pd.DataFrame,
    plasma_traces: dict[str, tuple[np.ndarray, np.ndarray]],
    standard_traces: dict[str, tuple[np.ndarray, np.ndarray]],
    show_xlabel: bool,
) -> None:
    sub = evidence[evidence["analyte"].eq(analyte)].sort_values("method_q3").reset_index(drop=True)
    fallback_p = float(sub.loc[sub["method_role"].str.lower().eq("quantifier"), "method_expected_rt_min"].iloc[0])
    fallback_s = float(sub.loc[sub["method_role"].str.lower().eq("quantifier"), "curve_expected_rt_min"].iloc[0])
    n = len(sub)
    for i, row in sub.iterrows():
        baseline = n - i - 1
        ax.hlines(baseline, -0.42, 0.42, color=LIGHT, lw=0.45, zorder=0)
        p_expected = row["method_expected_rt_min"] if pd.notna(row["method_expected_rt_min"]) else fallback_p
        s_expected = row["curve_expected_rt_min"] if pd.notna(row["curve_expected_rt_min"]) else fallback_s
        if row["component"] in plasma_traces:
            xp, yp = normalized_window(*plasma_traces[row["component"]], float(p_expected))
            ax.plot(xp, baseline + 0.72 * yp, color=PLASMA, lw=1.05 if row["method_role"].lower() == "quantifier" else 0.75)
        if row["component"] in standard_traces:
            xs, ys = normalized_window(*standard_traces[row["component"]], float(s_expected))
            ax.plot(xs, baseline + 0.72 * ys, color=STANDARD, lw=1.0, linestyle=(0, (2.1, 1.25)))
        ax.text(0.415, baseline + 0.08, f"{row['method_q3']:.3f}", ha="right", va="bottom", fontsize=5.1, color=MID)
    ax.axvline(0, color="#A3A3A3", lw=0.55, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(-0.42, 0.42)
    ax.set_ylim(-0.08, n - 0.18 + 0.75)
    ax.set_yticks([])
    ax.set_xticks([-0.4, 0, 0.4])
    ax.set_xlabel("RT − method expected RT (min)" if show_xlabel else "")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", pad=1.5)
    style_axis(ax)


def evidence_summary(axes, audit: pd.DataFrame) -> None:
    ax_names, ax_n, ax_sim, ax_rt = axes
    ordered = audit.set_index("analyte").loc[ALL_ANALYTES].reset_index()
    y = np.arange(len(ordered))[::-1]

    for ax in axes:
        ax.set_ylim(-0.6, len(ordered) - 0.4)
        for yi in y:
            if yi % 2 == 0:
                ax.axhspan(yi - 0.5, yi + 0.5, color=PALE, zorder=0)

    ax_names.set_xlim(0, 1)
    for yi, row in zip(y, ordered.itertuples()):
        multi = row.common_transitions_plasma_and_standard >= 2
        ax_names.add_patch(plt.Rectangle((0.015, yi - 0.28), 0.025, 0.56, color=MULTI if multi else SINGLE, lw=0))
        label = row.analyte.replace("3-Guanidinopropionic acid", "3-Guanidinopropionic acid")
        ax_names.text(0.06, yi, label, va="center", ha="left", fontsize=5.75, color=INK)
    ax_names.set_title("Analyte / evidence tier", loc="left", fontsize=6.1, fontweight="bold", pad=4)
    ax_names.axis("off")

    ax_n.set_xlim(0.5, 3.5)
    for yi, value in zip(y, ordered["common_transitions_plasma_and_standard"]):
        ax_n.scatter(value, yi, s=24, color=MULTI if value >= 2 else SINGLE, edgecolor="white", linewidth=0.4, zorder=3)
        ax_n.text(value, yi, str(int(value)), ha="center", va="center", fontsize=5.0, color="white", fontweight="bold")
    ax_n.set_xticks([1, 2, 3])
    ax_n.set_yticks([])
    ax_n.set_title("Transitions", fontsize=6.1, fontweight="bold", pad=4)
    ax_n.spines[["left", "bottom"]].set_visible(False)
    ax_n.tick_params(axis="x", length=0)

    ax_sim.set_xlim(0.985, 1.002)
    for yi, value in zip(y, ordered["transition_profile_cosine_similarity"]):
        if pd.notna(value):
            ax_sim.plot([0.985, value], [yi, yi], color=STANDARD_LIGHT, lw=2.4, solid_capstyle="round")
            ax_sim.scatter(value, yi, s=14, color=STANDARD, zorder=3)
        else:
            ax_sim.text(0.9935, yi, "n/a", ha="center", va="center", fontsize=5.2, color=SINGLE)
    ax_sim.axvline(1.0, color=LIGHT, lw=0.5)
    ax_sim.set_xticks([0.99, 1.00])
    ax_sim.set_yticks([])
    ax_sim.set_title("Profile cosine*", fontsize=6.1, fontweight="bold", pad=4)
    ax_sim.spines["left"].set_visible(False)
    style_axis(ax_sim)

    pdev = ordered["quantifier_plasma_rt_deviation_from_final_expected_min"].to_numpy(float)
    sdev = ordered["quantifier_standard_rt_deviation_from_curve_expected_min"].to_numpy(float)
    for yi, p, s in zip(y, pdev, sdev):
        ax_rt.plot([p, s], [yi, yi], color="#C9C9C9", lw=0.6, zorder=1)
        ax_rt.scatter(p, yi, s=15, color=PLASMA, zorder=3)
        ax_rt.scatter(s, yi, s=15, facecolor="white", edgecolor=STANDARD, linewidth=0.8, zorder=3)
    ax_rt.axvline(0, color=MID, lw=0.55, linestyle=(0, (2, 2)))
    ax_rt.set_xlim(-0.42, 0.08)
    ax_rt.set_xticks([-0.4, -0.2, 0])
    ax_rt.set_yticks([])
    ax_rt.set_title("Quantifier ΔRT (min)", fontsize=6.1, fontweight="bold", pad=4)
    ax_rt.set_xlabel("Observed − expected")
    ax_rt.spines["left"].set_visible(False)
    style_axis(ax_rt)

    add_panel_label(ax_names, "d", x=-0.02, y=1.05)


def main() -> None:
    evidence = pd.read_csv(TABLE, sep="\t")
    audit = pd.read_csv(AUDIT, sep="\t")
    wanted = set(evidence[evidence["analyte"].isin(MULTI_ANALYTES)]["component"])
    samples = evidence[evidence["analyte"].isin(MULTI_ANALYTES)]["representative_plasma_sample"].dropna().unique()
    plasma_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        sample_components = set(
            evidence[
                evidence["representative_plasma_sample"].eq(sample)
                & evidence["analyte"].isin(MULTI_ANALYTES)
            ]["component"]
        )
        plasma_traces.update(load_plasma_traces(str(sample), sample_components))
    standard_traces = load_standard_traces(wanted)

    fig = plt.figure(figsize=(7.2, 9.25), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.92, 5.45, 2.5], hspace=0.28)
    ax_a = fig.add_subplot(outer[0])
    draw_workflow(ax_a)

    mid = outer[1].subgridspec(5, 2, width_ratios=[1.02, 1.35], hspace=0.42, wspace=0.22)
    mirror_axes = []
    chrom_axes = []
    for i, analyte in enumerate(MULTI_ANALYTES):
        ax_m = fig.add_subplot(mid[i, 0])
        ax_c = fig.add_subplot(mid[i, 1])
        plot_mirror(ax_m, analyte, evidence, show_xlabel=i == len(MULTI_ANALYTES) - 1)
        plot_chromatograms(ax_c, analyte, evidence, plasma_traces, standard_traces, show_xlabel=i == len(MULTI_ANALYTES) - 1)
        mirror_axes.append(ax_m)
        chrom_axes.append(ax_c)
    add_panel_label(mirror_axes[0], "b", x=-0.12, y=1.08)
    add_panel_label(chrom_axes[0], "c", x=-0.08, y=1.08)
    mirror_axes[0].text(-0.12, 1.38, "Transition-intensity profile", transform=mirror_axes[0].transAxes, fontsize=7.0, fontweight="bold", ha="left")
    chrom_axes[0].text(-0.08, 1.38, "Method-relative chromatographic alignment", transform=chrom_axes[0].transAxes, fontsize=7.0, fontweight="bold", ha="left")

    bottom = outer[2].subgridspec(1, 4, width_ratios=[2.5, 0.75, 1.3, 1.8], wspace=0.18)
    summary_axes = [fig.add_subplot(bottom[0, i]) for i in range(4)]
    evidence_summary(summary_axes, audit)

    legend = [
        Line2D([0], [0], color=PLASMA, lw=1.5, label="Representative plasma"),
        Line2D([0], [0], color=STANDARD, lw=1.5, linestyle=(0, (2, 1.3)), label="Authentic standard (S6)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PLASMA, markeredgecolor=PLASMA, ms=4, label="Plasma ΔRT"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=STANDARD, ms=4, label="Standard ΔRT"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=4, frameon=False, handlelength=2.1, columnspacing=1.3)
    fig.text(0.055, 0.006, "*Descriptive transition-profile similarity; not a library-search score. MRM transition evidence, not full product-ion spectra.", fontsize=5.15, color=MID, ha="left")
    fig.suptitle("Authentic-standard support for eight targeted analytes", x=0.055, y=0.995, ha="left", fontsize=10.2, fontweight="bold", color=INK)
    fig.text(0.055, 0.975, "Integrated transition-profile, chromatographic and evidence-tier view", ha="left", va="top", fontsize=6.5, color=MID)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.055)

    stem = OUT / "Eight_targeted_analytes_MRM_evidence_composite"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
