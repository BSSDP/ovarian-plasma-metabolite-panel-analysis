from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
BASE = OUT_DIR.parent.parent
READABLE = BASE / "readable_raw"
PKG_DIR = READABLE / "python_pkgs"
if PKG_DIR.exists():
    sys.path.append(str(PKG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pyteomics import mzml

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


FINAL_EXPORT = BASE / "Quantitation Results" / "20260428OVxwfinal.txt"
MZML_DIR = READABLE / "final_area_ratio_semiquant" / "raw_trace_spotcheck" / "mzml"
FIG_DIR = OUT_DIR / "figures"
INDIV_DIR = FIG_DIR / "individual"
TABLE_DIR = OUT_DIR / "tables"

TARGETS = [
    "3-Guanidinopropionic acid",
    "Acetylcarnitine 1",
    "CREATINE 1",
    "Dehydroisoandrosterone sulfate",
    "L-ARGININE",
    "L-CARNITINE 1",
    "L-Phenylalanine 1",
    "Tryptophan 1",
]

SAMPLE_ORDER = ["qm1", "N1", "B1", "J1", "M1"]
SAMPLE_COLORS = {
    "qm1": "#6b7280",
    "N1": "#0f766e",
    "B1": "#3f8f3f",
    "J1": "#d97706",
    "M1": "#b91c1c",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def to_float(value) -> float:
    try:
        if value is None or str(value).strip() == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def mzml_component_name(chrom_id: str) -> str:
    match = re.search(r"name=(.*)$", chrom_id or "")
    return match.group(1).strip() if match else chrom_id


def selected_mz(chrom, part: str) -> float:
    entries = chrom.get(part) or []
    if not entries:
        return np.nan
    return to_float(entries[0].get("isolationWindow", {}).get("isolation window target m/z"))


def selected_ce(chrom) -> float:
    # pyteomics stores the first collision energy as a top-level key for SRM chromatograms.
    ce = chrom.get("collision energy")
    if ce is not None:
        return to_float(ce)
    chrom_id = chrom.get("id", "")
    match = re.search(r"\bce=([0-9.]+)", chrom_id)
    return to_float(match.group(1)) if match else np.nan


def find_mzml(sample_name: str) -> Path | None:
    sample = sample_name.lower()
    candidates = [
        p
        for p in MZML_DIR.glob("*.mzML")
        if p.stem.lower().startswith("20260415data-" + sample)
        or p.stem.lower().endswith("-" + sample)
    ]
    if not candidates:
        return None

    def duplicate_suffix_number(path: Path) -> int:
        match = re.search(r"\((\d+)\)$", path.stem)
        return int(match.group(1)) if match else 1

    return sorted(candidates, key=duplicate_suffix_number, reverse=True)[0]


def read_transition_manifest() -> pd.DataFrame:
    df = pd.read_csv(FINAL_EXPORT, sep="\t", dtype=str)
    for col in ["Precursor Mass", "Fragment Mass", "Expected RT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = [
        "Component Name",
        "Component Type",
        "Polarity",
        "Mass Info",
        "Precursor Mass",
        "Fragment Mass",
        "Expected RT",
        "IS Name",
        "IS Mass Info",
    ]
    uniq = df[cols].drop_duplicates().rename(
        columns={
            "Component Name": "component_name",
            "Component Type": "component_type",
            "Precursor Mass": "q1_precursor_mz",
            "Fragment Mass": "q3_product_mz",
            "Expected RT": "expected_rt_min",
            "IS Name": "internal_standard_name",
            "IS Mass Info": "internal_standard_mass_info",
        }
    )
    quant = uniq[uniq["component_name"].isin(TARGETS)].copy()
    is_rows = uniq[uniq["component_type"].eq("Internal Standards")].copy()
    is_rows = is_rows.add_prefix("is_")
    out = quant.merge(
        is_rows,
        left_on="internal_standard_name",
        right_on="is_component_name",
        how="left",
    )
    out["source_quantitation_export"] = str(FINAL_EXPORT)
    out["source_mzml_dir"] = str(MZML_DIR)
    return out.sort_values("component_name").reset_index(drop=True)


def read_xic_traces(components: set[str], samples: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []
    for sample in samples:
        path = find_mzml(sample)
        if path is None:
            summary_rows.append({"sample_name": sample, "mzml_file": "", "component_name": "", "read_status": "missing mzML"})
            continue
        reader = mzml.MzML(str(path))
        for chrom in reader.iterfind("chromatogram"):
            cid = chrom.get("id", "")
            component = mzml_component_name(cid)
            is_srm = "selected reaction monitoring chromatogram" in chrom
            if not is_srm or component not in components:
                continue
            times = np.asarray(chrom.get("time array", []), dtype=float)
            intens = np.asarray(chrom.get("intensity array", []), dtype=float)
            if times.size == 0 or intens.size == 0:
                continue
            apex_idx = int(np.nanargmax(intens))
            apex_rt = float(times[apex_idx])
            max_intensity = float(np.nanmax(intens))
            q1 = selected_mz(chrom, "precursor")
            q3 = selected_mz(chrom, "product")
            ce = selected_ce(chrom)
            summary_rows.append(
                {
                    "sample_name": sample,
                    "mzml_file": str(path),
                    "component_name": component,
                    "q1_mzml": q1,
                    "q3_mzml": q3,
                    "collision_energy_eV": ce,
                    "apex_rt_min": apex_rt,
                    "max_intensity": max_intensity,
                    "points": int(times.size),
                    "chromatogram_id": cid,
                    "read_status": "readable",
                }
            )
            for rt, intensity in zip(times, intens):
                long_rows.append(
                    {
                        "sample_name": sample,
                        "component_name": component,
                        "rt_min": float(rt),
                        "intensity": float(intensity),
                        "q1_mzml": q1,
                        "q3_mzml": q3,
                        "collision_energy_eV": ce,
                        "mzml_file": str(path),
                    }
                )
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def target_window(summary: pd.DataFrame, target: str, is_name: str | float) -> tuple[float, float]:
    comps = [target]
    if isinstance(is_name, str) and is_name:
        comps.append(is_name)
    sub = summary[summary["component_name"].isin(comps)].copy()
    centers = sub["apex_rt_min"].dropna().tolist()
    if not centers:
        return 0, 14
    lo = max(0, min(centers) - 0.7)
    hi = max(centers) + 0.7
    if hi - lo < 1.2:
        mid = (hi + lo) / 2
        lo, hi = max(0, mid - 0.6), mid + 0.6
    return lo, hi


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, length=2.5, width=0.6)
    ax.set_xlabel("Retention time (min)", fontsize=8)
    ax.set_ylabel("Intensity", fontsize=8)


def plot_individual_xics(long_df: pd.DataFrame, summary: pd.DataFrame, manifest: pd.DataFrame) -> list[Path]:
    outputs = []
    for _, row in manifest.iterrows():
        target = row["component_name"]
        is_name = row["internal_standard_name"]
        lo, hi = target_window(summary, target, is_name)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharex=True)
        for ax, component, label in [
            (axes[0], target, "Analyte"),
            (axes[1], is_name, "Internal standard"),
        ]:
            sub = long_df[long_df["component_name"].eq(component)].copy()
            for sample in SAMPLE_ORDER:
                ss = sub[sub["sample_name"].eq(sample)]
                if ss.empty:
                    continue
                ax.plot(
                    ss["rt_min"],
                    ss["intensity"],
                    lw=0.9 if sample == "qm1" else 0.75,
                    alpha=0.95 if sample == "qm1" else 0.75,
                    color=SAMPLE_COLORS.get(sample, "#111827"),
                    label=sample,
                )
            ax.set_xlim(lo, hi)
            ax.text(0.02, 0.96, f"{label}: {component}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
            style_axis(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_pdf = INDIV_DIR / f"{safe_name(target)}_XIC_analyte_IS.pdf"
        out_png = INDIV_DIR / f"{safe_name(target)}_XIC_analyte_IS.png"
        out_svg = INDIV_DIR / f"{safe_name(target)}_XIC_analyte_IS.svg"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
        fig.savefig(out_svg)
        plt.close(fig)
        outputs.append(out_pdf)
    return outputs


def normalized_trace(ss: pd.DataFrame) -> pd.DataFrame:
    ss = ss.copy()
    max_i = ss["intensity"].max()
    ss["normalized_intensity"] = ss["intensity"] / max_i if max_i and np.isfinite(max_i) else ss["intensity"]
    return ss


def plot_contact_sheet(long_df: pd.DataFrame, summary: pd.DataFrame, manifest: pd.DataFrame, sample: str = "qm1") -> tuple[Path, Path]:
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 8.6))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, manifest.iterrows()):
        target = row["component_name"]
        is_name = row["internal_standard_name"]
        lo, hi = target_window(summary[summary["sample_name"].eq(sample)], target, is_name)
        for component, color, label, lw in [
            (target, "#2563eb", "Analyte", 1.0),
            (is_name, "#dc2626", "IS", 0.95),
        ]:
            ss = long_df[(long_df["sample_name"].eq(sample)) & (long_df["component_name"].eq(component))]
            if ss.empty:
                continue
            ss = normalized_trace(ss)
            ax.plot(ss["rt_min"], ss["normalized_intensity"], color=color, lw=lw, label=label)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=-0.02)
        ax.text(0.02, 0.96, target, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.tick_params(labelsize=7, length=2.5, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("RT (min)", fontsize=7)
        ax.set_ylabel("Normalized XIC", fontsize=7)
    for ax in axes[len(manifest) :]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    pdf = FIG_DIR / "XIC_8metabolites_QM1_analyte_IS_contact_sheet.pdf"
    png = FIG_DIR / "XIC_8metabolites_QM1_analyte_IS_contact_sheet.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return pdf, png


def plot_analyte_only_8panel(long_df: pd.DataFrame, summary: pd.DataFrame, manifest: pd.DataFrame, sample: str = "qm1") -> tuple[Path, Path]:
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.8))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, manifest.iterrows()):
        target = row["component_name"]
        sub_summary = summary[(summary["sample_name"].eq(sample)) & (summary["component_name"].eq(target))]
        if not sub_summary.empty and pd.notna(sub_summary["apex_rt_min"].iloc[0]):
            mid = float(sub_summary["apex_rt_min"].iloc[0])
            lo, hi = max(0, mid - 0.65), mid + 0.65
        else:
            lo, hi = 0, 14
        ss = long_df[(long_df["sample_name"].eq(sample)) & (long_df["component_name"].eq(target))]
        if not ss.empty:
            ss = normalized_trace(ss)
            ax.plot(ss["rt_min"], ss["normalized_intensity"], color="#2563eb", lw=1.05)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=-0.02)
        ax.text(0.02, 0.96, target, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.tick_params(labelsize=7, length=2.5, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("RT (min)", fontsize=7)
        ax.set_ylabel("Normalized XIC", fontsize=7)
    fig.tight_layout()
    pdf = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_8panel.pdf"
    png = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_8panel.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return pdf, png


def plot_analyte_only_overlay(long_df: pd.DataFrame, manifest: pd.DataFrame, sample: str = "qm1") -> tuple[Path, Path]:
    label_map = {
        "3-Guanidinopropionic acid": "3-GPA",
        "Acetylcarnitine 1": "Acetyl-Carn",
        "CREATINE 1": "Creatine",
        "Dehydroisoandrosterone sulfate": "DHEA-S",
        "L-ARGININE": "Arginine",
        "L-CARNITINE 1": "Carnitine",
        "L-Phenylalanine 1": "Phe",
        "Tryptophan 1": "Tryptophan",
    }
    # High-contrast palette tuned for the close RT clusters:
    # 6.50 min: tryptophan vs phenylalanine; 7.57-7.67 min: creatine vs 3-GPA vs carnitine.
    colors = {
        "Dehydroisoandrosterone sulfate": "#7C3AED",
        "Tryptophan 1": "#111827",
        "L-Phenylalanine 1": "#F59E0B",
        "Acetylcarnitine 1": "#DC2626",
        "CREATINE 1": "#009E73",
        "3-Guanidinopropionic acid": "#2563EB",
        "L-CARNITINE 1": "#DB2777",
        "L-ARGININE": "#4F46E5",
    }
    label_positions = {
        "Dehydroisoandrosterone sulfate": (0.82, 1.05),
        "Tryptophan 1": (5.54, 1.05),
        "L-Phenylalanine 1": (5.58, 0.78),
        "Acetylcarnitine 1": (6.80, 0.63),
        "CREATINE 1": (7.02, 1.15),
        "3-Guanidinopropionic acid": (8.10, 0.99),
        "L-CARNITINE 1": (8.30, 0.70),
        "L-ARGININE": (9.25, 0.94),
    }
    label_align = {
        "Dehydroisoandrosterone sulfate": "left",
        "Tryptophan 1": "right",
        "L-Phenylalanine 1": "right",
        "Acetylcarnitine 1": "left",
        "CREATINE 1": "right",
        "3-Guanidinopropionic acid": "left",
        "L-CARNITINE 1": "left",
        "L-ARGININE": "left",
    }
    plot_order = [
        "Dehydroisoandrosterone sulfate",
        "Tryptophan 1",
        "L-Phenylalanine 1",
        "Acetylcarnitine 1",
        "CREATINE 1",
        "3-Guanidinopropionic acid",
        "L-CARNITINE 1",
        "L-ARGININE",
    ]
    fig, ax = plt.subplots(figsize=(7.35, 3.2))
    apex_rows = []
    manifest_by_target = manifest.set_index("component_name", drop=False)
    for target in plot_order:
        ss = long_df[(long_df["sample_name"].eq(sample)) & (long_df["component_name"].eq(target))]
        if ss.empty:
            continue
        ss = normalized_trace(ss)
        color = colors[target]
        lw = 1.65 if target in {"Tryptophan 1", "L-Phenylalanine 1", "CREATINE 1", "3-Guanidinopropionic acid", "L-CARNITINE 1"} else 1.45
        alpha = 0.92 if target != "CREATINE 1" else 0.82
        (line,) = ax.plot(
            ss["rt_min"],
            ss["normalized_intensity"],
            color=color,
            lw=lw,
            alpha=alpha,
            label=target,
            solid_capstyle="round",
        )
        line.set_path_effects([pe.Stroke(linewidth=lw + 0.65, foreground="white", alpha=0.25), pe.Normal()])

        is_name = manifest_by_target.loc[target, "internal_standard_name"] if target in manifest_by_target.index else ""
        if isinstance(is_name, str) and is_name:
            is_trace = long_df[
                (long_df["sample_name"].eq(sample)) & (long_df["component_name"].eq(is_name))
            ]
            if not is_trace.empty:
                is_trace = normalized_trace(is_trace)
                (is_line,) = ax.plot(
                    is_trace["rt_min"],
                    is_trace["normalized_intensity"],
                    color=color,
                    lw=max(0.95, lw - 0.35),
                    alpha=0.58,
                    ls=(0, (3.0, 1.7)),
                    solid_capstyle="round",
                )
                is_line.set_path_effects(
                    [pe.Stroke(linewidth=lw + 0.25, foreground="white", alpha=0.22), pe.Normal()]
                )
        apex_idx = int(np.nanargmax(ss["normalized_intensity"].to_numpy()))
        apex_rows.append(
            {
                "target": target,
                "rt": float(ss["rt_min"].iloc[apex_idx]),
                "height": float(ss["normalized_intensity"].iloc[apex_idx]),
                "color": color,
            }
        )
    for row in apex_rows:
        lx, ly = label_positions[row["target"]]
        ann = ax.annotate(
            label_map[row["target"]],
            xy=(row["rt"], min(row["height"], 1.0)),
            xytext=(lx, ly),
            textcoords="data",
            color=row["color"],
            fontsize=5.7,
            ha=label_align[row["target"]],
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": row["color"],
                "lw": 0.42,
                "alpha": 0.48,
                "shrinkA": 1.5,
                "shrinkB": 1.5,
            },
        )
        ann.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
    analyte_handle = plt.Line2D([0], [0], color="#333333", lw=1.5, label="Analyte")
    is_handle = plt.Line2D([0], [0], color="#333333", lw=1.2, ls=(0, (3.0, 1.7)), alpha=0.65, label="Internal standard")
    ax.legend(handles=[analyte_handle, is_handle], frameon=False, fontsize=6.2, loc="upper right")
    ax.set_xlim(0, 10.25)
    ax.set_ylim(-0.025, 1.20)
    ax.set_xlabel("Retention time (min)", fontsize=8)
    ax.set_ylabel("Normalized XIC", fontsize=8)
    ax.tick_params(labelsize=7, length=2.5, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.tight_layout()
    pdf = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_overlay.pdf"
    png = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_overlay.png"
    svg = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_overlay.svg"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    fig.savefig(svg)
    plt.close(fig)
    return pdf, png


def plot_analyte_only_stacked(long_df: pd.DataFrame, manifest: pd.DataFrame, sample: str = "qm1") -> tuple[Path, Path]:
    label_map = {
        "3-Guanidinopropionic acid": "3-GPA",
        "Acetylcarnitine 1": "Acetylcarnitine",
        "CREATINE 1": "Creatine",
        "Dehydroisoandrosterone sulfate": "DHEA-S",
        "L-ARGININE": "Arginine",
        "L-CARNITINE 1": "Carnitine",
        "L-Phenylalanine 1": "Phenylalanine",
        "Tryptophan 1": "Tryptophan",
    }
    colors = {
        "3-Guanidinopropionic acid": "#0072B2",
        "Acetylcarnitine 1": "#D55E00",
        "CREATINE 1": "#009E73",
        "Dehydroisoandrosterone sulfate": "#CC79A7",
        "L-ARGININE": "#E69F00",
        "L-CARNITINE 1": "#56B4E9",
        "L-Phenylalanine 1": "#6A3D9A",
        "Tryptophan 1": "#111827",
    }
    order = [
        "Dehydroisoandrosterone sulfate",
        "Tryptophan 1",
        "L-Phenylalanine 1",
        "Acetylcarnitine 1",
        "CREATINE 1",
        "3-Guanidinopropionic acid",
        "L-CARNITINE 1",
        "L-ARGININE",
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.95))
    spacing = 1.18
    rows = []
    for idx, target in enumerate(order):
        ss = long_df[(long_df["sample_name"].eq(sample)) & (long_df["component_name"].eq(target))]
        if ss.empty:
            continue
        ss = normalized_trace(ss)
        offset = (len(order) - 1 - idx) * spacing
        y = ss["normalized_intensity"] + offset
        color = colors[target]
        ax.plot(ss["rt_min"], y, color=color, lw=1.25, solid_joinstyle="round")
        ax.fill_between(ss["rt_min"], offset, y, color=color, alpha=0.08, linewidth=0)
        apex_idx = int(np.nanargmax(ss["normalized_intensity"].to_numpy()))
        apex_rt = float(ss["rt_min"].iloc[apex_idx])
        apex_y = float(y.iloc[apex_idx])
        ax.scatter([apex_rt], [apex_y], s=12, color=color, zorder=4)
        ax.hlines(offset, 0, 10.35, color="#e5e7eb", lw=0.45, zorder=0)
        ax.text(10.42, offset + 0.46, label_map.get(target, target), color=color, fontsize=10.8, va="center", ha="left")
        ax.text(apex_rt, offset + 1.04, f"{apex_rt:.2f}", color=color, fontsize=8.8, ha="center", va="bottom")
        rows.append(offset)
    ax.set_xlim(0, 10.85)
    ax.set_ylim(-0.15, max(rows) + 1.25 if rows else 1)
    ax.set_xlabel("Retention time (min)", fontsize=11.5)
    ax.set_ylabel("Normalized XIC, vertically offset", fontsize=11.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10.0, length=3.0, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    pdf = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_stacked.pdf"
    png = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_stacked.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return pdf, png


def plot_analyte_only_stacked_clean(long_df: pd.DataFrame, manifest: pd.DataFrame, sample: str = "qm1") -> tuple[Path, Path]:
    label_map = {
        "3-Guanidinopropionic acid": "3-GPA",
        "Acetylcarnitine 1": "Acetylcarnitine",
        "CREATINE 1": "Creatine",
        "Dehydroisoandrosterone sulfate": "DHEA-S",
        "L-ARGININE": "Arginine",
        "L-CARNITINE 1": "Carnitine",
        "L-Phenylalanine 1": "Phenylalanine",
        "Tryptophan 1": "Tryptophan",
    }
    colors = {
        "3-Guanidinopropionic acid": "#0072B2",
        "Acetylcarnitine 1": "#D55E00",
        "CREATINE 1": "#009E73",
        "Dehydroisoandrosterone sulfate": "#CC79A7",
        "L-ARGININE": "#E69F00",
        "L-CARNITINE 1": "#56B4E9",
        "L-Phenylalanine 1": "#6A3D9A",
        "Tryptophan 1": "#111827",
    }
    order = [
        "Dehydroisoandrosterone sulfate",
        "Tryptophan 1",
        "L-Phenylalanine 1",
        "Acetylcarnitine 1",
        "CREATINE 1",
        "3-Guanidinopropionic acid",
        "L-CARNITINE 1",
        "L-ARGININE",
    ]
    fig, ax = plt.subplots(figsize=(7.1, 4.15))
    spacing = 1.06
    rows = []
    for idx, target in enumerate(order):
        ss = long_df[(long_df["sample_name"].eq(sample)) & (long_df["component_name"].eq(target))]
        if ss.empty:
            continue
        ss = normalized_trace(ss)
        offset = (len(order) - 1 - idx) * spacing
        y = ss["normalized_intensity"] + offset
        color = colors[target]
        ax.plot(ss["rt_min"], y, color=color, lw=1.15, solid_capstyle="round", solid_joinstyle="round")
        ax.fill_between(ss["rt_min"], offset, y, color=color, alpha=0.07, linewidth=0)
        ax.hlines(offset, 0, 10.22, color="#e5e7eb", lw=0.35, zorder=0)
        ax.text(10.32, offset + 0.43, label_map.get(target, target), color=color, fontsize=7.5, va="center", ha="left")
        rows.append(offset)
    ax.set_xlim(0, 10.75)
    ax.set_ylim(-0.12, max(rows) + 1.12 if rows else 1)
    ax.set_xlabel("Retention time (min)", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, length=2.5, width=0.6)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.tight_layout()
    pdf = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_stacked_clean.pdf"
    png = FIG_DIR / "XIC_8metabolites_QM1_analyte_only_stacked_clean.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return pdf, png


def write_report(
    manifest: pd.DataFrame,
    long_df: pd.DataFrame,
    summary: pd.DataFrame,
    individual_pdfs: list[Path],
    contact_pdf: Path,
    contact_png: Path,
    analyte_8panel_pdf: Path,
    analyte_overlay_pdf: Path,
    analyte_stacked_pdf: Path,
    analyte_stacked_clean_pdf: Path,
) -> Path:
    report = OUT_DIR / "XIC_8metabolites_report.md"
    quant_cols = [
        "component_name",
        "Polarity",
        "q1_precursor_mz",
        "q3_product_mz",
        "expected_rt_min",
        "internal_standard_name",
        "is_q1_precursor_mz",
        "is_q3_product_mz",
        "is_expected_rt_min",
    ]
    qtab = manifest[quant_cols].copy()
    md_table = ["| " + " | ".join(qtab.columns) + " |", "| " + " | ".join(["---"] * len(qtab.columns)) + " |"]
    for _, r in qtab.iterrows():
        vals = []
        for c in qtab.columns:
            v = r[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        md_table.append("| " + " | ".join(vals) + " |")
    text = f"""# XIC Chromatogram Peak Plots for Eight IS-Matched Targeted Metabolites

## Summary

- Output folder: `{OUT_DIR}`
- Source mzML folder: `{MZML_DIR}`
- Quantitation export used for analyte-to-internal-standard mapping: `{FINAL_EXPORT}`
- Target metabolites: {len(manifest)}
- Representative samples plotted in individual panels: {", ".join(SAMPLE_ORDER)}
- Long trace rows exported: {len(long_df)}
- Readable chromatogram summary rows: {len(summary)}

The eight plotted metabolites correspond to the retained targeted model panel with matched internal standards. `2,4,6-Trimethylbenzoic acid` and `Glutamic acid` were not included in this XIC set because the requested set was the eight IS-matched retained targets used downstream.

## Main Outputs

- Contact sheet PDF: `{contact_pdf}`
- Contact sheet PNG: `{contact_png}`
- Analyte-only 8-panel PDF: `{analyte_8panel_pdf}`
- Analyte/IS overlay PDF: `{analyte_overlay_pdf}`
- Analyte-only stacked PDF: `{analyte_stacked_pdf}`
- Analyte-only stacked clean PDF: `{analyte_stacked_clean_pdf}`
- Individual analyte/IS PDFs: `{INDIV_DIR}`
- Trace table: `{TABLE_DIR / "XIC_8metabolites_trace_long.tsv"}`
- Manifest: `{TABLE_DIR / "XIC_8metabolites_transition_manifest.xlsx"}`

## Transition and Internal-Standard Mapping

{chr(10).join(md_table)}

## Notes

- Q1/Q3, polarity, expected RT and IS mapping come from the SCIEX final quantitation export.
- XIC time-intensity arrays come from ProteoWizard-converted SCIEX mzML files under `raw_trace_spotcheck`.
- Individual figures use raw intensity; the contact sheet normalizes analyte and IS traces separately to compare peak shapes.
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    INDIV_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = read_transition_manifest()
    all_components = set(manifest["component_name"]) | set(manifest["internal_standard_name"].dropna())
    long_df, summary = read_xic_traces(all_components, SAMPLE_ORDER)

    # Add mzML-confirmed CE to the manifest from the first available chromatogram per component.
    ce_map = summary.dropna(subset=["component_name"]).drop_duplicates("component_name").set_index("component_name")[
        ["q1_mzml", "q3_mzml", "collision_energy_eV", "apex_rt_min"]
    ]
    for prefix, col in [("", "component_name"), ("is_", "internal_standard_name")]:
        merged = manifest[col].map(ce_map["collision_energy_eV"]) if not ce_map.empty else np.nan
        manifest[f"{prefix}collision_energy_eV_mzML"] = merged
        manifest[f"{prefix}apex_rt_min_mzML"] = manifest[col].map(ce_map["apex_rt_min"]) if not ce_map.empty else np.nan

    long_df.to_csv(TABLE_DIR / "XIC_8metabolites_trace_long.tsv", sep="\t", index=False)
    summary.to_csv(TABLE_DIR / "XIC_8metabolites_chromatogram_summary.tsv", sep="\t", index=False)
    manifest.to_csv(TABLE_DIR / "XIC_8metabolites_transition_manifest.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(TABLE_DIR / "XIC_8metabolites_transition_manifest.xlsx", engine="openpyxl") as writer:
        manifest.to_excel(writer, sheet_name="transition_manifest", index=False)
        summary.to_excel(writer, sheet_name="chromatogram_summary", index=False)

    individual_pdfs = plot_individual_xics(long_df, summary, manifest)
    contact_pdf, contact_png = plot_contact_sheet(long_df, summary, manifest)
    analyte_8panel_pdf, analyte_8panel_png = plot_analyte_only_8panel(long_df, summary, manifest)
    analyte_overlay_pdf, analyte_overlay_png = plot_analyte_only_overlay(long_df, manifest)
    analyte_stacked_pdf, analyte_stacked_png = plot_analyte_only_stacked(long_df, manifest)
    analyte_stacked_clean_pdf, analyte_stacked_clean_png = plot_analyte_only_stacked_clean(long_df, manifest)

    with PdfPages(FIG_DIR / "XIC_8metabolites_individual_multipage.pdf") as pdf:
        for p in individual_pdfs:
            # Re-open the generated PNG companion through matplotlib for a compact multipage copy.
            png = p.with_suffix(".png")
            img = plt.imread(png)
            fig, ax = plt.subplots(figsize=(7.0, 2.35))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig)
            plt.close(fig)

    report = write_report(
        manifest,
        long_df,
        summary,
        individual_pdfs,
        contact_pdf,
        contact_png,
        analyte_8panel_pdf,
        analyte_overlay_pdf,
        analyte_stacked_pdf,
        analyte_stacked_clean_pdf,
    )
    print("output_dir", OUT_DIR)
    print("targets", len(manifest))
    print("trace_rows", len(long_df))
    print("summary_rows", len(summary))
    print("contact_pdf", contact_pdf)
    print("analyte_8panel_pdf", analyte_8panel_pdf)
    print("analyte_overlay_pdf", analyte_overlay_pdf)
    print("analyte_stacked_pdf", analyte_stacked_pdf)
    print("analyte_stacked_clean_pdf", analyte_stacked_clean_pdf)
    print("report", report)


if __name__ == "__main__":
    main()
