from __future__ import annotations

import re
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
TARGET_ROOT = PROJECT / "11_trarget"
OUT_ROOT = TARGET_ROOT / "analysis_results_requested" / "06_MRM_transition_standard_evidence"
FIG_ROOT = OUT_ROOT / "figures"
PDF_DIR = FIG_ROOT / "pdf"
SVG_DIR = FIG_ROOT / "svg"
PNG_DIR = FIG_ROOT / "png_preview"
TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"

for directory in [PDF_DIR, SVG_DIR, PNG_DIR, TABLE_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT / "11_trarget" / "readable_raw" / "python_pkgs"))
sys.path.append(str(PROJECT / "00_project_style"))

from pyteomics import mzml  # noqa: E402
from ov_publication_style import (  # noqa: E402
    SIGNAL_BLUE,
    SIGNAL_RED,
    TEXT_DARK,
    save_publication_figure,
    setup_matplotlib_style,
)


setup_matplotlib_style(matplotlib, base_size=7.5)

METHOD_TABLE = TARGET_ROOT / "靶向" / "SCIEX_targeted_ion_pairs_method_table_20260506.csv"
CURVE_LONG = TARGET_ROOT / "readable_raw" / "curve_chromatograms_long.tsv"
PLASMA_MZML_DIR = (
    TARGET_ROOT
    / "readable_raw"
    / "final_area_ratio_semiquant"
    / "raw_trace_spotcheck"
    / "mzml"
)

STANDARD_SAMPLE = "S6"
PLASMA_CANDIDATES = ["N1", "B1", "J1", "M1"]

ANALYTES = {
    "3-Guanidinopropionic acid": ["3-Guanidinopropionic acid"],
    "Acetylcarnitine": ["Acetylcarnitine 1", "Acetylcarnitine 2", "Acetylcarnitine 3"],
    "Creatine": ["CREATINE 1", "CREATINE 2"],
    "DHEA-S": ["Dehydroisoandrosterone sulfate"],
    "Arginine": ["L-ARGININE"],
    "Carnitine": ["L-CARNITINE 1", "L-CARNITINE 2"],
    "Phenylalanine": ["L-Phenylalanine 1", "L-Phenylalanine 2"],
    "Tryptophan": ["Tryptophan 1", "Tryptophan 2", "Tryptophan 3"],
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def component_name(chrom_id: str) -> str:
    match = re.search(r"name=(.*)$", chrom_id or "")
    return match.group(1).strip() if match else chrom_id


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def selected_mz(chrom: dict, part: str) -> float:
    entries = chrom.get(part) or []
    if not entries:
        return np.nan
    return to_float(entries[0].get("isolationWindow", {}).get("isolation window target m/z"))


def collision_energy(chrom: dict) -> float:
    value = chrom.get("collision energy")
    if value is not None:
        return to_float(value)
    match = re.search(r"\bce=([0-9.]+)", chrom.get("id", ""))
    return to_float(match.group(1)) if match else np.nan


def duplicate_suffix_number(path: Path) -> int:
    match = re.search(r"\((\d+)\)$", path.stem)
    return int(match.group(1)) if match else 1


def find_plasma_mzml(sample: str) -> Path:
    candidates = [
        path
        for path in PLASMA_MZML_DIR.glob("*.mzML")
        if path.stem.lower().startswith(f"20260415data-{sample.lower()}")
    ]
    if not candidates:
        raise FileNotFoundError(f"No mzML found for representative plasma candidate {sample}")
    return sorted(candidates, key=duplicate_suffix_number, reverse=True)[0]


def apex_in_expected_window(
    times: np.ndarray,
    intensities: np.ndarray,
    expected_rt: float,
    half_window_min: float = 0.75,
) -> tuple[int, str]:
    if np.isfinite(expected_rt):
        indices = np.flatnonzero(
            (times >= expected_rt - half_window_min) & (times <= expected_rt + half_window_min)
        )
        if indices.size:
            local_index = int(indices[int(np.nanargmax(intensities[indices]))])
            return local_index, f"expected RT +/- {half_window_min:.2f} min"
    return int(np.nanargmax(intensities)), "global apex fallback"


def extract_plasma_transitions(sample: str, wanted: set[str], method: pd.DataFrame) -> pd.DataFrame:
    path = find_plasma_mzml(sample)
    expected_map = method.set_index("component")["method_expected_rt_min"].to_dict()
    rows: list[dict] = []
    reader = mzml.MzML(str(path))
    for chrom in reader.iterfind("chromatogram"):
        if "selected reaction monitoring chromatogram" not in chrom:
            continue
        name = component_name(chrom.get("id", ""))
        if name not in wanted:
            continue
        times = np.asarray(chrom.get("time array", []), dtype=float)
        intensities = np.asarray(chrom.get("intensity array", []), dtype=float)
        if times.size == 0 or intensities.size == 0:
            continue
        expected_rt = to_float(expected_map.get(name))
        apex_index, apex_rule = apex_in_expected_window(times, intensities, expected_rt)
        rows.append(
            {
                "sample": sample,
                "source_type": "Representative plasma",
                "source_file": str(path),
                "component": name,
                "q1": selected_mz(chrom, "precursor"),
                "q3": selected_mz(chrom, "product"),
                "collision_energy_eV": collision_energy(chrom),
                "apex_rt_min": float(times[apex_index]),
                "apex_intensity": float(intensities[apex_index]),
                "expected_rt_min": expected_rt,
                "apex_selection_rule": apex_rule,
            }
        )
    return pd.DataFrame(rows)


def extract_standard_transitions(wanted: set[str], method: pd.DataFrame) -> pd.DataFrame:
    curve = pd.read_csv(CURVE_LONG, sep="\t")
    curve = curve[
        curve["Sample Name"].eq(STANDARD_SAMPLE)
        & curve["component_from_mzML"].isin(wanted)
    ].copy()
    expected_map = method.set_index("component")["curve_expected_rt_min"].to_dict()
    rows: list[dict] = []
    for component, sub in curve.groupby("component_from_mzML", sort=False):
        expected_rt = to_float(expected_map.get(component))
        times = sub["RT_min"].to_numpy(float)
        intensities = sub["intensity"].to_numpy(float)
        local_index, apex_rule = apex_in_expected_window(times, intensities, expected_rt)
        apex = sub.iloc[local_index]
        rows.append(
            {
                "sample": STANDARD_SAMPLE,
                "source_type": "Authentic-standard mixture",
                "source_file": str(CURVE_LONG),
                "component": component,
                "q1": to_float(apex["Q1_mzML"]),
                "q3": to_float(apex["Q3_mzML"]),
                "collision_energy_eV": np.nan,
                "apex_rt_min": to_float(apex["RT_min"]),
                "apex_intensity": to_float(apex["intensity"]),
                "expected_rt_min": expected_rt,
                "apex_selection_rule": apex_rule,
            }
        )
    return pd.DataFrame(rows)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def prepare_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = {component for components in ANALYTES.values() for component in components}
    method = pd.read_csv(METHOD_TABLE)
    method = method.rename(
        columns={
            "Component Name": "component",
            "Role for methods": "method_role",
            "Polarity": "polarity",
            "Q1 precursor m/z": "method_q1",
            "Q3 product m/z": "method_q3",
            "Collision energy CE (eV)": "method_collision_energy_eV",
            "Final Expected RT (min)": "method_expected_rt_min",
            "Curve Expected RT (min)": "curve_expected_rt_min",
        }
    )
    method_cols = [
        "component",
        "method_role",
        "polarity",
        "method_q1",
        "method_q3",
        "method_collision_energy_eV",
        "method_expected_rt_min",
        "curve_expected_rt_min",
    ]
    for column in [
        "method_q1",
        "method_q3",
        "method_collision_energy_eV",
        "method_expected_rt_min",
        "curve_expected_rt_min",
    ]:
        method[column] = pd.to_numeric(method[column], errors="coerce")
    standards = extract_standard_transitions(wanted, method)
    plasma_all = pd.concat(
        [extract_plasma_transitions(sample, wanted, method) for sample in PLASMA_CANDIDATES],
        ignore_index=True,
    )

    evidence_rows: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for analyte, components in ANALYTES.items():
        quantifier = components[0]
        quantifier_rows = plasma_all[plasma_all["component"].eq(quantifier)]
        if quantifier_rows.empty:
            representative = ""
        else:
            representative = str(
                quantifier_rows.sort_values("apex_intensity", ascending=False).iloc[0]["sample"]
            )
        plasma = plasma_all[
            plasma_all["sample"].eq(representative) & plasma_all["component"].isin(components)
        ].copy()
        standard = standards[standards["component"].isin(components)].copy()
        paired = plasma.merge(
            standard,
            on="component",
            how="outer",
            suffixes=("_plasma", "_standard"),
        )
        paired["analyte"] = analyte
        paired["representative_plasma_sample"] = representative
        paired = paired.merge(method[method_cols], on="component", how="left")

        paired["plasma_relative_intensity_pct"] = (
            paired["apex_intensity_plasma"] / paired["apex_intensity_plasma"].max() * 100
        )
        paired["standard_relative_intensity_pct"] = (
            paired["apex_intensity_standard"] / paired["apex_intensity_standard"].max() * 100
        )
        paired["plasma_rt_deviation_from_final_expected_min"] = (
            paired["apex_rt_min_plasma"] - paired["method_expected_rt_min"]
        )
        paired["standard_rt_deviation_from_curve_expected_min"] = (
            paired["apex_rt_min_standard"] - paired["curve_expected_rt_min"]
        )
        evidence_rows.append(paired)

        common = paired.dropna(
            subset=["plasma_relative_intensity_pct", "standard_relative_intensity_pct"]
        )
        similarity = (
            cosine_similarity(
                common["plasma_relative_intensity_pct"].to_numpy(float),
                common["standard_relative_intensity_pct"].to_numpy(float),
            )
            if len(common) >= 2
            else np.nan
        )
        quant = paired[paired["component"].eq(quantifier)]
        plasma_rt_deviation = (
            float(quant.iloc[0]["plasma_rt_deviation_from_final_expected_min"])
            if not quant.empty
            else np.nan
        )
        standard_rt_deviation = (
            float(quant.iloc[0]["standard_rt_deviation_from_curve_expected_min"])
            if not quant.empty
            else np.nan
        )
        audit_rows.append(
            {
                "analyte": analyte,
                "representative_plasma_sample": representative,
                "monitored_transitions_expected": len(components),
                "common_transitions_plasma_and_standard": len(common),
                "transition_profile_cosine_similarity": similarity,
                "quantifier_plasma_rt_deviation_from_final_expected_min": plasma_rt_deviation,
                "quantifier_standard_rt_deviation_from_curve_expected_min": standard_rt_deviation,
                "lc_method_note": "Plasma and standard acquisitions used different LC methods; direct RT difference is not interpreted.",
                "evidence_class": (
                    "multi-transition MRM profile plus RT"
                    if len(common) >= 2
                    else "single-transition MRM plus RT"
                ),
                "interpretation_limit": (
                    "Supports transition-ratio and RT agreement; not a full product-ion spectrum."
                    if len(common) >= 2
                    else "Supports monitored Q1/Q3 and RT only; no qualifier-ion ratio or full product-ion spectrum."
                ),
            }
        )
    return pd.concat(evidence_rows, ignore_index=True), pd.DataFrame(audit_rows)


def plot_mirror(ax, analyte: str, evidence: pd.DataFrame, audit: pd.Series, show_xlabel: bool) -> None:
    sub = evidence[evidence["analyte"].eq(analyte)].sort_values("q3_plasma")
    x = np.arange(len(sub), dtype=float)
    plasma = sub["plasma_relative_intensity_pct"].fillna(0).to_numpy(float)
    standard = sub["standard_relative_intensity_pct"].fillna(0).to_numpy(float)

    ax.vlines(x, 0, plasma, color=SIGNAL_RED, linewidth=2.2, label="Representative plasma")
    ax.vlines(x, 0, -standard, color=SIGNAL_BLUE, linewidth=2.2, label="Authentic standard")
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.set_ylim(-118, 118)
    ax.set_xlim(-0.55, max(len(sub) - 0.45, 0.55))
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_ylabel("Relative intensity (%)")
    ax.set_xticks(x)
    q3_labels = [
        f"{to_float(row.q3_plasma if pd.notna(row.q3_plasma) else row.q3_standard):.3f}"
        for row in sub.itertuples()
    ]
    ax.set_xticklabels(q3_labels)
    ax.set_xlabel("Product ion m/z" if show_xlabel else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)

    q1 = sub["q1_plasma"].dropna()
    q1_text = f"{q1.iloc[0]:.3f}" if not q1.empty else "NA"
    sample = audit["representative_plasma_sample"]
    ax.set_title(analyte, loc="left", fontweight="bold", fontsize=8.5, pad=8)
    ax.text(
        0.98,
        0.97,
        f"Q1 {q1_text} | plasma {sample} | standard {STANDARD_SAMPLE}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.9,
        color="#555555",
    )
    cosine = audit["transition_profile_cosine_similarity"]
    cosine_text = f"cosine={cosine:.3f}" if pd.notna(cosine) else "cosine=NA (single transition)"
    plasma_dev = audit["quantifier_plasma_rt_deviation_from_final_expected_min"]
    standard_dev = audit["quantifier_standard_rt_deviation_from_curve_expected_min"]
    plasma_dev_text = f"{plasma_dev:+.3f}" if pd.notna(plasma_dev) else "NA"
    standard_dev_text = f"{standard_dev:+.3f}" if pd.notna(standard_dev) else "NA"
    ax.text(
        0.99,
        0.04,
        f"{audit['common_transitions_plasma_and_standard']} transition(s); {cosine_text}\n"
        f"RT dev. plasma/standard: {plasma_dev_text}/{standard_dev_text} min",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#555555",
        linespacing=1.15,
    )


def make_figures(evidence: pd.DataFrame, audit: pd.DataFrame) -> None:
    for analyte in ANALYTES:
        fig, ax = plt.subplots()
        row = audit[audit["analyte"].eq(analyte)].iloc[0]
        plot_mirror(ax, analyte, evidence, row, show_xlabel=True)
        handles = [
            plt.Line2D([0], [0], color=SIGNAL_RED, lw=2.2, label="Representative plasma"),
            plt.Line2D([0], [0], color=SIGNAL_BLUE, lw=2.2, label="Authentic standard"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, bbox_to_anchor=(0, 0.98))
        fig.tight_layout()
        save_publication_figure(
            fig,
            PDF_DIR / f"{safe_name(analyte)}_MRM_transition_profile_mirror",
            width=4.2,
            height=2.75,
        )
        # save_publication_figure writes all formats beside the PDF; move format-specific copies below.
        plt.close(fig)

    # Move the format variants into the requested format directories.
    for pdf in PDF_DIR.glob("*.svg"):
        pdf.replace(SVG_DIR / pdf.name)
    for pdf in PDF_DIR.glob("*.png"):
        pdf.replace(PNG_DIR / pdf.name)

    fig, axes = plt.subplots(4, 2, figsize=(7.4, 10.0))
    for index, (ax, analyte) in enumerate(zip(axes.flat, ANALYTES)):
        row = audit[audit["analyte"].eq(analyte)].iloc[0]
        plot_mirror(ax, analyte, evidence, row, show_xlabel=index >= 6)
    handles = [
        plt.Line2D([0], [0], color=SIGNAL_RED, lw=2.2, label="Representative plasma"),
        plt.Line2D([0], [0], color=SIGNAL_BLUE, lw=2.2, label="Authentic standard"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Targeted analyte MRM transition evidence",
        fontsize=10,
        fontweight="bold",
        color=TEXT_DARK,
        y=1.015,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.4, w_pad=1.2)
    save_publication_figure(
        fig,
        PDF_DIR / "Eight_targeted_analytes_MRM_transition_profile_mirror_contact_sheet",
        width=7.4,
        height=10.0,
    )
    plt.close(fig)
    for path in PDF_DIR.glob("Eight_targeted_analytes_MRM_transition_profile_mirror_contact_sheet.svg"):
        path.replace(SVG_DIR / path.name)
    for path in PDF_DIR.glob("Eight_targeted_analytes_MRM_transition_profile_mirror_contact_sheet.png"):
        path.replace(PNG_DIR / path.name)


def write_report(evidence: pd.DataFrame, audit: pd.DataFrame) -> None:
    lines = [
        "# Eight targeted analytes: MRM transition and authentic-standard evidence",
        "",
        "## Evidence type",
        "",
        "The available SCIEX mzML files contain selected-reaction-monitoring chromatograms, not full product-ion scan spectra. "
        "The generated mirror plots therefore compare normalized monitored-transition responses in representative plasma samples "
        "with the S6 authentic-standard mixture. They must be described as **MRM transition-profile mirror plots**, not full MS/MS library matches.",
        "",
        "## Per-analyte audit",
        "",
        "| Analyte | Plasma sample | Common transitions | Cosine | Plasma RT dev. | Standard RT dev. | Evidence class |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit.itertuples():
        cosine = f"{row.transition_profile_cosine_similarity:.3f}" if pd.notna(row.transition_profile_cosine_similarity) else "NA"
        plasma_dev = f"{row.quantifier_plasma_rt_deviation_from_final_expected_min:+.3f}" if pd.notna(row.quantifier_plasma_rt_deviation_from_final_expected_min) else "NA"
        standard_dev = f"{row.quantifier_standard_rt_deviation_from_curve_expected_min:+.3f}" if pd.notna(row.quantifier_standard_rt_deviation_from_curve_expected_min) else "NA"
        lines.append(
            f"| {row.analyte} | {row.representative_plasma_sample} | "
            f"{row.common_transitions_plasma_and_standard} | {cosine} | {plasma_dev} | {standard_dev} | {row.evidence_class} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Acetylcarnitine, creatine, carnitine, phenylalanine and tryptophan have multiple monitored analyte transitions and can support qualifier-ion profile comparison.",
            "- 3-Guanidinopropionic acid, DHEA-S and arginine currently have only one monitored analyte transition in the method. These plots support Q1/Q3 and retention-time agreement only.",
            "- Full Level-1-style MS/MS spectral matching would require authentic-standard and plasma product-ion scan spectra acquired under comparable collision conditions.",
            "- Transition-profile cosine values are descriptive and should not be presented as library-search similarity scores.",
            "- The standard-curve and final plasma acquisitions used different LC methods. RT evidence is therefore reported as deviation from the expected RT within each method, not as a direct plasma-standard RT difference.",
            "",
            "## Recommended manuscript wording",
            "",
            "Authentic-standard support was assessed by retention-time agreement and comparison of monitored MRM transitions between the standard mixture and representative plasma samples. "
            "For analytes with multiple monitored transitions, relative transition-intensity profiles were compared descriptively.",
        ]
    )
    (REPORT_DIR / "MRM_transition_standard_evidence_QA.md").write_text("\n".join(lines), encoding="utf-8")

    readme = [
        "# 06 MRM transition and standard evidence",
        "",
        "This directory contains evidence figures for the eight targeted analytes.",
        "",
        "- `figures/pdf`: editable PDF mirror plots",
        "- `figures/svg`: editable SVG mirror plots",
        "- `figures/png_preview`: high-resolution previews",
        "- `tables`: transition-level source data and evidence audit",
        "- `reports`: evidence interpretation and limitations",
        "",
        "Important: these are MRM transition-profile mirror plots, not full MS/MS product-ion spectra.",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    evidence, audit = prepare_evidence()
    evidence.to_csv(TABLE_DIR / "eight_analyte_transition_level_evidence.tsv", sep="\t", index=False)
    audit.to_csv(TABLE_DIR / "eight_analyte_spectral_evidence_audit.tsv", sep="\t", index=False)
    make_figures(evidence, audit)
    write_report(evidence, audit)
    print(f"Wrote {len(audit)} analyte audits and figures to {OUT_ROOT}")


if __name__ == "__main__":
    main()
