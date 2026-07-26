from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
STYLE_DIR = ROOT / "00_project_style"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
from ov_publication_style import (
    BACKGROUND_GREY,
    GROUP_COLORS as OV_GROUP_COLORS,
    HEATMAP_BLUE,
    HEATMAP_RED,
    HEATMAP_WHITE,
    SIGNAL_BLUE,
    SIGNAL_RED,
    setup_matplotlib_style,
)

MOD = ROOT / "03B_pairwise_diff"
SOURCE = MOD / "data" / "source_data"
RESULTS = MOD / "data" / "results_tables"
FIG = MOD / "figure_final"
CLINICAL = ROOT / "01_cohort_and_design" / "data_clean" / "clinical_merged_analysis_ready.csv"
FIG2 = ROOT / "98figure" / "01_main_figure_candidates" / "Fig2"
QA = FIG / "Fig2_volcano_heatmap_publication_redraw_QA.md"

GROUP_ORDER = ["N", "B", "BD", "M"]
GROUP_LABEL = {"N": "Normal", "B": "Benign", "BD": "Borderline", "M": "Malignant"}
GROUP_COLORS = {"N": OV_GROUP_COLORS["N"], "B": OV_GROUP_COLORS["B"], "BD": OV_GROUP_COLORS["BD"], "M": OV_GROUP_COLORS["M"]}
COMP_ORDER = ["N_vs_B", "N_vs_BD", "N_vs_M", "B_vs_BD", "B_vs_M", "BD_vs_M"]
COMP_LABEL = {"N_vs_B": "N vs B", "N_vs_BD": "N vs BD", "N_vs_M": "N vs M", "B_vs_BD": "B vs BD", "B_vs_M": "B vs M", "BD_vs_M": "BD vs M"}
COMP_COLORS = {
    "N_vs_B": "#8DD3C7",
    "N_vs_BD": "#FDB462",
    "N_vs_M": "#FB8072",
    "B_vs_BD": "#BEBADA",
    "B_vs_M": "#FDCDE5",
    "BD_vs_M": "#B3B3B3",
}


def setup_style() -> None:
    setup_matplotlib_style(mpl, sns=sns, base_size=7)


def save_fig(fig: plt.Figure, stem: str, width: float, height: float, out_dir: Path = FIG) -> None:
    fig.set_size_inches(width, height)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def draw_multivolcano() -> pd.DataFrame:
    df = pd.read_csv(SOURCE / "MERGED__Fig3_grouped_multivolcano_source.csv")
    df = df[df["comparison"].isin(COMP_ORDER)].copy()
    df["comparison"] = pd.Categorical(df["comparison"], categories=COMP_ORDER, ordered=True)
    df["x"] = df["comparison"].cat.codes + 1
    rng = np.random.default_rng(123)
    df["x_jit"] = df["x"] + rng.uniform(-0.22, 0.22, len(df))
    df["direction"] = np.where(df["logFC"] >= 0, "Higher in second group", "Lower in second group")
    df["sig"] = df["adj.P.Val"] < 0.05

    counts = df[df["sig"]].assign(direction=np.where(df[df["sig"]]["logFC"] >= 0, "up", "down")).groupby(["comparison", "direction"], observed=False).size().unstack(fill_value=0)
    counts = counts.reindex(COMP_ORDER).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for i, comp in enumerate(COMP_ORDER, start=1):
        sub = df[df["comparison"].eq(comp)]
        q_hi = max(np.nanquantile(sub["logFC"], 0.975), 0)
        q_lo = min(np.nanquantile(sub["logFC"], 0.025), 0)
        ax.add_patch(plt.Rectangle((i - 0.38, 0), 0.76, q_hi, facecolor=BACKGROUND_GREY, alpha=0.16, edgecolor="none", zorder=0))
        ax.add_patch(plt.Rectangle((i - 0.38, q_lo), 0.76, -q_lo, facecolor=BACKGROUND_GREY, alpha=0.16, edgecolor="none", zorder=0))

    nonsig = df[~df["sig"]]
    sig_up = df[df["sig"] & (df["logFC"] >= 0)]
    sig_down = df[df["sig"] & (df["logFC"] < 0)]
    ax.scatter(nonsig["x_jit"], nonsig["logFC"], s=7, c=BACKGROUND_GREY, alpha=0.30, linewidths=0, label="FDR >= 0.05", zorder=1)
    ax.scatter(sig_up["x_jit"], sig_up["logFC"], s=13, c=SIGNAL_RED, alpha=0.90, marker="o", linewidths=0, label="FDR < 0.05, logFC > 0", zorder=2)
    ax.scatter(sig_down["x_jit"], sig_down["logFC"], s=15, c=SIGNAL_BLUE, alpha=0.90, marker="v", linewidths=0, label="FDR < 0.05, logFC < 0", zorder=2)

    ymin = float(df["logFC"].min())
    ymax = float(df["logFC"].max())
    span = ymax - ymin
    tile_y = ymin - span * 0.10
    for i, comp in enumerate(COMP_ORDER, start=1):
        ax.add_patch(plt.Rectangle((i - 0.38, tile_y - 0.12), 0.76, 0.24, facecolor=COMP_COLORS[comp], edgecolor="#333333", lw=0.45, clip_on=False))
        ax.text(i, tile_y, COMP_LABEL[comp], ha="center", va="center", color="white", fontsize=6.7, fontweight="bold", clip_on=False)
        up = int(counts.loc[comp, "up"]) if "up" in counts.columns else 0
        down = int(counts.loc[comp, "down"]) if "down" in counts.columns else 0
        ax.text(i, tile_y - 0.26, f"{up}/{down}", ha="center", va="top", fontsize=6.2, clip_on=False)

    ax.axhline(0, color="#9E9E9E", lw=0.55)
    ax.set_xlim(0.45, len(COMP_ORDER) + 0.55)
    ax.set_ylim(tile_y - 0.45, ymax + span * 0.06)
    ax.set_xticks([])
    ax.set_ylabel("Average log2 fold change")
    ax.set_title("Grouped multi-comparison differential features", loc="left")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.3)
    ax.text(0.01, -0.02, "Numbers below labels indicate significant features with positive/negative logFC.", transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
    sns.despine(ax=ax, bottom=True)
    save_fig(fig, "MERGED__Fig3_grouped_multivolcano", 7.2, 4.5)
    shutil.copy2(FIG / "MERGED__Fig3_grouped_multivolcano.pdf", FIG2 / "Fig2.D_Grouped_multi-comparison_volcano_summary_with_counts.pdf")
    if (FIG / "MERGED__Fig3_grouped_multivolcano.png").exists():
        shutil.copy2(FIG / "MERGED__Fig3_grouped_multivolcano.png", FIG2 / "Fig2.D_Grouped_multi-comparison_volcano_summary_with_counts.png")
    if (FIG / "MERGED__Fig3_grouped_multivolcano.svg").exists():
        shutil.copy2(FIG / "MERGED__Fig3_grouped_multivolcano.svg", FIG2 / "Fig2.D_Grouped_multi-comparison_volcano_summary_with_counts.svg")
    return counts.reset_index()


def clinical_annotations(sample_info: pd.DataFrame) -> pd.DataFrame:
    clin = pd.read_csv(CLINICAL)
    clin["sample_id"] = clin["sample_id"].astype(str)
    ann = sample_info.copy()
    ann["sample_id"] = ann["sample_id"].astype(str)
    ann = ann.merge(
        clin[["sample_id", "age_group", "CA125", "stage_binary", "pathology_class"]],
        on="sample_id",
        how="left",
    )
    ann["Group"] = ann["label"].map(GROUP_LABEL)
    ann["Age"] = ann["age_group"].fillna("Missing")
    ann["CA125"] = pd.cut(ann["CA125"], bins=[-np.inf, 35, np.inf], labels=["<=35", ">35"]).astype(object)
    ann.loc[ann["CA125"].isna(), "CA125"] = "Missing"
    ann["Stage"] = ann["stage_binary"].fillna("Not malignant/NA")
    ann["Pathology"] = ann["pathology_class"].fillna("Not malignant/NA")
    return ann


def select_heatmap_features(n: int = 30) -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS.glob("MERGED__*__diff_results.csv")):
        df = pd.read_csv(p)
        rows.append(df)
    allres = pd.concat(rows, ignore_index=True)
    allres["abs_logFC"] = allres["logFC"].abs()
    allres = allres[allres["adj.P.Val"] < 0.05].copy()
    ranked = (
        allres.sort_values(["adj.P.Val", "abs_logFC"], ascending=[True, False])
        .drop_duplicates("feature_id")
        .head(n)
        .copy()
    )
    return ranked


def draw_selected_heatmap() -> pd.DataFrame:
    proc = pd.read_csv(SOURCE / "MERGED_processed_matrix.csv")
    proc["label"] = pd.Categorical(proc["label"], categories=GROUP_ORDER, ordered=True)
    proc = proc.sort_values(["label", "sample_id"]).copy()
    selected = select_heatmap_features(30)
    features = [f for f in selected["feature_id"].tolist() if f in proc.columns]
    mat = proc[features].to_numpy(dtype=float)
    z = (mat - np.nanmean(mat, axis=0)) / np.nanstd(mat, axis=0)
    z = np.clip(z, -2.5, 2.5)
    heat = pd.DataFrame(z.T, index=features, columns=proc["sample_id"].astype(str).tolist())
    ann = clinical_annotations(proc[["sample_id", "label"]])

    group_codes = ann["label"].map({g: i for i, g in enumerate(GROUP_ORDER)}).astype(int).to_numpy()[None, :]
    age_levels = ["<40", "40-49", ">=50", "Missing"]
    ca_levels = ["<=35", ">35", "Missing"]
    stage_levels = ["Early (I-II)", "Advanced (III-IV)", "Not malignant/NA"]
    path_levels = ["Epithelial", "Sex cord-stromal", "Germ cell", "Carcinosarcoma", "Not malignant/NA"]
    ann_specs = [
        ("Group", ann["label"], GROUP_ORDER, [GROUP_COLORS[g] for g in GROUP_ORDER]),
        ("Age", ann["Age"], age_levels, ["#D9D9D9", "#BEBADA", "#80B1D3", "#F0F0F0"]),
        ("CA125", ann["CA125"], ca_levels, ["#8DD3C7", "#FB8072", "#F0F0F0"]),
        ("Stage", ann["Stage"], stage_levels, ["#8DD3C7", "#FB8072", "#F0F0F0"]),
        ("Pathology", ann["Pathology"], path_levels, ["#FB8072", "#FDB462", "#80B1D3", "#BEBADA", "#F0F0F0"]),
    ]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.36, 4.8], width_ratios=[6.3, 0.9], hspace=0.03, wspace=0.05)
    ax_ann = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[:, 1])

    ann_matrix = []
    ann_cmaps = []
    for _, values, levels, colors in ann_specs:
        code = np.array(pd.Categorical(values, categories=levels).codes, dtype=int)
        code[code < 0] = len(levels) - 1
        ann_matrix.append(code)
        ann_cmaps.append(ListedColormap(colors))

    for i, (name, _, levels, colors) in enumerate(ann_specs):
        ax_ann.imshow(np.array(ann_matrix[i])[None, :], aspect="auto", interpolation="nearest", cmap=ann_cmaps[i], vmin=0, vmax=len(levels) - 1, extent=[0, heat.shape[1], i, i + 1])
    ax_ann.set_xlim(0, heat.shape[1])
    ax_ann.set_ylim(0, len(ann_specs))
    ax_ann.set_yticks(np.arange(len(ann_specs)) + 0.5)
    ax_ann.set_yticklabels([s[0] for s in ann_specs], fontsize=6.1)
    ax_ann.set_xticks([])
    ax_ann.invert_yaxis()
    for spine in ax_ann.spines.values():
        spine.set_visible(False)

    display_names = []
    for name in heat.index:
        clean = str(name).replace(" (not validated)", "").replace("Not validated", "")
        display_names.append(clean if len(clean) <= 30 else clean[:27] + "...")
    cmap = LinearSegmentedColormap.from_list("ov_red_blue", [HEATMAP_BLUE, HEATMAP_WHITE, HEATMAP_RED])
    sns.heatmap(heat, cmap=cmap, center=0, vmin=-2.5, vmax=2.5, xticklabels=False, yticklabels=display_names, cbar_kws={"label": "Z-score"}, ax=ax_heat)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontsize=5.0)
    for boundary in np.cumsum(ann["label"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int).values)[:-1]:
        ax_heat.axvline(boundary, color="white", lw=1.0)
        ax_ann.axvline(boundary, color="white", lw=1.0)

    ax_leg.axis("off")
    y = 1.0
    for name, _, levels, colors in ann_specs:
        ax_leg.text(0, y, name, transform=ax_leg.transAxes, fontsize=6.3, fontweight="bold", va="top")
        y -= 0.04
        for level, color in zip(levels, colors):
            ax_leg.add_patch(plt.Rectangle((0, y - 0.018), 0.08, 0.022, transform=ax_leg.transAxes, color=color, clip_on=False))
            ax_leg.text(0.11, y - 0.006, str(level), transform=ax_leg.transAxes, fontsize=5.4, va="center")
            y -= 0.035
        y -= 0.03
    fig.suptitle(f"Selected differential features with clinical annotations (n={len(features)})", x=0.08, y=0.98, ha="left", fontsize=7.5)
    save_fig(fig, "MERGED__Heatmap_selected_features_clinical_annotations", 7.2, 4.8)
    selected.to_csv(SOURCE / "MERGED__Heatmap_selected_features_clinical_annotations_feature_list.csv", index=False)
    ann.to_csv(SOURCE / "MERGED__Heatmap_selected_features_clinical_annotations_sample_info.csv", index=False)
    heat.to_csv(SOURCE / "MERGED__Heatmap_selected_features_clinical_annotations_matrix.csv")
    return selected


def write_qa(counts: pd.DataFrame, selected: pd.DataFrame) -> None:
    pdf_lines = []
    for name in ["MERGED__Fig3_grouped_multivolcano.pdf", "MERGED__Heatmap_selected_features_clinical_annotations.pdf"]:
        p = FIG / name
        b = p.read_bytes()
        pdf_lines.append(f"- `{name}`: /Type3={b.count(b'/Type3')}, /Font={b.count(b'/Font')}")
    lines = [
        "# Fig. 2 volcano and selected heatmap redraw QA",
        "",
        "- Multi-volcano non-significant points were changed to light grey with low alpha.",
        "- Significant positive logFC points are red circles; significant negative logFC points are blue triangles.",
        "- Selected-feature heatmap uses top significant features across six pairwise comparisons and adds clinical annotations.",
        f"- Selected heatmap feature count: {len(selected)}.",
        "",
        "## Significant feature counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## PDF font screen",
        "",
        *pdf_lines,
    ]
    QA.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    counts = draw_multivolcano()
    selected = draw_selected_heatmap()
    write_qa(counts, selected)


if __name__ == "__main__":
    main()
