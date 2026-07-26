from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "12_scFEA"
OUT = ROOT / "04_downstream_analysis"
FIG = ROOT / "05_figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "00_project_style"))

try:
    from ov_publication_style import (
        HEATMAP_BLUE,
        HEATMAP_RED,
        HEATMAP_WHITE,
        SIGNAL_BLUE,
        SIGNAL_RED,
        TEXT_DARK,
        save_publication_figure,
        setup_matplotlib_style,
    )
except Exception:
    HEATMAP_BLUE = "#2166AC"
    HEATMAP_WHITE = "#F7F7F7"
    HEATMAP_RED = "#B2182B"
    SIGNAL_BLUE = "#80B1D3"
    SIGNAL_RED = "#FB8072"
    TEXT_DARK = "#2F2F2F"

    def setup_matplotlib_style(mpl, sns=None, base_size: float = 7.0) -> None:
        mpl.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": base_size})

    def save_publication_figure(fig, out_stem: Path, width: float, height: float, dpi: int = 600) -> None:
        fig.set_size_inches(width, height)
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in [".pdf", ".svg", ".png"]:
            fig.savefig(out_stem.with_suffix(ext), bbox_inches="tight", dpi=dpi)


setup_matplotlib_style(matplotlib, base_size=7.5)
NORMAL = SIGNAL_BLUE
TUMOUR = SIGNAL_RED
NS = "#D9D9D9"

DATASETS = {
    "GSE217517": {
        "comparison": "is_tumor_cell",
        "h5ad": PROJECT_ROOT
        / "08_multiomics_validation_scRNA_TCGA"
        / "08A_scRNA_preprocess"
        / "GSE217517"
        / "10_kegg_enrichment"
        / "adata_kegg_scored.h5ad",
    },
    "GSE184880": {
        "comparison": "is_tumor_cell",
        "h5ad": PROJECT_ROOT
        / "08_multiomics_validation_scRNA_TCGA"
        / "08A_scRNA_preprocess"
        / "scRNAGSE184880"
        / "10_kegg_enrichment"
        / "adata_kegg_scored.h5ad",
    },
}

TARGET_AXES = [
    {
        "axis": "Arginine/proline/creatine",
        "plasma_analytes": "arginine; creatine; 3-guanidinopropionic acid",
        "display_module": "M_19",
        "display_label": "GATM/GAMT module",
        "query_genes": [
            "GATM",
            "GAMT",
            "ARG1",
            "ARG2",
            "ASS1",
            "ASL",
            "OAT",
            "PRODH",
            "PRODH2",
            "PYCR1",
            "CKB",
            "CKM",
            "SLC6A8",
            "AGMAT",
            "ODC1",
        ],
    },
    {
        "axis": "Tryptophan-related",
        "plasma_analytes": "tryptophan",
        "display_module": "M_60",
        "display_label": "AADAT-linked module",
        "query_genes": ["IDO1", "IDO2", "TDO2", "KMO", "KYNU", "AADAT", "AFMID", "TPH1", "TPH2"],
    },
    {
        "axis": "Phenylalanine-related",
        "plasma_analytes": "phenylalanine",
        "display_module": "M_57",
        "display_label": "FAH/HPD/TAT module",
        "query_genes": ["PAH", "TAT", "HPD", "FAH", "AOC2", "AOC3", "MAOA", "MAOB"],
    },
    {
        "axis": "Steroid-related",
        "plasma_analytes": "DHEA-S",
        "display_module": "M_169",
        "display_label": "steroid-enzyme module",
        "query_genes": [
            "STS",
            "SULT2A1",
            "CYP11A1",
            "CYP17A1",
            "CYP19A1",
            "HSD3B1",
            "HSD3B2",
            "HSD17B1",
            "SRD5A1",
            "SRD5A2",
            "CYP51A1",
        ],
    },
    {
        "axis": "Carnitine/FAO-related",
        "plasma_analytes": "carnitine; acetylcarnitine",
        "display_module": "M_35",
        "display_label": "FAO/CPT module",
        "query_genes": [
            "CPT1A",
            "CPT1B",
            "CPT1C",
            "CPT2",
            "CRAT",
            "CROT",
            "SLC22A5",
            "ACADM",
            "ACADL",
            "ACADVL",
            "HADHA",
            "HADHB",
        ],
    },
]
DISPLAY_MODULES = [item["display_module"] for item in TARGET_AXES]


def clean_old_generic_figures() -> None:
    old_stems = [
        "01_flux_module_differential_overview",
        "02_cross_dataset_flux_direction_consistency",
        "03_top_differential_flux_sample_heatmap",
        "04_representative_sample_level_flux_modules",
        "05_GSE217517_UMAP_flux_overlays",
        "05_GSE184880_UMAP_flux_overlays",
    ]
    for stem in old_stems:
        for ext in [".pdf", ".svg", ".png"]:
            path = FIG / f"{stem}{ext}"
            if path.exists():
                path.unlink()


def decode(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr])
    return arr.astype(str)


def h5ad_obs_column(obs: h5py.Group, name: str) -> np.ndarray:
    obj = obs[name]
    if isinstance(obj, h5py.Dataset):
        return decode(obj[()])
    categories = decode(obj["categories"][()])
    codes = obj["codes"][()]
    out = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out.astype(str)


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.clip(q, 0, 1)
    return pd.Series(result, index=pvalues.index)


def module_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / "01_software" / "scFEA" / "data" / "module_gene_m168.csv", index_col=0)


def build_target_axis_module_map(module_gene: pd.DataFrame) -> pd.DataFrame:
    records = []
    for module, row in module_gene.iterrows():
        genes = [str(x) for x in row.dropna().tolist()]
        gene_set = set(genes)
        for axis in TARGET_AXES:
            hits = sorted(gene_set.intersection(axis["query_genes"]))
            if not hits:
                continue
            records.append(
                {
                    "axis": axis["axis"],
                    "module": module,
                    "module_genes": ";".join(genes),
                    "hit_genes": ";".join(hits),
                    "plasma_analytes": axis["plasma_analytes"],
                    "recommended_display": "yes" if module == axis["display_module"] else "no",
                    "display_label": axis["display_label"] if module == axis["display_module"] else "",
                    "interpretation_note": "target-axis scFEA module; transcriptome-inferred flux only",
                }
            )
    result = pd.DataFrame(records).sort_values(["axis", "recommended_display", "module"], ascending=[True, False, True])
    result.to_csv(OUT / "targeted_scFEA_axis_module_map.tsv", sep="\t", index=False)
    return result


def module_annotations(axis_map: pd.DataFrame) -> pd.DataFrame:
    module_gene = module_table()
    display_map = axis_map.groupby("module")["axis"].apply(lambda s: "; ".join(sorted(set(s)))).to_dict()
    records = []
    for module, row in module_gene.iterrows():
        genes = [str(x) for x in row.dropna().tolist()]
        records.append(
            {
                "module": module,
                "module_genes": ";".join(genes),
                "representative_genes": "/".join(genes[:4]),
                "focus_theme": display_map.get(module, "Other metabolic module"),
            }
        )
    result = pd.DataFrame(records).set_index("module")
    result.to_csv(OUT / "scFEA_module_annotation.tsv", sep="\t")
    return result


def load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    flux = pd.read_csv(
        ROOT / "03_results" / f"{dataset}_epithelial" / "full" / f"{dataset}_epithelial_flux_m168.csv",
        index_col=0,
    )
    meta = pd.read_csv(
        ROOT / "02_inputs" / f"{dataset}_epithelial" / f"{dataset}_epithelial_cell_metadata.tsv",
        sep="\t",
        index_col=0,
    )
    common = meta.index.intersection(flux.index)
    return flux.loc[common], meta.loc[common]


def paired_sample_state(matrix: pd.DataFrame, meta: pd.DataFrame, comparison: str) -> pd.DataFrame:
    sample_state = matrix.join(meta[["sample", comparison]]).groupby(["sample", comparison], observed=False).mean()
    return sample_state


def paired_stats_from_sample_state(sample_state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in sample_state.columns:
        wide = sample_state[feature].unstack(level=1)
        complete = wide.dropna(subset=["Normal", "Tumor"])
        normal = complete["Normal"].to_numpy()
        tumour = complete["Tumor"].to_numpy()
        try:
            pvalue = wilcoxon(tumour, normal, alternative="two-sided").pvalue
        except ValueError:
            pvalue = 1.0
        effect = float(np.median(tumour - normal)) if len(complete) else np.nan
        rows.append(
            {
                "feature": feature,
                "effect_tumour_minus_normal": effect,
                "p_value": pvalue,
                "test": "paired Wilcoxon signed-rank",
                "n_normal_samples": len(complete),
                "n_tumour_samples": len(complete),
                "normal_median": float(np.median(normal)) if len(complete) else np.nan,
                "tumour_median": float(np.median(tumour)) if len(complete) else np.nan,
            }
        )
    result = pd.DataFrame(rows).set_index("feature")
    result["fdr_bh"] = bh_fdr(result["p_value"])
    return result


def differential_flux(dataset: str, flux: pd.DataFrame, meta: pd.DataFrame, annotation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = DATASETS[dataset]["comparison"]
    sample_state = paired_sample_state(flux, meta, comparison)
    sample_state.to_csv(OUT / f"{dataset}_sample_state_flux_summary.tsv", sep="\t")
    result = paired_stats_from_sample_state(sample_state)
    result.index.name = "module"
    result = result.join(annotation)
    result.sort_values(["fdr_bh", "p_value"]).to_csv(OUT / f"{dataset}_sample_aware_differential_flux.tsv", sep="\t")
    return result, sample_state


def differential_balance(dataset: str, meta: pd.DataFrame) -> None:
    balance = pd.read_csv(
        ROOT / "03_results" / f"{dataset}_epithelial" / "full" / f"{dataset}_epithelial_metabolite_balance_c70.csv",
        index_col=0,
    )
    common = meta.index.intersection(balance.index)
    sample_state = paired_sample_state(balance.loc[common], meta.loc[common], DATASETS[dataset]["comparison"])
    sample_state.to_csv(OUT / f"{dataset}_sample_state_metabolite_balance_summary.tsv", sep="\t")
    result = paired_stats_from_sample_state(sample_state)
    result.index.name = "metabolite_balance"
    result.sort_values(["fdr_bh", "p_value"]).to_csv(
        OUT / f"{dataset}_sample_aware_differential_metabolite_balance.tsv", sep="\t"
    )


def extract_gatm_expression() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    stats_rows = []
    for dataset, config in DATASETS.items():
        with h5py.File(config["h5ad"], "r") as handle:
            obs = handle["obs"]
            cell_ids = decode(obs["_index"][()])
            var_names = decode(handle["var"]["_index"][()])
            gene_index = {gene: idx for idx, gene in enumerate(var_names)}
            if "GATM" not in gene_index:
                raise ValueError(f"GATM not found in {dataset}")
            meta = pd.DataFrame(index=cell_ids)
            for column in ["cell_type", "sample", "is_tumor_cell", "group"]:
                meta[column] = h5ad_obs_column(obs, column)
            epithelial = meta["cell_type"].str.startswith("Epithelial").to_numpy()
            values = np.asarray(handle["layers"]["log1p"][:, gene_index["GATM"]]).ravel()[epithelial]
            meta = meta.loc[epithelial].copy()
            meta["GATM_log_expr"] = values
        sample_state = meta.groupby(["sample", "is_tumor_cell"], observed=False)["GATM_log_expr"].mean().unstack("is_tumor_cell")
        complete = sample_state.dropna(subset=["Normal", "Tumor"]).copy()
        complete["paired_difference_tumour_minus_normal"] = complete["Tumor"] - complete["Normal"]
        try:
            pvalue = wilcoxon(complete["Tumor"], complete["Normal"], alternative="two-sided").pvalue
        except ValueError:
            pvalue = np.nan
        for sample, row in complete.iterrows():
            all_rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "normal_like_mean_log_expr": row["Normal"],
                    "tumour_like_mean_log_expr": row["Tumor"],
                    "paired_difference_tumour_minus_normal": row["paired_difference_tumour_minus_normal"],
                }
            )
        stats_rows.append(
            {
                "dataset": dataset,
                "gene": "GATM",
                "n_paired_samples": len(complete),
                "median_difference_tumour_minus_normal": complete["paired_difference_tumour_minus_normal"].median(),
                "wilcoxon_p_value": pvalue,
                "interpretation": "supportive trend only; not statistically significant",
            }
        )
    audit = pd.DataFrame(all_rows)
    stats = pd.DataFrame(stats_rows)
    audit = audit.merge(stats[["dataset", "wilcoxon_p_value"]], on="dataset", how="left")
    audit.to_csv(OUT / "GATM_expression_tumour_like_vs_normal_like.tsv", sep="\t", index=False)
    stats.to_csv(OUT / "GATM_expression_summary.tsv", sep="\t", index=False)
    return audit, stats


def paired_arrays(sample_state: pd.DataFrame, feature: str) -> pd.DataFrame:
    wide = sample_state[feature].unstack(level=1).dropna(subset=["Normal", "Tumor"]).copy()
    wide["paired_difference_tumour_minus_normal"] = wide["Tumor"] - wide["Normal"]
    return wide


def draw_paired_panel(ax, wide: pd.DataFrame, normal_label: str = "Normal-like", tumour_label: str = "Tumour-like") -> None:
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row["Normal"], row["Tumor"]], color="#777777", lw=0.6, alpha=0.65, zorder=1)
    ax.scatter(np.zeros(len(wide)), wide["Normal"], color=NORMAL, s=18, edgecolor="white", lw=0.4, zorder=2)
    ax.scatter(np.ones(len(wide)), wide["Tumor"], color=TUMOUR, s=18, edgecolor="white", lw=0.4, zorder=2)
    box = ax.boxplot(
        [wide["Normal"].to_numpy(), wide["Tumor"].to_numpy()],
        positions=[0, 1],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], [NORMAL, TUMOUR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(TEXT_DARK)
    for line in box["medians"]:
        line.set_color(TEXT_DARK)
        line.set_linewidth(0.8)
    ax.set_xticks([0, 1], [normal_label, tumour_label])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_target_axis_flux(dataset: str, sample_state: pd.DataFrame, result: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(3.65, 8.4), sharex=False)
    records = []
    for ax, axis in zip(axes, TARGET_AXES):
        module = axis["display_module"]
        wide = paired_arrays(sample_state, module)
        draw_paired_panel(ax, wide)
        stat = result.loc[module]
        ax.set_title(f"{axis['axis']} | {module} ({axis['display_label']})", loc="left", pad=2)
        ax.set_ylabel("scFEA flux")
        ax.text(
            0.02,
            0.96,
            f"FDR={stat['fdr_bh']:.3g}; median delta={stat['effect_tumour_minus_normal']:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.7,
        )
        records.append(
            {
                "dataset": dataset,
                "axis": axis["axis"],
                "module": module,
                "display_label": axis["display_label"],
                "plasma_analytes": axis["plasma_analytes"],
                "n_paired_samples": len(wide),
                "effect_tumour_minus_normal": stat["effect_tumour_minus_normal"],
                "p_value": stat["p_value"],
                "fdr_bh": stat["fdr_bh"],
            }
        )
    fig.suptitle(f"{dataset}: targeted scFEA axes", y=0.995, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.15)
    save_publication_figure(fig, FIG / f"{dataset}_targeted_axis_flux_boxplots", width=3.65, height=8.4)
    pd.DataFrame(records).to_csv(OUT / f"{dataset}_targeted_axis_display_statistics.tsv", sep="\t", index=False)


def plot_gatm_creatine_support(dataset: str, gatm_audit: pd.DataFrame, sample_state: pd.DataFrame, result: pd.DataFrame) -> None:
    gatm = gatm_audit.loc[gatm_audit["dataset"].eq(dataset)].set_index("sample")
    expr_wide = gatm.rename(
        columns={"normal_like_mean_log_expr": "Normal", "tumour_like_mean_log_expr": "Tumor"}
    )[["Normal", "Tumor"]]
    flux_wide = paired_arrays(sample_state, "M_19")
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.6))
    draw_paired_panel(axes[0], expr_wide)
    axes[0].set_title("GATM expression", loc="left")
    axes[0].set_ylabel("Mean log expression")
    p_expr = gatm["wilcoxon_p_value"].iloc[0]
    axes[0].text(0.03, 0.96, f"P={p_expr:.3g}", transform=axes[0].transAxes, va="top", fontsize=7)
    draw_paired_panel(axes[1], flux_wide)
    axes[1].set_title("M_19 GATM/GAMT flux", loc="left")
    axes[1].set_ylabel("scFEA flux")
    stat = result.loc["M_19"]
    axes[1].text(
        0.03,
        0.94,
        f"FDR={stat['fdr_bh']:.3g}\nmedian delta={stat['effect_tumour_minus_normal']:.2g}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=6.5,
    )
    fig.suptitle(f"{dataset}: creatine-linked GATM axis", y=1.02, fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    save_publication_figure(fig, FIG / f"{dataset}_GATM_creatine_axis_support", width=5.0, height=2.6)


def plot_target_axis_heatmap(dataset: str, sample_state: pd.DataFrame) -> None:
    modules = DISPLAY_MODULES
    labels = [f"{axis['display_module']} | {axis['axis']}" for axis in TARGET_AXES]
    wide_columns = []
    matrix_columns = []
    for sample in sample_state.index.get_level_values(0).unique():
        sample_slice = sample_state.loc[sample]
        if {"Normal", "Tumor"}.issubset(sample_slice.index):
            for state in ["Normal", "Tumor"]:
                matrix_columns.append(sample_slice.loc[state, modules])
                short_sample = sample.replace("GSM", "")
                wide_columns.append(f"{short_sample}\n{state}")
    matrix = pd.DataFrame(matrix_columns, index=wide_columns).T
    matrix.index = labels
    z = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ov_redblue", [HEATMAP_BLUE, HEATMAP_WHITE, HEATMAP_RED]
    )
    im = ax.imshow(z, aspect="auto", cmap=cmap, vmin=-2, vmax=2)
    ax.set_yticks(range(len(z.index)), z.index)
    ax.set_xticks(range(len(z.columns)), z.columns, rotation=90)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015)
    cbar.set_label("Sample-state flux Z-score")
    ax.set_title(f"{dataset}: target-axis sample-state flux", loc="left", fontweight="bold")
    save_publication_figure(fig, FIG / f"{dataset}_targeted_axis_sample_heatmap", width=7.2, height=2.45)


def write_targeted_qa(results: dict[str, pd.DataFrame], gatm_stats: pd.DataFrame) -> None:
    lines = [
        "# Targeted scFEA axis and GATM evidence QA",
        "",
        "## Interpretation rule",
        "",
        "- scFEA estimates transcriptome-inferred flux; it is not direct metabolic-flux measurement.",
        "- Both datasets are analysed as tumour-like versus normal-like epithelial-state comparisons using `is_tumor_cell`.",
        "- GATM is retained only as part of the arginine/glycine-to-creatine axis, not as a standalone significant gene result.",
        "",
        "## GATM expression",
        "",
    ]
    for _, row in gatm_stats.iterrows():
        lines.append(
            f"- {row['dataset']}: median paired difference={row['median_difference_tumour_minus_normal']:.4g}, "
            f"Wilcoxon P={row['wilcoxon_p_value']:.3g}; supportive trend only."
        )
    lines.extend(["", "## Display modules", ""])
    for axis in TARGET_AXES:
        lines.append(
            f"- {axis['axis']}: {axis['display_module']} ({axis['display_label']}), plasma analytes: {axis['plasma_analytes']}."
        )
    lines.extend(["", "## Dataset statistics for displayed modules", ""])
    for dataset, result in results.items():
        lines.append(f"### {dataset}")
        for axis in TARGET_AXES:
            stat = result.loc[axis["display_module"]]
            lines.append(
                f"- {axis['display_module']} {axis['axis']}: median delta={stat['effect_tumour_minus_normal']:.4g}, "
                f"P={stat['p_value']:.3g}, FDR={stat['fdr_bh']:.3g}, paired n={int(stat['n_normal_samples'])}."
            )
        lines.append("")
    (OUT / "targeted_scFEA_axis_and_GATM_QA.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    clean_old_generic_figures()
    axis_map = build_target_axis_module_map(module_table())
    annotation = module_annotations(axis_map)
    gatm_audit, gatm_stats = extract_gatm_expression()

    results: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    qa = {}
    for dataset in DATASETS:
        flux, meta = load_dataset(dataset)
        result, summary = differential_flux(dataset, flux, meta, annotation)
        differential_balance(dataset, meta)
        results[dataset] = result
        summaries[dataset] = summary
        plot_target_axis_flux(dataset, summary, result)
        plot_gatm_creatine_support(dataset, gatm_audit, summary, result)
        plot_target_axis_heatmap(dataset, summary)
        qa[dataset] = {
            "cells": len(flux),
            "modules": flux.shape[1],
            "nan_values": int(flux.isna().sum().sum()),
            "comparison_counts": meta[DATASETS[dataset]["comparison"]].value_counts().to_dict(),
            "paired_samples_for_state_comparison": int(
                summary.index.to_frame(index=False).pivot_table(
                    index="sample", columns=DATASETS[dataset]["comparison"], aggfunc="size", fill_value=0
                ).pipe(lambda x: ((x.get("Normal", 0) > 0) & (x.get("Tumor", 0) > 0)).sum())
            ),
        }
    consistency = results["GSE217517"][["effect_tumour_minus_normal", "fdr_bh"]].rename(
        columns={"effect_tumour_minus_normal": "GSE217517_effect", "fdr_bh": "GSE217517_fdr"}
    ).join(
        results["GSE184880"][["effect_tumour_minus_normal", "fdr_bh"]].rename(
            columns={"effect_tumour_minus_normal": "GSE184880_effect", "fdr_bh": "GSE184880_fdr"}
        )
    )
    consistency["direction_concordant"] = np.sign(consistency["GSE217517_effect"]) == np.sign(
        consistency["GSE184880_effect"]
    )
    consistency.to_csv(OUT / "cross_dataset_flux_direction_consistency.tsv", sep="\t")
    write_targeted_qa(results, gatm_stats)
    qa["cross_dataset_concordant_modules"] = int(consistency["direction_concordant"].sum())
    qa["targeted_axis_mode"] = "targeted plasma-analyte-linked axes; dataset-specific figures"
    qa["interpretation"] = (
        "scFEA estimates transcriptome-inferred flux. GATM expression is supportive only; "
        "M_19 GATM/GAMT flux is the creatine-linked module-level evidence."
    )
    (OUT / "scFEA_downstream_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
