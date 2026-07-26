#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from pathlib import Path

import gseapy as gp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import fisher_exact, ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
})

PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "08_multiomics_validation_scRNA_TCGA" / "08D_CPTAC_OV"
ANALYSIS_DIR = ROOT / "09_proteomics_tumor_vs_normal_kegg"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIGURE_DIR = ANALYSIS_DIR / "figures"
GMT_DIR = ANALYSIS_DIR / "gmt"

PROTEOMICS_FILE = ROOT / "exports" / "proteomics__umich.csv.gz"
SAMPLE_STATUS_FILE = ROOT / "summary" / "ov_proteomics_sample_status.csv"
LOCAL_KEGG_GMT = Path(os.environ.get("OV_KEGG_GMT", GMT_DIR / "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt"))

RANDOM_SEED = 20260330
MIN_PRESENT_FRACTION = 0.50
DIFF_FDR_CUTOFF = 0.05
DIFF_EFFECT_CUTOFF = 0.30
TOP_HEATMAP_GENES_PER_DIRECTION = 20
TARGET_KEGG_PATHWAYS = {
    "KEGG_ARGININE_AND_PROLINE_METABOLISM": {"kegg_id": "hsa00330", "label": "Arginine and proline metabolism"},
    "KEGG_PHENYLALANINE_TYROSINE_AND_TRYPTOPHAN_BIOSYNTHESIS": {"kegg_id": "hsa00400", "label": "Phenylalanine, tyrosine and tryptophan biosynthesis"},
    "KEGG_PHENYLALANINE_METABOLISM": {"kegg_id": "hsa00360", "label": "Phenylalanine metabolism"},
    "KEGG_ARGININE_BIOSYNTHESIS": {"kegg_id": "hsa00220", "label": "Arginine biosynthesis"},
    "KEGG_LYSINE_DEGRADATION": {"kegg_id": "hsa00310", "label": "Lysine degradation"},
    "KEGG_GLYCINE_SERINE_AND_THREONINE_METABOLISM": {"kegg_id": "hsa00260", "label": "Glycine, serine and threonine metabolism"},
    "KEGG_TRYPTOPHAN_METABOLISM": {"kegg_id": "hsa00380", "label": "Tryptophan metabolism"},
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS": {"kegg_id": "hsa00140", "label": "Steroid hormone biosynthesis"},
    "KEGG_TYROSINE_METABOLISM": {"kegg_id": "hsa00350", "label": "Tyrosine metabolism"},
    "KEGG_FATTY_ACID_DEGRADATION": {"kegg_id": "hsa00071", "label": "Fatty acid degradation"},
}
FOCUS_TERMS = [
    "KEGG_ARGININE_AND_PROLINE_METABOLISM",
    "KEGG_PHENYLALANINE_METABOLISM",
    "KEGG_TRYPTOPHAN_METABOLISM",
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS",
]


def pretty_term(term: str) -> str:
    label = str(term)
    if label.startswith("KEGG_"):
        label = label[5:]
    label = label.replace("_", " ").lower()
    return label


def sanitize_kegg_name(name: str) -> str:
    label = re.sub(r" - Homo sapiens \(human\)$", "", str(name)).strip()
    label = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"KEGG_{label}"


def ensure_dirs() -> None:
    for path in [ANALYSIS_DIR, TABLE_DIR, FIGURE_DIR, GMT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    prot = pd.read_csv(PROTEOMICS_FILE, header=[0, 1], index_col=0)
    prot.index = prot.index.astype(str)
    status = pd.read_csv(SAMPLE_STATUS_FILE)
    status["Patient_ID"] = status["Patient_ID"].astype(str)
    status = status.set_index("Patient_ID").loc[prot.index].reset_index()
    return prot, status


def aggregate_to_gene_level(prot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = prot.copy()
    tmp.columns = pd.Index([str(g).upper() for g in tmp.columns.get_level_values(0)], name="Gene")
    dup = pd.Series(tmp.columns).value_counts().rename_axis("gene").reset_index(name="n_protein_entries")
    dup.to_csv(TABLE_DIR / "gene_symbol_duplicate_counts.csv", index=False, encoding="utf-8-sig")
    gene_matrix = tmp.T.groupby(level=0).median().T
    gene_matrix.index.name = "Patient_ID"
    gene_matrix.to_csv(TABLE_DIR / "gene_level_proteomics_matrix.csv.gz", compression="gzip", encoding="utf-8-sig")
    return gene_matrix, dup


def summarize_qc(gene_matrix: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    qc = status.copy()
    qc["missing_fraction_gene_level"] = gene_matrix.isna().mean(axis=1).to_numpy()
    qc["n_detected_genes"] = gene_matrix.notna().sum(axis=1).to_numpy()
    qc = qc.sort_values(["sample_class", "Patient_ID"]).reset_index(drop=True)
    qc.to_csv(TABLE_DIR / "sample_qc_metrics.csv", index=False, encoding="utf-8-sig")
    return qc


def filter_genes(gene_matrix: pd.DataFrame, status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_class = status.set_index("Patient_ID")["sample_class"]
    tumor_ids = sample_class[sample_class.eq("tumor")].index
    normal_ids = sample_class[sample_class.eq("normal")].index
    tumor_present = gene_matrix.loc[tumor_ids].notna().mean(axis=0)
    normal_present = gene_matrix.loc[normal_ids].notna().mean(axis=0)
    keep = (tumor_present >= MIN_PRESENT_FRACTION) | (normal_present >= MIN_PRESENT_FRACTION)
    metrics = pd.DataFrame(
        {
            "gene": gene_matrix.columns,
            "tumor_present_fraction": tumor_present.to_numpy(),
            "normal_present_fraction": normal_present.to_numpy(),
            "keep_for_analysis": keep.to_numpy(),
        }
    )
    metrics.to_csv(TABLE_DIR / "gene_filter_metrics.csv", index=False, encoding="utf-8-sig")
    filtered = gene_matrix.loc[:, keep]
    filtered.to_csv(TABLE_DIR / "gene_level_proteomics_matrix.filtered.csv.gz", compression="gzip", encoding="utf-8-sig")
    return filtered, metrics


def compute_pca(filtered: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    variances = filtered.var(axis=0, skipna=True).sort_values(ascending=False)
    top_genes = variances.head(min(2000, len(variances))).index.tolist()
    pca_input = filtered[top_genes]
    imputed = SimpleImputer(strategy="median").fit_transform(pca_input)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(scaled)
    out = status.copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    out["explained_var_pc1"] = pca.explained_variance_ratio_[0]
    out["explained_var_pc2"] = pca.explained_variance_ratio_[1]
    out.to_csv(TABLE_DIR / "pca_coordinates.csv", index=False, encoding="utf-8-sig")
    return out


def differential_analysis(filtered: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    sample_class = status.set_index("Patient_ID")["sample_class"]
    tumor_ids = sample_class[sample_class.eq("tumor")].index
    normal_ids = sample_class[sample_class.eq("normal")].index
    rows = []
    for gene in filtered.columns:
        tumor_vals = filtered.loc[tumor_ids, gene].dropna()
        normal_vals = filtered.loc[normal_ids, gene].dropna()
        if tumor_vals.shape[0] < 3 or normal_vals.shape[0] < 3:
            continue
        stat = ttest_ind(tumor_vals, normal_vals, equal_var=False, nan_policy="omit")
        mean_t = float(tumor_vals.mean())
        mean_n = float(normal_vals.mean())
        rows.append(
            {
                "gene": gene,
                "n_tumor": int(tumor_vals.shape[0]),
                "n_normal": int(normal_vals.shape[0]),
                "mean_tumor": mean_t,
                "mean_normal": mean_n,
                "median_tumor": float(tumor_vals.median()),
                "median_normal": float(normal_vals.median()),
                "mean_diff_tumor_minus_normal": mean_t - mean_n,
                "t_statistic": float(stat.statistic) if pd.notna(stat.statistic) else np.nan,
                "pvalue": float(stat.pvalue) if pd.notna(stat.pvalue) else np.nan,
            }
        )
    diff = pd.DataFrame(rows)
    diff["fdr"] = multipletests(diff["pvalue"].fillna(1.0), method="fdr_bh")[1]
    diff["direction"] = np.where(diff["mean_diff_tumor_minus_normal"] > 0, "up_in_tumor", "down_in_tumor")
    diff["significant"] = (diff["fdr"] < DIFF_FDR_CUTOFF) & (diff["mean_diff_tumor_minus_normal"].abs() >= DIFF_EFFECT_CUTOFF)
    diff = diff.sort_values(["fdr", "pvalue", "mean_diff_tumor_minus_normal"], ascending=[True, True, False]).reset_index(drop=True)
    diff.to_csv(TABLE_DIR / "differential_proteins_tumor_vs_normal.csv", index=False, encoding="utf-8-sig")
    return diff


def parse_local_kegg() -> dict[str, list[str]]:
    out = {}
    with LOCAL_KEGG_GMT.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                out[parts[0]] = sorted(set(g.upper() for g in parts[2:] if g))
    return out


def fetch_kegg_pathway_genes(pathway_id: str) -> list[str]:
    with urllib.request.urlopen(f"https://rest.kegg.jp/get/{pathway_id}", timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    genes = []
    in_gene_block = False
    for raw in text.splitlines():
        if raw.startswith("GENE"):
            in_gene_block = True
            payload = raw[12:]
        elif in_gene_block and raw.startswith(" "):
            payload = raw[12:]
        else:
            in_gene_block = False
            continue
        payload = payload.strip()
        if not payload:
            continue
        parts = payload.split(None, 1)
        if len(parts) < 2:
            continue
        symbol = parts[1].split(";")[0].split(",")[0].strip().upper()
        if symbol:
            genes.append(symbol)
    return sorted(set(genes))


def fetch_metabolic_kegg_catalog() -> pd.DataFrame:
    with urllib.request.urlopen("https://rest.kegg.jp/list/pathway/hsa", timeout=120) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        pathway_id = parts[0].replace("path:", "")
        if not pathway_id.startswith("hsa00"):
            continue
        rows.append(
            {
                "kegg_id": pathway_id,
                "term": sanitize_kegg_name(parts[1]),
                "label": re.sub(r" - Homo sapiens \(human\)$", "", parts[1]).strip(),
            }
        )
    out = pd.DataFrame(rows).sort_values("kegg_id").reset_index(drop=True)
    out.to_csv(TABLE_DIR / "metabolic_kegg_catalog.csv", index=False, encoding="utf-8-sig")
    return out


def build_kegg_sets() -> tuple[dict[str, list[str]], pd.DataFrame]:
    kegg = parse_local_kegg()
    rows = []
    for term, meta in TARGET_KEGG_PATHWAYS.items():
        if term not in kegg:
            genes = fetch_kegg_pathway_genes(meta["kegg_id"])
            kegg[term] = genes
            rows.append({"term": term, "source": f"KEGG_REST:{meta['kegg_id']}", "n_genes": len(genes)})
        else:
            rows.append({"term": term, "source": "local_legacy_gmt", "n_genes": len(kegg[term])})
    source_df = pd.DataFrame(rows)
    source_df.to_csv(TABLE_DIR / "target_kegg_sources.csv", index=False, encoding="utf-8-sig")
    return kegg, source_df


def build_metabolic_kegg_sets(base_kegg: dict[str, list[str]]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    catalog = fetch_metabolic_kegg_catalog()
    gene_sets = {}
    rows = []
    for _, row in catalog.iterrows():
        term = row["term"]
        if term in base_kegg:
            genes = base_kegg[term]
            source = "local_legacy_gmt"
        else:
            genes = fetch_kegg_pathway_genes(row["kegg_id"])
            source = f"KEGG_REST:{row['kegg_id']}"
        gene_sets[term] = genes
        rows.append(
            {
                "term": term,
                "kegg_id": row["kegg_id"],
                "label": row["label"],
                "source": source,
                "n_genes": len(genes),
            }
        )
    meta = pd.DataFrame(rows).sort_values("kegg_id").reset_index(drop=True)
    meta.to_csv(TABLE_DIR / "metabolic_kegg_sources.csv", index=False, encoding="utf-8-sig")
    return gene_sets, meta


def write_gmt(gene_sets: dict[str, list[str]], out_file: Path, terms: list[str]) -> Path:
    with out_file.open("w", encoding="utf-8") as handle:
        for term in terms:
            genes = gene_sets.get(term, [])
            if genes:
                handle.write("\t".join([term, term] + genes) + "\n")
    return out_file


def fisher_enrichment(gene_list: list[str], universe: set[str], gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    genes = set(gene_list) & universe
    rows = []
    if not genes:
        return pd.DataFrame(columns=["term", "pathway_size", "overlap_size", "odds_ratio", "pvalue", "fdr", "overlap_genes"])
    for term, term_genes in gene_sets.items():
        pathway = set(term_genes) & universe
        if len(pathway) < 5:
            continue
        overlap = genes & pathway
        if not overlap:
            continue
        a = len(overlap)
        b = len(genes - pathway)
        c = len(pathway - genes)
        d = len(universe - genes - pathway)
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "term": term,
                "pathway_size": len(pathway),
                "overlap_size": a,
                "odds_ratio": float(odds_ratio),
                "pvalue": float(pvalue),
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = multipletests(out["pvalue"], method="fdr_bh")[1]
        out = out.sort_values(["fdr", "pvalue", "odds_ratio"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def run_prerank(diff: pd.DataFrame, gmt_file: Path) -> pd.DataFrame:
    ranking = diff[["gene", "t_statistic"]].dropna().drop_duplicates(subset=["gene"]).sort_values("t_statistic", ascending=False)
    pre = gp.prerank(
        rnk=ranking,
        gene_sets=str(gmt_file),
        min_size=5,
        max_size=500,
        permutation_num=1000,
        threads=4,
        seed=RANDOM_SEED,
        outdir=None,
        verbose=False,
    )
    res = pre.res2d.copy().reset_index(drop=True)
    if "Term" in res.columns:
        res = res.rename(columns={"Term": "term"})
    elif "term.1" in res.columns:
        res = res.rename(columns={"term.1": "term"})
    elif "Name" in res.columns and res["Name"].astype(str).str.startswith("KEGG_").any():
        res = res.rename(columns={"Name": "term"})
    elif "term" not in res.columns:
        res = res.rename(columns={res.columns[0]: "term"})
    if "term" in res.columns:
        res = res.loc[:, ~res.columns.duplicated()].copy()
        if pd.api.types.is_numeric_dtype(res["term"]):
            pathway_like = [c for c in res.columns if c != "term" and res[c].astype(str).str.startswith("KEGG_").any()]
            if pathway_like:
                res["term"] = res[pathway_like[0]].astype(str)
    drop_cols = [c for c in ["Name", "term.1"] if c in res.columns and c != "term"]
    if drop_cols:
        res = res.drop(columns=drop_cols)
    res.to_csv(TABLE_DIR / "kegg_gsea_prerank_results.csv", index=False, encoding="utf-8-sig")
    return res


def compute_target_scores(filtered: pd.DataFrame, target_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = filtered.apply(lambda col: (col - col.mean(skipna=True)) / (col.std(skipna=True) if col.std(skipna=True) else 1.0), axis=0)
    scores = pd.DataFrame(index=filtered.index)
    rows = []
    for term, genes in target_sets.items():
        available = [g for g in genes if g in z.columns]
        if len(available) < 3:
            continue
        scores[term] = z[available].mean(axis=1, skipna=True)
        rows.append({"term": term, "n_genes_available": len(available), "genes": ";".join(available)})
    scores.index.name = "Patient_ID"
    scores.to_csv(TABLE_DIR / "target_pathway_scores.csv", encoding="utf-8-sig")
    meta = pd.DataFrame(rows)
    meta.to_csv(TABLE_DIR / "target_pathway_gene_usage.csv", index=False, encoding="utf-8-sig")
    return scores, meta


def compare_target_scores(scores: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    sample_class = status.set_index("Patient_ID")["sample_class"]
    tumor_ids = sample_class[sample_class.eq("tumor")].index
    normal_ids = sample_class[sample_class.eq("normal")].index
    rows = []
    for term in scores.columns:
        tumor_vals = scores.loc[tumor_ids, term].dropna()
        normal_vals = scores.loc[normal_ids, term].dropna()
        if tumor_vals.shape[0] < 3 or normal_vals.shape[0] < 3:
            continue
        stat = ttest_ind(tumor_vals, normal_vals, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "term": term,
                "mean_tumor_score": float(tumor_vals.mean()),
                "mean_normal_score": float(normal_vals.mean()),
                "score_diff_tumor_minus_normal": float(tumor_vals.mean() - normal_vals.mean()),
                "t_statistic": float(stat.statistic),
                "pvalue": float(stat.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = multipletests(out["pvalue"], method="fdr_bh")[1]
        out = out.sort_values(["fdr", "pvalue"]).reset_index(drop=True)
    out.to_csv(TABLE_DIR / "target_pathway_score_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_target_genes(diff: pd.DataFrame, target_sets: dict[str, list[str]]) -> pd.DataFrame:
    lookup = diff.set_index("gene")
    rows = []
    for term, genes in target_sets.items():
        for gene in genes:
            if gene in lookup.index:
                row = lookup.loc[gene]
                rows.append(
                    {
                        "term": term,
                        "gene": gene,
                        "mean_diff_tumor_minus_normal": row["mean_diff_tumor_minus_normal"],
                        "fdr": row["fdr"],
                        "pvalue": row["pvalue"],
                        "significant": row["significant"],
                    }
                )
    out = pd.DataFrame(rows).sort_values(["term", "fdr", "pvalue"])
    out.to_csv(TABLE_DIR / "target_pathway_gene_differential_summary.csv", index=False, encoding="utf-8-sig")
    return out


def plot_target_gene_differential_summary(target_gene_df: pd.DataFrame) -> None:
    """Show protein-level drivers within the four manuscript-focus KEGG pathways."""
    if target_gene_df.empty:
        return
    focus = target_gene_df[target_gene_df["term"].isin(FOCUS_TERMS)].copy()
    if focus.empty:
        return
    focus["pathway_label"] = focus["term"].map(lambda t: TARGET_KEGG_PATHWAYS[t]["label"])
    focus["neg_log10_fdr"] = -np.log10(pd.to_numeric(focus["fdr"], errors="coerce").clip(lower=1e-300))
    focus["abs_effect"] = pd.to_numeric(focus["mean_diff_tumor_minus_normal"], errors="coerce").abs()
    focus = focus.dropna(subset=["mean_diff_tumor_minus_normal", "fdr", "neg_log10_fdr", "abs_effect"])

    selected_rows = []
    for term in FOCUS_TERMS:
        sub = focus[focus["term"].eq(term)].copy()
        if sub.empty:
            continue
        sig = sub[sub["significant"].astype(bool)].copy()
        use = sig if not sig.empty else sub
        use = use.sort_values(["fdr", "abs_effect"], ascending=[True, False]).head(10)
        selected_rows.append(use)
    if not selected_rows:
        return
    plot_df = pd.concat(selected_rows, ignore_index=True)
    plot_df["direction"] = np.where(plot_df["mean_diff_tumor_minus_normal"] >= 0, "Higher in tumor", "Lower in tumor")
    plot_df.to_csv(TABLE_DIR / "target_pathway_gene_differential_focus_plot_data.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)
    colors = {"Higher in tumor": "#FB8072", "Lower in tumor": "#80B1D3"}
    x_min = min(-1.8, float(plot_df["mean_diff_tumor_minus_normal"].min()) - 0.15)
    x_max = max(1.0, float(plot_df["mean_diff_tumor_minus_normal"].max()) + 0.15)
    for ax, term in zip(axes.flat, FOCUS_TERMS):
        sub = plot_df[plot_df["term"].eq(term)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("mean_diff_tumor_minus_normal", ascending=True)
        y = np.arange(len(sub))
        ax.hlines(y, 0, sub["mean_diff_tumor_minus_normal"], color="#B8C2C8", linewidth=1.0, zorder=1)
        sizes = np.clip(sub["neg_log10_fdr"], 1, 12) * 12
        ax.scatter(
            sub["mean_diff_tumor_minus_normal"],
            y,
            s=sizes,
            c=sub["direction"].map(colors),
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.axvline(0, color="#2F3337", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["gene"], fontsize=7.2)
        ax.set_title(TARGET_KEGG_PATHWAYS[term]["label"], fontsize=8.5, fontweight="bold", loc="left", pad=5)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="x", color="#E6ECEF", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#AAB4BA")
        ax.tick_params(axis="x", labelsize=7.2)
        ax.tick_params(axis="y", length=0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Protein abundance difference (tumor - normal)", fontsize=8)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["Higher in tumor"], markeredgecolor="white", markersize=6, label="Higher in tumor"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["Lower in tumor"], markeredgecolor="white", markersize=6, label="Lower in tumor"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#808A93", markeredgecolor="white", markersize=4, label="Smaller -log10 FDR"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#808A93", markeredgecolor="white", markersize=8, label="Larger -log10 FDR"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("CPTAC-OV protein-level changes in focus metabolic pathways", fontsize=10.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), w_pad=1.4, h_pad=1.2)
    fig.savefig(FIGURE_DIR / "09_target_pathway_gene_differential_focus_dotplot.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / "09_target_pathway_gene_differential_focus_dotplot.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sample_qc(qc: pd.DataFrame) -> None:
    palette = {"tumor": "#FB8072", "normal": "#80B1D3"}
    ordered = qc.sort_values(["sample_class", "missing_fraction_gene_level", "Patient_ID"]).reset_index(drop=True)
    ordered["plot_order"] = np.arange(ordered.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = ordered["sample_class"].map(palette).tolist()
    axes[0].bar(ordered["plot_order"], ordered["missing_fraction_gene_level"], color=colors, width=0.9)
    axes[0].set_ylabel("Missing fraction")
    axes[0].set_xlabel("")
    axes[0].set_title("CPTAC-OV Proteomics Sample QC")
    handles = [plt.Line2D([0], [0], color=palette[k], lw=6) for k in ["tumor", "normal"]]
    axes[0].legend(handles, ["tumor", "normal"], title="Sample class", frameon=False)
    axes[0].set_xticks([])
    axes[1].bar(ordered["plot_order"], ordered["n_detected_genes"], color=colors, width=0.9)
    axes[1].set_ylabel("Detected genes")
    axes[1].set_xlabel("Samples")
    axes[1].set_xticks([])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_sample_qc.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pca(pca_df: pd.DataFrame) -> None:
    matplotlib.rcParams.update({
        "pdf.use14corefonts": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    palette = {"tumor": "#FB8072", "normal": "#80B1D3"}
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="sample_class", palette=palette, s=70, edgecolor="white", linewidth=0.4, ax=ax)
    ax.set_xlabel(f"PC1 ({pca_df['explained_var_pc1'].iloc[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_df['explained_var_pc2'].iloc[0] * 100:.1f}% variance)")
    ax.set_title("CPTAC-OV Proteomics PCA")
    ax.legend(title="Sample class", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_pca_filtered_proteome.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_volcano(diff: pd.DataFrame) -> None:
    plot_df = diff.copy()
    plot_df["neg_log10_fdr"] = -np.log10(plot_df["fdr"].clip(lower=1e-300))
    plot_df["plot_group"] = "not_significant"
    plot_df.loc[plot_df["significant"] & (plot_df["mean_diff_tumor_minus_normal"] > 0), "plot_group"] = "up_in_tumor"
    plot_df.loc[plot_df["significant"] & (plot_df["mean_diff_tumor_minus_normal"] < 0), "plot_group"] = "down_in_tumor"
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sns.scatterplot(
        data=plot_df,
        x="mean_diff_tumor_minus_normal",
        y="neg_log10_fdr",
        hue="plot_group",
        hue_order=["up_in_tumor", "down_in_tumor", "not_significant"],
        palette={"up_in_tumor": "#FB8072", "down_in_tumor": "#80B1D3", "not_significant": "#D9D9D9"},
        s=24,
        linewidth=0,
        alpha=0.75,
        legend=False,
        ax=ax,
    )
    ax.axvline(DIFF_EFFECT_CUTOFF, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-DIFF_EFFECT_CUTOFF, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-math.log10(DIFF_FDR_CUTOFF), color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean difference (Tumor - Normal)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Differential Proteins: Tumor vs Normal")
    labels = pd.concat([
        plot_df.loc[plot_df["mean_diff_tumor_minus_normal"] > 0].nsmallest(8, "fdr"),
        plot_df.loc[plot_df["mean_diff_tumor_minus_normal"] < 0].nsmallest(8, "fdr"),
    ]).drop_duplicates("gene")
    for _, row in labels.iterrows():
        ax.text(row["mean_diff_tumor_minus_normal"], row["neg_log10_fdr"], row["gene"], fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_volcano_tumor_vs_normal.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(filtered: pd.DataFrame, status: pd.DataFrame, diff: pd.DataFrame) -> None:
    top_up = diff.loc[diff["mean_diff_tumor_minus_normal"] > 0].nsmallest(TOP_HEATMAP_GENES_PER_DIRECTION, "fdr")
    top_down = diff.loc[diff["mean_diff_tumor_minus_normal"] < 0].nsmallest(TOP_HEATMAP_GENES_PER_DIRECTION, "fdr")
    genes = pd.concat([top_up, top_down])["gene"].drop_duplicates().tolist()
    if not genes:
        return
    ordered_samples = status.sort_values(["sample_class", "Patient_ID"])["Patient_ID"].tolist()
    matrix = filtered.loc[ordered_samples, genes].fillna(filtered[genes].median())
    z = matrix.apply(lambda col: (col - col.mean()) / (col.std() if col.std() else 1.0), axis=0).T
    sample_status = status.set_index("Patient_ID").loc[ordered_samples]
    annotation = pd.DataFrame(
        [sample_status["sample_class"].map({"tumor": 1, "normal": 0}).to_numpy()],
        index=["sample class"],
        columns=ordered_samples,
    )
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.25, 9], hspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        annotation,
        cmap=ListedColormap(["#80B1D3", "#FB8072"]),
        cbar=False,
        xticklabels=False,
        yticklabels=["sample class"],
        ax=ax0,
    )
    ax0.tick_params(axis="y", labelrotation=0)
    sns.heatmap(z, cmap="RdBu_r", center=0, cbar_kws={"label": "Gene-level z-score"}, ax=ax1)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Genes")
    ax1.set_title("Top Differential Proteins")
    ax0.set_xlim(ax1.get_xlim())
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_top_differential_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gsea(gsea_df: pd.DataFrame) -> None:
    df = gsea_df.copy()
    fdr_col = "FDR q-val" if "FDR q-val" in df.columns else "FDR"
    if fdr_col not in df.columns or "NES" not in df.columns:
        return
    df["fdr"] = pd.to_numeric(df[fdr_col], errors="coerce")
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    df = df.dropna(subset=["fdr", "NES"]).copy()
    df["abs_nes"] = df["NES"].abs()
    top = df.sort_values(["fdr", "abs_nes"], ascending=[True, False]).head(20).sort_values("NES")
    if top.empty:
        return
    top["term_label"] = top["term"].map(pretty_term)
    fig, ax = plt.subplots(figsize=(8.5, max(6, 0.32 * top.shape[0])))
    sns.scatterplot(
        data=top,
        x="NES",
        y="term_label",
        size=-np.log10(top["fdr"].clip(lower=1e-300)),
        hue="NES",
        palette="RdBu_r",
        sizes=(40, 220),
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("NES")
    ax.set_ylabel("")
    ax.set_title("Top KEGG GSEA Results")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_kegg_gsea_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_target_boxplots(scores: pd.DataFrame, status: pd.DataFrame, comp: pd.DataFrame) -> None:
    long_df = scores.reset_index().melt(id_vars="Patient_ID", var_name="term", value_name="score")
    long_df = long_df.merge(status[["Patient_ID", "sample_class"]], on="Patient_ID", how="left")
    comp_map = comp.set_index("term")[["score_diff_tumor_minus_normal", "fdr"]].to_dict("index") if not comp.empty else {}
    ordered_terms = [t for t in TARGET_KEGG_PATHWAYS if t in scores.columns]
    n = len(ordered_terms)
    cols = 2
    rows = math.ceil(max(n, 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(8, 4 * rows)), squeeze=False)
    palette = {"tumor": "#FB8072", "normal": "#80B1D3"}
    for ax, term in zip(axes.flat, ordered_terms):
        sub = long_df[long_df["term"] == term]
        sns.boxplot(data=sub, x="sample_class", y="score", hue="sample_class", palette=palette, width=0.55, fliersize=0, ax=ax, legend=False)
        sns.stripplot(data=sub, x="sample_class", y="score", color="black", alpha=0.5, size=2.5, ax=ax)
        meta = comp_map.get(term, {})
        label = TARGET_KEGG_PATHWAYS[term]["label"]
        title = label
        if meta:
            title += f"\nDiff={meta['score_diff_tumor_minus_normal']:.2f}, FDR={meta['fdr']:.3g}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Pathway score")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_target_kegg_pathway_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_target_gsea(target_gsea: pd.DataFrame) -> None:
    if target_gsea.empty:
        return
    df = target_gsea.copy()
    fdr_col = "FDR q-val" if "FDR q-val" in df.columns else "FDR"
    nom_col = "NOM p-val" if "NOM p-val" in df.columns else None
    df["fdr"] = pd.to_numeric(df[fdr_col], errors="coerce")
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    if nom_col is not None:
        df["nom_p"] = pd.to_numeric(df[nom_col], errors="coerce")
    else:
        df["nom_p"] = np.nan
    df = df.dropna(subset=["fdr", "NES"]).sort_values("NES")
    df["term_label"] = df["term"].map(pretty_term)
    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.5 * df.shape[0])))
    bars = ax.barh(df["term_label"], df["NES"], color=np.where(df["NES"] >= 0, "#FB8072", "#80B1D3"))
    for bar, (_, row) in zip(bars, df.iterrows()):
        note = f"  p={row['nom_p']:.3g}" if pd.notna(row["nom_p"]) else ""
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, note, va="center", fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("NES")
    ax.set_ylabel("")
    ax.set_title("Target KEGG pathways in preranked GSEA\n(nominal p shown; global FDR reported in tables)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_target_kegg_nes_barplot.pdf", bbox_inches="tight")
    plt.close(fig)



def write_summary(qc: pd.DataFrame, filter_df: pd.DataFrame, diff: pd.DataFrame, up_enr: pd.DataFrame, down_enr: pd.DataFrame, comp: pd.DataFrame, target_gsea: pd.DataFrame) -> None:
    n_sig_up = int(((diff["significant"]) & (diff["mean_diff_tumor_minus_normal"] > 0)).sum())
    n_sig_down = int(((diff["significant"]) & (diff["mean_diff_tumor_minus_normal"] < 0)).sum())
    lines = [
        "CPTAC-OV proteomics tumor-vs-normal KEGG analysis summary",
        "========================================================",
        f"Samples: {qc.shape[0]} total ({(qc['sample_class'] == 'tumor').sum()} tumor, {(qc['sample_class'] == 'normal').sum()} normal)",
        f"Gene-level proteins before filtering: {filter_df.shape[0]}",
        f"Gene-level proteins retained: {int(filter_df['keep_for_analysis'].sum())}",
        f"Median missing fraction per sample: {qc['missing_fraction_gene_level'].median():.4f}",
        f"Significant proteins up in tumor: {n_sig_up}",
        f"Significant proteins down in tumor: {n_sig_down}",
        "",
        "Top differential proteins:",
        diff.head(20).to_string(index=False),
        "",
        "Top KEGG enrichment among tumor-up proteins:",
        up_enr.head(10).to_string(index=False) if not up_enr.empty else "None",
        "",
        "Top KEGG enrichment among tumor-down proteins:",
        down_enr.head(10).to_string(index=False) if not down_enr.empty else "None",
        "",
        "Target pathway score comparison:",
        comp.to_string(index=False) if not comp.empty else "None",
        "",
        "Target pathway GSEA:",
        target_gsea.to_string(index=False) if not target_gsea.empty else "None",
    ]
    (TABLE_DIR / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")
    np.random.seed(RANDOM_SEED)

    prot, status = read_inputs()
    gene_matrix, _ = aggregate_to_gene_level(prot)
    qc = summarize_qc(gene_matrix, status)
    filtered, filter_df = filter_genes(gene_matrix, status)
    pca_df = compute_pca(filtered, status)
    diff = differential_analysis(filtered, status)

    kegg_sets, _ = build_kegg_sets()
    metabolic_kegg_sets, metabolic_meta = build_metabolic_kegg_sets(kegg_sets)
    combined_terms = metabolic_meta["term"].tolist()
    target_terms = [t for t in TARGET_KEGG_PATHWAYS if t in metabolic_kegg_sets]
    combined_gmt = write_gmt(metabolic_kegg_sets, GMT_DIR / "combined_kegg_hsa00_metabolic.gmt", combined_terms)
    write_gmt(metabolic_kegg_sets, GMT_DIR / "target_kegg_pathways.gmt", target_terms)

    universe = set(filtered.columns.astype(str))
    sig_up = diff.loc[diff["significant"] & (diff["mean_diff_tumor_minus_normal"] > 0), "gene"].tolist()
    sig_down = diff.loc[diff["significant"] & (diff["mean_diff_tumor_minus_normal"] < 0), "gene"].tolist()
    up_enr = fisher_enrichment(sig_up, universe, kegg_sets)
    down_enr = fisher_enrichment(sig_down, universe, kegg_sets)
    up_enr.to_csv(TABLE_DIR / "kegg_overrepresentation_up_in_tumor.csv", index=False, encoding="utf-8-sig")
    down_enr.to_csv(TABLE_DIR / "kegg_overrepresentation_down_in_tumor.csv", index=False, encoding="utf-8-sig")

    gsea_df = run_prerank(diff, combined_gmt)
    target_gsea = gsea_df[gsea_df["term"].isin(target_terms)].copy()
    target_gsea.to_csv(TABLE_DIR / "target_kegg_gsea_results.csv", index=False, encoding="utf-8-sig")

    target_sets = {term: metabolic_kegg_sets[term] for term in target_terms}
    scores, _ = compute_target_scores(filtered, target_sets)
    comp = compare_target_scores(scores, status)
    target_gene_df = summarize_target_genes(diff, target_sets)

    # Current manuscript architecture retains only sample QC and filtered-proteome PCA
    # from this broad CPTAC analysis entry point. Four-pathway grouped/focus panels
    # are generated by export_individual_target_boxplots.py from the same tables.
    plot_sample_qc(qc)
    plot_pca(pca_df)
    write_summary(qc, filter_df, diff, up_enr, down_enr, comp, target_gsea)

    manifest = {
        "input_proteomics": str(PROTEOMICS_FILE),
        "input_sample_status": str(SAMPLE_STATUS_FILE),
        "tumor_samples": int((status["sample_class"] == "tumor").sum()),
        "normal_samples": int((status["sample_class"] == "normal").sum()),
        "genes_before_filter": int(filter_df.shape[0]),
        "genes_after_filter": int(filter_df["keep_for_analysis"].sum()),
        "combined_gmt": str(combined_gmt),
        "metabolic_kegg_pathway_count": int(len(combined_terms)),
    }
    with (TABLE_DIR / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
