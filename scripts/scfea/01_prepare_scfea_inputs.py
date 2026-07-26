from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "12_scFEA"
MODULE_FILE = ROOT / "01_software" / "scFEA" / "data" / "module_gene_m168.csv"
SEED = 42

DATASETS = {
    "GSE217517": {
        "h5ad": Path(
            PROJECT_ROOT / "08_multiomics_validation_scRNA_TCGA"
            / "08A_scRNA_preprocess" / "GSE217517" / "10_kegg_enrichment" / "adata_kegg_scored.h5ad"
        ),
        "comparison_col": "is_tumor_cell",
    },
    "GSE184880": {
        "h5ad": Path(
            PROJECT_ROOT / "08_multiomics_validation_scRNA_TCGA"
            / "08A_scRNA_preprocess" / "scRNAGSE184880" / "10_kegg_enrichment" / "adata_kegg_scored.h5ad"
        ),
        "comparison_col": "is_tumor_cell",
    },
}


def decode(values):
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.array(
            [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values]
        )
    return values.astype(str)


def read_frame_column(group: h5py.Group, name: str) -> np.ndarray:
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return decode(obj[()])
    codes = obj["codes"][()]
    categories = decode(obj["categories"][()])
    out = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out.astype(str)


def read_csr(group: h5py.Group) -> sparse.csr_matrix:
    shape = tuple(int(x) for x in group.attrs["shape"])
    return sparse.csr_matrix(
        (group["data"][()], group["indices"][()], group["indptr"][()]), shape=shape
    )


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_pilot(meta: pd.DataFrame, comparison_col: str, n_per_group: int = 250) -> list[str]:
    rng = np.random.default_rng(SEED)
    selected: list[str] = []
    for label in ["Normal", "Tumor"]:
        pool = meta.index[meta[comparison_col].eq(label)].to_numpy()
        if len(pool) < n_per_group:
            raise ValueError(f"{label} has only {len(pool)} cells; cannot select {n_per_group}")
        selected.extend(rng.choice(pool, n_per_group, replace=False).tolist())
    return selected


def save_gene_by_cell(matrix: sparse.csr_matrix, genes: list[str], cells: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.sparse.from_spmatrix(matrix.T, index=genes, columns=cells)
    frame.to_csv(path)


def prepare_dataset(dataset: str, config: dict, module_genes: list[str]) -> dict:
    source = config["h5ad"]
    out_dir = ROOT / "02_inputs" / f"{dataset}_epithelial"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        obs_group = handle["obs"]
        var_group = handle["var"]
        cells = decode(obs_group["_index"][()])
        genes = decode(var_group["_index"][()])
        meta = pd.DataFrame(index=cells)
        for col in ["cell_type", "sample", "group", "is_tumor_cell", "cnv_score"]:
            if col in obs_group:
                meta[col] = read_frame_column(obs_group, col)

        epithelial_mask = meta["cell_type"].str.startswith("Epithelial", na=False).to_numpy()
        epithelial_meta = meta.loc[epithelial_mask].copy()
        epithelial_meta.index.name = "cell_id"
        if "obsm" in handle and "X_umap" in handle["obsm"]:
            umap = np.asarray(handle["obsm"]["X_umap"][()])[epithelial_mask]
            epithelial_meta["UMAP_1"] = umap[:, 0]
            epithelial_meta["UMAP_2"] = umap[:, 1]
        comparison_col = config["comparison_col"]
        epithelial_meta["formal_comparison"] = epithelial_meta[comparison_col]

        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        overlap = [gene for gene in module_genes if gene in gene_to_idx]
        gene_indices = np.array([gene_to_idx[gene] for gene in overlap], dtype=int)
        row_indices = np.flatnonzero(epithelial_mask)
        counts = read_csr(handle["layers"]["counts"])[row_indices][:, gene_indices].tocsr()

    full_counts = out_dir / f"{dataset}_epithelial_scFEA_counts.csv"
    full_meta = out_dir / f"{dataset}_epithelial_cell_metadata.tsv"
    save_gene_by_cell(counts, overlap, epithelial_meta.index.tolist(), full_counts)
    epithelial_meta.to_csv(full_meta, sep="\t")

    pilot_cells = balanced_pilot(epithelial_meta, comparison_col)
    pilot_positions = epithelial_meta.index.get_indexer(pilot_cells)
    pilot_counts = out_dir / f"{dataset}_epithelial_pilot500_scFEA_counts.csv"
    pilot_meta = out_dir / f"{dataset}_epithelial_pilot500_cell_metadata.tsv"
    save_gene_by_cell(counts[pilot_positions], overlap, pilot_cells, pilot_counts)
    epithelial_meta.loc[pilot_cells].to_csv(pilot_meta, sep="\t")

    gene_audit = pd.DataFrame(
        {
            "module_gene": module_genes,
            "present_in_h5ad": [gene in gene_to_idx for gene in module_genes],
        }
    )
    gene_audit.to_csv(out_dir / f"{dataset}_scFEA_gene_overlap_audit.tsv", sep="\t", index=False)

    summary = {
        "dataset": dataset,
        "source_h5ad": str(source),
        "source_h5ad_sha256": sha256(source),
        "source_shape": [len(meta), len(genes)],
        "epithelial_cells": int(len(epithelial_meta)),
        "formal_comparison_column": comparison_col,
        "formal_comparison_counts": epithelial_meta[comparison_col].value_counts().to_dict(),
        "sample_counts": epithelial_meta["sample"].value_counts().to_dict(),
        "scfea_module_genes": len(module_genes),
        "overlap_genes": len(overlap),
        "nonzero_fraction": float(counts.nnz / np.prod(counts.shape)),
        "pilot_cells": len(pilot_cells),
        "input_files": {
            "full_counts": str(full_counts),
            "full_metadata": str(full_meta),
            "pilot_counts": str(pilot_counts),
            "pilot_metadata": str(pilot_meta),
        },
    }
    (out_dir / f"{dataset}_input_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    module_table = pd.read_csv(MODULE_FILE, index_col=0)
    module_genes = sorted(
        {
            str(value)
            for value in module_table.to_numpy().ravel()
            if pd.notna(value) and str(value) != "nan"
        }
    )
    summaries = [prepare_dataset(name, cfg, module_genes) for name, cfg in DATASETS.items()]
    manifest = ROOT / "00_manifest" / "input_preparation_summary.json"
    manifest.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
