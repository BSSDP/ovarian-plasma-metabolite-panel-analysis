from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
BASE = PROJECT_ROOT / "08_multiomics_validation_scRNA_TCGA" / "08A_scRNA_preprocess" / "25_epithelial_four_pathway_activity_combined"
PREPROCESS = BASE.parent
TABLE_DIR = BASE / "tables"
LOG_DIR = BASE / "logs"

CELL_LEVEL_LONG = TABLE_DIR / "combined_epithelial_four_pathway_long_data.csv"
ANNOTATIONS = {
    "GSE217517": PREPROCESS / "GSE217517" / "04_annotated" / "cell_annotations.csv",
    "GSE184880": PREPROCESS / "scRNAGSE184880" / "04_annotated" / "cell_annotations.csv",
}

SAMPLE_LEVEL_OUT = TABLE_DIR / "sample_level_four_pathway_scores.tsv"
SAMPLE_COUNTS_OUT = TABLE_DIR / "sample_level_four_pathway_sample_counts.tsv"
STATS_OUT = TABLE_DIR / "sample_level_four_pathway_paired_statistics.tsv"
CLINICAL_STATS_OUT = TABLE_DIR / "sample_level_four_pathway_clinical_group_statistics.tsv"
RUN_LOG = LOG_DIR / "sample_level_four_pathway_paired_analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_wilcoxon_signed_rank(differences: pd.Series) -> dict[str, float | int]:
    """Two-sided exact Wilcoxon signed-rank test by complete sign enumeration.

    Zero differences are removed. Average ranks are used for tied absolute
    differences. The exact null distribution conditions on the observed ranks.
    """
    diff = pd.Series(differences, dtype=float).dropna()
    diff = diff[~np.isclose(diff, 0.0)]
    n = int(diff.size)
    if n == 0:
        return {
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "w_statistic": 0.0,
            "exact_p_value": 1.0,
        }

    ranks = diff.abs().rank(method="average").to_numpy(dtype=float)
    observed_w_plus = float(ranks[diff.to_numpy() > 0].sum())
    total_rank = float(ranks.sum())
    observed_stat = min(observed_w_plus, total_rank - observed_w_plus)

    null_stats = []
    for signs in product((0, 1), repeat=n):
        w_plus = float(ranks[np.asarray(signs, dtype=bool)].sum())
        null_stats.append(min(w_plus, total_rank - w_plus))
    null_stats = np.asarray(null_stats, dtype=float)
    p_value = float(np.mean(null_stats <= observed_stat + 1e-12))

    return {
        "n_nonzero": n,
        "w_plus": observed_w_plus,
        "w_minus": total_rank - observed_w_plus,
        "w_statistic": observed_stat,
        "exact_p_value": min(1.0, p_value),
    }


def exact_mann_whitney(first: pd.Series, second: pd.Series) -> dict[str, float | int]:
    """Two-sided exact Mann-Whitney test by enumerating group assignments."""
    x = pd.Series(first, dtype=float).dropna().to_numpy()
    y = pd.Series(second, dtype=float).dropna().to_numpy()
    n_x, n_y = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    observed_u = float(ranks[:n_x].sum() - n_x * (n_x + 1) / 2)
    observed_stat = min(observed_u, n_x * n_y - observed_u)

    null_stats = []
    all_indices = np.arange(n_x + n_y)
    for x_indices in combinations(all_indices, n_x):
        rank_sum = float(ranks[np.asarray(x_indices, dtype=int)].sum())
        u_value = rank_sum - n_x * (n_x + 1) / 2
        null_stats.append(min(u_value, n_x * n_y - u_value))
    null_stats = np.asarray(null_stats, dtype=float)
    p_value = float(np.mean(null_stats <= observed_stat + 1e-12))
    return {
        "n_first": int(n_x),
        "n_second": int(n_y),
        "u_statistic": observed_u,
        "exact_p_value": min(1.0, p_value),
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    p = pd.Series(values, dtype=float)
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for index in range(m - 1, -1, -1):
        running = min(running, ranked[index] * m / (index + 1))
        adjusted[index] = running
    restored = np.empty(m, dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    return pd.Series(restored, index=p.index)


def load_with_samples() -> pd.DataFrame:
    data = pd.read_csv(CELL_LEVEL_LONG)
    required = {
        "cell_id",
        "dataset",
        "is_tumor_cell",
        "pathway_id",
        "pathway_label",
        "pathway_score",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    pieces = []
    for dataset, annotation_path in ANNOTATIONS.items():
        subset = data.loc[data["dataset"].eq(dataset)].copy()
        annotations = pd.read_csv(annotation_path, index_col=0)
        if "sample" not in annotations.columns:
            raise KeyError(f"Missing sample column: {annotation_path}")
        subset["sample"] = subset["cell_id"].map(annotations["sample"])
        if subset["sample"].isna().any():
            missing_cells = int(subset["sample"].isna().sum())
            raise ValueError(f"{dataset}: {missing_cells} pathway rows lack a sample mapping")
        subset["clinical_group"] = np.where(
            subset["sample"].astype(str).str.contains("Cancer", case=False),
            "Cancer",
            np.where(
                subset["sample"].astype(str).str.contains("Norm", case=False),
                "Normal",
                "Tumour cohort",
            ),
        )
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_with_samples()

    unique_cells = data.drop_duplicates(["dataset", "cell_id"])
    sample_counts = (
        unique_cells.groupby(
            ["dataset", "sample", "clinical_group", "is_tumor_cell"], observed=True
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sample_counts.to_csv(SAMPLE_COUNTS_OUT, sep="\t", index=False)

    sample_level = (
        data.groupby(
            [
                "dataset",
                "sample",
                "clinical_group",
                "is_tumor_cell",
                "pathway_id",
                "pathway_label",
            ],
            observed=True,
        )["pathway_score"]
        .agg(pathway_score_median="median", n_cells="size")
        .reset_index()
    )
    sample_level.to_csv(SAMPLE_LEVEL_OUT, sep="\t", index=False)

    # Paired CNV-state contrast: GSE217517 consists of tumour samples. For
    # GSE184880, restrict the within-sample comparison to cancer samples.
    paired_eligible = data.loc[
        data["dataset"].eq("GSE217517")
        | (data["dataset"].eq("GSE184880") & data["clinical_group"].eq("Cancer"))
    ].copy()
    states_per_sample = (
        paired_eligible.drop_duplicates(["dataset", "sample", "cell_id"])
        .groupby(["dataset", "sample"])["is_tumor_cell"]
        .agg(lambda values: set(values.astype(str)))
    )
    paired_samples = states_per_sample[
        states_per_sample.map(lambda values: {"Normal", "Tumor"}.issubset(values))
    ].index
    paired_index = pd.MultiIndex.from_frame(sample_level[["dataset", "sample"]])
    paired_sample_level = sample_level.loc[paired_index.isin(paired_samples)].copy()

    rows = []
    for (dataset, pathway_id, pathway_label), group in paired_sample_level.groupby(
        ["dataset", "pathway_id", "pathway_label"], observed=True
    ):
        paired = group.pivot(index="sample", columns="is_tumor_cell", values="pathway_score_median")
        paired = paired.dropna(subset=["Normal", "Tumor"])
        differences = paired["Tumor"] - paired["Normal"]
        test = exact_wilcoxon_signed_rank(differences)
        rows.append(
            {
                "dataset": dataset,
                "pathway_id": pathway_id,
                "pathway_label": pathway_label,
                "n_paired_samples": int(len(paired)),
                "normal_sample_median": float(paired["Normal"].median()),
                "tumor_sample_median": float(paired["Tumor"].median()),
                "paired_median_difference_tumor_minus_normal": float(differences.median()),
                "n_tumor_higher": int((differences > 0).sum()),
                "n_tumor_lower": int((differences < 0).sum()),
                **test,
                "analysis_unit": "sample",
                "within_sample_summary": "median cell-level pathway score",
                "test": "two-sided exact paired Wilcoxon signed-rank",
            }
        )

    stats = pd.DataFrame(rows).sort_values(["dataset", "pathway_id"]).reset_index(drop=True)
    stats["bh_fdr_within_dataset"] = stats.groupby("dataset", group_keys=False)[
        "exact_p_value"
    ].apply(benjamini_hochberg)
    stats["bh_fdr_across_eight_tests"] = benjamini_hochberg(stats["exact_p_value"])
    stats.to_csv(STATS_OUT, sep="\t", index=False)

    # Independent clinical-sample contrast available in GSE184880: normal-like
    # epithelial summaries from five normal samples versus tumour-like
    # epithelial summaries from seven cancer samples.
    clinical_rows = []
    gse184880 = sample_level.loc[sample_level["dataset"].eq("GSE184880")].copy()
    for (pathway_id, pathway_label), group in gse184880.groupby(
        ["pathway_id", "pathway_label"], observed=True
    ):
        normal = group.loc[
            group["clinical_group"].eq("Normal") & group["is_tumor_cell"].eq("Normal"),
            "pathway_score_median",
        ]
        cancer = group.loc[
            group["clinical_group"].eq("Cancer") & group["is_tumor_cell"].eq("Tumor"),
            "pathway_score_median",
        ]
        test = exact_mann_whitney(normal, cancer)
        clinical_rows.append(
            {
                "dataset": "GSE184880",
                "pathway_id": pathway_id,
                "pathway_label": pathway_label,
                "normal_samples_median": float(normal.median()),
                "cancer_samples_median": float(cancer.median()),
                "median_difference_cancer_minus_normal": float(cancer.median() - normal.median()),
                **test,
                "analysis_unit": "sample",
                "normal_definition": "normal-like epithelial cells in clinical normal samples",
                "cancer_definition": "tumour-like epithelial cells in clinical cancer samples",
                "test": "two-sided exact Mann-Whitney U",
            }
        )
    clinical_stats = pd.DataFrame(clinical_rows).sort_values("pathway_id").reset_index(drop=True)
    clinical_stats["bh_fdr_within_dataset"] = benjamini_hochberg(
        clinical_stats["exact_p_value"]
    )
    clinical_stats.to_csv(CLINICAL_STATS_OUT, sep="\t", index=False)

    expected_n = {"GSE217517": 8, "GSE184880": 7}
    observed_n = stats.groupby("dataset")["n_paired_samples"].unique().to_dict()
    for dataset, expected in expected_n.items():
        values = observed_n.get(dataset)
        if values is None or len(values) != 1 or int(values[0]) != expected:
            raise AssertionError(f"{dataset}: expected {expected} paired samples, observed {values}")

    log = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "sample",
        "aggregation": "median cell-level pathway score per sample and CNV-defined epithelial state",
        "GSE217517_rule": "all tumour-cohort samples with both CNV-defined states",
        "GSE184880_rule": "clinically annotated cancer samples with both CNV-defined states",
        "paired_samples": expected_n,
        "multiple_testing": {
            "primary": "Benjamini-Hochberg within each dataset across four predefined pathways",
            "audit": "Benjamini-Hochberg across all eight dataset-pathway tests",
        },
        "inputs": {
            str(CELL_LEVEL_LONG): sha256(CELL_LEVEL_LONG),
            **{str(path): sha256(path) for path in ANNOTATIONS.values()},
        },
        "outputs": [
            str(SAMPLE_LEVEL_OUT),
            str(SAMPLE_COUNTS_OUT),
            str(STATS_OUT),
            str(CLINICAL_STATS_OUT),
        ],
    }
    RUN_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(stats.to_string(index=False))
    print("\nGSE184880 independent clinical-sample contrast")
    print(clinical_stats.to_string(index=False))
    print(f"\nWrote: {STATS_OUT}")


if __name__ == "__main__":
    main()
