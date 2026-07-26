"""
單細胞轉錄組細胞類型註釋腳本
使用CellTypist進行自動註釋，標記基因識別作為補充
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from publication_plotting import (
    plot_composition,
    plot_embedding_categorical,
    plot_embedding_grid,
    plot_fraction_heatmap,
    plot_marker_dotplot,
    set_publication_style,
)

# 用於可視化的額外導入
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[WARNING] seaborn或matplotlib未安裝，部分可視化功能可能不可用")

# 設置scanpy參數
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, facecolor='white', figsize=(4.2, 3.7), fontsize=8)
set_publication_style()


def celltypist_marker_dict():
    """Canonical marker genes used only to display CellTypist annotation support."""
    return {
        "T cells": ["CD3D", "CD3E"],
        "B cells": ["MS4A1", "CD79A"],
        "Plasma cells": ["JCHAIN", "MZB1"],
        "Macrophages": ["C1QA", "APOE"],
        "Monocytes": ["LST1", "FCN1"],
        "DC": ["FCER1A", "CST3"],
        "pDC": ["GZMB", "IRF7"],
        "ILC": ["NKG7", "GNLY"],
        "Endothelial cells": ["PECAM1", "VWF"],
        "Fibroblasts": ["COL1A1", "DCN"],
        "Epithelial cells": ["EPCAM", "KRT18", "KRT19"],
    }


def load_clustered_data(input_file):
    """加載聚類後的數據"""
    print("=" * 60)
    print("加載聚類後的數據...")
    print("=" * 60)
    
    if not Path(input_file).exists():
        print(f"[ERROR] 找不到數據文件: {input_file}")
        return None
    
    adata = sc.read_h5ad(input_file)
    print(f"[OK] 數據加載成功")
    print(f"  - 細胞數: {adata.n_obs:,}")
    print(f"  - 基因數: {adata.n_vars:,}")
    
    if 'leiden' in adata.obs.columns:
        print(f"  - 聚類數: {adata.obs['leiden'].nunique()}")
    
    return adata

def find_marker_genes(adata, groupby='leiden', n_genes=20):
    """尋找每個聚類的標記基因"""
    print("\n" + "=" * 60)
    print("尋找標記基因...")
    print("=" * 60)
    
    if groupby not in adata.obs.columns:
        print(f"[ERROR] 未找到分組變量 '{groupby}'")
        return adata
    
    print(f"\n為每個聚類尋找前 {n_genes} 個標記基因...")
    print("使用Wilcoxon秩和檢驗...")
    
    # 使用Wilcoxon秩和檢驗尋找標記基因
    sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon',
                            n_genes=n_genes, use_raw=True)
    
    # 獲取結果
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    print(f"\n[OK] 找到 {len(groups)} 個聚類的標記基因")
    
    # 顯示每個聚類的前5個標記基因
    print("\n各聚類的前5個標記基因:")
    for group in groups[:min(10, len(groups))]:  # 只顯示前10個聚類
        print(f"\n聚類 {group}:")
        genes = result['names'][group][:5]
        scores = result['scores'][group][:5]
        pvals = result['pvals_adj'][group][:5] if 'pvals_adj' in result else [None] * 5
        for i, (gene, score) in enumerate(zip(genes, scores)):
            pval_str = f", p_adj={pvals[i]:.2e}" if pvals[i] is not None else ""
            print(f"  {gene}: score={score:.2f}{pval_str}")
    
    return adata

def celltypist_annotation(adata, model='auto', majority_voting=True):
    """使用CellTypist進行細胞類型註釋

    參數:
        adata: AnnData對象
        model: 預訓練模型名稱或路徑
            - 'auto': 直接使用高精度免疫模型（推薦）
            - 'Immune_All_High.pkl': 高精度免疫細胞模型
            - 'Immune_All_Low.pkl': 標準免疫細胞模型
            - 或其他預訓練模型
        majority_voting: 是否使用多數投票（基於聚類）
    """
    print("\n" + "=" * 60)
    print("使用CellTypist進行細胞類型註釋...")
    print("=" * 60)
    
    try:
        import celltypist
        from celltypist import models
    except ImportError:
        print("[ERROR] celltypist未安裝，請運行: pip install celltypist")
        print("[INFO] 改用基於標記基因的註釋方法...")
        return None
    
    # 確保使用原始計數數據
    if adata.raw is None:
        print("[WARNING] 未找到原始數據，使用當前數據")
        adata.raw = adata
    
    # 準備數據（CellTypist需要log1p轉換的數據）
    print("\n準備數據...")
    if 'log1p' not in adata.layers.keys():
        # 如果沒有log1p層，創建一個
        import scipy.sparse as sp
        if sp.issparse(adata.raw.X):
            adata.layers['log1p'] = np.log1p(adata.raw.X.toarray())
        else:
            adata.layers['log1p'] = np.log1p(adata.raw.X)
    
    # 使用log1p層進行註釋
    adata_for_annotation = adata.copy()
    adata_for_annotation.X = adata_for_annotation.layers['log1p']

    # 直接使用高精度免疫模型
    if model == 'auto':
        print("\n直接使用高精度免疫模型進行細胞類型註釋...")
        model = 'Immune_All_High.pkl'
    else:
        print(f"\n使用指定模型: {model}")

    print(f"最終使用的模型: {model}")
    
    # 下載或加載模型
    try:
        print("加載預訓練模型...")
        # CellTypist會自動下載模型（如果不存在）
        
        # 進行註釋
        print("進行細胞類型預測...")
        predictions = celltypist.annotate(adata_for_annotation, 
                                          model=model,
                                          majority_voting=majority_voting,
                                          mode='best match')
        
        # 獲取註釋結果
        predicted_labels = predictions.predicted_labels
        
        if majority_voting and 'leiden' in adata.obs.columns:
            # 多數投票模式：基於聚類
            if 'majority_voting' in predicted_labels.columns:
                adata.obs['cell_type'] = predicted_labels['majority_voting'].values
                print("[OK] 使用多數投票模式（基於聚類）")
            else:
                # 如果沒有majority_voting列，使用predicted_labels
                adata.obs['cell_type'] = predicted_labels['predicted_labels'].values
                print("[OK] 使用單細胞模式（majority_voting列不存在）")
        else:
            # 單細胞模式
            adata.obs['cell_type'] = predicted_labels['predicted_labels'].values
            print("[OK] 使用單細胞模式")
        
        # 添加置信度分數（如果可用）
        if 'conf_score' in predicted_labels.columns:
            adata.obs['cell_type_confidence'] = predicted_labels['conf_score'].values
        elif 'score' in predicted_labels.columns:
            adata.obs['cell_type_confidence'] = predicted_labels['score'].values
        
        # 顯示註釋結果統計
        print("\n各細胞類型的細胞數:")
        annotation_counts = adata.obs['cell_type'].value_counts()
        for cell_type, count in annotation_counts.items():
            pct = count / adata.n_obs * 100
            print(f"  {cell_type}: {count:,} 個細胞 ({pct:.2f}%)")
        
        # 顯示置信度統計（如果可用）
        if 'cell_type_confidence' in adata.obs.columns:
            print(f"\n平均置信度: {adata.obs['cell_type_confidence'].mean():.3f}")
            print(f"置信度範圍: [{adata.obs['cell_type_confidence'].min():.3f}, {adata.obs['cell_type_confidence'].max():.3f}]")
        
        return adata
        
    except Exception as e:
        print(f"[ERROR] CellTypist註釋失敗: {e}")
        print("[INFO] 改用基於標記基因的註釋方法...")
        return None

def annotation_visualization(adata):
    """Publication-grade CellTypist annotation figures."""
    print("\n" + "=" * 60)
    print("Generating publication-grade CellTypist annotation figures...")
    print("=" * 60)

    if 'cell_type' not in adata.obs.columns:
        print("[WARNING] cell_type annotation not found; skipping annotation figures")
        return adata

    if 'X_umap' not in adata.obsm.keys():
        print("[WARNING] UMAP not found; computing UMAP with existing graph")
        sc.tl.umap(adata, min_dist=0.5, spread=1.0)

    figdir = Path(sc.settings.figdir)

    print("\n1. CellTypist cell type UMAP...")
    plot_embedding_categorical(
        adata, 'umap', 'cell_type', figdir / 'umap_annotation_cell_type.pdf',
        figsize=(4.8, 3.9), label_on_data=False,
    )
    print("  [OK] CellTypist UMAP saved")

    if 'leiden' in adata.obs.columns:
        print("\n2. CellTypist annotation versus Leiden clusters...")
        plot_embedding_grid(
            adata, 'umap', ['cell_type', 'leiden'],
            figdir / 'umap_annotation_vs_cluster.pdf', ncols=2,
        )
        plot_fraction_heatmap(
            adata, 'leiden', 'cell_type',
            figdir / 'cluster_celltype_fraction_heatmap.pdf',
            figsize=(5.6, 4.6),
        )
        print("  [OK] Annotation versus cluster figures saved")

    if 'sample' in adata.obs.columns:
        print("\n3. Cell type composition by sample...")
        plot_composition(
            adata, 'cell_type', 'sample',
            figdir / 'celltype_composition_by_sample.pdf', normalize='index',
            figsize=(5.8, 3.4),
        )
        print("  [OK] Cell type by sample composition saved")

    print("\n4. CellTypist marker support dotplot...")
    try:
        plot_marker_dotplot(
            adata,
            celltypist_marker_dict(),
            groupby='cell_type',
            out_file=figdir / 'celltype_marker_dotplot_celltypist.pdf',
            use_raw=True,
        )
        print("  [OK] Cell type marker dotplot saved")
    except Exception as e:
        print(f"  [WARNING] Cell type marker dotplot failed: {e}")

    print("\n5. Leiden marker genes...")
    try:
        if 'rank_genes_groups' in adata.uns:
            sc.pl.rank_genes_groups(
                adata, n_genes=5, sharey=False, show=False,
                save='_annotation_markers.pdf', fontsize=7,
            )
            plt.close('all')
            print("  [OK] Leiden marker gene figure saved")
    except Exception as e:
        print(f"  [WARNING] Rank-gene marker plot failed: {e}")

    return adata

def save_results(adata, output_dir):
    """Save CellTypist-only annotation outputs."""
    print("\n" + "=" * 60)
    print("Saving CellTypist annotation results...")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if 'cell_type_literature' in adata.obs.columns:
        del adata.obs['cell_type_literature']

    output_file = output_dir / "adata_annotated.h5ad"
    adata.write(str(output_file))
    print(f"[OK] Annotated AnnData saved to: {output_file}")

    annotation_file = output_dir / "cell_annotations.csv"
    annotation_cols = ['cell_type', 'leiden']
    if 'sample' in adata.obs.columns:
        annotation_cols.append('sample')
    if 'cell_type_confidence' in adata.obs.columns:
        annotation_cols.append('cell_type_confidence')
    adata.obs[annotation_cols].to_csv(annotation_file)
    print(f"[OK] Annotation table saved to: {annotation_file}")

    if 'rank_genes_groups' in adata.uns:
        markers_file = output_dir / "marker_genes.csv"
        result = adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        markers_df = pd.DataFrame({group: result['names'][group] for group in groups})
        markers_df.to_csv(markers_file)
        print(f"[OK] Leiden marker genes saved to: {markers_file}")

    stats_file = output_dir / "annotation_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("CellTypist-only cell-type annotation statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Cells: {adata.n_obs}\n")
        f.write(f"Genes: {adata.n_vars}\n")

        if 'cell_type' in adata.obs.columns:
            f.write(f"\nCell types: {adata.obs['cell_type'].nunique()}\n")
            f.write("\nCell type counts:\n")
            for cell_type, count in adata.obs['cell_type'].value_counts().items():
                pct = count / adata.n_obs * 100
                f.write(f"  {cell_type}: {count} cells ({pct:.2f}%)\n")

            if 'leiden' in adata.obs.columns:
                f.write("\nCell type by Leiden cluster:\n")
                f.write(pd.crosstab(adata.obs['cell_type'], adata.obs['leiden']).to_string())
                f.write("\n")

        if 'sample' in adata.obs.columns and 'cell_type' in adata.obs.columns:
            f.write("\nCell type by sample:\n")
            f.write(pd.crosstab(adata.obs['cell_type'], adata.obs['sample']).to_string())
            f.write("\n")

    print(f"[OK] Annotation statistics saved to: {stats_file}")

def main():
    """Run CellTypist-only annotation and publication-grade visualization."""
    base_dir = Path(__file__).parent
    input_file = base_dir / "03_clustered" / "adata_clustered.h5ad"
    cnv_file = base_dir / "04b_cnv_inferred" / "adata_cnv_inferred.h5ad"
    output_dir = base_dir / "04_annotated"
    output_dir.mkdir(exist_ok=True)

    sc.settings.figdir = str(output_dir / "figures")
    Path(sc.settings.figdir).mkdir(exist_ok=True)

    if cnv_file.exists():
        print("[INFO] Loading CNV-inferred AnnData for annotation context...")
        adata = load_clustered_data(str(cnv_file))
    else:
        print("[INFO] CNV-inferred AnnData not found; loading clustered AnnData...")
        adata = load_clustered_data(str(input_file))

    if adata is None:
        return

    annotation_result = celltypist_annotation(adata, model='auto', majority_voting=True)
    if annotation_result is None:
        raise RuntimeError("CellTypist annotation failed. No alternative annotation branch is enabled in this script.")

    adata = annotation_result
    if 'cell_type_literature' in adata.obs.columns:
        del adata.obs['cell_type_literature']

    print("\n[INFO] Finding Leiden marker genes for annotation support tables/plots...")
    adata = find_marker_genes(adata, groupby='leiden', n_genes=20)

    adata = annotation_visualization(adata)
    save_results(adata, output_dir)

    print("\n" + "=" * 60)
    print("CellTypist-only annotation completed")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {sc.settings.figdir}")
    if 'cell_type_confidence' in adata.obs.columns:
        print("\n[INFO] CellTypist confidence scores are stored in 'cell_type_confidence'.")

if __name__ == "__main__":
    main()
