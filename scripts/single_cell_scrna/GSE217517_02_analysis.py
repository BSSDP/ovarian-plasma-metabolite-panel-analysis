"""
單細胞轉錄組降維分析腳本
進行PCA、UMAP、t-SNE降維分析
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from publication_plotting import (
    plot_embedding_categorical,
    plot_embedding_grid,
    plot_pca_variance,
    set_publication_style,
)

# 設置scanpy參數
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, facecolor='white', figsize=(4.2, 3.7), fontsize=8)
set_publication_style()

# 檢查可選的性能優化庫
try:
    import pynndescent
    PNN_AVAILABLE = True
    print("[INFO] PyNNDescent可用，用於UMAP近鄰搜索加速")
except ImportError:
    PNN_AVAILABLE = False
    print("[INFO] PyNNDescent不可用，使用標準UMAP")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def load_processed_data(input_file):
    """加載預處理後的數據"""
    print("=" * 60)
    print("加載預處理後的數據...")
    print("=" * 60)
    
    if not Path(input_file).exists():
        print(f"[ERROR] 找不到數據文件: {input_file}")
        return None
    
    adata = sc.read_h5ad(input_file)
    print(f"[OK] 數據加載成功")
    print(f"  - 細胞數: {adata.n_obs:,}")
    print(f"  - 基因數: {adata.n_vars:,}")
    
    return adata

def dimensionality_reduction(adata, use_harmony=True, compute_tsne=False):
    """降維分析（PCA, UMAP, t-SNE）"""
    print("\n" + "=" * 60)
    print("降維分析...")
    print("=" * 60)
    
    # 1. PCA降維
    print("\n1. 進行PCA降維...")
    if 'X_pca' not in adata.obsm.keys():
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    print(f"  PCA維度: {adata.obsm['X_pca'].shape}")
    
    # 計算PCA的解釋方差
    print("\n2. PCA解釋方差分析...")
    plot_pca_variance(adata, Path(sc.settings.figdir) / 'pca_variance.pdf', n_pcs=50)
    print("  [OK] PCA方差圖已保存")
    
    # 2. 構建鄰居圖
    print("\n3. 構建鄰居圖...")
    if use_harmony and 'X_pca_harmony' in adata.obsm.keys():
        print("  使用Harmony矯正後的PCA空間")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, use_rep='X_pca_harmony')
    else:
        print("  使用標準PCA空間")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    print("  [OK] 鄰居圖構建完成")
    
    # 3. UMAP降維（可選使用ANN加速）
    print("\n4. 進行UMAP降維...")
    if PNN_AVAILABLE:
        print("  [PERF] 使用PyNNDescent進行近鄰搜索加速")
        try:
            # 使用ANN進行近鄰搜索
            from pynndescent import NNDescent
            import umap

            # 準備數據
            if use_harmony and 'X_pca_harmony' in adata.obsm.keys():
                data = adata.obsm['X_pca_harmony'][:, :40]
            else:
                data = adata.obsm['X_pca'][:, :40]

            # 構建ANN索引
            n_neighbors = 15
            index = NNDescent(data, n_neighbors=n_neighbors, metric='euclidean')
            neighbors, distances = index.query(data, k=n_neighbors)

            # 使用UMAP with precomputed neighbors
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.5,
                spread=1.0,
                metric='euclidean',
                random_state=42
            )

            # 設置預計算的近鄰
            reducer.embedding_ = None
            reducer._raw_data = data
            reducer._n_neighbors = n_neighbors

            # 進行降維
            embedding = reducer.fit_transform(data)
            adata.obsm['X_umap'] = embedding

            print("  [OK] UMAP降維完成（使用ANN加速）")
        except Exception as e:
            print(f"  [WARNING] ANN加速失敗，使用標準UMAP: {e}")
            sc.tl.umap(adata, min_dist=0.5, spread=1.0)
            print("  [OK] UMAP降維完成（標準方法）")
    else:
        sc.tl.umap(adata, min_dist=0.5, spread=1.0)
        print("  [OK] UMAP降維完成（標準方法）")
    
    if compute_tsne:
        print("\n5. Running t-SNE...")
        try:
            sc.tl.tsne(adata, n_pcs=40, perplexity=30)
            print("  [OK] t-SNE completed")
        except Exception as e:
            print(f"  [WARNING] t-SNE failed: {e}")
    else:
        print("\n5. Skipping t-SNE (UMAP is the primary embedding for publication figures)")

    return adata

def visualization(adata):
    """Publication-grade dimensionality reduction figures."""
    print("\n" + "=" * 60)
    print("Generating publication-grade dimensionality reduction figures...")
    print("=" * 60)
    figdir = Path(sc.settings.figdir)

    if 'X_pca' in adata.obsm.keys() and 'sample' in adata.obs.columns:
        print("\n1. PCA by sample...")
        plot_embedding_categorical(
            adata, 'pca', 'sample', figdir / 'pca_by_sample.pdf',
            figsize=(4.4, 3.8), label_on_data=False,
        )
        print("  [OK] PCA by sample saved")

    if 'sample' in adata.obs.columns:
        print("\n2. UMAP by sample...")
        plot_embedding_categorical(
            adata, 'umap', 'sample', figdir / 'umap_by_sample.pdf',
            figsize=(4.5, 3.8), label_on_data=False,
        )
        print("  [OK] UMAP by sample saved")

    qc_vars = [v for v in ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'] if v in adata.obs.columns]
    if qc_vars:
        print("\n3. UMAP QC metrics...")
        plot_embedding_grid(adata, 'umap', qc_vars, figdir / 'umap_qc_metrics.pdf', ncols=3)
        print("  [OK] UMAP QC metric grid saved")

    return adata

def save_results(adata, output_dir):
    """保存降維分析結果"""
    print("\n" + "=" * 60)
    print("保存降維分析結果...")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 保存完整的AnnData對象（包含降維結果）
    output_file = output_dir / "adata_dimreduced.h5ad"
    adata.write(str(output_file))
    print(f"[OK] 降維數據已保存至: {output_file}")
    
    # 2. 保存降維統計信息
    stats_file = output_dir / "dimension_reduction_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("降維分析統計\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"細胞數: {adata.n_obs}\n")
        f.write(f"基因數: {adata.n_vars}\n")
        
        if 'X_pca' in adata.obsm.keys():
            f.write(f"\nPCA維度: {adata.obsm['X_pca'].shape}\n")
            if 'variance_ratio' in adata.uns['pca']:
                var_ratio = adata.uns['pca']['variance_ratio']
                f.write(f"前10個主成分解釋方差:\n")
                for i, ratio in enumerate(var_ratio[:10]):
                    f.write(f"  PC{i+1}: {ratio:.4f}\n")
                f.write(f"前10個主成分累積解釋方差: {sum(var_ratio[:10]):.4f}\n")
        
        if 'X_umap' in adata.obsm.keys():
            f.write(f"\nUMAP維度: {adata.obsm['X_umap'].shape}\n")
        
        if 'X_tsne' in adata.obsm.keys():
            f.write(f"\nt-SNE維度: {adata.obsm['X_tsne'].shape}\n")
        
        if 'sample' in adata.obs.columns:
            f.write(f"\n樣本數: {adata.obs['sample'].nunique()}\n")
            f.write(f"樣本分布:\n{adata.obs['sample'].value_counts().to_string()}\n")
        
        if 'group' in adata.obs.columns:
            f.write(f"\n組別分布:\n{adata.obs['group'].value_counts().to_string()}\n")
    
    print(f"[OK] 統計信息已保存至: {stats_file}")

def main():
    """主函數"""
    # 設置路徑
    base_dir = Path(__file__).parent
    input_file = base_dir / "01_processed" / "adata_raw.h5ad"
    output_dir = base_dir / "02_dimreduced"
    output_dir.mkdir(exist_ok=True)

    # GSE217517數據集說明
    print("\n" + "=" * 60)
    print("GSE217517 單細胞轉錄組數據降維分析")
    print("數據集: 人類卵巢癌單細胞RNA-seq數據")
    print("=" * 60)
    
    # 設置圖形保存目錄
    sc.settings.figdir = str(output_dir / "figures")
    Path(sc.settings.figdir).mkdir(exist_ok=True)
    
    # 1. 加載數據
    adata = load_processed_data(str(input_file))
    if adata is None:
        return
    
    # 2. 降維分析
    use_harmony = 'X_pca_harmony' in adata.obsm.keys()
    adata = dimensionality_reduction(adata, use_harmony=use_harmony)
    
    # 3. 降維結果可視化
    adata = visualization(adata)
    
    # 4. 保存結果
    save_results(adata, output_dir)
    
    print("\n" + "=" * 60)
    print("降維分析完成！")
    print("=" * 60)
    print(f"\n結果文件保存在: {output_dir}")
    print(f"圖形文件保存在: {sc.settings.figdir}")
    print("\n降維數據已保存，可以繼續進行聚類分析\n")

if __name__ == "__main__":
    main()
