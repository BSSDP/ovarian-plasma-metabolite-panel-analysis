"""
單細胞轉錄組數據讀取與預處理腳本
讀取GSE184880數據並進行質控和預處理
"""

import os
import tarfile
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import warnings
warnings.filterwarnings('ignore')

from publication_plotting import (
    plot_count_bar,
    plot_qc_filtering_summary,
    plot_qc_violin,
    set_publication_style,
)

# 設置scanpy參數
sc.settings.verbosity = 3  # 顯示詳細信息
sc.settings.set_figure_params(dpi=300, facecolor='white', fontsize=8)
set_publication_style()

def extract_tar_file(tar_path, extract_dir):
    """解壓tar文件"""
    print("=" * 60)
    print("解壓數據文件...")
    print("=" * 60)
    
    if os.path.exists(extract_dir):
        print(f"[OK] 解壓目錄已存在: {extract_dir}")
        return True
    
    if not os.path.exists(tar_path):
        print(f"[ERROR] 錯誤: 找不到數據文件 {tar_path}")
        return False
    
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)
        print(f"[OK] 數據已解壓至: {extract_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] 解壓失敗: {e}")
        return False

def check_data_structure(data_dir):
    """檢查解壓後的數據結構"""
    print("\n" + "=" * 60)
    print("檢查數據結構...")
    print("=" * 60)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] 錯誤: 數據目錄不存在 {data_dir}")
        return None
    
    # 列出所有文件和文件夾
    items = list(data_path.iterdir())
    print(f"\n找到 {len(items)} 個項目:")
    for item in items:
        print(f"  - {item.name} ({'目錄' if item.is_dir() else '文件'})")
    
    return items

def load_10x_from_files(matrix_file, genes_file, barcodes_file, sample_name):
    """從指定的10x格式文件加載數據"""
    print(f"  讀取10x格式文件:")
    print(f"    - Matrix: {Path(matrix_file).name}")
    print(f"    - Genes: {Path(genes_file).name}")
    print(f"    - Barcodes: {Path(barcodes_file).name}")
    
    temp_dir = None
    try:
        # 創建臨時目錄
        temp_dir = tempfile.mkdtemp(prefix=f"scRNA_{sample_name}_")
        temp_path = Path(temp_dir)
        
        # 複製文件並重命名為標準名稱
        shutil.copy2(matrix_file, temp_path / "matrix.mtx.gz")
        shutil.copy2(genes_file, temp_path / "genes.tsv.gz")
        shutil.copy2(barcodes_file, temp_path / "barcodes.tsv.gz")
        
        # scanpy的新版本期望features.tsv.gz，所以也創建一個副本
        # 這樣可以同時支持舊版本（genes.tsv）和新版本（features.tsv）
        shutil.copy2(genes_file, temp_path / "features.tsv.gz")
        
        # 讀取數據
        adata = sc.read_10x_mtx(
            str(temp_path),
            var_names='gene_symbols',
            cache=True
        )
        # 確保變量名稱唯一
        adata.var_names_make_unique()
        
        # 清理臨時目錄
        shutil.rmtree(temp_dir)
        temp_dir = None
        
        return adata
    except Exception as e:
        print(f"  [ERROR] 讀取失敗: {e}")
        # 清理臨時目錄（如果存在）
        if temp_dir is not None and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return None

def load_single_sample(sample_path, sample_name):
    """加載單個樣本的數據"""
    sample_path = Path(sample_path)
    
    # 嘗試不同的數據格式
    # 1. 10x Genomics格式 (matrix.mtx + genes.tsv + barcodes.tsv)
    matrix_file = None
    genes_file = None
    barcodes_file = None
    
    # 檢查標準名稱
    if (sample_path / "matrix.mtx").exists() or (sample_path / "matrix.mtx.gz").exists():
        matrix_file = sample_path / ("matrix.mtx.gz" if (sample_path / "matrix.mtx.gz").exists() else "matrix.mtx")
        genes_file = sample_path / ("genes.tsv.gz" if (sample_path / "genes.tsv.gz").exists() else "genes.tsv")
        barcodes_file = sample_path / ("barcodes.tsv.gz" if (sample_path / "barcodes.tsv.gz").exists() else "barcodes.tsv")
        
        if matrix_file.exists() and genes_file.exists() and barcodes_file.exists():
            print(f"  檢測到10x Genomics格式（標準文件名）")
            try:
                adata = sc.read_10x_mtx(
                    str(sample_path),
                    var_names='gene_symbols',
                    cache=True
                )
                adata.var_names_make_unique()
                return adata
            except Exception as e:
                print(f"  讀取10x格式失敗: {e}")
    
    # 2. 10x Genomics格式 (filtered_feature_bc_matrix文件夾)
    filtered_path = sample_path / "filtered_feature_bc_matrix"
    if filtered_path.exists():
        print(f"  檢測到filtered_feature_bc_matrix文件夾")
        try:
            adata = sc.read_10x_mtx(
                str(filtered_path),
                var_names='gene_symbols',
                cache=True
            )
            adata.var_names_make_unique()
            return adata
        except Exception as e:
            print(f"  讀取失敗: {e}")
    
    # 3. CSV格式
    csv_files = list(sample_path.glob("*.csv"))
    if csv_files:
        print(f"  檢測到CSV文件: {csv_files[0].name}")
        try:
            adata = sc.read_csv(str(csv_files[0]))
            return adata
        except Exception as e:
            print(f"  讀取CSV失敗: {e}")
    
    # 4. TSV格式
    tsv_files = list(sample_path.glob("*.tsv"))
    if tsv_files:
        print(f"  檢測到TSV文件: {tsv_files[0].name}")
        try:
            adata = sc.read_csv(str(tsv_files[0]), delimiter='\t')
            return adata
        except Exception as e:
            print(f"  讀取TSV失敗: {e}")
    
    # 5. H5AD格式
    h5ad_files = list(sample_path.glob("*.h5ad"))
    if h5ad_files:
        print(f"  檢測到H5AD文件: {h5ad_files[0].name}")
        try:
            adata = sc.read_h5ad(str(h5ad_files[0]))
            return adata
        except Exception as e:
            print(f"  讀取H5AD失敗: {e}")
    
    return None

def identify_samples_from_files(data_dir):
    """從文件列表中識別樣本（GSE217517格式）"""
    data_path = Path(data_dir)
    files = list(data_path.glob("*.gz"))

    # GSE217517文件名格式: GSM6720925_single_cell_matrix_2251.mtx.gz
    samples = {}
    for file in files:
        parts = file.name.split('_')
        if len(parts) >= 4 and parts[1] == 'single' and parts[2] == 'cell':
            file_type = parts[3]  # matrix, features, barcodes
            sample_id = parts[0]  # GSM6720925

            if sample_id not in samples:
                samples[sample_id] = {}

            if file_type == 'matrix':
                samples[sample_id]['matrix'] = file
            elif file_type == 'features':
                samples[sample_id]['genes'] = file
            elif file_type == 'barcodes':
                samples[sample_id]['barcodes'] = file

    # 只返回完整的樣本（有3個文件）
    complete_samples = {}
    for sample_name, files_dict in samples.items():
        if 'matrix' in files_dict and 'genes' in files_dict and 'barcodes' in files_dict:
            complete_samples[sample_name] = files_dict

    return complete_samples

def load_all_samples(data_dir):
    """加載所有樣本的數據"""
    print("\n" + "=" * 60)
    print("加載所有樣本數據...")
    print("=" * 60)
    
    data_path = Path(data_dir)
    items = check_data_structure(data_dir)
    
    if items is None:
        return None
    
    adatas = []
    sample_names = []
    
    # 首先嘗試識別文件格式的樣本（所有文件在同一目錄）
    samples_dict = identify_samples_from_files(data_dir)
    
    if len(samples_dict) > 0:
        print(f"\n識別到 {len(samples_dict)} 個樣本（文件格式）")
        for sample_name, files_dict in samples_dict.items():
            print(f"\n處理樣本: {sample_name}")
            adata = load_10x_from_files(
                files_dict['matrix'],
                files_dict['genes'],
                files_dict['barcodes'],
                sample_name
            )
            if adata is not None:
                adatas.append(adata)
                sample_names.append(sample_name)
                print(f"  [OK] 成功加載，細胞數: {adata.n_obs}, 基因數: {adata.n_vars}")
            else:
                print(f"  [ERROR] 無法加載此樣本")
    
    # 如果沒有找到文件格式的樣本，嘗試文件夾格式
    if len(adatas) == 0:
        print("\n嘗試文件夾格式...")
        for item in items:
            if item.is_dir():
                print(f"\n處理樣本: {item.name}")
                adata = load_single_sample(item, item.name)
                if adata is not None:
                    adatas.append(adata)
                    sample_names.append(item.name)
                    print(f"  [OK] 成功加載，細胞數: {adata.n_obs}, 基因數: {adata.n_vars}")
                else:
                    print(f"  [ERROR] 無法加載此樣本")
            elif item.suffix in ['.h5ad', '.h5']:
                # 直接是h5ad文件
                print(f"\n處理文件: {item.name}")
                try:
                    adata = sc.read_h5ad(str(item))
                    adatas.append(adata)
                    sample_names.append(item.stem)
                    print(f"  [OK] 成功加載，細胞數: {adata.n_obs}, 基因數: {adata.n_vars}")
                except Exception as e:
                    print(f"  [ERROR] 讀取失敗: {e}")
    
    if len(adatas) == 0:
        print("\n[ERROR] 未能加載任何樣本數據")
        return None
    
    # 合併所有樣本
    print("\n" + "=" * 60)
    print("合併所有樣本...")
    print("=" * 60)
    
    # 添加樣本名稱到obs
    for i, adata in enumerate(adatas):
        adata.obs['sample'] = sample_names[i]
    
    # 合併數據
    if len(adatas) == 1:
        adata = adatas[0]
    else:
        # 使用concatenate合併，使用臨時的batch_key
        # 注意：concatenate會將batch_key的值轉換為索引，所以我們先用臨時列
        adata = adatas[0].concatenate(adatas[1:], join='outer', batch_key='batch_temp')
        
        # 恢復實際的樣本名稱
        # concatenate會創建batch_temp列，值為'0', '1', '2'等字符串
        if 'batch_temp' in adata.obs.columns:
            # 將batch_temp轉換為整數索引，然後映射到實際樣本名稱
            batch_indices = adata.obs['batch_temp'].astype(str)
            # 提取批次索引（可能是'0', '1', '2'或'0-0', '1-0'等格式）
            def extract_batch_idx(x):
                x_str = str(x)
                if '-' in x_str:
                    return int(x_str.split('-')[0])
                else:
                    return int(x_str)
            
            batch_indices_int = batch_indices.apply(extract_batch_idx)
            # 映射到實際樣本名稱
            adata.obs['sample'] = batch_indices_int.apply(lambda idx: sample_names[idx] if idx < len(sample_names) else f'Unknown_{idx}')
            
            # 刪除臨時的batch_temp列
            adata.obs.drop(columns=['batch_temp'], inplace=True)
        else:
            # 如果沒有batch_temp列，手動根據細胞順序分配
            print("[WARNING] 未找到batch_temp列，手動分配樣本名稱...")
            start_idx = 0
            for i, name in enumerate(sample_names):
                end_idx = start_idx + adatas[i].n_obs
                adata.obs.iloc[start_idx:end_idx, adata.obs.columns.get_loc('sample')] = name
                start_idx = end_idx
    
    print(f"[OK] 合併完成，總細胞數: {adata.n_obs}, 總基因數: {adata.n_vars}")
    
    # 驗證樣本名稱
    if 'sample' in adata.obs.columns:
        unique_samples = adata.obs['sample'].unique()
        print(f"\n樣本名稱驗證:")
        print(f"  找到 {len(unique_samples)} 個唯一樣本")
        for s in unique_samples[:5]:  # 顯示前5個
            count = (adata.obs['sample'] == s).sum()
            print(f"    {s}: {count} 個細胞")
    
    return adata

def add_sample_metadata(adata, data_dir):
    """添加樣本元數據（GSE217517只有腫瘤樣本）"""
    print("\n" + "=" * 60)
    print("添加樣本元數據...")
    print("=" * 60)

    # GSE217517數據集只包含卵巢癌腫瘤樣本
    def infer_group(sample_name):
        # GSE217517的所有樣本都是腫瘤樣本
        # 樣本名稱格式：GSM6720925, GSM6720926, etc.
        return 'Tumor'

    if 'sample' in adata.obs.columns:
        # 顯示樣本名稱以便調試
        print(f"\n樣本名稱列表:")
        unique_samples = adata.obs['sample'].unique()
        for s in unique_samples:
            print(f"  - {s}")

        # 所有樣本都設為Tumor組
        adata.obs['group'] = adata.obs['sample'].apply(infer_group)
        print(f"\n樣本分組統計:")
        print(adata.obs['group'].value_counts())

        # 添加樣本描述信息
        print(f"\nGSE217517數據集信息:")
        print(f"  - 這是人類卵巢癌單細胞轉錄組數據集")
        print(f"  - 包含 {len(unique_samples)} 個腫瘤樣本")
        print(f"  - 總細胞數: {len(adata)}")

    else:
        print("[WARNING] 無法添加樣本分組信息，請手動檢查樣本名稱")

    return adata

def quality_control(adata):
    """Quality control and filtering with publication-ready summary figures."""
    print("\n" + "=" * 60)
    print("Running quality control...")
    print("=" * 60)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    before_stats = {"cells": adata.n_obs, "genes": adata.n_vars}
    print("\nQC metrics before filtering:")
    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Mean detected genes per cell: {adata.obs['n_genes_by_counts'].mean():.1f}")
    print(f"  Mean total counts per cell: {adata.obs['total_counts'].mean():.1f}")
    print(f"  Mean mitochondrial percentage: {adata.obs['pct_counts_mt'].mean():.2f}%")

    print("\n" + "-" * 60)
    print("Filtering low-quality cells and genes...")
    print("-" * 60)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    adata = adata[adata.obs['pct_counts_mt'] < 20, :].copy()
    after_stats = {
        "cells": adata.n_obs,
        "genes": adata.n_vars,
        "thresholds": "min genes >= 200\nmitochondrial counts < 20%",
    }
    print(f"  Cells after filtering: {adata.n_obs}")
    print(f"  Genes after filtering: {adata.n_vars}")

    try:
        figdir = Path(sc.settings.figdir) if sc.settings.figdir else Path("figures")
        plot_qc_violin(
            adata,
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
            figdir / 'violin_qc_after_filtering.pdf',
        )
        if 'sample' in adata.obs.columns:
            plot_count_bar(adata, 'sample', figdir / 'sample_cell_counts_after_filtering.pdf')
        plot_qc_filtering_summary(before_stats, after_stats, figdir / 'qc_filtering_summary.pdf')
        print("  [OK] QC figures saved")
    except Exception as e:
        print(f"  [WARNING] QC figure generation failed: {e}")

    return adata

def normalize_data(adata):
    """數據歸一化和標準化"""
    print("\n" + "=" * 60)
    print("數據歸一化和標準化...")
    print("=" * 60)
    
    # 保存原始計數
    adata.raw = adata
    adata.layers['counts'] = adata.X.copy()
    
    # 歸一化到每個細胞總計數為10,000
    print("\n1. 歸一化到每個細胞總計數為10,000")
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 對數轉換
    print("2. 對數轉換 (log1p)")
    sc.pp.log1p(adata)
    
    # 尋找高變異基因
    print("3. 識別高變異基因")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    print(f"  找到 {sum(adata.var.highly_variable)} 個高變異基因")
    
    # 標準化（可選，用於某些下游分析）
    print("4. 標準化數據")
    adata.layers['normalized'] = adata.X.copy()
    sc.pp.scale(adata, max_value=10)
    adata.layers['scaled'] = adata.X.copy()
    
    print("[OK] 歸一化和標準化完成")
    
    return adata

def batch_correction(adata, method='harmony', batch_key='sample'):
    """批次矯正（可選）
    
    參數:
        adata: AnnData對象
        method: 批次矯正方法 ('harmony', 'scanorama', 'bbknn', 'none')
        batch_key: 批次變量名稱（默認'sample'）
    """
    print("\n" + "=" * 60)
    print("批次矯正...")
    print("=" * 60)
    
    if method == 'none' or batch_key not in adata.obs.columns:
        print("[INFO] 跳過批次矯正")
        return adata
    
    if adata.obs[batch_key].nunique() <= 1:
        print(f"[INFO] 只有1個批次，跳過批次矯正")
        return adata
    
    print(f"\n使用 {method} 方法進行批次矯正")
    print(f"批次變量: {batch_key}")
    print(f"批次數: {adata.obs[batch_key].nunique()}")
    
    try:
        if method == 'harmony':
            # 使用Harmony進行批次矯正
            # 需要先進行PCA
            print("\n1. 進行PCA降維...")
            sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
            
            print("2. 使用Harmony進行批次矯正...")
            try:
                import scanpy.external as sce
                sce.pp.harmony_integrate(adata, key=batch_key, max_iter_harmony=30)
                print("[OK] Harmony批次矯正完成")
            except ImportError:
                print("[WARNING] harmonypy未安裝，嘗試使用scanorama...")
                method = 'scanorama'
        
        if method == 'scanorama':
            # 使用Scanorama進行批次矯正
            print("\n使用Scanorama進行批次矯正...")
            try:
                import scanpy.external as sce
                # Scanorama需要按批次分組的數據列表
                batches = adata.obs[batch_key].unique()
                adatas = [adata[adata.obs[batch_key] == batch].copy() for batch in batches]
                sce.pp.scanorama_integrate(adatas, var_names=adata.var_names, dimred=50)
                # 合併結果
                adata = adatas[0].concatenate(adatas[1:], join='outer')
                print("[OK] Scanorama批次矯正完成")
            except ImportError:
                print("[WARNING] scanorama未安裝，跳過批次矯正")
                return adata
        
        if method == 'bbknn':
            # 使用BBKNN進行批次矯正（在構建鄰居圖時）
            print("\n[INFO] BBKNN將在構建鄰居圖時使用")
            print("請在後續分析中使用 sc.external.pp.bbknn()")
            return adata
            
    except Exception as e:
        print(f"[WARNING] 批次矯正失敗: {e}")
        print("[INFO] 繼續使用未矯正的數據")
        return adata
    
    return adata

def main():
    """主函數"""
    print("開始執行數據加載腳本...")

    # GSE217517數據集說明
    print("\n" + "=" * 60)
    print("GSE217517 單細胞轉錄組數據加載和預處理")
    print("數據集: 人類卵巢癌單細胞RNA-seq數據")
    print("=" * 60)

    # 設置路徑
    base_dir = Path(__file__).parent
    print(f"腳本目錄: {base_dir}")
    # GSE217517數據直接是.gz文件，不需要解壓tar
    # 數據目錄在父目錄中
    data_dir = base_dir.parent / "GSE217517_RAW"
    output_dir = base_dir / "01_processed"
    output_dir.mkdir(exist_ok=True)
    sc.settings.figdir = str(output_dir / "figures")
    Path(sc.settings.figdir).mkdir(exist_ok=True)

    print(f"數據目錄: {data_dir}")
    print(f"輸出目錄: {output_dir}")
    
    # 1. 檢查數據目錄
    if not data_dir.exists():
        print(f"\n[ERROR] 找不到數據目錄: {data_dir}")
        print("請確保GSE217517_RAW目錄存在於當前腳本目錄下")
        return

    # 2. 加載所有樣本
    print("\n2. 加載樣本數據...")
    adata = load_all_samples(str(data_dir))
    if adata is None:
        print("\n[ERROR] 無法加載數據，請檢查數據格式")
        return
    
    # 3. 添加樣本元數據
    print("\n3. 添加樣本元數據...")
    adata = add_sample_metadata(adata, str(data_dir))

    # 4. 質控和過濾
    print("\n4. 進行質控和過濾...")
    adata = quality_control(adata)

    # 5. 歸一化和標準化
    print("\n5. 進行歸一化和標準化...")
    adata = normalize_data(adata)
    
    # 6. 批次矯正（可選，如果有多個樣本/批次）
    # 可以選擇 'harmony', 'scanorama', 'bbknn', 或 'none'
    adata = batch_correction(adata, method='harmony', batch_key='sample')
    
    # 7. 保存處理後的數據
    print("\n" + "=" * 60)
    print("保存處理後的數據...")
    print("=" * 60)
    
    output_file = output_dir / "adata_raw.h5ad"

    # 數據壓縮保存（減少存儲空間）
    print("  使用壓縮格式保存數據...")
    try:
        # 使用gzip壓縮（默認設置）
        adata.write(str(output_file), compression='gzip', compression_opts=6)
        print(f"[OK] 數據已壓縮保存至: {output_file}")

        # 檢查文件大小
        file_size = output_file.stat().st_size / (1024**2)  # MB
        print(f"  文件大小: {file_size:.1f} MB")

    except Exception as e:
        print(f"  [WARNING] 壓縮保存失敗，使用標準格式: {e}")
        adata.write(str(output_file))
        print(f"[OK] 數據已保存至: {output_file}（未壓縮）")
    
    # 保存統計信息
    stats_file = output_dir / "loading_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("數據加載和預處理統計\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"總細胞數: {adata.n_obs}\n")
        f.write(f"總基因數: {adata.n_vars}\n")
        f.write(f"樣本數: {adata.obs['sample'].nunique()}\n")
        if 'group' in adata.obs.columns:
            f.write(f"\n樣本分組:\n{adata.obs['group'].value_counts().to_string()}\n")
        f.write(f"\n質控指標:\n")
        f.write(f"  平均每個細胞的基因數: {adata.obs['n_genes_by_counts'].mean():.1f}\n")
        f.write(f"  平均每個細胞的UMI數: {adata.obs['total_counts'].mean():.1f}\n")
        f.write(f"  平均線粒體基因比例: {adata.obs['pct_counts_mt'].mean():.2f}%\n")
    
    print(f"[OK] 統計信息已保存至: {stats_file}")
    
    print("\n" + "=" * 60)
    print("數據加載和預處理完成！")
    print("=" * 60)
    print(f"\n數據摘要:")
    print(f"  - 細胞數: {adata.n_obs:,}")
    print(f"  - 基因數: {adata.n_vars:,}")
    print(f"  - 樣本數: {adata.obs['sample'].nunique()}")
    if 'group' in adata.obs.columns:
        print(f"  - 樣本分組: {dict(adata.obs['group'].value_counts())}")
    print(f"\n處理後的數據文件: {output_file}")
    print("\n可以繼續進行後續分析（降維、聚類等）\n")

if __name__ == "__main__":
    main()

