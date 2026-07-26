"""
單細胞轉錄組CNV推斷腳本
使用infercnvpy從單細胞轉錄組數據推斷拷貝數變異（CNV）事件
用於識別腫瘤細胞和正常細胞

針對卵巢癌優化的策略：
- 只對上皮細胞應用嚴格的CNV閾值進行腫瘤分類
- 對非上皮細胞（如免疫細胞）使用寬鬆閾值或直接標記為正常
- 這樣可以顯著減少假陽性率，提高分類準確性
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from publication_plotting import plot_embedding_categorical, plot_embedding_continuous
warnings.filterwarnings('ignore')

# 嘗試導入infercnvpy
try:
    import infercnvpy as cnv
    INFERCNVPY_AVAILABLE = True
except ImportError:
    INFERCNVPY_AVAILABLE = False
    print("[WARNING] infercnvpy未安裝，請運行: pip install infercnvpy")

# 設置scanpy參數
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(10, 8))

def load_annotated_data(input_file):
    """加載註釋後的數據"""
    print("=" * 60)
    print("加載註釋後的數據...")
    print("=" * 60)
    
    if not Path(input_file).exists():
        print(f"[ERROR] 找不到數據文件: {input_file}")
        return None
    
    adata = sc.read_h5ad(input_file)
    print(f"[OK] 數據加載成功")
    print(f"  - 細胞數: {adata.n_obs:,}")
    print(f"  - 基因數: {adata.n_vars:,}")
    
    if 'cell_type' in adata.obs.columns:
        print(f"  - 細胞類型數: {adata.obs['cell_type'].nunique()}")
        print(f"  - 細胞類型: {list(adata.obs['cell_type'].unique())}")
    
    if 'group' in adata.obs.columns:
        print(f"  - 組別: {list(adata.obs['group'].unique())}")
    
    return adata

def prepare_reference_cells(adata):
    """準備參考細胞（正常細胞）"""
    print("\n" + "=" * 60)
    print("準備參考細胞...")
    print("=" * 60)
    
    # 方法1: 使用Normal組的細胞作為參考
    if 'group' in adata.obs.columns:
        normal_cells = adata.obs['group'] == 'Normal'
        n_normal = normal_cells.sum()
        if n_normal > 0:
            print(f"  使用Normal組作為參考: {n_normal} 個細胞")
            return normal_cells
        else:
            print(f"  [WARNING] Normal組沒有細胞，嘗試其他方法...")
    
    # 方法2: 使用非上皮細胞作為參考（如果沒有Normal組或Normal組為空）
    if 'cell_type' in adata.obs.columns:
        # 排除上皮細胞，使用其他細胞類型作為參考
        non_epithelial = ~adata.obs['cell_type'].str.contains('Epithelial', case=False, na=False)
        n_ref = non_epithelial.sum()
        if n_ref > 0:
            print(f"  使用非上皮細胞作為參考: {n_ref} 個細胞")
            print(f"    參考細胞類型: {adata.obs.loc[non_epithelial, 'cell_type'].unique()}")
            return non_epithelial
        else:
            print(f"  [WARNING] 沒有非上皮細胞，嘗試其他方法...")
    
    # 方法3: 使用免疫細胞作為參考（T細胞、B細胞等）
    if 'cell_type' in adata.obs.columns:
        immune_types = ['T', 'B', 'NK', 'Monocyte', 'Macrophage', 'Dendritic']
        immune_mask = adata.obs['cell_type'].str.contains('|'.join(immune_types), case=False, na=False)
        n_immune = immune_mask.sum()
        if n_immune > 0:
            print(f"  使用免疫細胞作為參考: {n_immune} 個細胞")
            return immune_mask
    
    # 方法4: 隨機選擇一部分細胞作為參考
    print("  [WARNING] 未找到合適的參考細胞，隨機選擇20%細胞作為參考")
    n_ref = max(int(adata.n_obs * 0.2), 100)  # 至少100個細胞
    ref_mask = np.zeros(adata.n_obs, dtype=bool)
    ref_indices = np.random.choice(adata.n_obs, min(n_ref, adata.n_obs), replace=False)
    ref_mask[ref_indices] = True
    return ref_mask

def add_genomic_positions(adata):
    """添加基因的染色體位置信息到adata.var"""
    print("\n" + "=" * 60)
    print("添加基因染色體位置信息...")
    print("=" * 60)
    
    # 檢查是否已經有位置信息
    if all(col in adata.var.columns for col in ['chromosome', 'start', 'end']):
        print("[OK] 基因位置信息已存在")
        return adata
    
    print("從Ensembl REST API獲取基因位置信息...")
    
    try:
        import requests
        import json
        import time
        import pandas as pd
        
        # 獲取基因符號列表
        gene_symbols = adata.var_names.tolist()
        print(f"  查詢 {len(gene_symbols)} 個基因的位置信息...")
        
        # 使用Ensembl REST API
        server = "https://rest.ensembl.org"
        ext = "/lookup/symbol/homo_sapiens"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        # 批量查詢（每次最多100個基因，避免超時）
        batch_size = 100
        all_positions = []
        failed_batches = 0
        
        for i in range(0, len(gene_symbols), batch_size):
            batch_genes = gene_symbols[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(gene_symbols)-1)//batch_size + 1
            print(f"  查詢批次 {batch_num}/{total_batches} ({len(batch_genes)} 個基因)...")
            
            try:
                data = json.dumps({"symbols": batch_genes})
                r = requests.post(server + ext, headers=headers, data=data, timeout=60)
                
                if r.ok:
                    results = r.json()
                    batch_results = []
                    for gene_name, gene_info in results.items():
                        if isinstance(gene_info, dict) and 'seq_region_name' in gene_info and 'start' in gene_info:
                            batch_results.append({
                                'external_gene_name': gene_name,
                                'chromosome_name': str(gene_info['seq_region_name']),
                                'start_position': int(gene_info['start']),
                                'end_position': int(gene_info.get('end', gene_info['start']))
                            })
                    
                    if batch_results:
                        all_positions.append(pd.DataFrame(batch_results))
                        print(f"    [OK] 獲取到 {len(batch_results)} 個基因的位置信息")
                    else:
                        print(f"    [WARNING] 批次 {batch_num} 未獲取到位置信息")
                        failed_batches += 1
                else:
                    print(f"    [WARNING] 批次 {batch_num} API請求失敗: {r.status_code}")
                    failed_batches += 1
                
                # 避免請求過快，添加短暫延遲
                time.sleep(0.1)
                
            except Exception as e:
                print(f"    [WARNING] 批次 {batch_num} 查詢失敗: {e}")
                failed_batches += 1
                continue
        
        if not all_positions:
            print("[WARNING] 無法從Ensembl獲取基因位置信息，嘗試其他方法...")
            raise Exception("Ensembl REST API查詢失敗")
        
        # 合併所有查詢結果
        positions_df = pd.concat(all_positions, ignore_index=True)
        print(f"[INFO] 成功獲取 {len(positions_df)} 個基因的位置信息（失敗批次: {failed_batches}/{total_batches}）")
        
        # 重命名列（處理可能的列名差異）
        rename_dict = {}
        if 'hgnc_symbol' in positions_df.columns:
            rename_dict['hgnc_symbol'] = 'gene_symbol'
        elif 'external_gene_name' in positions_df.columns:
            rename_dict['external_gene_name'] = 'gene_symbol'
        
        rename_dict.update({
            'chromosome_name': 'chromosome',
            'start_position': 'start',
            'end_position': 'end'
        })
        
        positions_df = positions_df.rename(columns=rename_dict)
        
        # 處理染色體名稱（確保格式為chr1, chr2等）
        positions_df['chromosome'] = positions_df['chromosome'].astype(str)
        positions_df = positions_df[~positions_df['chromosome'].isin(['nan', 'None'])]
        positions_df['chromosome'] = 'chr' + positions_df['chromosome']
        
        # 移除性染色體（如果需要的話，但infercnvpy會自動處理）
        # positions_df = positions_df[~positions_df['chromosome'].isin(['chrX', 'chrY'])]
        
        # 合併到adata.var
        # 創建一個臨時的DataFrame，索引為gene_symbol
        positions_df = positions_df.set_index('gene_symbol')
        
        # 只保留需要的列
        positions_df = positions_df[['chromosome', 'start', 'end']]
        
        # 合併到adata.var
        adata.var = adata.var.join(positions_df, how='left')
        
        # 檢查有多少基因獲得了位置信息
        n_with_pos = adata.var[['chromosome', 'start', 'end']].notna().all(axis=1).sum()
        print(f"[OK] {n_with_pos}/{len(adata.var)} 個基因獲得了位置信息 ({n_with_pos/len(adata.var)*100:.1f}%)")
        
        if n_with_pos < len(adata.var) * 0.5:
            print("[WARNING] 少於50%的基因獲得了位置信息，CNV推斷可能不準確")
        
        return adata
        
    except Exception as e:
        print(f"[WARNING] 從Ensembl獲取位置信息失敗: {e}")
        print("嘗試使用備用方法...")
        
        # 備用方法：使用預定義的基因位置映射（如果可用）
        # 或者使用簡化的方法：根據基因名稱推斷（不準確，但可以運行）
        print("[INFO] 使用簡化方法：為缺少位置信息的基因分配假位置")
        print("      注意：這會影響CNV推斷的準確性")
        
        # 為缺少位置信息的基因創建假位置（僅用於測試）
        if 'chromosome' not in adata.var.columns:
            adata.var['chromosome'] = None
        if 'start' not in adata.var.columns:
            adata.var['start'] = None
        if 'end' not in adata.var.columns:
            adata.var['end'] = None
        
        # 填充缺失值（使用假值，僅用於測試）
        missing_mask = adata.var['chromosome'].isna()
        if missing_mask.sum() > 0:
            print(f"[WARNING] {missing_mask.sum()} 個基因缺少位置信息，將使用假值")
            # 為這些基因分配假位置（按順序分配到不同染色體）
            n_missing = missing_mask.sum()
            chromosomes = [f'chr{i}' for i in range(1, 23)]  # chr1-chr22
            for i, idx in enumerate(adata.var[missing_mask].index):
                chr_idx = i % len(chromosomes)
                adata.var.loc[idx, 'chromosome'] = chromosomes[chr_idx]
                # 確保start和end是整數類型
                adata.var.loc[idx, 'start'] = int((i // len(chromosomes)) * 1000000 + 1)
                adata.var.loc[idx, 'end'] = int((i // len(chromosomes)) * 1000000 + 100000)
        
        # 確保start和end是數值類型（整數）
        if 'start' in adata.var.columns:
            adata.var['start'] = pd.to_numeric(adata.var['start'], errors='coerce').astype('Int64')
        if 'end' in adata.var.columns:
            adata.var['end'] = pd.to_numeric(adata.var['end'], errors='coerce').astype('Int64')
        
        print("[WARNING] 已使用假位置信息，CNV推斷結果可能不準確")
        return adata

def infer_cnv_infercnvpy(adata, reference_cells, gene_order_file=None):
    """使用infercnvpy推斷CNV"""
    print("\n" + "=" * 60)
    print("使用infercnvpy推斷CNV...")
    print("=" * 60)
    
    if not INFERCNVPY_AVAILABLE:
        print("[ERROR] infercnvpy未安裝，請運行: pip install infercnvpy")
        return None
    
    print(f"  參考細胞數: {reference_cells.sum()}")
    print(f"  待分析細胞數: {(~reference_cells).sum()}")
    
    # 檢查並添加基因位置信息
    adata = add_genomic_positions(adata)
    
    # 檢查位置信息是否完整
    required_cols = ['chromosome', 'start', 'end']
    if not all(col in adata.var.columns for col in required_cols):
        print("[ERROR] 缺少必需的基因位置信息列")
        return None
    
    # 檢查是否有足夠的基因有位置信息
    n_with_pos = adata.var[required_cols].notna().all(axis=1).sum()
    if n_with_pos < 100:
        print(f"[ERROR] 只有 {n_with_pos} 個基因有位置信息，不足以進行CNV推斷")
        return None
    
    # 設置參考細胞標記
    # infercnvpy需要參考細胞的標記在adata.obs中
    # 我們需要創建一個標記列，標記哪些是參考細胞
    adata.obs['is_reference'] = reference_cells.astype(str)
    
    # 運行CNV推斷
    print("\n運行CNV推斷（這可能需要一些時間）...")
    print("  參數: reference_key='is_reference', reference_cat=['True']")
    print("  窗口大小: 100個基因, 步長: 10")
    try:
        # infercnvpy的主要函數
        # reference_cat應該是參考細胞的標記值（字符串）
        cnv.tl.infercnv(
            adata,
            reference_key="is_reference",
            reference_cat=["True"],  # 參考細胞的標記值
            window_size=100,  # 滑動窗口大小（基因數）
            step=10,  # 步長
            lfc_clip=3,  # log fold change裁剪值
            dynamic_threshold=1.5,  # 動態閾值
            exclude_chromosomes=('chrX', 'chrY'),  # 排除性染色體
            inplace=True,  # 將結果保存到adata
            key_added='cnv',  # 結果保存的鍵名
        )
        
        print("[OK] CNV推斷完成")
        print(f"  CNV矩陣已保存到: adata.obsm['X_cnv']")
        
        # 進行CNV聚類（可選）
        print("\n進行CNV聚類...")
        try:
            # 提取CNV矩陣並進行PCA
            if 'X_cnv' in adata.obsm:
                cnv_matrix = adata.obsm['X_cnv']
                # 創建臨時的AnnData對象用於PCA
                adata_cnv = sc.AnnData(cnv_matrix)
                sc.pp.pca(adata_cnv, n_comps=50)
                # 將PCA結果複製回原始adata
                adata.obsm['X_cnv_pca'] = adata_cnv.obsm['X_pca']
                
                # 構建鄰居圖
                sc.pp.neighbors(adata, use_rep='X_cnv_pca', n_neighbors=15, n_pcs=50)
                sc.tl.leiden(adata, key_added='cnv_leiden', resolution=0.5)
                print("[OK] CNV聚類完成")
            else:
                print("[WARNING] 未找到CNV矩陣，跳過聚類")
        except Exception as e:
            print(f"[WARNING] CNV聚類失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 計算CNV評分
        print("\n計算CNV評分...")
        try:
            cnv.tl.cnv_score(
                adata,
                groupby='cnv_leiden' if 'cnv_leiden' in adata.obs.columns else None,
                use_rep='cnv',
                key_added='cnv_score',
                inplace=True
            )
            print("[OK] CNV評分已計算")
        except Exception as e:
            print(f"[WARNING] CNV評分計算失敗: {e}")
            # 如果評分失敗，嘗試使用CNV矩陣的平均值作為評分
            if 'X_cnv' in adata.obsm:
                print("  使用CNV矩陣的平均值作為評分...")
                cnv_matrix = adata.obsm['X_cnv']
                if hasattr(cnv_matrix, 'toarray'):
                    cnv_matrix = cnv_matrix.toarray()
                adata.obs['cnv_score'] = np.mean(np.abs(cnv_matrix), axis=1)
                print("[OK] 使用CNV矩陣平均值作為評分")
        
        # 檢查結果
        cnv_cols = [col for col in adata.obs.columns if 'cnv' in col.lower()]
        print(f"\n可用CNV相關列: {cnv_cols}")
        
        return adata
        
    except Exception as e:
        import traceback
        print(f"[ERROR] CNV推斷失敗: {e}")
        print(f"詳細錯誤:")
        traceback.print_exc()
        return None

def classify_tumor_cells_epithelial_only(adata, cnv_threshold=None, epithelial_threshold=None, non_epithelial_threshold=None):
    """只針對上皮細胞進行CNV腫瘤分類的改進版本

    參數:
        adata: AnnData對象
        cnv_threshold: 通用CNV閾值（向後兼容）
        epithelial_threshold: 上皮細胞專用CNV閾值
        non_epithelial_threshold: 非上皮細胞CNV閾值（默認使用較寬鬆的閾值）
    """

    # 檢查是否有細胞類型信息
    if 'cell_type' not in adata.obs.columns:
        print("  [WARNING] 未找到細胞類型信息，使用傳統方法...")
        return classify_tumor_cells(adata, cnv_threshold, 'optimal')

    print("  [INFO] 使用上皮細胞專用CNV分類策略")
    print("    - 上皮細胞：應用嚴格CNV閾值")
    print("    - 非上皮細胞：應用寬鬆閾值或直接標記為正常")

    # 獲取CNV評分
    cnv_score_col = None
    for col in adata.obs.columns:
        if 'cnv' in col.lower() and 'score' in col.lower():
            cnv_score_col = col
            break

    if cnv_score_col is None:
        print("  [ERROR] 未找到CNV評分列")
        return adata

    cnv_scores = adata.obs[cnv_score_col]
    print(f"  使用CNV評分列: {cnv_score_col}")

    # 確定上皮細胞
    epithelial_mask = adata.obs['cell_type'].str.contains('Epithelial', case=False, na=False)
    epithelial_count = epithelial_mask.sum()
    non_epithelial_count = (~epithelial_mask).sum()

    print(f"  細胞類型統計:")
    print(f"    上皮細胞: {epithelial_count} ({epithelial_count/adata.n_obs*100:.1f}%)")
    print(f"    非上皮細胞: {non_epithelial_count} ({non_epithelial_count/adata.n_obs*100:.1f}%)")

    # 確定閾值 - 基於樣本組別的差異化策略
    if epithelial_threshold is None:
        # 如果沒有指定閾值，使用自動計算
        if 'group' in adata.obs.columns:
            print("  [INFO] 使用基於樣本組別的差異化閾值策略")

            # 分離正常和腫瘤樣本的上皮細胞
            normal_mask = adata.obs['group'] == 'Normal'
            tumor_mask = adata.obs['group'] == 'Tumor'

            normal_epithelial_scores = cnv_scores[epithelial_mask & normal_mask]
            tumor_epithelial_scores = cnv_scores[epithelial_mask & tumor_mask]

            if len(normal_epithelial_scores) > 10 and len(tumor_epithelial_scores) > 10:
                # 使用統計分佈方法：正常樣本均值 + 2倍標準差
                normal_mean = normal_epithelial_scores.mean()
                normal_std = normal_epithelial_scores.std()
                tumor_mean = tumor_epithelial_scores.mean()

                # 閾值設置為正常樣本的均值 + 2倍標準差（覆蓋95%的正常細胞）
                epithelial_threshold = normal_mean + 2 * normal_std

                print(f"  上皮細胞閾值: {epithelial_threshold:.4f} (正常均值+2倍標準差)")
                print(f"    正常樣本均值: {normal_mean:.4f}, 標準差: {normal_std:.4f}")
                print(f"    腫瘤樣本均值: {tumor_mean:.4f}")

                # 檢查閾值合理性
                normal_above = (normal_epithelial_scores > epithelial_threshold).mean()
                tumor_above = (tumor_epithelial_scores > epithelial_threshold).mean()
                print(f"    正常樣本高於閾值: {normal_above:.1%}")
                print(f"    腫瘤樣本高於閾值: {tumor_above:.1%}")

                # 如果正常樣本假陽性率過高，調整為更保守的閾值
                if normal_above > 0.05:  # 超過5%
                    epithelial_threshold = normal_mean + 2.5 * normal_std
                    print(f"  調整上皮細胞閾值: {epithelial_threshold:.4f} (正常均值+2.5倍標準差)")
            else:
                print("  [WARNING] 上皮細胞樣本數不足，使用全局分位數方法")
                all_epithelial_scores = cnv_scores[epithelial_mask]
                epithelial_threshold = np.percentile(all_epithelial_scores, 80)
                print(f"  上皮細胞閾值: {epithelial_threshold:.4f} (80%分位數)")
        else:
            # 沒有組別信息時的回退策略
            epithelial_scores = cnv_scores[epithelial_mask]
            if len(epithelial_scores) > 0:
                epithelial_threshold = np.percentile(epithelial_scores, 85)
                print(f"  上皮細胞閾值: {epithelial_threshold:.4f} (85%分位數)")
            else:
                epithelial_threshold = cnv_scores.median() + 2 * cnv_scores.std()
                print(f"  上皮細胞閾值: {epithelial_threshold:.4f} (默認值)")
    else:
        # 使用用戶指定的閾值
        print(f"  [INFO] 使用用戶指定的上皮細胞CNV閾值: {epithelial_threshold:.4f}")

    # 對於非上皮細胞，使用非常寬鬆的閾值或完全不分類為腫瘤
    if non_epithelial_threshold is None:
        non_epithelial_scores = cnv_scores[~epithelial_mask]
        if len(non_epithelial_scores) > 0:
            # 使用95%分位數作為非上皮細胞的閾值（非常保守）
            non_epithelial_threshold = np.percentile(non_epithelial_scores, 95)
            print(f"  非上皮細胞閾值: {non_epithelial_threshold:.4f} (95%分位數，非常保守)")
        else:
            non_epithelial_threshold = cnv_scores.median() + 4 * cnv_scores.std()
            print(f"  非上皮細胞閾值: {non_epithelial_threshold:.4f} (極其保守)")

    # 分類腫瘤細胞
    tumor_predictions = pd.Series('Normal', index=adata.obs.index)

    # 上皮細胞：使用嚴格閾值
    epithelial_tumor = (cnv_scores > epithelial_threshold) & epithelial_mask
    tumor_predictions[epithelial_tumor] = 'Tumor'
    epithelial_tumor_count = epithelial_tumor.sum()

    # 非上皮細胞：使用寬鬆閾值（如果指定），否則保持為正常
    if non_epithelial_threshold < float('inf'):  # 如果不是無限大
        non_epithelial_tumor = (cnv_scores > non_epithelial_threshold) & (~epithelial_mask)
        tumor_predictions[non_epithelial_tumor] = 'Tumor'
        non_epithelial_tumor_count = non_epithelial_tumor.sum()
    else:
        non_epithelial_tumor_count = 0

    adata.obs['is_tumor_cell'] = tumor_predictions

    # 統計結果
    tumor_count = (adata.obs['is_tumor_cell'] == 'Tumor').sum()
    normal_count = (adata.obs['is_tumor_cell'] == 'Normal').sum()

    print(f"\n  分類結果:")
    print(f"    總腫瘤細胞: {tumor_count} ({tumor_count/adata.n_obs*100:.1f}%)")
    print(f"    總正常細胞: {normal_count} ({normal_count/adata.n_obs*100:.1f}%)")
    print(f"    上皮細胞腫瘤: {epithelial_tumor_count} ({epithelial_tumor_count/epithelial_count*100:.1f}%)")
    print(f"    非上皮細胞腫瘤: {non_epithelial_tumor_count} ({non_epithelial_tumor_count/non_epithelial_count*100:.1f}%)")

    # 按組別統計
    if 'group' in adata.obs.columns:
        print(f"\n  按組別統計:")
        for group in adata.obs['group'].unique():
            group_mask = adata.obs['group'] == group
            group_tumor = (adata.obs.loc[group_mask, 'is_tumor_cell'] == 'Tumor').sum()
            group_total = group_mask.sum()
            print(f"    {group}: {group_tumor}/{group_total} 腫瘤細胞 ({group_tumor/group_total*100:.1f}%)")

    return adata

def classify_tumor_cells(adata, cnv_threshold=None, threshold_method='optimal'):
    """根據CNV評分分類腫瘤細胞

    參數:
        adata: AnnData對象
        cnv_threshold: CNV閾值，如果為None則自動確定
        threshold_method: 閾值確定方法
            - 'auto': 自動方法（中位數+2倍標準差）
            - 'percentile': 分位數方法（使用指定百分位數）
            - 'manual': 手動指定閾值
            - 'optimal': 基於樣本組別優化閾值
    """
    print("\n" + "=" * 60)
    print("分類腫瘤細胞...")
    print("=" * 60)

    # 查找CNV評分列
    cnv_score_col = None
    for col in adata.obs.columns:
        if 'cnv' in col.lower() and 'score' in col.lower():
            cnv_score_col = col
            break

    if cnv_score_col is None:
        print("[WARNING] 未找到CNV評分列，嘗試其他方法...")
        # 嘗試使用CNV聚類結果
        cnv_cluster_col = None
        for col in adata.obs.columns:
            if 'cnv' in col.lower() and 'leiden' in col.lower():
                cnv_cluster_col = col
                break

        if cnv_cluster_col:
            print(f"  使用CNV聚類結果: {cnv_cluster_col}")
            # 根據CNV聚類結果推斷（需要根據實際結果調整）
            adata.obs['is_tumor_cell'] = 'Unknown'
            print("  [INFO] 請手動檢查CNV聚類結果以確定腫瘤細胞")
            return adata
        else:
            print("[ERROR] 無法找到CNV相關結果")
            return adata

    cnv_scores = adata.obs[cnv_score_col]
    print(f"  使用CNV評分列: {cnv_score_col}")
    print(f"  CNV評分統計:")
    print(f"    平均: {cnv_scores.mean():.4f}")
    print(f"    中位數: {cnv_scores.median():.4f}")
    print(f"    標準差: {cnv_scores.std():.4f}")
    print(f"    範圍: [{cnv_scores.min():.4f}, {cnv_scores.max():.4f}]")

    # 確定閾值
    if cnv_threshold is None:
        if threshold_method == 'auto':
            # 自動方法：中位數+2倍標準差
            cnv_threshold = cnv_scores.median() + 2 * cnv_scores.std()
            print(f"  自動確定閾值: {cnv_threshold:.4f} (中位數+2×標準差)")

        elif threshold_method == 'percentile':
            # 分位數方法：使用90百分位數
            cnv_threshold = np.percentile(cnv_scores, 90)
            print(f"  分位數閾值: {cnv_threshold:.4f} (90th百分位數)")

        elif threshold_method == 'optimal':
            # 基於樣本組別優化的閾值
            cnv_threshold = optimize_cnv_threshold(adata, cnv_scores, cnv_score_col)
            print(f"  優化閾值: {cnv_threshold:.4f} (基於樣本組別)")

        else:
            # 默認使用自動方法
            cnv_threshold = cnv_scores.median() + 2 * cnv_scores.std()
            print(f"  默認閾值: {cnv_threshold:.4f} (中位數+2×標準差)")

    else:
        print(f"  使用指定閾值: {cnv_threshold}")

    # 分類腫瘤細胞
    adata.obs['is_tumor_cell'] = (cnv_scores > cnv_threshold).astype(str)
    adata.obs['is_tumor_cell'] = adata.obs['is_tumor_cell'].replace({'True': 'Tumor', 'False': 'Normal'})

    tumor_count = (adata.obs['is_tumor_cell'] == 'Tumor').sum()
    normal_count = (adata.obs['is_tumor_cell'] == 'Normal').sum()

    print(f"\n  分類結果:")
    print(f"    腫瘤細胞: {tumor_count} ({tumor_count/adata.n_obs*100:.1f}%)")
    print(f"    正常細胞: {normal_count} ({normal_count/adata.n_obs*100:.1f}%)")

    # 按組別統計
    if 'group' in adata.obs.columns:
        print(f"\n  按組別統計:")
        for group in adata.obs['group'].unique():
            group_mask = adata.obs['group'] == group
            group_tumor = (adata.obs.loc[group_mask, 'is_tumor_cell'] == 'Tumor').sum()
            group_total = group_mask.sum()
            print(f"    {group}: {group_tumor}/{group_total} 腫瘤細胞 ({group_tumor/group_total*100:.1f}%)")

    # 按細胞類型統計
    if 'cell_type' in adata.obs.columns:
        print(f"\n  按細胞類型統計（前10個）:")
        cell_type_stats = adata.obs.groupby('cell_type')['is_tumor_cell'].apply(
            lambda x: (x == 'Tumor').sum() / len(x) * 100
        ).sort_values(ascending=False)
        for cell_type, pct in cell_type_stats.head(10).items():
            count = (adata.obs['cell_type'] == cell_type).sum()
            tumor_count = ((adata.obs['cell_type'] == cell_type) &
                          (adata.obs['is_tumor_cell'] == 'Tumor')).sum()
            print(f"    {cell_type}: {tumor_count}/{count} ({pct:.1f}%)")

    return adata

def optimize_cnv_threshold(adata, cnv_scores, cnv_score_col, test_thresholds=None):
    """基於樣本組別優化CNV閾值

    策略：選擇能最好區分腫瘤和正常樣本的閾值
    優化目標：最大化 (腫瘤樣本準確率 + 正常樣本準確率)
    """

    # 添加更保守的閾值選項
    print("  [INFO] CNV閾值優化策略說明:")
    print("    目標: 最大化 (腫瘤樣本中腫瘤細胞比例 + 正常樣本中正常細胞比例)")
    print("    這是一個平衡精確率和召回率的策略")
    if 'group' not in adata.obs.columns:
        print("  [WARNING] 未找到group列，使用默認閾值")
        return cnv_scores.median() + 2 * cnv_scores.std()

    if test_thresholds is None:
        # 測試一系列閾值
        min_score = cnv_scores.min()
        max_score = cnv_scores.max()
        test_thresholds = np.linspace(min_score, max_score, 50)

    best_threshold = cnv_scores.median() + cnv_scores.std()
    best_score = 0

    print("  正在優化CNV閾值...")

    for threshold in test_thresholds:
        # 計算分類結果
        predictions = (cnv_scores > threshold)

        # 計算腫瘤樣本中腫瘤細胞的比例
        tumor_samples = adata.obs['group'] == 'Tumor'
        if tumor_samples.sum() > 0:
            tumor_sample_tumor_cells = predictions[tumor_samples].mean()
        else:
            tumor_sample_tumor_cells = 0

        # 計算正常樣本中正常細胞的比例
        normal_samples = adata.obs['group'] == 'Normal'
        if normal_samples.sum() > 0:
            normal_sample_normal_cells = (1 - predictions[normal_samples]).mean()
        else:
            normal_sample_normal_cells = 0

        # 綜合評分：腫瘤樣本中腫瘤細胞比例 + 正常樣本中正常細胞比例
        score = tumor_sample_tumor_cells + normal_sample_normal_cells

        # 添加更嚴格的約束條件：正常樣本中的"腫瘤細胞"比例不應超過15%
        false_positive_rate = 1 - normal_sample_normal_cells
        if false_positive_rate > 0.15:  # 如果假陽性率超過15%，降低評分
            penalty = (false_positive_rate - 0.15) * 2  # 懲罰係數
            score = max(0, score - penalty)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold

def visualize_cnv_results(adata, output_dir):
    """可視化CNV結果"""
    print("\n" + "=" * 60)
    print("可視化CNV結果...")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 設置scanpy的圖形保存目錄
    sc.settings.figdir = str(fig_dir)
    
    # 查找CNV評分列
    cnv_score_col = None
    for col in adata.obs.columns:
        if 'cnv' in col.lower() and 'score' in col.lower():
            cnv_score_col = col
            break
    
    if cnv_score_col is None:
        print("[WARNING] 未找到CNV評分列，跳過可視化")
        return
    
    # 確保有UMAP結果
    if 'X_umap' not in adata.obsm:
        print("[WARNING] 未找到UMAP結果，先進行UMAP降維...")
        sc.tl.umap(adata, min_dist=0.5, spread=1.0)
    
    # 1. UMAP上顯示CNV評分
    if 'X_umap' in adata.obsm:
        print("  繪製CNV評分UMAP圖...")
        try:
            sc.pl.umap(adata, color=cnv_score_col, 
                      save='_cnv_score.pdf',
                      show=False,
                      title='CNV Score')
            print("    [OK] CNV評分UMAP圖已保存")
        except Exception as e:
            print(f"    [ERROR] CNV評分UMAP圖保存失敗: {e}")
        
        # 按組別顯示
        if 'group' in adata.obs.columns:
            try:
                sc.pl.umap(adata, color=cnv_score_col, 
                          groups=['Normal', 'Tumor'],
                          save='_cnv_score_by_group.pdf',
                          show=False,
                          title='CNV Score by Group')
                print("    [OK] CNV評分按組別UMAP圖已保存")
            except Exception as e:
                print(f"    [ERROR] CNV評分按組別UMAP圖保存失敗: {e}")
        
        # 顯示腫瘤細胞分類
        if 'is_tumor_cell' in adata.obs.columns:
            try:
                sc.pl.umap(adata, color='is_tumor_cell',
                          save='_tumor_cell_classification.pdf',
                          show=False,
                          title='Tumor Cell Classification')
                print("    [OK] 腫瘤細胞分類UMAP圖已保存")
            except Exception as e:
                print(f"    [ERROR] 腫瘤細胞分類UMAP圖保存失敗: {e}")
    
    # 2. 按細胞類型顯示CNV評分
    if 'cell_type' in adata.obs.columns and cnv_score_col:
        print("  繪製CNV評分按細胞類型分組圖...")
        try:
            sc.pl.violin(adata, keys=cnv_score_col, groupby='cell_type',
                        save='_cnv_score_by_celltype.pdf',
                        show=False)
            print("    [OK] CNV評分按細胞類型小提琴圖已保存")
        except Exception as e:
            print(f"    [ERROR] CNV評分按細胞類型小提琴圖保存失敗: {e}")
    
    # 3. 按組別顯示CNV評分分布
    if 'group' in adata.obs.columns and cnv_score_col:
        print("  繪製CNV評分按組別分組圖...")
        try:
            sc.pl.violin(adata, keys=cnv_score_col, groupby='group',
                        save='_cnv_score_by_group_violin.pdf',
                        show=False)
            print("    [OK] CNV評分按組別小提琴圖已保存")
        except Exception as e:
            print(f"    [ERROR] CNV評分按組別小提琴圖保存失敗: {e}")
    
    # 4. 腫瘤患者的腫瘤細胞UMAP圖（按患者區分）
    if 'is_tumor_cell' in adata.obs.columns and 'group' in adata.obs.columns and 'sample' in adata.obs.columns:
        print("\n  繪製腫瘤患者的腫瘤細胞UMAP圖（按患者區分）...")
        try:
            # 篩選腫瘤患者的腫瘤細胞
            tumor_patient_mask = (
                (adata.obs['is_tumor_cell'] == 'Tumor') & 
                (adata.obs['group'] == 'Tumor')
            )
            tumor_patient_cells = tumor_patient_mask.sum()
            
            if tumor_patient_cells > 50:  # 至少需要50個細胞才能繪製UMAP
                adata_tumor_patients = adata[tumor_patient_mask].copy()
                
                # 為腫瘤細胞子集重新計算UMAP，以更好地展示內部亞群結構
                print("    為腫瘤細胞子集重新計算UMAP（展示內部亞群）...")
                
                # 確保有PCA結果，如果沒有則先計算PCA
                if 'X_pca' not in adata_tumor_patients.obsm:
                    print("      先計算PCA...")
                    if adata_tumor_patients.n_vars > 50:
                        sc.pp.pca(adata_tumor_patients, n_comps=50, svd_solver='arpack')
                    else:
                        sc.pp.pca(adata_tumor_patients, n_comps=min(adata_tumor_patients.n_vars-1, 30), svd_solver='arpack')
                
                # 構建鄰域圖
                print("      構建鄰域圖...")
                n_neighbors = min(15, tumor_patient_cells // 10)  # 根據細胞數調整鄰居數
                n_neighbors = max(5, n_neighbors)  # 至少5個鄰居
                sc.pp.neighbors(adata_tumor_patients, n_neighbors=n_neighbors, n_pcs=min(50, adata_tumor_patients.obsm['X_pca'].shape[1]))
                
                # 計算UMAP
                print("      計算UMAP...")
                sc.tl.umap(adata_tumor_patients, min_dist=0.3, spread=1.5)
                print("      [OK] UMAP計算完成")
                
                # 獲取腫瘤患者的樣本列表
                tumor_samples = sorted(adata_tumor_patients.obs['sample'].unique())
                print(f"    找到 {len(tumor_samples)} 個腫瘤患者樣本: {tumor_samples}")
                print(f"    腫瘤細胞總數: {tumor_patient_cells}")
                
                # 繪製UMAP圖，按sample著色
                sc.pl.umap(adata_tumor_patients, color='sample',
                          save='_tumor_cells_by_patient.pdf',
                          show=False,
                          title=f'Tumor Cells by Patient (n={tumor_patient_cells})',
                          legend_loc='right margin',
                          legend_fontsize=8)
                print("    [OK] 腫瘤患者的腫瘤細胞UMAP圖已保存")
                
                # 顯示每個患者的細胞數
                sample_counts = adata_tumor_patients.obs['sample'].value_counts()
                print(f"    各患者腫瘤細胞數:")
                for sample, count in sample_counts.items():
                    print(f"      {sample}: {count}")
            else:
                print(f"    [WARNING] 腫瘤患者的腫瘤細胞數量太少 ({tumor_patient_cells})，跳過")
        except Exception as e:
            print(f"    [ERROR] 腫瘤患者腫瘤細胞UMAP圖保存失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 正常個體的正常細胞UMAP圖（按個體區分）
    if 'is_tumor_cell' in adata.obs.columns and 'group' in adata.obs.columns and 'sample' in adata.obs.columns:
        print("\n  繪製正常個體的正常細胞UMAP圖（按個體區分）...")
        try:
            # 篩選正常個體的正常細胞
            normal_person_mask = (
                (adata.obs['is_tumor_cell'] == 'Normal') & 
                (adata.obs['group'] == 'Normal')
            )
            normal_person_cells = normal_person_mask.sum()
            
            if normal_person_cells > 50:  # 至少需要50個細胞才能繪製UMAP
                adata_normal_persons = adata[normal_person_mask].copy()
                
                # 為正常細胞子集重新計算UMAP，以更好地展示內部亞群結構
                print("    為正常細胞子集重新計算UMAP（展示內部亞群）...")
                
                # 確保有PCA結果，如果沒有則先計算PCA
                if 'X_pca' not in adata_normal_persons.obsm:
                    print("      先計算PCA...")
                    if adata_normal_persons.n_vars > 50:
                        sc.pp.pca(adata_normal_persons, n_comps=50, svd_solver='arpack')
                    else:
                        sc.pp.pca(adata_normal_persons, n_comps=min(adata_normal_persons.n_vars-1, 30), svd_solver='arpack')
                
                # 構建鄰域圖
                print("      構建鄰域圖...")
                n_neighbors = min(15, normal_person_cells // 10)  # 根據細胞數調整鄰居數
                n_neighbors = max(5, n_neighbors)  # 至少5個鄰居
                sc.pp.neighbors(adata_normal_persons, n_neighbors=n_neighbors, n_pcs=min(50, adata_normal_persons.obsm['X_pca'].shape[1]))
                
                # 計算UMAP
                print("      計算UMAP...")
                sc.tl.umap(adata_normal_persons, min_dist=0.3, spread=1.5)
                print("      [OK] UMAP計算完成")
                
                # 獲取正常個體的樣本列表
                normal_samples = sorted(adata_normal_persons.obs['sample'].unique())
                print(f"    找到 {len(normal_samples)} 個正常個體樣本: {normal_samples}")
                print(f"    正常細胞總數: {normal_person_cells}")
                
                # 繪製UMAP圖，按sample著色
                sc.pl.umap(adata_normal_persons, color='sample',
                          save='_normal_cells_by_person.pdf',
                          show=False,
                          title=f'Normal Cells by Person (n={normal_person_cells})',
                          legend_loc='right margin',
                          legend_fontsize=8)
                print("    [OK] 正常個體的正常細胞UMAP圖已保存")
                
                # 顯示每個個體的細胞數
                sample_counts = adata_normal_persons.obs['sample'].value_counts()
                print(f"    各正常個體細胞數:")
                for sample, count in sample_counts.items():
                    print(f"      {sample}: {count}")
            else:
                print(f"    [WARNING] 正常個體的正常細胞數量太少 ({normal_person_cells})，跳過")
        except Exception as e:
            print(f"    [ERROR] 正常個體正常細胞UMAP圖保存失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 檢查文件是否真的保存了
    saved_files = list(fig_dir.glob("*.pdf"))
    if saved_files:
        print(f"\n[OK] 成功保存 {len(saved_files)} 個圖形文件:")
        for f in saved_files[:10]:  # 只顯示前10個
            print(f"  - {f.name}")
        if len(saved_files) > 10:
            print(f"  ... 還有 {len(saved_files) - 10} 個文件")
    else:
        print("[WARNING] 未找到保存的圖形文件，請檢查路徑設置")
    
    print(f"[OK] 可視化結果已保存至: {fig_dir}")

def redraw_cnv_publication_umaps(adata, output_dir):
    """Redraw key CNV UMAPs using the same style as CellTypist annotation UMAPs."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    cnv_score_col = None
    for col in adata.obs.columns:
        if "cnv" in col.lower() and "score" in col.lower():
            cnv_score_col = col
            break
    if cnv_score_col is None or "X_umap" not in adata.obsm:
        print("  [WARNING] Missing CNV score or UMAP coordinates; skipping publication-style CNV UMAP redraw")
        return

    print("  Redrawing publication-style CNV UMAPs...")
    plot_embedding_continuous(
        adata,
        "umap",
        cnv_score_col,
        fig_dir / "umap_cnv_score.pdf",
        cmap="viridis",
        figsize=(4.8, 3.9),
        alpha=0.85,
    )
    if "is_tumor_cell" in adata.obs.columns:
        plot_embedding_categorical(
            adata,
            "umap",
            "is_tumor_cell",
            fig_dir / "umap_tumor_cell_classification.pdf",
            palette={"Normal": "#80B1D3", "Tumor": "#FB8072"},
            figsize=(4.8, 3.9),
            label_on_data=False,
        )
    print("  [OK] Publication-style CNV UMAPs saved")


def save_results(adata, output_dir):
    """保存CNV推斷結果"""
    print("\n" + "=" * 60)
    print("保存CNV推斷結果...")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 保存完整的AnnData對象
    output_file = output_dir / "adata_cnv_inferred.h5ad"
    
    # 在保存前，確保chromosome, start, end列的數據類型正確
    if 'chromosome' in adata.var.columns:
        adata.var['chromosome'] = adata.var['chromosome'].astype(str)
    if 'start' in adata.var.columns:
        # 轉換為可空整數類型
        adata.var['start'] = pd.to_numeric(adata.var['start'], errors='coerce').astype('Int64')
    if 'end' in adata.var.columns:
        adata.var['end'] = pd.to_numeric(adata.var['end'], errors='coerce').astype('Int64')
    
    try:
        adata.write(str(output_file))
        print(f"[OK] CNV推斷數據已保存至: {output_file}")
    except Exception as e:
        print(f"[WARNING] 保存h5ad文件失敗: {e}")
        print("嘗試移除有問題的列後重新保存...")
        # 移除有問題的列
        if 'start' in adata.var.columns:
            adata.var = adata.var.drop(columns=['start'])
        if 'end' in adata.var.columns:
            adata.var = adata.var.drop(columns=['end'])
        adata.write(str(output_file))
        print(f"[OK] CNV推斷數據已保存（已移除位置信息列）: {output_file}")
    
    # 2. 保存CNV評分和分類結果
    cnv_cols = [col for col in adata.obs.columns if 'cnv' in col.lower() or 'tumor' in col.lower()]
    if cnv_cols:
        cnv_df = adata.obs[cnv_cols].copy()
        cnv_file = output_dir / "cnv_results.csv"
        cnv_df.to_csv(cnv_file)
        print(f"[OK] CNV結果已保存至: {cnv_file}")
    
    # 3. 保存統計信息
    stats_file = output_dir / "cnv_inference_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("CNV推斷統計\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"細胞數: {adata.n_obs}\n")
        f.write(f"基因數: {adata.n_vars}\n\n")
        
        if 'is_tumor_cell' in adata.obs.columns:
            f.write("腫瘤細胞分類結果:\n")
            tumor_counts = adata.obs['is_tumor_cell'].value_counts()
            for cell_type, count in tumor_counts.items():
                f.write(f"  {cell_type}: {count} ({count/adata.n_obs*100:.1f}%)\n")
        
        if 'group' in adata.obs.columns and 'is_tumor_cell' in adata.obs.columns:
            f.write("\n按組別統計:\n")
            for group in adata.obs['group'].unique():
                group_mask = adata.obs['group'] == group
                group_tumor = (adata.obs.loc[group_mask, 'is_tumor_cell'] == 'Tumor').sum()
                group_total = group_mask.sum()
                f.write(f"  {group}: {group_tumor}/{group_total} 腫瘤細胞 ({group_tumor/group_total*100:.1f}%)\n")
        
        if 'cell_type' in adata.obs.columns and 'is_tumor_cell' in adata.obs.columns:
            f.write("\n按細胞類型統計:\n")
            cell_type_stats = adata.obs.groupby('cell_type')['is_tumor_cell'].apply(
                lambda x: f"{(x == 'Tumor').sum()}/{len(x)} ({(x == 'Tumor').sum()/len(x)*100:.1f}%)"
            )
            for cell_type, stat in cell_type_stats.items():
                f.write(f"  {cell_type}: {stat}\n")
    
    print(f"[OK] 統計信息已保存至: {stats_file}")

def main():
    """主函數"""
    # GSE217517數據集說明
    print("\n" + "=" * 60)
    print("GSE217517 CNV推斷分析")
    print("數據集: 人類卵巢癌單細胞RNA-seq數據")
    print("=" * 60)

    # 設置路徑
    base_dir = Path(__file__).parent
    input_file = base_dir / "04_annotated" / "adata_annotated.h5ad"
    output_dir = base_dir / "04b_cnv_inferred"
    output_dir.mkdir(exist_ok=True)
    
    # 設置圖形保存目錄（在開始時設置，確保所有圖形都保存到正確位置）
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    sc.settings.figdir = str(fig_dir)
    
    # 1. 加載數據
    adata = load_annotated_data(str(input_file))
    if adata is None:
        return
    
    # 2. 準備參考細胞
    reference_cells = prepare_reference_cells(adata)
    
    # 3. CNV推斷
    adata = infer_cnv_infercnvpy(adata, reference_cells)
    if adata is None:
        print("\n[ERROR] CNV推斷失敗")
        return
    
    # 4. 分類腫瘤細胞
    # 使用上皮細胞專用CNV分類策略（針對卵巢癌優化）
    # - 只對上皮細胞應用嚴格CNV閾值
    # - 對非上皮細胞使用寬鬆閾值或直接標記為正常
    # 這樣可以減少假陽性，特別是免疫細胞的干擾
    print("\n[INFO] 使用上皮細胞專用CNV分類策略")
    # 指定CNV閾值為0.04（基於雙峰分佈）
    adata = classify_tumor_cells_epithelial_only(adata, epithelial_threshold=0.04)
    
    # 5. 可視化
    visualize_cnv_results(adata, output_dir)
    redraw_cnv_publication_umaps(adata, output_dir)
    
    # 6. 保存結果
    save_results(adata, output_dir)
    
    print("\n" + "=" * 60)
    print("CNV推斷完成！")
    print("=" * 60)
    print(f"\n結果文件保存在: {output_dir}")
    print("\nCNV推斷結果已保存，可以繼續進行通路分析\n")
    print("\n[注意] 請檢查is_tumor_cell分類結果，必要時手動調整閾值\n")

if __name__ == "__main__":
    main()
