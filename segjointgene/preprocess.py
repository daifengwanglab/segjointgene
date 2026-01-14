# Set number of threads to use
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

from tqdm import tqdm  # <--- 必须导入这个
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter

import cv2  # OpenCV 用于高效的图像缩放
from skimage.segmentation import find_boundaries

def scale_coordinates(row, row_name, min, max, coodinate_size=1000, scale_ratio=1, floor_float=False):
    assert row_name == 'x' or row_name == 'y'
    row_value = row[row_name]
    assert row_value >= min and row_value <= max
    new_row_value = coodinate_size * (row_value - min)/(max - min)
    new_row_value = new_row_value * scale_ratio
    if floor_float:
        new_row_value = math.floor(new_row_value)
    return new_row_value


def quality_control_check(mask, csv_df, dapi_img, crop_coords=None):
    offset_x, offset_y = crop_coords if crop_coords else (0, 0)

    # 随机抽查 1000 个细胞，或者全部检查
    check_sample = csv_df.sample(n=min(len(csv_df), 1000))
    correct_match = 0
    lost_cells = 0
    mismatch = 0

    for _, row in check_sample.iterrows():
        # 转换到 mask 坐标系
        mx = int(row['x'] - offset_x)
        my = int(row['y'] - offset_y)
        true_id = int(row['id'])
        if 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]:
            mask_val = mask[my, mx]
            if mask_val == -1 or mask_val == 0:
                lost_cells += 1
            elif mask_val == true_id:
                correct_match += 1
            else:
                mismatch += 1

    total_checked = len(check_sample)
    print(f"抽查样本数: {total_checked}")
    print(f"✅ 完美匹配: {correct_match} ({correct_match / total_checked * 100:.1f}%)")
    print(f"⚠️ 丢失细胞 (因DAPI太暗): {lost_cells} ({lost_cells / total_checked * 100:.1f}%)")
    print(f"❌ ID错位: {mismatch} ({mismatch / total_checked * 100:.1f}%)")

    if (lost_cells / total_checked) > 0.1:
        print("警告：超过 10% 的细胞丢失，建议调低 threshold_otsu 的阈值或改用 local_threshold！")
    print("正在生成可视化抽查图...")
    valid_center = csv_df.iloc[len(csv_df) // 2]
    cx, cy = int(valid_center['x'] - offset_x), int(valid_center['y'] - offset_y)

    viz_size = 128
    x1 = max(0, cx - viz_size // 2)
    x2 = min(mask.shape[1], cx + viz_size // 2)
    y1 = max(0, cy - viz_size // 2)
    y2 = min(mask.shape[0], cy + viz_size // 2)

    # 切片
    sub_mask = mask[y1:y2, x1:x2]
    sub_dapi = dapi_img[y1 + int(offset_y):y2 + int(offset_y), x1 + int(offset_x):x2 + int(offset_x)]

    # 准备 Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1: DAPI 原图 + CSV 坐标点
    axes[0].imshow(sub_dapi, cmap='gray')
    axes[0].set_title("Raw DAPI + CSV Centroids")

    # 筛选出视野内的点
    local_spots = csv_df[
        (csv_df['x'] - offset_x >= x1) & (csv_df['x'] - offset_x < x2) &
        (csv_df['y'] - offset_y >= y1) & (csv_df['y'] - offset_y < y2)
        ]
    axes[0].scatter(local_spots['x'] - offset_x - x1, local_spots['y'] - offset_y - y1, c='r', s=10, alpha=0.6, label='CSV Seed')
    colored_mask = np.zeros((*sub_mask.shape, 3))
    unique_ids = np.unique(sub_mask)
    for uid in unique_ids:
        if uid <= 0: continue  # 背景黑
        # 简单的哈希颜色生成
        np.random.seed(uid)
        col = np.random.rand(3)
        colored_mask[sub_mask == uid] = col

    axes[1].imshow(colored_mask)
    axes[1].set_title("Generated Watershed Mask")

    # 图3: 叠加对比 (DAPI + 边界 + 点)
    axes[2].imshow(sub_dapi, cmap='gray')
    # 提取边界
    boundaries = find_boundaries(sub_mask, mode='thick')
    # 创建一个透明红色的边界层
    boundary_overlay = np.zeros((*sub_mask.shape, 4))
    boundary_overlay[boundaries] = [1, 1, 0, 1]  # 黄色边界

    axes[2].imshow(boundary_overlay)
    axes[2].scatter(local_spots['x'] - offset_x - x1, local_spots['y'] - offset_y - y1, c='r', s=5, label='Center')
    axes[2].set_title("Overlay: DAPI(Gray) + Mask(Yellow) + Center(Red)")

    plt.tight_layout()
    plt.savefig("check_segmentation_quality.png")
    print("可视化结果已保存为 'check_segmentation_quality.png'，请查看！")
    plt.close()


# -----------------------------------------------------------------------------
# 1. 单个 Patch 处理函数 (_worker)
# -----------------------------------------------------------------------------
def _worker(r, c, patch_size, spots, global_mask, global_dapi, lookup_array,
            npz_dir, image_dir, viz_targets, n_genes, gene_to_idx, sigma,
            dir_seg, dir_raw, dir_dapi, dir_inst):
    # 1. 坐标准备
    y1, y2 = r * patch_size, (r + 1) * patch_size
    x1, x2 = c * patch_size, (c + 1) * patch_size

    # Padding 设置
    padding = int(sigma * 3)

    # ============================================================
    # 2. 现场分割 & DAPI 切片
    # ============================================================
    patch_cell_ids = global_mask[y1:y2, x1:x2]
    dapi_patch = global_dapi[y1:y2, x1:x2]

    seg_patch = np.zeros_like(patch_cell_ids, dtype=np.int32)  # 初始化为0 (Background)
    valid_pixels = patch_cell_ids > 0  # 只有大于0的才是细胞

    if np.any(valid_pixels):
        seg_patch[valid_pixels] = lookup_array[patch_cell_ids[valid_pixels] - 1]

    # ============================================================
    # 3. Density 计算 (保持不变)
    # ============================================================
    mask = ((spots['x'] >= x1 - padding) & (spots['x'] < x2 + padding) &
            (spots['y'] >= y1 - padding) & (spots['y'] < y2 + padding))
    local_spots = spots[mask]

    # --- A. Raw Spots ---
    raw_spots, _, _ = np.histogram2d(
        local_spots['y'].values - y1, local_spots['x'].values - x1,
        bins=[patch_size, patch_size], range=[[0, patch_size], [0, patch_size]])

    # --- B. Gene Density ---
    density = np.zeros((n_genes, patch_size, patch_size), dtype=np.float32)

    if not local_spots.empty:
        padded_size = patch_size + 2 * padding
        grouped = local_spots.groupby('gene')
        for gene, group in grouped:
            if gene in gene_to_idx:
                g_y = group['y'].values - (y1 - padding)
                g_x = group['x'].values - (x1 - padding)

                H_padded, _, _ = np.histogram2d(
                    g_y, g_x,
                    bins=[padded_size, padded_size],
                    range=[[0, padded_size], [0, padded_size]]
                )

                if sigma > 0:
                    H_blurred = gaussian_filter(H_padded, sigma=sigma, mode='constant')
                    density[gene_to_idx[gene]] = H_blurred[padding:padding + patch_size, padding:padding + patch_size]
                else:
                    density[gene_to_idx[gene]] = H_padded[padding:padding + patch_size, padding:padding + patch_size]

    # 4. 保存
    name = f"p_{r}_{c}"

    # 保存 NPZ (包含新增的 instance 字段)
    dapi_to_save = dapi_patch.astype(np.int32)
    np.savez_compressed(os.path.join(npz_dir, name + ".npz"),
                        image=density,
                        label=seg_patch,
                        instance=patch_cell_ids,
                        spots=raw_spots,
                        dapi=dapi_to_save)

    # --- 改进：Instance 可视化颜色打散逻辑 ---
    inst_viz = np.zeros_like(patch_cell_ids, dtype=np.float32)
    unique_ids = np.unique(patch_cell_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    if len(unique_ids) > 0:
        rs = np.random.RandomState(42)
        colors = rs.rand(len(unique_ids))
        id_to_color = dict(zip(unique_ids, colors))
        for uid, color in id_to_color.items():
            inst_viz[patch_cell_ids == uid] = color

    # 保存类别图
    plt.imsave(os.path.join(dir_seg, name + ".png"), seg_patch, cmap='nipy_spectral', origin='upper')
    plt.imsave(os.path.join(dir_inst, name + ".png"), inst_viz, cmap='nipy_spectral', origin='upper')
    plt.imsave(os.path.join(dir_raw, name + ".png"), raw_spots, cmap='inferno', origin='upper')
    plt.imsave(os.path.join(dir_dapi, name + ".png"), dapi_patch, cmap='gray', origin='upper')

    for g_name, idx in viz_targets:
        plt.imsave(os.path.join(image_dir, str(g_name), name + ".png"), density[idx], cmap='viridis', origin='upper')

# -----------------------------------------------------------------------------
# 2. 批量处理与划分函数 (process_and_save_patches_optimized)
# -----------------------------------------------------------------------------
def process_and_save_patches_optimized(spots, global_mask, global_dapi, celltype_lookup,
                                       npz_dir, image_dir, total_size, patch_size=128, sigma=2.0, split_ratio=0.8):
    # 1. 准备工作
    print(f"正在准备任务列表 (16核并行)...")
    all_genes = np.sort(spots['gene'].unique())
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n_genes = len(all_genes)

    viz_genes = all_genes[:max(1, int(n_genes * 0.1))]
    viz_targets = [(g, gene_to_idx[g]) for g in viz_genes]

    # 2. 建立物理子目录
    train_dir = os.path.join(npz_dir, "train")
    test_dir = os.path.join(npz_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # [核心修改] 保存 metadata，增加 celltype_lookup 用于后续 Instance->Type 的转换
    np.savez(os.path.join(npz_dir, "metadata.npz"),
             gene_names=all_genes,
             celltype_lookup=celltype_lookup)

    # 建立可视化图像目录
    dir_seg = os.path.join(image_dir, "cell_type")
    dir_raw = os.path.join(image_dir, "mRNA_spots")
    dir_dapi = os.path.join(image_dir, "DAPI")
    dir_inst = os.path.join(image_dir, "instance")
    for d in [dir_seg, dir_raw, dir_dapi, dir_inst]: os.makedirs(d, exist_ok=True)
    for g in viz_genes: os.makedirs(os.path.join(image_dir, str(g)), exist_ok=True)

    # 3. 计算切片网格
    Real_H, Real_W = global_mask.shape
    n_rows = Real_H // patch_size
    n_cols = Real_W // patch_size

    print(f"  -> 输入图像尺寸: {Real_W}x{Real_H} (WxH)")
    print(f"  -> 切片网格: {n_rows} 行 x {n_cols} 列，总计 {n_rows * n_cols} 个 Patch")

    # 4. 物理划分坐标
    all_coords = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    random.seed(42)
    random.shuffle(all_coords)

    num_train = int(len(all_coords) * split_ratio)
    train_coords_set = set(all_coords[:num_train])

    print(f"  -> 物理划分完成: Train {num_train} 个, Test {len(all_coords) - num_train} 个")

    # 5. 构建任务列表
    tasks = []
    for (r, c) in all_coords:
        target_save_path = train_dir if (r, c) in train_coords_set else test_dir

        tasks.append(delayed(_worker)(
            r, c, patch_size, spots, global_mask, global_dapi, np.asarray(celltype_lookup),
            target_save_path,
            image_dir, viz_targets, n_genes, gene_to_idx, sigma,
            dir_seg, dir_raw, dir_dapi, dir_inst
        ))

    # 6. 执行并行任务
    for _ in tqdm(Parallel(n_jobs=16, return_as='generator')(tasks), total=len(tasks), desc="Processing Patches"):
        pass

    print(f"===== 数据集已成功导出至: {train_dir} 和 {test_dir} =====")