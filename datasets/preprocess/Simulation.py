"""Preprocess pipeline for Simulation spatial transcriptomics dataset.

Reads raw data from ``{data_dir}/`` (``image_*.png``, ``label_*.png``,
``spots_*.csv``), writes NPZ patches to ``{data_dir}/Simulation/{train,val,test}/``
and visualization images to ``{data_dir}/Simulation_image/``.

Four conditions are treated as one unified dataset with group indices:
  (0,0) LowNoise_DenseCells
  (0,1) LowNoise_SparseCells
  (1,0) HighNoise_DenseCells
  (1,1) HighNoise_SparseCells

**Patch split:** all grid slots ``(group_row, group_col, r, c)`` across the four
conditions are pooled, shuffled with ``Simulation_split_seed``, then assigned to
**train / val / test** with approximate ratio **7:1:2** (integer partition of ``N``).
See ``_build_random_patch_split_map`` and Phase 2 in ``step_preprocess_Simulation``.

``global_scale`` (default 1.0) scales spot ``(x,y)`` and label/DAPI rasters together, like WMB/CA1.

Multislice: Simulation does **not** call ``visum_multi_slice``; density voxels reuse the same
histogram + Gaussian recipe as that module (comment in ``_simulation_worker``).

Celltypes are derived via KMeans clustering on per-cell gene expression.

Requires optional dependencies: ``scikit-learn``, ``scikit-image``.
"""

from __future__ import annotations

import math
import os
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from dlbase.training.seed import apply_lightning_reproducibility

CONDITIONS = [
    "LowNoise_DenseCells",
    "LowNoise_SparseCells",
    "HighNoise_DenseCells",
    "HighNoise_SparseCells",
]

CONDITION_GRID = {
    "LowNoise_DenseCells":  (0, 0),
    "LowNoise_SparseCells": (0, 1),
    "HighNoise_DenseCells":  (1, 0),
    "HighNoise_SparseCells": (1, 1),
}


def _apply_global_scale_spots_label(
    spots_df: pd.DataFrame,
    label_img: np.ndarray,
    global_scale: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Scale spot coordinates and label raster together (same contract as ``visum`` global scale)."""
    gs = float(global_scale)
    if abs(gs - 1.0) < 1e-12:
        return spots_df, label_img
    h, w = int(label_img.shape[0]), int(label_img.shape[1])
    new_w, new_h = int(w * gs), int(h * gs)
    out = spots_df.copy()
    out["x"] = out["x"].to_numpy(dtype=np.float64) * gs
    out["y"] = out["y"].to_numpy(dtype=np.float64) * gs
    lab = cv2.resize(
        label_img.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return out, lab.astype(np.int32)


def _resize_raster_global_scale(img: np.ndarray, global_scale: float, nearest: bool) -> np.ndarray:
    """Resize a 2D image after global scale (DAPI uses linear; labels use nearest above)."""
    gs = float(global_scale)
    if abs(gs - 1.0) < 1e-12:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    new_w, new_h = int(w * gs), int(h * gs)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=interp).astype(np.float32)


def dapi_closed_foreground(
    dapi: np.ndarray,
    gaussian_sigma: float = 1.5,
    close_radius: int = 9,
    min_hole_area: int = 256,
) -> np.ndarray:
    """Binary foreground mask: tissue vs black background, with closed boundaries.

    Uses Gaussian smoothing + Otsu on non-black pixels, then morphological closing
    and small-hole filling so foreground forms contiguous closed regions (not a
    single intensity cutoff like ``> 0``).
    """
    try:
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import closing, disk, remove_small_holes
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise ImportError(
            "Simulation preprocess requires ``scikit-image`` and ``scipy``."
        ) from exc

    img = np.asarray(dapi, dtype=np.float64)
    if img.max() <= 0:
        return np.zeros_like(img, dtype=bool)

    smooth = gaussian(img, sigma=gaussian_sigma, preserve_range=True)
    # Otsu on bright pixels only so the black border does not dominate the histogram
    lo = float(np.percentile(smooth, 2.0))
    hi = float(smooth.max())
    valid = smooth > lo + 1e-6
    if valid.sum() > 5000:
        t = threshold_otsu(smooth[valid].astype(np.float64))
    else:
        t = threshold_otsu(smooth.astype(np.float64))

    fg = smooth > t
    r = max(1, int(close_radius))
    fg = closing(fg, footprint=disk(r))
    _mha = int(min_hole_area)
    try:
        fg = remove_small_holes(fg, max_size=_mha)
    except TypeError:
        fg = remove_small_holes(fg, area_threshold=_mha)
    fg = ndi.binary_fill_holes(fg)
    return fg.astype(bool)


def cluster_cells_balanced(
    log_matrix: np.ndarray,
    min_large_clusters: int = 5,
    min_fraction: float = 0.10,
    k_min: int = 10,
    k_max: int = 28,
    random_state: int = 42,
) -> tuple[np.ndarray, int]:
    """KMeans with search so at least *min_large_clusters* clusters have
    ``count >= min_fraction * n_cells``. Falls back to quantile split on PC1
    if KMeans cannot satisfy the constraint.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n = log_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), 0

    # Floor so equal-mass bins (e.g. 10% of 4822 -> 482 cells/bin) still qualify
    min_count = max(1, int(math.floor(min_fraction * n)))

    def _count_large(lab: np.ndarray, k: int) -> int:
        cnt = np.bincount(lab, minlength=k)
        return int(np.sum(cnt >= min_count))

    k_hi = min(k_max, n)
    k_lo = max(k_min, min_large_clusters, 2)

    chosen_lab: np.ndarray | None = None
    chosen_k = 0
    best_fallback: tuple[int, np.ndarray] | None = None

    for k_try in range(k_lo, k_hi + 1):
        km = KMeans(n_clusters=k_try, random_state=random_state, n_init=15, max_iter=300)
        lab = km.fit_predict(log_matrix)
        n_large = _count_large(lab, k_try)
        if best_fallback is None or n_large > best_fallback[0]:
            best_fallback = (n_large, lab.copy())
        if n_large >= min_large_clusters:
            chosen_lab = lab
            chosen_k = k_try
            print(
                f"  KMeans: k={k_try} -> {n_large} clusters >= {min_fraction:.0%} "
                f"(min count {min_count})"
            )
            break

    if chosen_lab is None and best_fallback is not None:
        chosen_lab = best_fallback[1]
        chosen_k = int(chosen_lab.max()) + 1
        print(
            f"  KMeans: no k reached {min_large_clusters} large clusters; "
            f"using best attempt ({best_fallback[0]} large) at k≈{chosen_k}"
        )

    if chosen_lab is None:
        chosen_k = min(max(8, k_lo), n)
        km = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=15)
        chosen_lab = km.fit_predict(log_matrix)

    lab = chosen_lab.astype(np.int64).copy()
    k_cur = int(lab.max()) + 1

    # Merge clusters smaller than min_count into nearest centroid neighbor
    for _ in range(300):
        cnt = np.bincount(lab, minlength=k_cur)
        small_idx = np.where((cnt > 0) & (cnt < min_count))[0]
        if len(small_idx) == 0:
            break
        s = int(small_idx[np.argmin(cnt[small_idx])])
        centers = np.stack(
            [log_matrix[lab == j].mean(axis=0) if np.any(lab == j) else log_matrix.mean(axis=0) for j in range(k_cur)]
        )
        dists = np.linalg.norm(centers - centers[s], axis=1)
        dists[s] = np.inf
        t = int(np.argmin(dists))
        lab[lab == s] = t
        uniq = np.unique(lab)
        remap = {int(u): i for i, u in enumerate(uniq)}
        lab = np.vectorize(remap.get)(lab)
        k_cur = int(lab.max()) + 1

    n_large_after = _count_large(lab, k_cur)
    if n_large_after < min_large_clusters:
        print(
            f"  [warn] After merge only {n_large_after} clusters >= {min_fraction:.0%}; "
            "using PC1 equal-mass bins (each ~1/B of cells)."
        )
        pc1 = PCA(n_components=1, random_state=random_state).fit_transform(log_matrix).ravel()
        # ~equal cells per bin so each bin is typically >= min_fraction
        n_bins = max(min_large_clusters + 2, 10)
        order = np.argsort(pc1)
        lab = np.empty(n, dtype=np.int64)
        for b in range(n_bins):
            i0 = b * n // n_bins
            i1 = (b + 1) * n // n_bins if b < n_bins - 1 else n
            lab[order[i0:i1]] = b
        k_cur = n_bins

    _, lab_final = np.unique(lab, return_inverse=True)
    lab_final = lab_final.astype(np.int64)
    k_final = int(lab_final.max()) + 1

    cnt_final = np.bincount(lab_final, minlength=k_final)
    fracs = cnt_final / max(n, 1)
    print(f"  Final cluster count: {k_final}, fraction per cluster: " + ", ".join(f"{fr:.1%}" for fr in fracs))
    # Use count threshold, not raw fraction >= 0.1 (482/4822 is slightly below 10%)
    n_ge = int(np.sum(cnt_final >= min_count))
    print(f"  Clusters with count >= {min_fraction:.0%} of cells (min_count={min_count}): {n_ge}")

    return lab_final, k_final


def _simulation_worker(
    group_row, group_col, r, c, patch_size,
    spots, training_mask, full_mask, global_dapi,
    lookup_array, gt_lookup_array,
    target_save_path, image_dir, viz_targets,
    global_gene_list, global_gene_to_idx,
    sigma, dir_seg, dir_raw, dir_dapi, dir_inst, dir_gt,
):
    """Patch worker for Simulation with separate training and ground-truth masks.

    ``training_mask`` = label intersected with DAPI foreground (for ``label`` /
    ``instance``).  ``full_mask`` = complete label image (for ``ground_truth``).
    """
    y1, y2 = r * patch_size, (r + 1) * patch_size
    x1, x2 = c * patch_size, (c + 1) * patch_size
    padding = int(sigma * 3)

    train_ids = training_mask[y1:y2, x1:x2]
    full_ids = full_mask[y1:y2, x1:x2]
    dapi_patch = global_dapi[y1:y2, x1:x2]

    # Skip uniform training-mask patches (e.g. all background or single constant label)
    if train_ids.min() == train_ids.max():
        return

    # Training label (celltype from DAPI-foreground intersection only)
    seg_patch = np.zeros_like(train_ids, dtype=np.int32)
    valid_train = train_ids > 0
    if np.any(valid_train):
        seg_patch[valid_train] = lookup_array[train_ids[valid_train] - 1]

    # Ground truth (celltype from the FULL label mask)
    gt_patch = np.zeros_like(full_ids, dtype=np.int32)
    valid_full = full_ids > 0
    if np.any(valid_full):
        gt_patch[valid_full] = gt_lookup_array[full_ids[valid_full] - 1]
    os.makedirs(dir_gt, exist_ok=True)

    # Density calculation (same as visum_multi_slice._worker)
    mask = (
        (spots['x'] >= x1 - padding) & (spots['x'] < x2 + padding) &
        (spots['y'] >= y1 - padding) & (spots['y'] < y2 + padding)
    )
    local_spots = spots[mask]

    weights_raw = local_spots['expression'].values if 'expression' in local_spots.columns else None
    raw_spots, _, _ = np.histogram2d(
        local_spots['y'].values - y1, local_spots['x'].values - x1,
        bins=[patch_size, patch_size], range=[[0, patch_size], [0, patch_size]],
        weights=weights_raw,
    )

    n_global_genes = len(global_gene_list)
    density = np.zeros((n_global_genes, patch_size, patch_size), dtype=np.float32)

    if not local_spots.empty:
        padded_size = patch_size + 2 * padding
        grouped = local_spots.groupby('gene')
        offset_y = y1 - padding
        offset_x = x1 - padding

        for gene, group in grouped:
            if gene in global_gene_to_idx:
                g_y = group['y'].values - offset_y
                g_x = group['x'].values - offset_x
                g_weights = group['expression'].values if 'expression' in group.columns else None

                H_padded, _, _ = np.histogram2d(
                    g_y, g_x,
                    bins=[padded_size, padded_size],
                    range=[[0, padded_size], [0, padded_size]],
                    weights=g_weights,
                )

                idx = global_gene_to_idx[gene]
                if sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    H_blurred = gaussian_filter(H_padded, sigma=sigma, mode='constant')
                    density[idx] = H_blurred[padding:padding + patch_size, padding:padding + patch_size]
                else:
                    density[idx] = H_padded[padding:padding + patch_size, padding:padding + patch_size]

    # Save NPZ
    name = f"p_{group_row}_{group_col}_{r}_{c}"
    np.savez_compressed(
        os.path.join(target_save_path, name + ".npz"),
        image=density,
        label=seg_patch,
        instance=train_ids,
        spots=raw_spots,
        dapi=dapi_patch.astype(np.int32),
        ground_truth=gt_patch,
        # Full-label instance ids (same space as ``full_mask``); used for Gene mIoU vs pred instance.
        ground_truth_instance=full_ids.astype(np.int32),
    )

    # Visualization
    inst_viz = np.zeros_like(train_ids, dtype=np.float32)
    unique_ids = np.unique(train_ids)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) > 0:
        rs = np.random.RandomState(42)
        colors = rs.rand(len(unique_ids))
        for uid, color in zip(unique_ids, colors):
            inst_viz[train_ids == uid] = color

    plt.imsave(os.path.join(dir_seg, name + ".png"), seg_patch, cmap='nipy_spectral', origin='upper')
    plt.imsave(os.path.join(dir_inst, name + ".png"), inst_viz, cmap='nipy_spectral', origin='upper')
    plt.imsave(os.path.join(dir_dapi, name + ".png"), dapi_patch, cmap='gray', origin='upper')
    plt.imsave(os.path.join(dir_gt, name + ".png"), gt_patch, cmap='nipy_spectral', origin='upper')

    vmax_raw = raw_spots.max()
    vmax_raw = float(vmax_raw) if vmax_raw > 0 else 1.0
    plt.imsave(os.path.join(dir_raw, name + ".png"), raw_spots, cmap='inferno', origin='upper', vmin=0, vmax=vmax_raw)

    for g_name, idx in viz_targets:
        gene_dir = os.path.join(image_dir, str(g_name))
        os.makedirs(gene_dir, exist_ok=True)
        vmax_den = density[idx].max()
        vmax_den = float(vmax_den) if vmax_den > 0 else 1.0
        plt.imsave(os.path.join(gene_dir, name + ".png"), density[idx], cmap='viridis', origin='upper', vmin=0, vmax=vmax_den)


def _assign_spots_to_cells(spots_df: pd.DataFrame, label_img: np.ndarray) -> pd.DataFrame:
    """Add a ``cell_id`` column by looking up the instance label at each spot's pixel."""
    h, w = label_img.shape
    xs = spots_df['x'].values.astype(int).clip(0, w - 1)
    ys = spots_df['y'].values.astype(int).clip(0, h - 1)
    spots_df = spots_df.copy()
    spots_df['cell_id'] = label_img[ys, xs].astype(int)
    return spots_df


def _build_cell_gene_matrix(spots_df: pd.DataFrame, global_gene_to_idx: dict, n_genes: int):
    """Build (n_cells, n_genes) count matrix from spots assigned to cells.

    Returns the count matrix and a mapping from matrix-row index to cell instance ID.
    Background spots (cell_id == 0) are excluded.
    """
    fg = spots_df[spots_df['cell_id'] > 0]
    if fg.empty:
        return np.zeros((0, n_genes), dtype=np.float32), np.array([], dtype=int)

    cell_ids = fg['cell_id'].values
    gene_names = fg['gene'].values

    unique_cells = np.unique(cell_ids)
    cell_id_to_row = {cid: i for i, cid in enumerate(unique_cells)}

    mat = np.zeros((len(unique_cells), n_genes), dtype=np.float32)
    for cid, gene in zip(cell_ids, gene_names):
        gidx = global_gene_to_idx.get(gene)
        if gidx is not None:
            mat[cell_id_to_row[cid], gidx] += 1.0

    return mat, unique_cells


def _build_random_patch_split_map(
    all_slots: list[tuple[int, int, int, int]],
    *,
    split_seed: int,
    train_tenths: int = 7,
    val_tenths: int = 1,
) -> tuple[dict[tuple[int, int, int, int], str], int, int, int]:
    """Map each patch slot to ``train`` / ``val`` / ``test`` (~7:1:2 by default).

    ``test`` size is ``N - n_train - n_val`` so the three counts sum to ``N``.
    """
    n = len(all_slots)
    if n == 0:
        return {}, 0, 0, 0
    rng = np.random.default_rng(int(split_seed))
    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)
    tt, vt = int(train_tenths), int(val_tenths)
    if tt < 0 or vt < 0 or tt + vt > 10:
        raise ValueError(f"Invalid split tenths: train={tt}, val={vt} (must be non-negative and sum <= 10)")
    n_train = (n * tt) // 10
    n_val = (n * vt) // 10
    n_test = n - n_train - n_val
    split_code = np.empty(n, dtype=np.int8)
    split_code[order[:n_train]] = 0
    split_code[order[n_train : n_train + n_val]] = 1
    split_code[order[n_train + n_val :]] = 2
    names = ("train", "val", "test")
    out: dict[tuple[int, int, int, int], str] = {}
    for i, slot in enumerate(all_slots):
        out[slot] = names[int(split_code[i])]
    return out, n_train, n_val, n_test


def _process_simulation_patches(
    spots,
    training_mask,
    full_mask,
    global_dapi,
    celltype_lookup,
    npz_dir,
    image_dir,
    group_row,
    group_col,
    global_gene_list,
    global_gene_to_idx,
    patch_size,
    sigma,
    patch_split_map: dict[tuple[int, int, int, int], str],
):
    """Slice one condition into patches; each slot's split comes from ``patch_split_map``.

    Uses ``training_mask`` (label AND DAPI foreground) for ``label`` / ``instance``,
    and ``full_mask`` for ``ground_truth``.
    """
    n_genes = len(global_gene_list)
    viz_genes = global_gene_list[:max(30, int(n_genes * 0.1))]
    viz_targets = [(g, global_gene_to_idx[g]) for g in viz_genes]

    train_dir = os.path.join(npz_dir, "train")
    val_dir = os.path.join(npz_dir, "val")
    test_dir = os.path.join(npz_dir, "test")
    for d in (train_dir, val_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    dir_seg = os.path.join(image_dir, "cell_type")
    dir_raw = os.path.join(image_dir, "mRNA_spots")
    dir_dapi = os.path.join(image_dir, "DAPI")
    dir_inst = os.path.join(image_dir, "instance")
    dir_gt = os.path.join(image_dir, "ground_truth")

    for d in [dir_seg, dir_raw, dir_dapi, dir_inst, dir_gt]:
        os.makedirs(d, exist_ok=True)

    Real_H, Real_W = training_mask.shape
    n_rows = Real_H // patch_size
    n_cols = Real_W // patch_size

    lookup_array = np.asarray(celltype_lookup)
    gt_lookup_array = np.asarray(celltype_lookup)

    split_dirs = {"train": train_dir, "val": val_dir, "test": test_dir}

    tasks = []
    for r in range(n_rows):
        for c in range(n_cols):
            key = (int(group_row), int(group_col), int(r), int(c))
            split_name = patch_split_map[key]
            target_save_path = split_dirs[split_name]

            tasks.append(
                delayed(_simulation_worker)(
                    group_row,
                    group_col,
                    r,
                    c,
                    patch_size,
                    spots,
                    training_mask,
                    full_mask,
                    global_dapi,
                    lookup_array,
                    gt_lookup_array,
                    target_save_path,
                    image_dir,
                    viz_targets,
                    global_gene_list,
                    global_gene_to_idx,
                    sigma,
                    dir_seg,
                    dir_raw,
                    dir_dapi,
                    dir_inst,
                    dir_gt,
                )
            )

    print(f"  Grid {n_rows}x{n_cols} = {n_rows * n_cols} patch slots (split by global map)")

    for _ in tqdm(
        Parallel(n_jobs=16, return_as="generator")(tasks),
        total=len(tasks),
        desc=f"Slicing Group ({group_row}, {group_col})",
    ):
        pass


def step_preprocess_Simulation(data_dir: str, args):
    """Preprocess Simulation dataset.

    Reads raw files from ``{data_dir}/Simulation_raw`` (``image_*.png``, ``label_*.png``,
    ``spots_*.csv``), clusters cells by gene expression, and writes NPZ
    patches to ``{data_dir}/Simulation/{train,val,test}/``.
    """
    assert args.datasets_name == 'Simulation'
    patch_size = args.patch_size
    sigma = args.density_sigma
    min_large = int(getattr(args, "Simulation_min_large_clusters", 5))
    min_frac = float(getattr(args, "Simulation_cluster_min_frac", 0.10))
    k_min = int(getattr(args, "Simulation_kmeans_k_min", 10))
    k_max = int(getattr(args, "Simulation_kmeans_k_max", 28))
    dapi_close_r = int(getattr(args, "Simulation_dapi_close_radius", 9))
    dapi_gs = float(getattr(args, "Simulation_dapi_gaussian_sigma", 1.5))
    dapi_hole = int(getattr(args, "Simulation_dapi_min_hole_area", 256))
    global_scale = float(getattr(args, "global_scale", 1.0))

    print(
        f"===== Simulation preprocess (patch_size={patch_size}, global_scale={global_scale}, "
        f"cluster: >={min_large} clusters @ >={min_frac:.0%} each) ====="
    )
    apply_lightning_reproducibility(0, workers=False, verbose=False)

    raw_dir = os.path.join(data_dir, "Simulation_raw")
    npz_dir = os.path.join(data_dir, 'Simulation')
    image_dir = os.path.join(data_dir, 'Simulation_image')
    os.makedirs(raw_dir, exist_ok=True)
    print(f"  Raw input dir: {raw_dir}")

    # Only clear generated outputs. Never modify ``Simulation_raw``.
    for d in [npz_dir, image_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # =====================================================================
    # PHASE 1: Global Scan — genes, cell-gene matrix, KMeans clustering
    # =====================================================================
    print(">>> Phase 1: Global Scan + KMeans Clustering...")

    global_gene_set: set = set()
    condition_spots: dict[str, pd.DataFrame] = {}
    condition_labels: dict[str, np.ndarray] = {}

    for cond in CONDITIONS:
        spots_csv = os.path.join(raw_dir, f"spots_{cond}.csv")
        if not os.path.isfile(spots_csv):
            raise FileNotFoundError(
                f"Missing raw Simulation spots file: {spots_csv}. "
                f"Please copy CSV/PNG files from SegJointGene_dataset into {raw_dir}."
            )
        df = pd.read_csv(spots_csv)
        df = df.rename(columns={'spotX': 'x', 'spotY': 'y'})

        label_path = os.path.join(raw_dir, f"label_{cond}.png")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(
                f"Missing raw Simulation label file: {label_path}. "
                f"Please copy CSV/PNG files from SegJointGene_dataset into {raw_dir}."
            )
        from PIL import Image
        label_img = np.array(Image.open(label_path)).astype(np.int32)

        df, label_img = _apply_global_scale_spots_label(df, label_img, global_scale)
        global_gene_set.update(df['gene'].unique())
        condition_spots[cond] = df
        condition_labels[cond] = label_img
        print(f"  {cond}: {len(df)} spots, {label_img.max()} cells, image {label_img.shape}")

    global_gene_list = sorted(list(global_gene_set))
    global_gene_to_idx = {g: i for i, g in enumerate(global_gene_list)}
    n_genes = len(global_gene_list)
    print(f"  Global gene set: {n_genes} genes")

    # Build cell x gene expression matrix across all conditions
    all_matrices = []
    cell_meta = []  # (condition, cell_instance_ids_array)

    for cond in CONDITIONS:
        spots_with_cells = _assign_spots_to_cells(condition_spots[cond], condition_labels[cond])
        condition_spots[cond] = spots_with_cells

        mat, cell_ids = _build_cell_gene_matrix(spots_with_cells, global_gene_to_idx, n_genes)
        if mat.shape[0] > 0:
            all_matrices.append(mat)
            cell_meta.append((cond, cell_ids))
            print(f"  {cond}: {mat.shape[0]} cells with spots")

    pooled_matrix = np.concatenate(all_matrices, axis=0)
    print(f"  Pooled cell-gene matrix: {pooled_matrix.shape}")

    # Log-normalize and cluster (balanced: many clusters each with sufficient mass)
    log_matrix = np.log1p(pooled_matrix)

    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Simulation preprocess requires ``scikit-learn``: pip install scikit-learn"
        ) from exc

    cluster_labels, k_final = cluster_cells_balanced(
        log_matrix,
        min_large_clusters=min_large,
        min_fraction=min_frac,
        k_min=k_min,
        k_max=k_max,
        random_state=42,
    )

    # Map cluster labels to 1-indexed celltype IDs (0 = background)
    cluster_labels_1indexed = cluster_labels.astype(np.int64) + 1

    # Build per-condition celltype lookup: instance_id -> celltype_id
    condition_celltype_lookup: dict[str, np.ndarray] = {}
    offset = 0
    for cond, cell_ids in cell_meta:
        n_cells_cond = len(cell_ids)
        cond_cluster = cluster_labels_1indexed[offset:offset + n_cells_cond]
        offset += n_cells_cond

        max_instance = int(cell_ids.max())
        lookup = np.zeros(max_instance, dtype=np.int32)
        for cell_id, ct_id in zip(cell_ids, cond_cluster):
            lookup[cell_id - 1] = ct_id
        condition_celltype_lookup[cond] = lookup

    # Generate celltype names
    kept_type_names = [f"Cluster_{i}" for i in range(k_final)]

    # Save metadata
    celltype_txt = os.path.join(npz_dir, "celltype_id_map.txt")
    with open(celltype_txt, "w") as f:
        f.write("0\tBackground\n")
        for i, name in enumerate(kept_type_names, start=1):
            f.write(f"{i}\t{name}\n")

    gene_txt = os.path.join(npz_dir, "gene_id_map.txt")
    with open(gene_txt, "w") as f:
        for i, g in enumerate(global_gene_list):
            f.write(f"{i}\t{g}\n")

    np.savez(
        os.path.join(npz_dir, "metadata.npz"),
        gene_names=global_gene_list,
        celltype_names=kept_type_names,
    )

    n_class_total = k_final + 1
    cfg_cap = int(getattr(args, "n_celltype", n_class_total))
    if n_class_total > cfg_cap:
        raise ValueError(
            f"Simulation preprocess produced {k_final} foreground celltypes (+ background => "
            f"{n_class_total} classes), but dynamic_seg.n_celltype={cfg_cap}. "
            "Increase n_celltype, output_channel, and model.num_classes in the Hydra config."
        )

    summary_path = os.path.join(npz_dir, "preprocess_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(f"n_celltype_total={n_class_total}\n")
        sf.write(f"n_fg_celltypes={k_final}\n")
        sf.write(f"patch_size={patch_size}\n")
        sf.write("# Phase-2 patch split lines appended after slicing\n")
    ntype_path = os.path.join(npz_dir, "n_celltype.txt")
    with open(ntype_path, "w", encoding="utf-8") as nf:
        nf.write(str(n_class_total))
    print(f"  Wrote {summary_path} and {ntype_path} (set yaml n_celltype/output_channel/model.num_classes={n_class_total})")
    print(f"  Maps saved to {npz_dir}")

    # =====================================================================
    # PHASE 2: Patch slicing — random train / val / test (~7:1:2) over all patch slots
    # =====================================================================
    split_seed = int(getattr(args, "Simulation_split_seed", 0))
    train_tenths = int(getattr(args, "Simulation_train_tenths", 7))
    val_tenths = int(getattr(args, "Simulation_val_tenths", 1))
    print(
        f">>> Phase 2: Patch slicing (random train/val/test ~{train_tenths}:{val_tenths}:"
        f"{10 - train_tenths - val_tenths}, seed={split_seed})..."
    )

    all_slots: list[tuple[int, int, int, int]] = []
    cond_work: list[dict] = []

    for cond in CONDITIONS:
        group_row, group_col = CONDITION_GRID[cond]
        print(f"\n>>> Preparing {cond} | Grid: ({group_row}, {group_col})")

        label_img = condition_labels[cond]
        full_mask = label_img.astype(np.int32)

        dapi_path = os.path.join(raw_dir, f"image_{cond}.png")
        if not os.path.isfile(dapi_path):
            raise FileNotFoundError(
                f"Missing raw Simulation image file: {dapi_path}. "
                f"Please copy CSV/PNG files from SegJointGene_dataset into {raw_dir}."
            )
        from PIL import Image
        dapi_img = np.array(Image.open(dapi_path)).astype(np.float32)
        dapi_img = _resize_raster_global_scale(dapi_img, global_scale, nearest=False)
        if dapi_img.shape[0] != label_img.shape[0] or dapi_img.shape[1] != label_img.shape[1]:
            raise ValueError(
                f"{cond}: after global_scale={global_scale}, DAPI {dapi_img.shape} != label {label_img.shape}"
            )

        dapi_foreground = dapi_closed_foreground(
            dapi_img,
            gaussian_sigma=dapi_gs,
            close_radius=dapi_close_r,
            min_hole_area=dapi_hole,
        )
        training_mask = full_mask.copy()
        training_mask[~dapi_foreground] = 0

        n_train_px = (training_mask > 0).sum()
        n_full_px = (full_mask > 0).sum()
        ring_px = n_full_px - n_train_px
        print(
            f"  Label split: training={n_train_px} px, ring (GT only)={ring_px} px "
            f"({100 * ring_px / max(n_full_px, 1):.1f}%) [closed DAPI foreground]"
        )

        spots = condition_spots[cond]
        celltype_lookup = condition_celltype_lookup[cond]

        h, w = full_mask.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            training_mask = cv2.copyMakeBorder(
                training_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
            full_mask = cv2.copyMakeBorder(
                full_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
            dapi_img = cv2.copyMakeBorder(
                dapi_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
            print(f"  Padded {h}x{w} -> {full_mask.shape[0]}x{full_mask.shape[1]}")

        Real_H, Real_W = training_mask.shape
        n_rows = Real_H // patch_size
        n_cols = Real_W // patch_size
        for r in range(n_rows):
            for c in range(n_cols):
                all_slots.append((int(group_row), int(group_col), int(r), int(c)))

        cond_work.append(
            {
                "cond": cond,
                "group_row": group_row,
                "group_col": group_col,
                "spots": spots,
                "training_mask": training_mask,
                "full_mask": full_mask,
                "global_dapi": dapi_img,
                "celltype_lookup": celltype_lookup,
            }
        )

    patch_split_map, n_train_slots, n_val_slots, n_test_slots = _build_random_patch_split_map(
        all_slots,
        split_seed=split_seed,
        train_tenths=train_tenths,
        val_tenths=val_tenths,
    )
    n_tot = len(all_slots)
    print(
        f"  Global patch-slot split: N={n_tot} | train={n_train_slots}, val={n_val_slots}, "
        f"test={n_test_slots} (empty grid cells still get a split; workers may skip saving)"
    )
    with open(summary_path, "a", encoding="utf-8") as sf:
        sf.write(f"Simulation_split_seed={split_seed}\n")
        sf.write(f"n_patch_slots={n_tot}\n")
        sf.write(f"n_slots_train={n_train_slots}\n")
        sf.write(f"n_slots_val={n_val_slots}\n")
        sf.write(f"n_slots_test={n_test_slots}\n")

    for w in cond_work:
        cond = w["cond"]
        group_row = w["group_row"]
        group_col = w["group_col"]
        print(f"\n>>> Slicing NPZ {cond} | Grid: ({group_row}, {group_col})")
        _process_simulation_patches(
            spots=w["spots"],
            training_mask=w["training_mask"],
            full_mask=w["full_mask"],
            global_dapi=w["global_dapi"],
            celltype_lookup=w["celltype_lookup"],
            npz_dir=npz_dir,
            image_dir=image_dir,
            group_row=group_row,
            group_col=group_col,
            global_gene_list=global_gene_list,
            global_gene_to_idx=global_gene_to_idx,
            patch_size=patch_size,
            sigma=sigma,
            patch_split_map=patch_split_map,
        )

        training_mask = w["training_mask"]
        full_mask = w["full_mask"]
        spots = w["spots"]
        preview_scale = 0.1
        H_final, W_final = full_mask.shape
        p_h = max(1, int(H_final * preview_scale))
        p_w = max(1, int(W_final * preview_scale))

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        preview_train = cv2.resize(training_mask.astype(np.float32), (p_w, p_h), interpolation=cv2.INTER_NEAREST)
        preview_full = cv2.resize(full_mask.astype(np.float32), (p_w, p_h), interpolation=cv2.INTER_NEAREST)
        sample_spots = spots.sample(n=min(len(spots), 10000))

        axes[0].imshow(preview_train, cmap="nipy_spectral", origin="upper")
        axes[0].scatter(
            sample_spots["x"] * preview_scale,
            sample_spots["y"] * preview_scale,
            s=1,
            c="white",
            alpha=0.5,
        )
        axes[0].set_title("Training mask (label AND DAPI)")

        axes[1].imshow(preview_full, cmap="nipy_spectral", origin="upper")
        axes[1].scatter(
            sample_spots["x"] * preview_scale,
            sample_spots["y"] * preview_scale,
            s=1,
            c="white",
            alpha=0.5,
        )
        axes[1].set_title("Ground truth (full label)")

        fig.suptitle(f"Simulation: {cond} ({W_final}x{H_final})")
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"align_{cond}.png"))
        plt.close(fig)

    print("===== Simulation preprocess done =====")
