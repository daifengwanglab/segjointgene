import matplotlib
matplotlib.use('Agg')
import pandas as pd

from utils import setup_seed, save_csv
import tifffile
import cv2
from skimage.segmentation import watershed
from skimage.filters import rank, gaussian, threshold_otsu
from SegJointGene.preprocess import *

def scale_coordinates(row, row_name, min, max, coodinate_size=1000, scale_ratio=1, floor_float=False):
    assert row_name == 'x' or row_name == 'y'
    row_value = row[row_name]
    assert row_value >= min and row_value <= max
    new_row_value = coodinate_size * (row_value - min)/(max - min)
    new_row_value = new_row_value * scale_ratio
    if floor_float:
        new_row_value = math.floor(new_row_value)
    return new_row_value

def step_preprocess_CA1(root_path, args):
    assert args.datasets_name == 'CA1'
    print("===== CA1 preprocess (No-Scale Padding Version) =====")
    setup_seed(0)

    ca1_pref = os.path.join(root_path, "data", "CA1_raw", "3_1_left")
    cells_csv = os.path.join(ca1_pref, "cells_leftCA1_3-1.csv")
    spots_csv = os.path.join(ca1_pref, "spots_w_segmentation_leftCA1_3-1.csv")
    celltype_csv = os.path.join(ca1_pref, "celltype_leftCA1_3-1.csv")
    DAPI_tif = os.path.join(ca1_pref, "CA1DapiBoundaries_3-1_left.tif")

    print("读取 CSV 文件...")
    cells = pd.read_csv(cells_csv)
    spots = pd.read_csv(spots_csv)
    celltypes = pd.read_csv(celltype_csv)

    nuclei = cells[['cell', 'cellX', 'cellY']].copy()
    nuclei = nuclei.rename(columns={'cellX': 'x', 'cellY': 'y'})
    nuclei = nuclei.sort_values(by='cell')
    nuclei_raw_coords = nuclei[['x', 'y']].values.astype(int)

    spots = spots[['gene', 'spotX', 'spotY']].copy()
    spots = spots.rename(columns={'spotX': 'x', 'spotY': 'y'})

    nuclei['cell_id'] = nuclei['cell']
    nuclei['id'] = range(len(nuclei))

    celltype_cat = celltypes['original_celltype'].astype('category')
    celltype_names_by_id = list(celltype_cat.cat.categories)

    cell_to_type = dict(zip(celltypes['cell'], celltype_cat.cat.codes + 1))

    celltype_lookup = np.array(
        [cell_to_type.get(cid, 0) for cid in nuclei['cell_id']]
    )
    print(celltype_lookup)

    celltype_txt = os.path.join(root_path, "data", "CA1", "celltype_id_map.txt")
    os.makedirs(os.path.dirname(celltype_txt), exist_ok=True)

    with open(celltype_txt, "w") as f:
        f.write("0\tBackground\n")
        for i, name in enumerate(celltype_names_by_id, start=1):
            f.write(f"{i}\t{name}\n")

    all_genes = np.sort(spots['gene'].unique())
    gene_txt = os.path.join(root_path, "data", "CA1", "gene_id_map.txt")

    with open(gene_txt, "w") as f:
        for i, g in enumerate(all_genes):
            f.write(f"{i}\t{g}\n")

    print(f"已保存 gene 映射: {gene_txt}")

    print(f"正在加载 DAPI 并运行 Seeded Watershed: {DAPI_tif}")
    dapi_img = tifffile.imread(DAPI_tif)
    H_raw, W_raw = dapi_img.shape

    image_smooth = gaussian(dapi_img, sigma=1.0, preserve_range=True)
    thresh_val = threshold_otsu(image_smooth) * 0.5
    binary_mask = image_smooth > thresh_val

    markers = np.zeros_like(dapi_img, dtype=np.int32)
    y_raw = nuclei_raw_coords[:, 1]
    x_raw = nuclei_raw_coords[:, 0]
    ids = nuclei['id'].values
    valid = (y_raw >= 0) & (y_raw < H_raw) & (x_raw >= 0) & (x_raw < W_raw)
    markers[y_raw[valid], x_raw[valid]] = ids[valid] + 1

    labels = watershed(-image_smooth, markers, mask=binary_mask)
    raw_global_mask = labels.astype(np.int32)

    print("正在执行几何对齐 (Crop -> Pad)...")
    x_min = min(nuclei['x'].min(), spots['x'].min())
    y_min = min(nuclei['y'].min(), spots['y'].min())
    x_max = max(nuclei['x'].max(), spots['x'].max())
    y_max = max(nuclei['y'].max(), spots['y'].max())

    crop_x1 = max(0, int(np.floor(x_min)))
    crop_y1 = max(0, int(np.floor(y_min)))
    crop_x2 = min(W_raw, int(np.ceil(x_max)) + 5)
    crop_y2 = min(H_raw, int(np.ceil(y_max)) + 5)

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    patch_size = args.patch_size
    pad_w = (patch_size - crop_w % patch_size) % patch_size
    pad_h = (patch_size - crop_h % patch_size) % patch_size

    roi_mask = raw_global_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    roi_dapi = dapi_img[crop_y1:crop_y2, crop_x1:crop_x2]

    final_global_mask = cv2.copyMakeBorder(
        roi_mask, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )
    final_global_dapi = cv2.copyMakeBorder(
        roi_dapi, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )

    nuclei['x'] -= crop_x1
    nuclei['y'] -= crop_y1
    spots['x'] -= crop_x1
    spots['y'] -= crop_y1

    image_dir = os.path.join(root_path, 'data', 'CA1_images')
    npz_dir = os.path.join(root_path, 'data', 'CA1')

    process_and_save_patches_optimized(
        spots=spots,
        global_mask=final_global_mask,
        global_dapi=final_global_dapi,
        celltype_lookup=celltype_lookup,
        npz_dir=npz_dir,
        image_dir=image_dir,
        total_size=None,
        patch_size=args.patch_size,
        sigma=args.density_sigma
    )

    print("===== CA1 preprocess done =====")