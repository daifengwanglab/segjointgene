"""Gene mIoU (mean Gene IoU): Jaccard index on RNA spot-ID sets for matched pred/GT cells (Simulation).

Uses stitched **pseudo-label** instance maps (``pred_insts`` / ``new_inst`` in the training loop), not
raw ``argmax(logits)``, compared to ``ground_truth_instance``. Spots are loaded from
``{data_dir}/spots_{condition}.csv`` (same coordinates as preprocess).
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets.preprocess.Simulation import CONDITION_GRID

# Max pred_unique * gt_unique for dense joint histogram; above this use naive mask IoU.
_MAX_JOINT_HIST_ELEMENTS = 50_000_000


def _condition_for_grid(group_row: int, group_col: int) -> Optional[str]:
    for name, (gr, gc) in CONDITION_GRID.items():
        if gr == group_row and gc == group_col:
            return name
    return None


def spot_rna_id(x: float, y: float, gene: str) -> str:
    return f"{int(round(float(x)))}_{int(round(float(y)))}_{gene}"


def build_cell_spot_id_sets(inst_map: np.ndarray, spots_df: pd.DataFrame) -> Dict[int, set]:
    """Map instance id -> set of unique RNA spot keys (coord + gene)."""
    h, w = inst_map.shape
    xs = spots_df["x"].values
    ys = spots_df["y"].values
    xi = np.clip(np.rint(xs).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int64), 0, h - 1)
    ci = inst_map[yi, xi]
    out: Dict[int, set] = defaultdict(set)
    genes = spots_df["gene"].values
    for k in range(len(spots_df)):
        cid = int(ci[k])
        if cid <= 0:
            continue
        out[cid].add(spot_rna_id(float(xs[k]), float(ys[k]), str(genes[k])))
    return dict(out)


def _pairwise_instance_mask_iou_naive(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    """O(P*G*H*W) reference: full-mask intersection/union per pair."""
    iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    pred_enum = enumerate(pred_ids)
    if show_progress:
        pred_enum = tqdm(
            pred_enum,
            total=len(pred_ids),
            desc="Gene mIoU: pred instances (naive)",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
        )
    for i, p in pred_enum:
        pm = pred_map == p
        for j, g in enumerate(gt_ids):
            gm = gt_map == g
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            if union > 0:
                iou[i, j] = inter / union
    return iou


def pairwise_instance_mask_iou(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    *,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_ids = np.unique(pred_map)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids > 0]
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((0, 0)), pred_ids, gt_ids

    pred_flat = pred_map.ravel()
    gt_flat = gt_map.ravel()
    pred_unique, pred_inv = np.unique(pred_flat, return_inverse=True)
    gt_unique, gt_inv = np.unique(gt_flat, return_inverse=True)
    npu, ngu = int(pred_unique.size), int(gt_unique.size)
    joint_size = npu * ngu
    if joint_size > _MAX_JOINT_HIST_ELEMENTS:
        warnings.warn(
            f"pairwise_instance_mask_iou: joint histogram would have {joint_size} elements "
            f"(limit {_MAX_JOINT_HIST_ELEMENTS}); using naive full-mask IoU.",
            RuntimeWarning,
            stacklevel=2,
        )
        iou = _pairwise_instance_mask_iou_naive(
            pred_map, gt_map, pred_ids, gt_ids, show_progress=show_progress
        )
        return iou, pred_ids, gt_ids

    combined = pred_inv.astype(np.int64) * ngu + gt_inv.astype(np.int64)
    hist = np.bincount(combined, minlength=joint_size).reshape(npu, ngu)

    pos_pred = np.searchsorted(pred_unique, pred_ids)
    pos_gt = np.searchsorted(gt_unique, gt_ids)
    if not (np.all(pred_unique[pos_pred] == pred_ids) and np.all(gt_unique[pos_gt] == gt_ids)):
        warnings.warn(
            "pairwise_instance_mask_iou: label index mismatch; using naive IoU.",
            RuntimeWarning,
            stacklevel=2,
        )
        iou = _pairwise_instance_mask_iou_naive(
            pred_map, gt_map, pred_ids, gt_ids, show_progress=show_progress
        )
        return iou, pred_ids, gt_ids

    inter = hist[np.ix_(pos_pred, pos_gt)].astype(np.float64)
    area_pred = hist[pos_pred, :].sum(axis=1, dtype=np.float64)
    area_gt = hist[:, pos_gt].sum(axis=0, dtype=np.float64)
    union = area_pred[:, None] + area_gt[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou, pred_ids, gt_ids


def match_instances_greedy_max_iou(
    iou: np.ndarray, pred_ids: np.ndarray, gt_ids: np.ndarray
) -> List[Tuple[int, int]]:
    if iou.size == 0:
        return []
    hi, wi = iou.shape
    flat_order = np.argsort(-iou.reshape(-1))
    pairs: List[Tuple[int, int]] = []
    used_p: set = set()
    used_g: set = set()
    for k in flat_order:
        i = k // wi
        j = k % wi
        if iou[i, j] <= 0:
            break
        pi = int(pred_ids[i])
        gj = int(gt_ids[j])
        if pi in used_p or gj in used_g:
            continue
        used_p.add(pi)
        used_g.add(gj)
        pairs.append((pi, gj))
    return pairs


def calc_single_gene_iou(pred_ids: set, gt_ids: set) -> float:
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0
    union = len(pred_ids | gt_ids)
    if union == 0:
        return 0.0
    return len(pred_ids & gt_ids) / union


def mean_gene_iou_for_pairs(
    pairs: Sequence[Tuple[int, int]],
    pred_sets: Dict[int, set],
    gt_sets: Dict[int, set],
) -> float:
    if not pairs:
        return 0.0
    scores = [calc_single_gene_iou(pred_sets.get(p, set()), gt_sets.get(g, set())) for p, g in pairs]
    return float(np.mean(scores))


def compute_gene_miou_one_tile(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    spots_df: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> float:
    pred_sets = build_cell_spot_id_sets(pred_inst, spots_df)
    gt_sets = build_cell_spot_id_sets(gt_inst, spots_df)
    iou, pred_ids, gt_ids = pairwise_instance_mask_iou(
        pred_inst, gt_inst, show_progress=show_progress
    )
    pairs = match_instances_greedy_max_iou(iou, pred_ids, gt_ids)
    return mean_gene_iou_for_pairs(pairs, pred_sets, gt_sets)


def compute_simulation_gene_miou_from_maps(
    pred_t: torch.Tensor,
    gt_t: torch.Tensor,
    data_dir: str,
    test_group_row: int,
    *,
    global_scale: float = 1.0,
    show_progress: bool = False,
    slab_desc: str = "Gene mIoU: condition slabs",
) -> float:
    """
    Mean Gene mIoU over ``group_col`` slabs using stitched pred/gt instance maps (same layout as stitcher).
    """
    scores: List[float] = []
    n_gc = int(pred_t.shape[0])
    gc_candidates: List[int] = []
    for gc in range(n_gc):
        cond = _condition_for_grid(test_group_row, gc)
        if cond is None:
            continue
        csv_path = os.path.join(data_dir, "Simulation_raw", f"spots_{cond}.csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(data_dir, f"spots_{cond}.csv")
        if not os.path.isfile(csv_path):
            continue
        pred = pred_t[gc].numpy().astype(np.int32)
        gt = gt_t[gc].numpy().astype(np.int32)
        if pred.max() == 0 and gt.max() == 0:
            continue
        gc_candidates.append(gc)

    slab_iter: Sequence[int] = gc_candidates
    if show_progress and gc_candidates:
        slab_iter = tqdm(
            gc_candidates,
            desc=slab_desc,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
        )

    for gc in slab_iter:
        cond = _condition_for_grid(test_group_row, gc)
        if cond is None:
            continue
        csv_path = os.path.join(data_dir, "Simulation_raw", f"spots_{cond}.csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(data_dir, f"spots_{cond}.csv")
        if not os.path.isfile(csv_path):
            continue
        pred = pred_t[gc].numpy().astype(np.int32)
        gt = gt_t[gc].numpy().astype(np.int32)
        df = pd.read_csv(csv_path)
        if "spotX" in df.columns:
            df = df.rename(columns={"spotX": "x", "spotY": "y"})
        if "x" not in df.columns or "y" not in df.columns or "gene" not in df.columns:
            continue
        gs = float(global_scale)
        if abs(gs - 1.0) > 1e-12:
            df["x"] = df["x"].astype(np.float64) * gs
            df["y"] = df["y"].astype(np.float64) * gs
        scores.append(
            compute_gene_miou_one_tile(pred, gt, df, show_progress=show_progress)
        )
    return float(np.mean(scores)) if scores else 0.0


def compute_simulation_gene_miou_from_stitcher(
    stitcher,
    data_dir: str,
    test_group_row: int,
    *,
    split: Optional[Literal["train", "val", "test"]] = None,
    global_scale: float = 1.0,
    show_progress: bool = False,
) -> float:
    """
    Mean Gene mIoU from a stitcher's stitched instance maps.

    ``split``:
      - ``\"train\"``: ``global_pred_inst_train`` / ``global_gt_inst_train`` (train-loader patches).
      - ``\"val\"``: ``global_pred_inst_val`` / ``global_gt_inst_val`` (val-loader patches).
      - ``\"test\"``: ``global_pred_inst_test`` / ``global_gt_inst_test`` (test-loader patches).
      - ``None``: legacy combined ``global_pred_inst`` / ``global_gt_inst``.

    ``data_dir`` is the parent of ``Simulation/`` (where ``spots_*.csv`` live).
    """
    if split == "train":
        pred_t = stitcher.global_pred_inst_train
        gt_t = stitcher.global_gt_inst_train
        desc = "Gene mIoU train: condition slabs"
    elif split == "val":
        pred_t = stitcher.global_pred_inst_val
        gt_t = stitcher.global_gt_inst_val
        desc = "Gene mIoU val: condition slabs"
    elif split == "test":
        pred_t = stitcher.global_pred_inst_test
        gt_t = stitcher.global_gt_inst_test
        desc = "Gene mIoU test: condition slabs"
    else:
        pred_t = stitcher.global_pred_inst
        gt_t = stitcher.global_gt_inst
        desc = "Gene mIoU: condition slabs"
    return compute_simulation_gene_miou_from_maps(
        pred_t,
        gt_t,
        data_dir,
        test_group_row,
        global_scale=global_scale,
        show_progress=show_progress,
        slab_desc=desc,
    )


def compute_simulation_gene_miou_bar_from_stitcher_by_gr(
    stitcher,
    data_dir: str,
    *,
    global_scale: float = 1.0,
    show_progress: bool = False,
) -> np.ndarray:
    """Gene mIoU per Simulation condition (test split), length-4 vector for eval_b bars."""
    from tasks.dynamic_segmentation.simulation_eval_bars import SIMULATION_BAR_CONDITION_KEYS

    out = np.zeros(4, dtype=np.float64)
    pred_t = getattr(stitcher, "global_pred_inst_test_by_gr", None)
    gt_t = getattr(stitcher, "global_gt_inst_test_by_gr", None)
    if pred_t is None or gt_t is None:
        return out

    for idx, cond in enumerate(SIMULATION_BAR_CONDITION_KEYS):
        gr, gc = CONDITION_GRID[cond]
        if gr >= int(pred_t.shape[0]) or gc >= int(pred_t.shape[1]):
            continue
        pred = pred_t[gr, gc].detach().cpu().numpy().astype(np.int32)
        gt = gt_t[gr, gc].detach().cpu().numpy().astype(np.int32)
        if pred.max() == 0 and gt.max() == 0:
            continue
        csv_path = os.path.join(data_dir, "Simulation_raw", f"spots_{cond}.csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(data_dir, f"spots_{cond}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "spotX" in df.columns:
            df = df.rename(columns={"spotX": "x", "spotY": "y"})
        if "x" not in df.columns or "y" not in df.columns or "gene" not in df.columns:
            continue
        gs = float(global_scale)
        if abs(gs - 1.0) > 1e-12:
            df["x"] = df["x"].astype(np.float64) * gs
            df["y"] = df["y"].astype(np.float64) * gs
        out[idx] = float(
            compute_gene_miou_one_tile(pred, gt, df, show_progress=show_progress)
        )
    return out
