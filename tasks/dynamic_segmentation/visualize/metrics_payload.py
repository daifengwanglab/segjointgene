"""Payload I/O for dynamic-segmentation visualize panels."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SegEvalPayload:
    """Serializable payload for six-panel visualization."""

    method_name: str
    epoch_tag: str
    epoch_id: int
    image_miou_test: float
    gene_miou_test: float
    cell_calling_scores: Dict[int, float]
    cell_area_values: np.ndarray
    cell_elongation_values: np.ndarray
    cell_convexity_values: np.ndarray
    sim_image_miou_bar: Optional[np.ndarray] = None
    sim_gene_miou_bar: Optional[np.ndarray] = None


def save_payload(
    output_dir: str,
    payload: SegEvalPayload,
    payload_filename: str = "segjointgene_eval_payload.npz",
    meta_filename: str = "meta.json",
) -> tuple[str, str]:
    """Persist payload arrays/scalars and readable metadata."""
    os.makedirs(output_dir, exist_ok=True)
    payload_path = os.path.join(output_dir, payload_filename)
    meta_path = os.path.join(output_dir, meta_filename)

    save_kw: dict = {
        "method_name": np.array([payload.method_name], dtype=object),
        "epoch_tag": np.array([payload.epoch_tag], dtype=object),
        "epoch_id": np.array([int(payload.epoch_id)], dtype=np.int64),
        "image_miou_test": np.array([float(payload.image_miou_test)], dtype=np.float64),
        "gene_miou_test": np.array([float(payload.gene_miou_test)], dtype=np.float64),
        "cell_calling_d3": np.array([float(payload.cell_calling_scores.get(3, 0.0))], dtype=np.float64),
        "cell_calling_d5": np.array([float(payload.cell_calling_scores.get(5, 0.0))], dtype=np.float64),
        "cell_calling_d7": np.array([float(payload.cell_calling_scores.get(7, 0.0))], dtype=np.float64),
        "cell_area_values": np.asarray(payload.cell_area_values, dtype=np.float64),
        "cell_elongation_values": np.asarray(payload.cell_elongation_values, dtype=np.float64),
        "cell_convexity_values": np.asarray(payload.cell_convexity_values, dtype=np.float64),
    }
    if payload.sim_image_miou_bar is not None:
        save_kw["sim_image_miou_bar"] = np.asarray(payload.sim_image_miou_bar, dtype=np.float64).reshape(4)
    if payload.sim_gene_miou_bar is not None:
        save_kw["sim_gene_miou_bar"] = np.asarray(payload.sim_gene_miou_bar, dtype=np.float64).reshape(4)
    np.savez_compressed(payload_path, **save_kw)

    meta = {
        "method_name": payload.method_name,
        "epoch_tag": payload.epoch_tag,
        "epoch_id": int(payload.epoch_id),
        "image_miou_test": float(payload.image_miou_test),
        "gene_miou_test": float(payload.gene_miou_test),
        "cell_calling_scores": {
            "3": float(payload.cell_calling_scores.get(3, 0.0)),
            "5": float(payload.cell_calling_scores.get(5, 0.0)),
            "7": float(payload.cell_calling_scores.get(7, 0.0)),
        },
        "cell_count_for_distribution": int(np.asarray(payload.cell_area_values).size),
    }
    if payload.sim_image_miou_bar is not None:
        meta["sim_image_miou_bar"] = [float(x) for x in np.asarray(payload.sim_image_miou_bar).reshape(-1)]
    if payload.sim_gene_miou_bar is not None:
        meta["sim_gene_miou_bar"] = [float(x) for x in np.asarray(payload.sim_gene_miou_bar).reshape(-1)]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return payload_path, meta_path


def load_payload(
    output_dir: str,
    payload_filename: str = "segjointgene_eval_payload.npz",
) -> SegEvalPayload:
    """Load payload from NPZ file."""
    payload_path = os.path.join(output_dir, payload_filename)
    with np.load(payload_path, allow_pickle=True) as npz:
        sim_img = (
            np.asarray(npz["sim_image_miou_bar"], dtype=np.float64).reshape(4)
            if "sim_image_miou_bar" in npz.files
            else None
        )
        sim_gene = (
            np.asarray(npz["sim_gene_miou_bar"], dtype=np.float64).reshape(4)
            if "sim_gene_miou_bar" in npz.files
            else None
        )
        meta_path = os.path.join(output_dir, "meta.json")
        if (sim_img is None or sim_gene is None) and os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if sim_img is None and "sim_image_miou_bar" in meta:
                sim_img = np.asarray(meta["sim_image_miou_bar"], dtype=np.float64).reshape(4)
            if sim_gene is None and "sim_gene_miou_bar" in meta:
                sim_gene = np.asarray(meta["sim_gene_miou_bar"], dtype=np.float64).reshape(4)
        return SegEvalPayload(
            method_name=str(npz["method_name"][0]),
            epoch_tag=str(npz["epoch_tag"][0]),
            epoch_id=int(npz["epoch_id"][0]),
            image_miou_test=float(npz["image_miou_test"][0]),
            gene_miou_test=float(npz["gene_miou_test"][0]),
            cell_calling_scores={
                3: float(npz["cell_calling_d3"][0]),
                5: float(npz["cell_calling_d5"][0]),
                7: float(npz["cell_calling_d7"][0]),
            },
            cell_area_values=np.asarray(npz["cell_area_values"], dtype=np.float64),
            cell_elongation_values=np.asarray(npz["cell_elongation_values"], dtype=np.float64),
            cell_convexity_values=np.asarray(npz["cell_convexity_values"], dtype=np.float64),
            sim_image_miou_bar=sim_img,
            sim_gene_miou_bar=sim_gene,
        )
