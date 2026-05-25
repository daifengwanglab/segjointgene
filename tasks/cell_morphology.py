"""Shared cell morphology metrics: area, convexity, elongation.

Works on integer instance segmentation maps (0 = background, 1..N = cell IDs).
Uses ``cv2`` for contour extraction and ``numpy`` for covariance eigenvalue analysis.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def calculate_cell_metrics(cell_mask: np.ndarray) -> Tuple[int, float, float]:
    """Compute morphology metrics for a single binary cell mask.

    Returns (area, convexity, elongation).
    """
    contours, _ = cv2.findContours(
        cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0, 0.0, 0.0

    contour = max(contours, key=cv2.contourArea)

    area = int(np.count_nonzero(cell_mask))

    r_cell = cv2.arcLength(contour, True)
    if r_cell > 0:
        hull = cv2.convexHull(contour)
        r_convex = cv2.arcLength(hull, True)
        convexity = r_convex / r_cell
    else:
        convexity = 0.0

    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) >= 2:
        cov_matrix = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        e_a = max(eigenvalues)
        e_b = min(eigenvalues)
        elongation = float(e_b / e_a) if e_a > 0 else 0.0
    else:
        elongation = 0.0

    return area, convexity, elongation


def compute_morphology_from_instance_map(inst_map: np.ndarray) -> Dict[str, float]:
    """Aggregate morphology stats over all cells in a 2-D instance map.

    Handles potential ID collisions (e.g. stitched maps) by splitting each
    unique ID into connected components.

    Returns dict with keys ``cell_area_mean``, ``cell_convexity_mean``,
    ``cell_elongation_mean``, ``cell_count``.
    """
    areas: List[int] = []
    convexities: List[float] = []
    elongations: List[float] = []

    for cell_id in np.unique(inst_map):
        if cell_id == 0:
            continue
        cell_binary = (inst_map == cell_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(cell_binary, connectivity=8)
        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8)
            if comp_mask.sum() < 4:
                continue
            a, c, e = calculate_cell_metrics(comp_mask)
            areas.append(a)
            convexities.append(c)
            elongations.append(e)

    if not areas:
        return {
            "cell_area_mean": 0.0,
            "cell_convexity_mean": 0.0,
            "cell_elongation_mean": 0.0,
            "cell_count": 0.0,
        }

    return {
        "cell_area_mean": float(np.mean(areas)),
        "cell_convexity_mean": float(np.mean(convexities)),
        "cell_elongation_mean": float(np.mean(elongations)),
        "cell_count": float(len(areas)),
    }


def compute_morphology_from_instance_maps(inst_maps: np.ndarray) -> Dict[str, float]:
    """Average morphology over multiple 2-D instance maps (e.g. group columns).

    ``inst_maps`` shape: ``(N, H, W)`` integer array.
    Skips all-zero slices.  Returns the same keys as
    :func:`compute_morphology_from_instance_map`, averaged across non-empty slices.
    """
    per_slice: List[Dict[str, float]] = []
    for i in range(inst_maps.shape[0]):
        s = inst_maps[i]
        if not s.any():
            continue
        per_slice.append(compute_morphology_from_instance_map(s))

    if not per_slice:
        return {
            "cell_area_mean": 0.0,
            "cell_convexity_mean": 0.0,
            "cell_elongation_mean": 0.0,
            "cell_count": 0.0,
        }

    return {
        "cell_area_mean": float(np.mean([d["cell_area_mean"] for d in per_slice])),
        "cell_convexity_mean": float(np.mean([d["cell_convexity_mean"] for d in per_slice])),
        "cell_elongation_mean": float(np.mean([d["cell_elongation_mean"] for d in per_slice])),
        "cell_count": float(np.mean([d["cell_count"] for d in per_slice])),
    }
