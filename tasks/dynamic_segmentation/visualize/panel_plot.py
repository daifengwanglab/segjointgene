"""Render evaluation figures from saved payload (six separate PNGs: eval_a–f)."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
from matplotlib.colors import to_rgb

from tasks.dynamic_segmentation.simulation_eval_bars import SIMULATION_BAR_LEGEND_LINES
from tasks.dynamic_segmentation.visualize.metrics_payload import SegEvalPayload, load_payload


def _safe_box_values(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64)
    if vals.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return vals


def _color_to_hex(rgb: tuple[float, float, float]) -> str:
    r = max(0.0, min(1.0, float(rgb[0])))
    g = max(0.0, min(1.0, float(rgb[1])))
    b = max(0.0, min(1.0, float(rgb[2])))
    return "#{:02x}{:02x}{:02x}".format(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _darken_color(color: str, factor: float) -> str:
    base = np.asarray(to_rgb(color), dtype=np.float64)
    out = base * (1.0 - float(factor))
    return _color_to_hex((float(out[0]), float(out[1]), float(out[2])))


def _lighten_color(color: str, factor: float) -> str:
    base = np.asarray(to_rgb(color), dtype=np.float64)
    out = base + (1.0 - base) * float(factor)
    return _color_to_hex((float(out[0]), float(out[1]), float(out[2])))


def _sim_four_bar_facecolors(base_hex: str, n: int = 4) -> list[str]:
    """Face colors for Simulation slice bars: same hue family, light → dark along bar order."""
    lit = np.asarray(to_rgb(_lighten_color(base_hex, 0.40)), dtype=np.float64)
    drk = np.asarray(to_rgb(_darken_color(base_hex, 0.32)), dtype=np.float64)
    out: list[str] = []
    for i in range(int(n)):
        t = i / max(1, int(n) - 1)
        rgb = lit * (1.0 - t) + drk * t
        out.append(_color_to_hex((float(rgb[0]), float(rgb[1]), float(rgb[2]))))
    return out


def render_panels_from_payload(
    payload: SegEvalPayload,
    output_dir: str,
    *,
    figure_dpi: float = 600.0,
) -> list[str]:
    """Write ``eval_a.png`` … ``eval_f.png`` under ``output_dir``; return their paths."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    _title_kw = {"fontsize": 23, "fontweight": "normal", "pad": 15}
    _dpi = float(figure_dpi)
    _figsize = (6, 5)

    method = str(payload.method_name)
    image_miou = float(payload.image_miou_test)
    gene_miou = float(payload.gene_miou_test)
    cc_scores = payload.cell_calling_scores
    base_color = "#c96b1d"
    dark_color = _darken_color(base_color, factor=0.20)
    light_color = _lighten_color(base_color, factor=0.25)

    names = ["eval_a.png", "eval_b.png", "eval_c.png", "eval_d.png", "eval_e.png", "eval_f.png"]
    paths: list[str] = []

    def _save(
        fig,
        ax,
        basename: str,
        *,
        box_aspect_one: bool = True,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        bbox_inches: str | None = None,
    ) -> None:
        if box_aspect_one:
            ax.set_box_aspect(1)
        if tight_layout_rect is not None:
            fig.tight_layout(rect=tight_layout_rect)
        else:
            fig.tight_layout()
        p = os.path.join(output_dir, basename)
        save_kw: dict = {"dpi": _dpi}
        if bbox_inches is not None:
            save_kw["bbox_inches"] = bbox_inches
        fig.savefig(p, **save_kw)
        plt.close(fig)
        paths.append(p)

    _bar_figsize = (9, 7)
    sim_img = payload.sim_image_miou_bar
    sim_gene = payload.sim_gene_miou_bar

    # eval_a — Image mIoU (Simulation: four conditions; else single bar)
    if sim_img is not None and int(np.asarray(sim_img).size) == 4:
        vals = np.asarray(sim_img, dtype=np.float64).reshape(4)
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 7.0))
        y = np.arange(4)
        colors = _sim_four_bar_facecolors(base_color, 4)
        legend_labels = list(SIMULATION_BAR_LEGEND_LINES)
        for i in range(4):
            ax.barh(
                y[i],
                vals[i],
                height=0.58,
                left=0.0,
                color=colors[i],
                label=legend_labels[i],
                align="center",
            )
        ax.set_yticks([])
        ax.tick_params(left=False, labelleft=False)
        ax.invert_yaxis()
        ax.set_title("Image mIoU (per slice)", **_title_kw)
        ax.set_xlim(0.0, max(1.0, float(vals.max()) * 1.15))
        _leg_font = 26
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderpad=1.1,
            labelspacing=1.45,
            handlelength=3.2,
            handleheight=2.6,
            fontsize=_leg_font,
            prop={"size": _leg_font},
        )
        _save(
            fig,
            ax,
            names[0],
            box_aspect_one=False,
            tight_layout_rect=(0.0, 0.03, 0.78, 0.98),
            bbox_inches="tight",
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=_figsize)
        ax.barh([method], [image_miou], color=base_color)
        ax.set_title("Image mIoU", **_title_kw)
        ax.set_xlim(0.0, max(1.0, image_miou * 1.15))
        _save(fig, ax, names[0])

    # eval_b — Gene mIoU
    if sim_gene is not None and int(np.asarray(sim_gene).size) == 4:
        vals = np.asarray(sim_gene, dtype=np.float64).reshape(4)
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 7.0))
        y = np.arange(4)
        colors = _sim_four_bar_facecolors(base_color, 4)
        legend_labels = list(SIMULATION_BAR_LEGEND_LINES)
        for i in range(4):
            ax.barh(
                y[i],
                vals[i],
                height=0.58,
                left=0.0,
                color=colors[i],
                label=legend_labels[i],
                align="center",
            )
        ax.set_yticks([])
        ax.tick_params(left=False, labelleft=False)
        ax.invert_yaxis()
        ax.set_title("Gene mIoU (per slice)", **_title_kw)
        ax.set_xlim(0.0, max(1.0, float(vals.max()) * 1.15))
        _leg_font = 26
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderpad=1.1,
            labelspacing=1.45,
            handlelength=3.2,
            handleheight=2.6,
            fontsize=_leg_font,
            prop={"size": _leg_font},
        )
        _save(
            fig,
            ax,
            names[1],
            box_aspect_one=False,
            tight_layout_rect=(0.0, 0.03, 0.78, 0.98),
            bbox_inches="tight",
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=_figsize)
        ax.barh([method], [gene_miou], color=base_color)
        ax.set_title("Gene mIoU", **_title_kw)
        ax.set_xlim(0.0, max(1.0, gene_miou * 1.15))
        _save(fig, ax, names[1])

    # eval_c — Cell Calling Score (no method name on y-axis)
    fig, ax = plt.subplots(1, 1, figsize=_figsize)
    d3 = float(cc_scores.get(3, 0.0))
    d5 = float(cc_scores.get(5, 0.0))
    d7 = float(cc_scores.get(7, 0.0))
    center_y = 0.0
    cluster_step = 0.24
    bar_h = 0.20
    y3 = center_y - cluster_step
    y5 = center_y
    y7 = center_y + cluster_step
    ax.barh([y3], [d3], height=bar_h, color=dark_color, label="d=3")
    ax.barh([y5], [d5], height=bar_h, color=base_color, label="d=5")
    ax.barh([y7], [d7], height=bar_h, color=light_color, label="d=7")
    ax.set_yticks([])
    ax.tick_params(left=False, labelleft=False)
    ax.set_title("Cell Calling Score", **_title_kw)
    ax.set_xlim(0.0, max(1.0, max(d3, d5, d7) * 1.15))
    ax.legend(loc="upper left", frameon=True, fontsize=12)
    _save(fig, ax, names[2])

    # eval_d — Cell Area
    fig, ax = plt.subplots(1, 1, figsize=_figsize)
    bp = ax.boxplot(
        [_safe_box_values(payload.cell_area_values)],
        vert=False,
        positions=[0.0],
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(base_color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color(dark_color)
        median.set_linewidth(1.5)
    ax.set_yticks([])
    ax.tick_params(left=False, labelleft=False)
    ax.set_title("Cell Area", **_title_kw)
    _save(fig, ax, names[3])

    # eval_e — Cell Elongation
    fig, ax = plt.subplots(1, 1, figsize=_figsize)
    bp = ax.boxplot(
        [_safe_box_values(payload.cell_elongation_values)],
        vert=False,
        positions=[0.0],
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(base_color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color(dark_color)
        median.set_linewidth(1.5)
    ax.set_yticks([])
    ax.tick_params(left=False, labelleft=False)
    ax.set_title("Cell Elongation", **_title_kw)
    _save(fig, ax, names[4])

    # eval_f — Cell Convexity
    fig, ax = plt.subplots(1, 1, figsize=_figsize)
    bp = ax.boxplot(
        [_safe_box_values(payload.cell_convexity_values)],
        vert=False,
        positions=[0.0],
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(base_color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color(dark_color)
        median.set_linewidth(1.5)
    ax.set_yticks([])
    ax.tick_params(left=False, labelleft=False)
    ax.set_title("Cell Convexity", **_title_kw)
    _save(fig, ax, names[5])

    return paths


def render_panels_from_saved_payload(
    output_dir: str,
    payload_filename: str = "segjointgene_eval_payload.npz",
    *,
    figure_dpi: float = 600.0,
) -> list[str]:
    payload = load_payload(output_dir=output_dir, payload_filename=payload_filename)
    return render_panels_from_payload(
        payload=payload, output_dir=output_dir, figure_dpi=figure_dpi
    )
