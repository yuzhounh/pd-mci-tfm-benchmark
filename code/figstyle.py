"""Common style of the manuscript and supplementary figures.

All 2-D plots (forest plots, calibration and decision curves, SHAP dependence plots, ROC/PR
panels, LASSO path) use the same font sizes and the same axes frame, so that the figures look
alike in the paper. The frame (the axes rectangle, not the whole figure with its labels) has a
fixed size in millimetres and an aspect ratio of about 1.4 (width : height).

Sizes (points): axis labels 8, tick labels 7.5, panel titles 9, legend text 6.5 (smaller than the
labels), text annotations 7, panel letters 11 (bold).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
AX_W = 72.0  # width of one axes frame in mm
AX_H = 52.0  # height of one axes frame in mm (aspect ratio 1.38)
LEGEND_SIZE = 6.5
ANNOTATION_SIZE = 7.0
LETTER_SIZE = 11


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
    })


def axes_grid(nrows: int, ncols: int, ax_w: float = AX_W, ax_h: float = AX_H, left: float = 15.0,
              right: float = 3.0, top: float = 7.0, bottom: float = 11.0, wgap: float = 17.0,
              hgap: float = 16.0):
    """Figure with an nrows x ncols grid of axes frames of exactly ax_w x ax_h mm.

    The margins (left, right, top, bottom) and gaps (wgap, hgap) are in mm and hold the tick
    labels, axis labels and titles, so labels never change the size of the frames.
    """
    width = left + ncols * ax_w + (ncols - 1) * wgap + right
    height = top + nrows * ax_h + (nrows - 1) * hgap + bottom
    fig = plt.figure(figsize=(width * MM, height * MM))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            x0 = left + c * (ax_w + wgap)
            y0 = bottom + (nrows - 1 - r) * (ax_h + hgap)
            axes[r, c] = fig.add_axes([x0 / width, y0 / height, ax_w / width, ax_h / height])
    return fig, axes


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=LETTER_SIZE, fontweight="bold")
