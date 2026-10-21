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
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
        }
    )


def axes_grid(
    nrows: int,
    ncols: int,
    ax_w: float = AX_W,
    ax_h: float = AX_H,
    left: float = 15.0,
    right: float = 3.0,
    top: float = 7.0,
    bottom: float = 11.0,
    wgap: float = 17.0,
    hgap: float = 16.0,
):
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
            axes[r, c] = fig.add_axes(
                [x0 / width, y0 / height, ax_w / width, ax_h / height]
            )
    return fig, axes


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.13,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=LETTER_SIZE,
        fontweight="bold",
    )


def table_legend(
    ax,
    handles,
    names,
    columns,
    header="Model",
    fontsize=LEGEND_SIZE,
    loc="best",
    **kwargs,
):
    """Framed Arial legend laid out as a table: handle, left-aligned name, right-aligned value columns.

    columns is a list of (column header, list of value strings); header is the heading of the name
    column (None for no heading row). Handles may be lines or filled areas; the heading row uses the
    same column positions as the rows below it.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
    from matplotlib.patches import Rectangle
    from matplotlib.textpath import TextPath

    props = {"fontfamily": "Arial", "fontsize": fontsize, "color": "black"}
    bold_props = {**props, "fontweight": "bold"}
    regular = FontProperties(family="Arial", size=fontsize)
    bold = FontProperties(family="Arial", size=fontsize, weight="bold")

    def width(label, prop):
        return TextPath((0, 0), label, prop=prop).get_extents().width if label else 0.0

    show_header = header is not None
    name_width = max(
        [width(n, regular) for n in names]
        + ([width(header, bold)] if show_header else [])
    )
    value_widths = [
        max(
            [width(v, regular) for v in values]
            + ([width(head, bold)] if show_header else [])
        )
        for head, values in columns
    ]
    row_height = fontsize * 1.55
    handle_width = fontsize * 3.4

    def cell(label, w, p, ha):
        box = DrawingArea(w, row_height, 0, 0)
        if label:
            box.add_artist(
                plt.Text(
                    0 if ha == "left" else w,
                    row_height / 2,
                    label,
                    ha=ha,
                    va="center",
                    **p,
                )
            )
        return box

    def draw_handle(handle):
        area = DrawingArea(handle_width, row_height, 0, 0)
        if handle is None:
            return area
        if isinstance(handle, Line2D):
            area.add_artist(
                Line2D(
                    [0, handle_width],
                    [row_height / 2] * 2,
                    color=handle.get_color(),
                    linestyle=handle.get_linestyle(),
                    linewidth=handle.get_linewidth(),
                    alpha=handle.get_alpha(),
                )
            )
        else:  # filled area (e.g. a confidence band)
            color = handle.get_facecolor()[0]
            area.add_artist(
                Rectangle(
                    (0, row_height * 0.2),
                    handle_width,
                    row_height * 0.6,
                    facecolor=color[:3],
                    alpha=color[3],
                    edgecolor="none",
                )
            )
        return area

    def make_row(handle, name, values, p):
        children = [draw_handle(handle), cell(name, name_width, p, "left")]
        children += [cell(v, w, p, "right") for v, w in zip(values, value_widths)]
        return HPacker(children=children, align="center", pad=0, sep=fontsize * 0.5)

    rows = []
    if show_header:
        rows.append(make_row(None, header, [head for head, _ in columns], bold_props))
    for i, (handle, name) in enumerate(zip(handles, names)):
        rows.append(make_row(handle, name, [values[i] for _, values in columns], props))
    anchored = AnchoredOffsetbox(
        loc=loc,
        child=VPacker(children=rows, align="left", pad=0, sep=1),
        frameon=True,
        pad=0.4,
        borderpad=0.4,
        **kwargs,
    )
    anchored.patch.set_boxstyle("round,pad=0,rounding_size=0.2")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.8")
    anchored.patch.set_alpha(0.8)
    ax.add_artist(anchored)
    return anchored
