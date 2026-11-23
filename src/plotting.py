"""Shared figure style.

Figures follow a minimal, high data-ink style: no in-figure titles, thin axes
without boxes, no gridlines, small open or filled markers, direct labels in
place of legends, and a restrained palette (dark ink plus two muted accents
from the Nord palette). Cohort identity is always encoded by marker shape as
well as colour. Figures are written as vector PDF, SVG and EPS plus 600 dpi TIFF and
PNG.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"] if f in _AVAILABLE), "DejaVu Sans")

INK = "#2E3440"        # nord0, primary ink
MUTED = "#6B7385"      # secondary ink
FAINT = "#C9CED8"      # reference lines
BLUE = "#5E81AC"       # nord10, accent 1
RED = "#BF616A"        # nord11, accent 2

COHORT_STYLE = {
    "TCGA_internal": dict(color=INK, marker="o", mfc="white", label="TCGA (internal)"),
    "mRNAseq_693": dict(color=BLUE, marker="s", mfc=BLUE, label="CGGA mRNAseq_693"),
    "mRNAseq_325": dict(color=RED, marker="^", mfc=RED, label="CGGA mRNAseq_325"),
    "pooled_seq": dict(color=BLUE, marker="D", mfc="white", label="CGGA pooled"),
}
MODEL_ORDER = ["B:cox", "B:coxnet", "B:rsf", "B:gbsa", "B:stacked"]
REF_ORDER = ["R:cox", "RA:cox", "RAG:cox"]
MODEL_LABEL = {
    "B:cox": "Cox", "B:coxnet": "Penalised Cox", "B:rsf": "Random survival forest",
    "B:gbsa": "Gradient boosting", "B:stacked": "Stacked ensemble",
    "A:cox": "Cox", "A:coxnet": "Penalised Cox", "A:rsf": "Random survival forest",
    "A:gbsa": "Gradient boosting", "A:stacked": "Stacked ensemble",
    "R:cox": "Integrated class only", "RA:cox": "Integrated class + age", "RAG:cox": "Class + age + grade",
}

MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM


def set_style() -> None:
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 0.7,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    })


def panel_label(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="normal",
            va="bottom", ha="left", color=INK)


def savefig(fig, stem, pad=0.04):
    """Save as PDF, SVG, 600 dpi TIFF and PNG. ``stem`` is a Path without suffix."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "eps", "png", "tiff"):
        kw = dict(bbox_inches="tight", pad_inches=pad, facecolor="white")
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(stem.with_suffix(f".{ext}"), **kw)
    plt.close(fig)
    return stem
