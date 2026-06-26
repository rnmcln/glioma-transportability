"""Shared plotting style and helpers for the figures.

A single Arial-like sans-serif font (Liberation Sans, metric-compatible with
Arial) and one cohesive pastel palette are used across all figures. Figures
carry no in-panel titles; titles and full legends are given in the manuscript
captions.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Register Liberation Sans (Arial-compatible) if present.
for _p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    try:
        font_manager.fontManager.addfont(_p)
    except Exception:
        pass

SANS = "Liberation Sans"

MODEL_ORDER = ["cox", "coxnet", "rsf", "gbsa", "stacked"]
MODEL_LABEL = {"cox": "Cox PH", "coxnet": "Penalised Cox", "rsf": "Random survival forest",
               "gbsa": "Gradient boosting", "stacked": "Stacked ensemble"}

# Cohesive pastel palette (muted, print-friendly, colour-blind aware ordering).
PALETTE = {
    "cox": "#7FA8D0",      # pastel blue
    "coxnet": "#8FBF9F",   # pastel green
    "rsf": "#E8CE8A",      # pastel amber
    "gbsa": "#E8A48E",     # pastel coral
    "stacked": "#B59FCB",  # pastel purple
}
# Cohort colours (used in cohort-level figures).
COHORT = {
    "internal": "#A9B6C7",   # soft slate
    "cgga693": "#7FA8D0",    # pastel blue
    "cgga325": "#E8A48E",    # pastel coral
}
INK = "#3A4452"  # text/edge ink


def set_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": [SANS, "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
