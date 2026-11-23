#!/usr/bin/env python
"""11_figures_tables.py - figures and summary tables.

Main figures
  1  Study flow.
  2  Discrimination: Harrell C by model and cohort, and paired differences
     versus the penalised Cox benchmark.
  3  Calibration at 24 months (penalised Cox) in TCGA and the primary CGGA
     releases, and calibration metrics across the updating ladder.
  4  Discrimination within and between integrated molecular classes.
Supplementary figures
  S1 covariate shift (standardised mean differences); S2 marker missingness;
  S3 calibration at 24 months for all models; S4 Kaplan-Meier curves by
  TCGA-derived risk tertile; S5 calibration at 12 and 36 months.
Tables are written as CSV with display-ready formatting.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.plotting import (BLUE, COHORT_STYLE, DOUBLE_COL, FAINT, INK, MM, MODEL_LABEL,  # noqa: E402
                          MODEL_ORDER, MUTED, RED, REF_ORDER, SINGLE_COL, panel_label,
                          savefig, set_style)
from src.utils import (CGGA_ALL_RELEASES, DAYS_PER_MONTH, GRID_DAYS, GRID_MONTHS,  # noqa: E402
                       PRIMARY_GRID_IDX, PRIMARY_HORIZON_DAYS, get_logger, get_paths)
from src.validation import calibration_table, smoothed_calibration  # noqa: E402

log = get_logger("11_figures")
PRIMARY = ["TCGA_internal", "mRNAseq_693", "mRNAseq_325"]


# ----------------------------------------------------------------------------- helpers
def _ext_module():
    from importlib import import_module
    return import_module("08_external_validation")


def _cohort_frame(paths, coh):
    if coh == "TCGA_internal":
        return pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    return _ext_module().cohorts(paths)[coh]


def _preds(paths, coh):
    f = "oof_tcga.npz" if coh == "TCGA_internal" else f"pred_{coh}.npz"
    z = np.load(paths.data_processed / f, allow_pickle=True)
    keys = sorted({k.split("|")[0] for k in z.files if "|" in k})
    return {k: (z[f"{k}|risk"], z[f"{k}|surv"]) for k in keys}


def _metrics(paths):
    a = pd.read_csv(paths.tables / "internal_validation_metrics.csv")
    b = pd.read_csv(paths.tables / "external_validation_metrics.csv")
    return pd.concat([a, b], ignore_index=True)


def _diffs(paths):
    a = pd.read_csv(paths.tables / "internal_validation_differences.csv")
    b = pd.read_csv(paths.tables / "external_validation_differences.csv")
    return pd.concat([a, b], ignore_index=True)


def _clean(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(left=False)


# ----------------------------------------------------------------------------- Figure 1
def fig_flow(paths):
    ch = pd.read_csv(paths.tables / "cohort_characteristics.csv", index_col=0)
    hx = pd.read_csv(paths.tables / "exclusions_hierarchical.csv")
    lab = {"recurrent_or_secondary": "recurrent or secondary", "missing_age_or_survival": "missing age or survival",
           "survival_time_nonpositive": "survival time \u2264 0", "age_under_18": "age < 18 y"}
    order = list(lab)

    def excl(coh):
        d = hx[hx.cohort == coh].set_index("category").n
        return [f"{lab[c]}: {int(d[c])}" for c in order if c in d.index]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 122 * MM))
    ax.set_xlim(0, 183); ax.set_ylim(0, 122); ax.axis("off")

    def box(x, y, w, h, lines, fs=6.3):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.5, ec=INK))
        n = len(lines)
        for i, s_ in enumerate(lines):
            ax.text(x + w / 2, y + h * (1 - (i + 0.5) / n), s_, ha="center", va="center",
                    fontsize=fs if i == 0 else fs - 0.5, color=INK if i == 0 else MUTED)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK, mutation_scale=6, shrinkA=0, shrinkB=0))

    def line(xs, ys):
        ax.plot(xs, ys, lw=0.5, color=INK, solid_capstyle="butt")

    ax.text(40, 119, "Development", ha="center", fontsize=6.6, color=INK)
    ax.text(142, 119, "External validation", ha="center", fontsize=6.6, color=INK)
    n_t, n_i, d_i = (int(ch.loc[k, "TCGA"]) for k in ["n_total", "n_included", "n_deaths"])
    box(6, 102, 68, 12, ["TCGA lgggbm_tcga_pub", f"{n_t} patients"])
    arrow(40, 102, 40, 82)
    line([40, 52], [92, 92])
    box(52, 84, 42, 16, [f"Excluded, n = {n_t - n_i}"] + excl("TCGA"), fs=6.0)
    box(6, 70, 68, 12, ["Development cohort", f"n = {n_i}; {d_i} deaths"])
    arrow(40, 70, 40, 62)
    box(6, 44, 68, 18, ["Training in TCGA only", "Cox, penalised Cox, random survival forest,",
                        "gradient boosting, stacked ensemble;", "reference: class only, class + age"], fs=6.3)
    arrow(40, 44, 40, 36)
    box(6, 22, 68, 14, ["Frozen models", "no refitting, retuning or recalibration in CGGA"])

    ys = [97, 78, 59, 40]
    for r, y in zip(CGGA_ALL_RELEASES, ys):
        role = "primary" if r in ("mRNAseq_693", "mRNAseq_325") else "sensitivity"
        e = excl(r)
        box(106, y, 72, 17, [f"CGGA {r} ({role})",
                             f"included {int(ch.loc['n_included', r])} of {int(ch.loc['n_total', r])}; "
                             f"{int(ch.loc['n_deaths', r])} deaths",
                             "excluded: " + e[0], "; ".join(e[1:])], fs=6.1)
    # outputs
    box(64, 4, 54, 12, ["Frozen external validation", "discrimination, calibration, overall accuracy"], fs=6.1)
    box(124, 4, 54, 12, ["Recalibration ladder", "cross-validated within CGGA"], fs=6.1)
    line([74, 91], [29, 29]); arrow(91, 29, 91, 16)
    line([142, 142], [40, 24]); line([100, 151], [24, 24])
    arrow(100, 24, 100, 16); arrow(151, 24, 151, 16)
    return savefig(fig, paths.figures / "Figure1")


# ----------------------------------------------------------------------------- Figure 2
def fig_discrimination(paths):
    met = _metrics(paths); dif = _diffs(paths)
    c = met[met.metric == "harrell_c"]
    rows = MODEL_ORDER + REF_ORDER
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 70 * MM),
                                 gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08))
    offs = {"TCGA_internal": -0.22, "mRNAseq_693": 0.0, "mRNAseq_325": 0.22}
    ypos = {k: i + (0.6 if k in REF_ORDER else 0) for i, k in enumerate(rows)}
    for coh in PRIMARY:
        st = COHORT_STYLE[coh]
        for k in rows:
            r = c[(c.cohort == coh) & (c.model == k)]
            if r.empty:
                continue
            r = r.iloc[0]; y = ypos[k] + offs[coh]
            a1.plot([r.lo, r.hi], [y, y], color=st["color"], lw=0.6)
            a1.plot(r.point, y, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.2, mew=0.6, ls="")
    a1.set_yticks([ypos[k] for k in rows]); a1.set_yticklabels([MODEL_LABEL[k] for k in rows])
    a1.invert_yaxis(); a1.set_xlim(0.62, 0.84); a1.set_xticks([0.64, 0.68, 0.72, 0.76, 0.80, 0.84])
    a1.set_xlabel("Harrell C-index (95% CI)")
    a1.axhline((ypos[MODEL_ORDER[-1]] + ypos[REF_ORDER[0]]) / 2, color=FAINT, lw=0.5)
    _clean(a1)
    a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    lab = {"TCGA_internal": "TCGA, cross-validated", "mRNAseq_693": "CGGA mRNAseq_693", "mRNAseq_325": "CGGA mRNAseq_325"}
    for j, coh in enumerate(PRIMARY):
        st = COHORT_STYLE[coh]
        xx = 0.626 + j * 0.080
        a1.plot(xx, -1.05, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.2, mew=0.6, ls="", clip_on=False)
        a1.text(xx + 0.0035, -1.05, lab[coh], va="center", fontsize=6, color=INK)
    a1.set_ylim(ypos[REF_ORDER[-1]] + 0.6, -1.45)
    panel_label(a1, "A", x=-0.42)

    # Panel B: paired differences vs penalised Cox
    d = dif[(dif.metric == "harrell_c") & (dif.reference == "B:coxnet")]
    # reference models: invert the stored (penalised Cox minus reference) differences
    inv = dif[(dif.metric == "harrell_c") & (dif.model == "B:coxnet") & dif.reference.isin(REF_ORDER)].copy()
    inv["model"], inv["reference"] = inv.reference, "B:coxnet"
    inv["diff"], inv["lo"], inv["hi"] = -inv["diff"], -inv["hi"], -inv["lo"]
    d = pd.concat([d, inv], ignore_index=True)
    rowsB = [k for k in rows if k != "B:coxnet"]
    for coh in PRIMARY:
        st = COHORT_STYLE[coh]
        for k in rowsB:
            r = d[(d.cohort == coh) & (d.model == k)]
            if r.empty:
                continue
            r = r.iloc[0]; y = ypos[k] + offs[coh]
            a2.plot([r.lo, r.hi], [y, y], color=st["color"], lw=0.6)
            a2.plot(r["diff"], y, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.2, mew=0.6, ls="")
    a2.axvline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    a2.axhline((ypos[MODEL_ORDER[-1]] + ypos[REF_ORDER[0]]) / 2, color=FAINT, lw=0.5)
    a2.set_ylim(a1.get_ylim()); a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("Difference in C-index vs penalised Cox (95% CI)")
    a2.set_xlim(-0.13, 0.05); a2.set_xticks([-0.12, -0.08, -0.04, 0, 0.04])
    a2.set_xticklabels(["\u22120.12", "\u22120.08", "\u22120.04", "0", "0.04"])
    a2.text(0.0, ypos["B:coxnet"], "reference", ha="center", va="center", fontsize=5.8, color=MUTED)
    panel_label(a2, "B", x=-0.04)
    return savefig(fig, paths.figures / "Figure2")


# ----------------------------------------------------------------------------- Figure 3
def _calib_panel(ax, ev, tm, sp, show_y=True, t=PRIMARY_HORIZON_DAYS, color=INK, marker="o", mfc="white"):
    tab = calibration_table(ev, tm, sp, t, n_bins=5)
    pr, ob = smoothed_calibration(ev, tm, sp, t)
    ax.plot([0, 1], [0, 1], color=FAINT, lw=0.6, zorder=0)
    if pr is not None:
        o = np.argsort(pr)
        ax.plot(pr[o], ob[o], color=color, lw=0.8)
    pred_risk = 1 - tab.pred.to_numpy(); obs_risk = 1 - tab.obs.to_numpy()
    lo = 1 - tab.obs_hi.to_numpy(); hi = 1 - tab.obs_lo.to_numpy()
    for x, l_, h_ in zip(pred_risk, lo, hi):
        ax.plot([x, x], [l_, h_], color=color, lw=0.5)
    ax.plot(pred_risk, obs_risk, ls="", marker=marker, color=color, mfc=mfc, ms=3.0, mew=0.6)
    # rug of predicted risk
    rr = 1 - np.asarray(sp)
    ax.plot(rr, np.full_like(rr, -0.025), ls="", marker="|", ms=2.2, mew=0.25, color=MUTED, alpha=0.5, clip_on=False)
    ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"] if show_y else [])
    ax.set_aspect("equal")
    ax.spines["left"].set_bounds(0, 1)


def fig_calibration(paths):
    met = _metrics(paths)
    lad = pd.read_csv(paths.tables / "updating_ladder.csv")
    key = "B:coxnet"
    fig = plt.figure(figsize=(DOUBLE_COL, 92 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], hspace=0.55, wspace=0.18)
    lab = {"TCGA_internal": "TCGA, cross-validated", "mRNAseq_693": "CGGA mRNAseq_693, frozen",
           "mRNAseq_325": "CGGA mRNAseq_325, frozen"}
    for j, coh in enumerate(PRIMARY):
        ax = fig.add_subplot(gs[0, j])
        df = _cohort_frame(paths, coh); pr = _preds(paths, coh)[key]
        st = COHORT_STYLE[coh]
        _calib_panel(ax, df.os_event.to_numpy(), df.os_time_days.to_numpy(), pr[1][:, PRIMARY_GRID_IDX],
                     show_y=(j == 0), color=st["color"], marker=st["marker"], mfc=st["mfc"])
        m = met[(met.cohort == coh) & (met.model == key)].set_index("metric").point
        ax.text(0.02, 0.97, lab[coh], transform=ax.transAxes, fontsize=6.2, va="top", color=INK)
        ax.text(0.98, 0.04, f"slope {m['slope_24m']:.2f}\nCITL {m['citl_24m']:+.3f}\nICI {m['ici_24m']:.3f}",
                transform=ax.transAxes, fontsize=5.8, va="bottom", ha="right", color=MUTED, linespacing=1.3)
        ax.set_xlabel("Predicted 24-month mortality")
        if j == 0:
            ax.set_ylabel("Observed 24-month mortality")
        panel_label(ax, "ABC"[j], x=-0.2 if j == 0 else -0.08)

    # Panels D and E: ladder metrics (all models, primary cohorts)
    gs_bottom = gs[1, :].subgridspec(1, 2, wspace=0.32)
    stages = ["0_frozen", "1_in_large", "2_slope", "3_revision", "4_refit"]
    slab = ["Frozen", "Recalibration\nin the large", "Slope\nupdate", "Class-specific\nbaseline", "Refit in\nCGGA"]
    for j, (metric, ylabel, ref) in enumerate([("ici_24m", "ICI, 24 months", None),
                                               ("slope_24m", "Calibration slope, 24 months", 1.0)]):
        ax = fig.add_subplot(gs_bottom[0, j])
        for ci, coh in enumerate(["mRNAseq_693", "mRNAseq_325"]):
            st = COHORT_STYLE[coh]
            d = lad[(lad.cohort == coh)]
            for k in MODEL_ORDER:
                dd = d[d.model == k].set_index("stage").reindex(stages)
                x = np.arange(len(stages)) + (ci - 0.5) * 0.22
                lw, alpha, ms = (0.8, 1.0, 3.0) if k == key else (0.4, 0.45, 2.0)
                ax.plot(x, dd[metric], color=st["color"], lw=lw, alpha=alpha, marker=st["marker"],
                        mfc=st["mfc"] if k == key else "white", ms=ms, mew=0.5)
        if ref is not None:
            ax.axhline(ref, color=FAINT, lw=0.6, zorder=0)
        ax.set_xticks(range(len(stages))); ax.set_xticklabels(slab, fontsize=5.6)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0)
        panel_label(ax, "DE"[j], x=-0.13)
    return savefig(fig, paths.figures / "Figure3")


# ----------------------------------------------------------------------------- Figure 4
def fig_within_class(paths):
    dec = pd.read_csv(paths.tables / "c_index_decomposition.csv")
    sub = pd.read_csv(paths.tables / "subgroup_performance.csv")
    cohorts = ["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]
    clab = {"TCGA_internal": "TCGA, cross-validated", "mRNAseq_693": "CGGA mRNAseq_693",
            "mRNAseq_325": "CGGA mRNAseq_325", "pooled_seq": "CGGA pooled sequencing"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 62 * MM),
                                 gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.75))
    key = "B:coxnet"
    for i, coh in enumerate(cohorts):
        r = dec[(dec.cohort == coh) & (dec.model == key)].iloc[0]
        a1.plot([r.c_within, r.c_between], [i, i], color=FAINT, lw=0.6, zorder=0)
        a1.plot(r.c_between, i, marker="o", color=INK, mfc=INK, ms=3.2, ls="")
        a1.plot(r.c_within, i, marker="o", color=INK, mfc="white", ms=3.2, mew=0.6, ls="")
        a1.plot(r.c_overall, i, marker="|", color=BLUE, ms=6, mew=0.9, ls="")
        a1.text(1.08, i, f"{100 * r.share_between:.0f}%", va="center", ha="left", fontsize=5.8, color=MUTED,
                transform=a1.get_yaxis_transform(), clip_on=False)
    a1.set_yticks(range(len(cohorts))); a1.set_yticklabels([clab[c] for c in cohorts])
    a1.invert_yaxis(); a1.set_xlim(0.55, 0.95)
    a1.set_xlabel("Harrell C-index, penalised Cox")
    a1.spines["left"].set_visible(False); a1.tick_params(axis="y", length=0)
    r0 = dec[(dec.cohort == cohorts[0]) & (dec.model == key)].iloc[0]
    a1.text(r0.c_within, -0.45, "within-class\npairs", ha="center", va="bottom", fontsize=5.6, color=INK)
    a1.text(r0.c_between, -0.45, "between-class\npairs", ha="center", va="bottom", fontsize=5.6, color=INK)
    a1.text(r0.c_overall, -0.45, "all", ha="center", va="bottom", fontsize=5.6, color=BLUE)
    a1.text(1.08, -0.45, "share of pairs\nbetween classes", ha="left", va="bottom", fontsize=5.6, color=MUTED,
            transform=a1.get_yaxis_transform(), clip_on=False)
    a1.set_ylim(len(cohorts) - 0.5, -1.25)
    panel_label(a1, "A", x=-0.62, y=1.0)

    sgs = ["gbm_idhwt", "astro_idhmut", "oligo_codel"]
    sglab = {"gbm_idhwt": "Glioblastoma,\nIDH-wildtype", "astro_idhmut": "Astrocytoma,\nIDH-mutant",
             "oligo_codel": "Oligodendroglioma,\nIDH-mutant, 1p/19q-codeleted"}
    offs = {"TCGA_internal": -0.24, "mRNAseq_693": -0.08, "mRNAseq_325": 0.08, "pooled_seq": 0.24}
    for i, sg in enumerate(sgs):
        for coh in cohorts:
            st = COHORT_STYLE[coh]
            r = sub[(sub.cohort == coh) & (sub.subgroup == sg) & (sub.model == key)]
            if r.empty or pd.isna(r.iloc[0].get("harrell_c")):
                continue
            r = r.iloc[0]; y = i + offs[coh]
            a2.plot([r.harrell_c_lo, r.harrell_c_hi], [y, y], color=st["color"], lw=0.6)
            a2.plot(r.harrell_c, y, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.0, mew=0.6, ls="")
            a2.text(1.0, y, f"{int(r.n)}/{int(r.n_events)}", va="center", ha="left", fontsize=5.2, color=MUTED,
                    transform=a2.get_yaxis_transform())
    a2.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    a2.set_yticks(range(len(sgs))); a2.set_yticklabels([sglab[s] for s in sgs])
    a2.invert_yaxis(); a2.set_xlim(0.3, 1.0)
    a2.set_xlabel("Harrell C-index within class (95% CI)")
    a2.spines["left"].set_visible(False); a2.tick_params(axis="y", length=0)
    a2.text(1.0, -0.55, "n/deaths", ha="left", va="center", fontsize=5.2, color=MUTED, transform=a2.get_yaxis_transform())
    # direct cohort key below the axis
    for j, coh in enumerate(cohorts):
        st = COHORT_STYLE[coh]
        xx = 0.32 + j * 0.17
        a2.plot(xx, 2.78, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.0, mew=0.6, ls="", clip_on=False)
        a2.text(xx + 0.014, 2.78, {"TCGA_internal": "TCGA", "mRNAseq_693": "693", "mRNAseq_325": "325",
                                   "pooled_seq": "pooled"}[coh], va="center", fontsize=5.6)
    a2.set_ylim(2.95, -0.7)
    panel_label(a2, "B", x=-0.5, y=1.0)
    return savefig(fig, paths.figures / "Figure4")


# ----------------------------------------------------------------------------- supplementary figures
def figS_love(paths):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.35, 105 * MM))
    frames = {}
    for rel in ["mRNAseq_693", "mRNAseq_325"]:
        d = pd.read_csv(paths.tables / f"cohort_shift_smd_{rel}.csv")
        d = d[d.abs_smd.notna()].copy()
        vlab = {"age_years": "Age", "sex": "Sex", "histology_harmonised": "Histology", "grade_harmonised": "WHO grade",
                "idh_status": "IDH", "codeletion_1p19q": "1p/19q", "mgmt_promoter_methylated": "MGMT promoter",
                "integrated_class_harmonised": "Integrated class", "os_event_rate": "Death proportion"}
        llab = {"(mean)": "mean", "(rate)": "", "glioblastoma_IDHwt": "glioblastoma, IDH-wildtype",
                "astrocytoma_IDHmut": "astrocytoma, IDH-mutant",
                "oligodendroglioma_IDHmut_codel": "oligodendroglioma, IDH-mutant, codeleted",
                "diffuse_glioma_NOS": "NOS", "oligoastrocytoma_legacy": "oligoastrocytoma (legacy)"}
        d = d[d.variable != "median_os_months"]
        d["label"] = [f"{vlab.get(v, v)}: {llab.get(l, l)}".rstrip(": ") for v, l in zip(d.variable, d.level.astype(str))]
        frames[rel] = d.set_index("label").abs_smd
    order = frames["mRNAseq_693"].add(frames["mRNAseq_325"], fill_value=0).sort_values(ascending=True).index
    for rel in frames:
        st = COHORT_STYLE[rel]
        v = frames[rel].reindex(order)
        ax.plot(v.values, range(len(order)), ls="", marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=2.8, mew=0.5)
    for x in (0.1, 0.25):
        ax.axvline(x, color=FAINT, lw=0.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5.4)
    ax.set_xlabel("Absolute standardised mean difference vs TCGA")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    top = len(order) - 1
    for j, rel in enumerate(["mRNAseq_693", "mRNAseq_325"]):
        st = COHORT_STYLE[rel]
        ax.plot(0.56, 0.08 - 0.045 * j, transform=ax.transAxes, marker=st["marker"], color=st["color"], mfc=st["mfc"],
                ms=2.8, ls="", clip_on=False)
        ax.text(0.58, 0.08 - 0.045 * j, f"CGGA {rel}", transform=ax.transAxes, va="center", fontsize=5.6)
    ax.set_ylim(-0.8, top + 0.8)
    return savefig(fig, paths.figures / "FigureS1")


def figS_missing(paths):
    ch = pd.read_csv(paths.tables / "cohort_characteristics.csv", index_col=0)
    cols = ["TCGA"] + list(CGGA_ALL_RELEASES)
    rows = [("idh_unknown_pct", "IDH"), ("codel_unknown_pct", "1p/19q"), ("mgmt_unknown_pct", "MGMT promoter"),
            ("grade4_pct", None)]
    rows = rows[:3]
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, 40 * MM))
    for j, c in enumerate(cols):
        for i, (r, lab) in enumerate(rows):
            v = float(ch.loc[r, c])
            ax.add_patch(Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=INK, alpha=0.05 + 0.9 * v / 100, lw=0))
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if v > 45 else INK)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["TCGA"] + [c for c in CGGA_ALL_RELEASES], fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, lab in rows])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return savefig(fig, paths.figures / "FigureS2")


def figS_calibration_all(paths, months=24, stem="FigureS3", keys=None):
    keys = keys or MODEL_ORDER
    t = round(months * DAYS_PER_MONTH)
    gi = GRID_MONTHS.index(months)
    fig, axes = plt.subplots(len(keys), 3, figsize=(DOUBLE_COL * 0.78, len(keys) * 45 * MM))
    axes = np.atleast_2d(axes)
    for j, coh in enumerate(PRIMARY):
        df = _cohort_frame(paths, coh); pr = _preds(paths, coh)
        st = COHORT_STYLE[coh]
        for i, k in enumerate(keys):
            ax = axes[i, j]
            _calib_panel(ax, df.os_event.to_numpy(), df.os_time_days.to_numpy(), pr[k][1][:, gi], show_y=(j == 0),
                         t=t, color=st["color"], marker=st["marker"], mfc=st["mfc"])
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABEL[k]}\nobserved", fontsize=6)
            if i == len(keys) - 1:
                ax.set_xlabel(f"Predicted {months}-month mortality", fontsize=6)
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.text(0.5, 1.06, {"TCGA_internal": "TCGA, cross-validated", "mRNAseq_693": "CGGA mRNAseq_693",
                                    "mRNAseq_325": "CGGA mRNAseq_325"}[coh], transform=ax.transAxes,
                        ha="center", fontsize=6.2)
    fig.subplots_adjust(hspace=0.25, wspace=0.12)
    return savefig(fig, paths.figures / stem)


def figS_km(paths):
    key = "B:coxnet"
    tc = _preds(paths, "TCGA_internal")[key][0]
    import pickle
    from src.registry import load_models
    m = load_models(paths, [key])[key]
    tcga = _cohort_frame(paths, "TCGA_internal")
    r_full = np.asarray(m.risk_score(tcga))
    cuts = np.quantile(r_full, [1 / 3, 2 / 3])
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 62 * MM), sharey=True)
    shades = [MUTED, BLUE, RED]
    names = ["Low", "Intermediate", "High"]
    for j, coh in enumerate(PRIMARY):
        ax = axes[j]
        df = _cohort_frame(paths, coh)
        r = r_full if coh == "TCGA_internal" else _preds(paths, coh)[key][0]
        grp = np.digitize(r, cuts)
        tm = df.os_time_days.to_numpy() / DAYS_PER_MONTH; ev = df.os_event.to_numpy()
        at_risk_rows = []
        for g in range(3):
            msk = grp == g
            if msk.sum() == 0:
                at_risk_rows.append([0] * 6); continue
            kmf = KaplanMeierFitter().fit(tm[msk], ev[msk])
            sf = kmf.survival_function_.iloc[:, 0]
            ax.step(sf.index, sf.values, where="post", color=shades[g], lw=0.8)
            ci = kmf.confidence_interval_survival_function_
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=shades[g], alpha=0.12, lw=0)
            xl = [100, 54, 30][g]
            ax.text(xl, float(kmf.predict(xl)) + 0.03, names[g], fontsize=5.6, color=shades[g], ha="left")
            at_risk_rows.append([int((tm[msk] >= x).sum()) for x in [0, 24, 48, 72, 96, 120]])
        ax.set_xlim(0, 120); ax.set_xticks([0, 24, 48, 72, 96, 120]); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Months from diagnosis")
        if j == 0:
            ax.set_ylabel("Overall survival")
        ax.text(0.5, 1.04, {"TCGA_internal": "TCGA (apparent)", "mRNAseq_693": "CGGA mRNAseq_693",
                            "mRNAseq_325": "CGGA mRNAseq_325"}[coh], transform=ax.transAxes, ha="center", fontsize=6.2)
        for g in range(3):
            for xi, x in enumerate([0, 24, 48, 72, 96, 120]):
                ax.text(x, -0.40 - 0.07 * g, str(at_risk_rows[g][xi]), ha="center", fontsize=5.2, color=shades[g],
                        transform=ax.get_xaxis_transform())
        if j == 0:
            ax.text(-0.02, -0.33, "Number at risk", transform=ax.transAxes, fontsize=5.4, color=MUTED, ha="left")
    fig.subplots_adjust(bottom=0.40, wspace=0.08)
    return savefig(fig, paths.figures / "FigureS4")


def main() -> int:
    set_style()
    paths = get_paths()
    for f in [fig_flow, fig_discrimination, fig_calibration, fig_within_class, figS_love, figS_missing,
              figS_calibration_all, figS_km]:
        f(paths); log.info("wrote %s", f.__name__)
    figS_calibration_all(paths, months=12, stem="FigureS5a", keys=["B:coxnet", "B:stacked"])
    figS_calibration_all(paths, months=36, stem="FigureS5b", keys=["B:coxnet", "B:stacked"])
    log.info("figures complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
