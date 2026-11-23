#!/usr/bin/env python
"""12_figures_tables.py - figures and summary tables.

Main figures
  1  Overall C of the reference and full models, its decomposition into
     within-class and between-class pairs, and the share of pairs and of
     above-chance concordance attributable to between-class pairs.
  2  Discrimination within integrated classes.
  3  Observed versus case-mix-expected C; class-specific associations of age,
     grade and MGMT status with survival in TCGA and CGGA.
  4  Calibration at 24 months (penalised Cox) and class-specific
     calibration-in-the-large across the updating ladder.
Supplementary figures
  S1 study flow; S2 covariate shift; S3 marker missingness; S4 comparison of
  model classes; S5 calibration at 24 months, all models; S6 and S7
  calibration at 12 and 36 months; S8 ICI and slope across the updating
  ladder; S9 elastic-net penalty path; S10 Kaplan-Meier curves by
  TCGA-derived risk tertile.
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

log = get_logger("12_figures")
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
                        "gradient boosting, stacked ensemble;", "reference: class; class + age; class + age + grade"], fs=6.3)
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
    return savefig(fig, paths.figures / "FigureS1")


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
    return savefig(fig, paths.figures / "FigureS4")


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
    return savefig(fig, paths.figures / "FigureS2")


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
    return savefig(fig, paths.figures / "FigureS3")


def figS_calibration_all(paths, months=24, stem="FigureS5", keys=None):
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
    return savefig(fig, paths.figures / "FigureS10")



# ----------------------------------------------------------------------------- main figures
COH4 = ["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]
OFF4 = {"TCGA_internal": -0.27, "mRNAseq_693": -0.09, "mRNAseq_325": 0.09, "pooled_seq": 0.27}
SHORT = {"TCGA_internal": "TCGA, cross-validated", "mRNAseq_693": "CGGA mRNAseq_693",
         "mRNAseq_325": "CGGA mRNAseq_325", "pooled_seq": "CGGA pooled (693 + 325)"}
FIG_MODELS = ["R:cox", "RA:cox", "RAG:cox", "B:cox", "B:coxnet"]
FIG_LABEL = {"R:cox": "Class only", "RA:cox": "Class + age", "RAG:cox": "Class + age + grade",
             "B:cox": "Cox, full predictor set", "B:coxnet": "Penalised Cox, full predictor set"}


def _dot(ax, x, lo, hi, y, st, ms=3.0, filled=None):
    if lo is not None and np.isfinite(lo):
        ax.plot([lo, hi], [y, y], color=st["color"], lw=0.6)
    mfc = st["mfc"] if filled is None else (st["color"] if filled else "white")
    ax.plot(x, y, marker=st["marker"], color=st["color"], mfc=mfc, ms=ms, mew=0.6, ls="")


def _key_row(fig, rect, cohorts=COH4, labels=None, fs=6.0):
    """Cohort key as one row in an invisible axes placed in figure coordinates."""
    labels = labels or SHORT
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(cohorts)
    for j, coh in enumerate(cohorts):
        st = COHORT_STYLE[coh]
        xx = j / n
        ax.plot(xx, 0.5, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.2, mew=0.6, ls="", clip_on=False)
        ax.text(xx + 0.012, 0.5, labels[coh], va="center", fontsize=fs, color=INK)
    return ax


def _dec(paths, variant="full_follow_up"):
    d = pd.read_csv(paths.tables / "c_decomposition_inference.csv")
    d = d[d.variant == variant]

    def get(coh, model, metric):
        r = d[(d.cohort == coh) & (d.model == model) & (d.metric == metric)].iloc[0]
        return float(r.point), float(r.lo), float(r.hi)
    return get


def fig_decomposition(paths):
    """Figure 1. Overall C of reference and full models, its within/between
    decomposition, and the share of pairs versus the share of above-chance
    concordance attributable to between-class pairs."""
    get = _dec(paths)
    fig = plt.figure(figsize=(DOUBLE_COL, 118 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.66], hspace=0.55, wspace=0.06,
                          left=0.20, right=0.97, top=0.93, bottom=0.08)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1])
    ypos = {k: i for i, k in enumerate(FIG_MODELS)}
    for coh in COH4:
        st = COHORT_STYLE[coh]
        for k in FIG_MODELS:
            y = ypos[k] + OFF4[coh]
            _dot(a1, *get(coh, k, "c_all"), y, st)
            pw, pb = get(coh, k, "c_within"), get(coh, k, "c_between")
            a2.plot([pw[0], pb[0]], [y, y], color=FAINT, lw=0.5, zorder=0)
            _dot(a2, *pw, y, st)
            _dot(a2, *pb, y, st)
    for ax in (a1, a2):
        ax.set_ylim(len(FIG_MODELS) - 0.55, -0.85)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.axhline(2.5, color=FAINT, lw=0.5)
    a1.set_yticks(range(len(FIG_MODELS))); a1.set_yticklabels([FIG_LABEL[k] for k in FIG_MODELS])
    a1.set_xlim(0.64, 0.84); a1.set_xticks([0.65, 0.70, 0.75, 0.80])
    a1.set_xlabel("Harrell C-index, all comparable pairs")
    a2.set_yticks([]); a2.set_xlim(0.44, 0.96); a2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    a2.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    a2.set_xlabel("Harrell C-index within one class and across classes")
    a2.text(0.58, -0.62, "pairs within one class", ha="center", va="center", fontsize=5.8, color=MUTED)
    a2.text(0.86, -0.62, "pairs spanning two classes", ha="center", va="center", fontsize=5.8, color=MUTED)
    a1.text(-0.02, 2.5, "reference models", transform=a1.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=5.4, color=MUTED)
    panel_label(a1, "A", x=-0.62, y=1.0)
    panel_label(a2, "B", x=-0.02, y=1.0)

    a3 = fig.add_subplot(gs[1, :])
    for i, coh in enumerate(COH4):
        st = COHORT_STYLE[coh]
        sp = get(coh, "B:coxnet", "share_between")
        a3.plot([sp[1], sp[2]], [i, i], color=st["color"], lw=0.6)
        a3.plot(sp[0], i, marker="|", color=st["color"], ms=7, mew=1.0, ls="")
        for k, off, lab in [("B:coxnet", -0.17, "penalised Cox"), ("B:cox", 0.17, "Cox")]:
            at = get(coh, k, "attrib_between")
            y = i + off
            a3.plot([sp[0], at[0]], [y, y], color=FAINT, lw=0.5, zorder=0)
            a3.plot([at[1], at[2]], [y, y], color=st["color"], lw=0.6)
            a3.plot(at[0], y, marker=st["marker"], color=st["color"], mfc=st["color"], ms=3.0, ls="")
            a3.text(min(at[2], 0.95) + 0.005, y, lab, va="center", fontsize=5.2, color=MUTED)
    a3.set_yticks(range(len(COH4))); a3.set_yticklabels([SHORT[c] for c in COH4])
    a3.set_ylim(len(COH4) - 0.45, -0.95)
    xt = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    a3.set_xlim(0.55, 1.0); a3.set_xticks(xt); a3.set_xticklabels([f"{round(100 * x)}%" for x in xt])
    a3.spines["bottom"].set_bounds(0.55, 0.95)
    a3.set_xlabel("Proportion attributable to pairs spanning two classes")
    a3.spines["left"].set_visible(False); a3.tick_params(axis="y", length=0)
    a3.text(0.645, -0.72, "of comparable pairs (vertical marks)", ha="center", fontsize=5.8, color=MUTED)
    a3.text(0.86, -0.72, "of concordance above 0.5 (filled markers)", ha="center", fontsize=5.8, color=MUTED)
    panel_label(a3, "C", x=-0.225, y=1.0)
    _key_row(fig, [0.20, 0.965, 0.77, 0.025])
    return savefig(fig, paths.figures / "Figure1")


def fig_within(paths):
    """Figure 2. Discrimination within integrated classes."""
    sub = pd.read_csv(paths.tables / "subgroup_performance.csv")
    models = ["RA:cox", "RAG:cox", "B:cox", "B:coxnet"]
    sgs = [("gbm_idhwt", "Glioblastoma, IDH-wildtype"), ("astro_idhmut", "Astrocytoma, IDH-mutant"),
           ("oligo_codel", "Oligodendroglioma, IDH-mutant,\n1p/19q-codeleted")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 70 * MM), sharey=True,
                             gridspec_kw=dict(wspace=0.10, left=0.20, right=0.98, top=0.84, bottom=0.24))
    for j, (sg, title) in enumerate(sgs):
        ax = axes[j]
        for coh in COH4:
            st = COHORT_STYLE[coh]
            for i, k in enumerate(models):
                r = sub[(sub.cohort == coh) & (sub.subgroup == sg) & (sub.model == k)]
                if r.empty or pd.isna(r.iloc[0].harrell_c):
                    continue
                r = r.iloc[0]
                _dot(ax, r.harrell_c, r.harrell_c_lo, r.harrell_c_hi, i + OFF4[coh], st)
        ax.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(0.28, 1.0); ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.axhline(1.5, color=FAINT, lw=0.5)
        nn = []
        for coh, lab in [("TCGA_internal", "TCGA"), ("pooled_seq", "CGGA pooled")]:
            r = sub[(sub.cohort == coh) & (sub.subgroup == sg)].iloc[0]
            nn.append(f"{lab} {int(r.n)}/{int(r.n_events)}")
        ax.set_title(title + "\n" + "; ".join(nn), fontsize=6.2, loc="left", color=INK, pad=4)
        if j == 1:
            ax.set_xlabel("Harrell C-index within class (95% CI)")
    axes[0].set_yticks(range(len(models))); axes[0].set_yticklabels([FIG_LABEL[k] for k in models])
    axes[0].set_ylim(len(models) - 0.55, -0.55)
    _key_row(fig, [0.20, 0.005, 0.77, 0.04])
    for j in range(3):
        panel_label(axes[j], "ABC"[j], x=-0.06 if j else -0.72, y=1.21)
    return savefig(fig, paths.figures / "Figure2")


def fig_transport(paths):
    """Figure 3. Observed versus case-mix-expected C, and class-specific
    predictor-outcome associations in TCGA and CGGA."""
    bm = pd.read_csv(paths.tables / "casemix_benchmark.csv")
    asc = pd.read_csv(paths.tables / "class_specific_associations.csv")
    fig = plt.figure(figsize=(DOUBLE_COL, 112 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8], hspace=0.62, wspace=0.10,
                          left=0.21, right=0.97, top=0.93, bottom=0.08)
    cohs = ["TCGA_apparent", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]
    clab = {"TCGA_apparent": "TCGA, apparent", "mRNAseq_693": "CGGA mRNAseq_693", "mRNAseq_325": "CGGA mRNAseq_325",
            "pooled_seq": "CGGA pooled"}
    sty = {"TCGA_apparent": COHORT_STYLE["TCGA_internal"], **{c: COHORT_STYLE[c] for c in cohs[1:]}}
    models = [("B:cox", "Cox"), ("B:coxnet", "Penalised Cox")]
    rows = [(c, k) for c in cohs for k, _ in models]
    for j, (met, title) in enumerate([("c_all", "All pairs"), ("c_within", "Within one class"),
                                      ("c_between", "Across classes")]):
        ax = fig.add_subplot(gs[0, j])
        for i, (c, k) in enumerate(rows):
            r = bm[(bm.cohort == c) & (bm.model == k) & (bm.metric == met)].iloc[0]
            y = i + (i // 2) * 0.5
            ax.add_patch(Rectangle((r.expected_lo, y - 0.28), r.expected_hi - r.expected_lo, 0.56, color=FAINT, lw=0))
            ax.plot([r.expected, r.expected], [y - 0.28, y + 0.28], color=MUTED, lw=0.6)
            st = sty[c]
            ax.plot(r.observed, y, marker=st["marker"], color=st["color"], mfc=st["mfc"], ms=3.2, mew=0.6, ls="")
        ax.set_ylim(len(rows) + 1.5 - 0.4, -0.7)
        ax.set_xlim({"c_all": (0.68, 0.84), "c_within": (0.55, 0.77), "c_between": (0.72, 0.92)}[met])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=6.4, loc="left", pad=3)
        if j == 0:
            ax.set_yticks([i + (i // 2) * 0.5 for i in range(len(rows))])
            ax.set_yticklabels([f"{clab[c]}, {dict(models)[k]}" for c, k in rows], fontsize=5.8)
        else:
            ax.set_yticks([])
        if j == 1:
            ax.set_xlabel("Harrell C-index: observed (markers) and expected if the model were correct (grey band, 95% range)")
        panel_label(ax, "A" if j == 0 else "", x=-1.05, y=1.03)

    ax = fig.add_subplot(gs[1, :])
    terms = [("glioblastoma_IDHwt", "age_per_10y", "Glioblastoma, IDH-wildtype: age, per 10 years"),
             ("glioblastoma_IDHwt", "mgmt_methylated", "Glioblastoma, IDH-wildtype: MGMT methylated"),
             ("astrocytoma_IDHmut", "age_per_10y", "Astrocytoma, IDH-mutant: age, per 10 years"),
             ("astrocytoma_IDHmut", "grade_3_or_4", "Astrocytoma, IDH-mutant: grade 3 or 4 vs 2"),
             ("oligodendroglioma_IDHmut_codel", "age_per_10y", "Oligodendroglioma: age, per 10 years"),
             ("oligodendroglioma_IDHmut_codel", "grade_3_or_4", "Oligodendroglioma: grade 3 vs 2")]
    for i, (cl, t, lab) in enumerate(terms):
        for coh, off, st in [("TCGA", -0.15, COHORT_STYLE["TCGA_internal"]), ("CGGA pooled", 0.15, COHORT_STYLE["pooled_seq"])]:
            r = asc[(asc.cohort == coh) & (asc.integrated_class == cl) & (asc.term == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            _dot(ax, r.hr, r.lo, r.hi, i + off, st)
    ax.set_xscale("log"); ax.axvline(1, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(0.3, 10); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_yticks(range(len(terms))); ax.set_yticklabels([x[2] for x in terms], fontsize=5.8)
    ax.set_ylim(len(terms) - 0.5, -0.6)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Hazard ratio within class (95% CI), Cox model fitted separately in each cohort")
    panel_label(ax, "B", x=-0.235, y=1.02)
    _key_row(fig, [0.21, 0.965, 0.76, 0.025], cohorts=["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"],
             labels={"TCGA_internal": "TCGA", "mRNAseq_693": "CGGA mRNAseq_693", "mRNAseq_325": "CGGA mRNAseq_325",
                     "pooled_seq": "CGGA pooled"})
    return savefig(fig, paths.figures / "Figure3")


def fig_calibration_main(paths):
    """Figure 4. Pooled calibration of the penalised Cox model and class-specific
    calibration-in-the-large across the updating ladder."""
    met = _metrics(paths)
    cc = pd.read_csv(paths.tables / "class_specific_recalibration.csv")
    key = "B:coxnet"
    fig = plt.figure(figsize=(DOUBLE_COL, 112 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.75], hspace=0.55, wspace=0.18,
                          left=0.08, right=0.97, top=0.96, bottom=0.10)
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
        ax.text(0.98, 0.04, f"slope {m['slope_24m']:.2f}\nCITL {m['citl_24m']:+.3f}\nICI {m['ici_24m']:.3f}".replace("-", "\u2212"),
                transform=ax.transAxes, fontsize=5.8, va="bottom", ha="right", color=MUTED, linespacing=1.3)
        ax.set_xlabel("Predicted 24-month mortality")
        if j == 0:
            ax.set_ylabel("Observed 24-month mortality")
        panel_label(ax, "ABC"[j], x=-0.2 if j == 0 else -0.08)

    stages = ["0_frozen", "1_in_large", "2_slope", "3_revision"]
    slab = ["Frozen", "Baseline\nre-estimated", "Baseline and\nslope updated", "Class-specific\nbaselines"]
    groups = [("all", "All patients", INK, "o"), ("glioblastoma_IDHwt", "Glioblastoma, IDH-wildtype", BLUE, "s"),
              ("IDH_mutant", "IDH-mutant", RED, "^")]
    sub = gs[1, :].subgridspec(1, 2, wspace=0.30)
    for j, coh in enumerate(["mRNAseq_693", "mRNAseq_325"]):
        ax = fig.add_subplot(sub[0, j])
        d = cc[(cc.cohort == coh) & (cc.model == key)]
        for gi, (g, gl, col, mk) in enumerate(groups):
            dd = d[d.group == g].set_index("stage").reindex(stages)
            x = np.arange(len(stages)) + (gi - 1) * 0.16
            for xi, (_, r) in zip(x, dd.iterrows()):
                ax.plot([xi, xi], [r.citl_lo, r.citl_hi], color=col, lw=0.6)
            ax.plot(x, dd.citl_24m, color=col, lw=0.6, marker=mk, mfc=col if g != "all" else "white", ms=3.0, mew=0.6)
            ax.text(x[-1] + 0.12, dd.citl_24m.iloc[-1], "", fontsize=5.4)
        ax.axhline(0, color=FAINT, lw=0.6, zorder=0)
        ax.set_xticks(range(len(stages))); ax.set_xticklabels(slab, fontsize=5.6)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(-0.20, 0.16)
        ax.set_ylabel("Observed minus predicted\n24-month survival")
        ax.text(0.0, 1.04, {"mRNAseq_693": "CGGA mRNAseq_693", "mRNAseq_325": "CGGA mRNAseq_325"}[coh] +
                ", penalised Cox, held-out folds", transform=ax.transAxes, fontsize=6.0, color=INK)
        panel_label(ax, "DE"[j], x=-0.20, y=1.04)
        if j == 1:
            for gi, (g, gl, col, mk) in enumerate(groups):
                ax.text(1.02, 0.80 - 0.12 * gi, gl, transform=ax.transAxes, fontsize=5.6, color=col, va="center")
    return savefig(fig, paths.figures / "Figure4")


# ----------------------------------------------------------------------------- further supplementary figures
def figS_ladder(paths):
    lad = pd.read_csv(paths.tables / "updating_ladder.csv")
    key = "B:coxnet"
    stages = ["0_frozen", "1_in_large", "2_slope", "3_revision", "4_refit"]
    slab = ["Frozen", "Baseline\nre-estimated", "Slope\nupdate", "Class-specific\nbaselines", "Refit in\nCGGA"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 55 * MM), gridspec_kw=dict(wspace=0.30))
    for j, (metric, ylabel, ref) in enumerate([("ici_24m", "ICI, 24 months", None),
                                               ("slope_24m", "Calibration slope, 24 months", 1.0)]):
        ax = axes[j]
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
        ax.set_ylabel(ylabel); ax.tick_params(axis="x", length=0)
        panel_label(ax, "AB"[j], x=-0.15)
    _key_row(fig, [0.15, 0.97, 0.6, 0.03], cohorts=["mRNAseq_693", "mRNAseq_325"])
    return savefig(fig, paths.figures / "FigureS8")


def figS_penalty(paths):
    pp = pd.read_csv(paths.tables / "penalty_path.csv")
    pp = pp[pp.n_nonzero > 0]
    from src.registry import load_models
    sel = float(np.ravel(load_models(paths, ["B:coxnet"])["B:coxnet"].estimator.alphas)[0])
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, 65 * MM))
    series = [("tcga_cv_c", "TCGA, cross-validated, all pairs", INK, "o"),
              ("cgga_c", "CGGA pooled, all pairs", BLUE, "D"),
              ("cgga_c_idhmut", "CGGA pooled, IDH-mutant", RED, "^"),
              ("cgga_c_gbm", "CGGA pooled, glioblastoma IDH-wildtype", MUTED, "s")]
    pos = {"tcga_cv_c": (1, "bottom", 0.006), "cgga_c": (1, "top", -0.008), "cgga_c_idhmut": (-1, "bottom", 0.006),
           "cgga_c_gbm": (-1, "bottom", 0.006)}
    for col, lab, c, mk in series:
        ax.plot(pp.alpha, pp[col], color=c, lw=0.7, marker=mk, ms=2.6, mfc="white", mew=0.5)
        i, va, dy = pos[col]
        ax.text(pp.alpha.iloc[i], pp[col].iloc[i] + dy, lab, fontsize=5.4, color=c, ha="left" if i == 1 else "right", va=va)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.minorticks_off()
    ax.set_xticks([0.3, 0.1, 0.03, 0.01, 0.003]); ax.set_xticklabels(["0.3", "0.1", "0.03", "0.01", "0.003"])
    ax.axvline(sel, color=FAINT, lw=0.8, zorder=0)
    ax.text(sel, 0.505, "penalty selected\nin TCGA", fontsize=5.4, color=MUTED, ha="center", va="bottom")
    ax.set_xlabel("Elastic-net penalty (log scale; weaker penalty to the right)")
    ax.set_ylabel("Harrell C-index")
    ax.set_ylim(0.50, 0.82)
    return savefig(fig, paths.figures / "FigureS9")


def main() -> int:
    set_style()
    paths = get_paths()
    for f in [fig_decomposition, fig_within, fig_transport, fig_calibration_main,
              fig_flow, figS_love, figS_missing, fig_discrimination, figS_calibration_all, figS_ladder,
              figS_penalty, figS_km]:
        f(paths); log.info("wrote %s", f.__name__)
    figS_calibration_all(paths, months=12, stem="FigureS6", keys=["B:coxnet", "B:stacked"])
    figS_calibration_all(paths, months=36, stem="FigureS7", keys=["B:coxnet", "B:stacked"])
    log.info("figures complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
