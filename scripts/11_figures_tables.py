#!/usr/bin/env python
"""11_figures_tables.py - assemble manuscript figures and the main results table.

Figures carry no in-panel titles (titles live in the manuscript captions), use a
single Arial-like font and one pastel palette, and place legends outside the data
area. Main figures show the interpretively essential content; fuller versions are
written as supplementary figures.
"""
from __future__ import annotations

import glob
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from src.plotting import COHORT, INK, MODEL_LABEL, MODEL_ORDER, PALETTE, savefig, set_style  # noqa: E402
from src.utils import (CGGA_ALL_RELEASES, GLOBAL_SEED, HORIZONS_DAYS,  # noqa: E402
                       PRIMARY_HORIZON_DAYS, get_logger, get_paths)

log = get_logger("11_figures")
FS = "B_clinical_molecular"
PRIM_IDX = list(HORIZONS_DAYS).index(PRIMARY_HORIZON_DAYS)


def _load_cohort(paths, release):
    key = release.lower().replace("_", "")
    df = pd.read_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet")
    return df[df.included_primary_analysis].reset_index(drop=True)


def _km_at(event, time, t):
    return float(KaplanMeierFitter().fit(time, event).predict(t))


def _calib_points(event, time, pred_s, t, q=5):
    try:
        bins = pd.qcut(pred_s, q=q, duplicates="drop")
    except Exception:
        return np.array([]), np.array([])
    px, ox = [], []
    for _, idx in pd.Series(range(len(pred_s))).groupby(bins, observed=True):
        i = idx.values
        px.append(np.mean(pred_s[i])); ox.append(_km_at(event[i].astype(int), time[i], t))
    return np.array(px), np.array(ox)


# ----------------------------------------------------------------------------- Fig 1 roadmap
def fig_flow(paths):
    tcga = pd.read_parquet(paths.data_processed / "tcga_harmonised.parquet")
    n_excl = int((~tcga.included_primary_analysis).sum())
    fig, ax = plt.subplots(figsize=(8.6, 6.4)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(cx, cy, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.015", fc=fc, ec=INK, lw=1.1))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=14, lw=1.3, color=INK, shrinkA=0, shrinkB=0))

    BLUE, GREY, GREEN, AMBER = "#DCE7F2", "#E9ECF1", "#E2F0E6", "#FBF1DD"
    # vertical development spine at x=0.34
    box(0.34, 0.92, 0.46, 0.085, "TCGA merged LGG + GBM cohort\nlgggbm_tcga_pub (1122 samples)", BLUE)
    arrow(0.34, 0.877, 0.34, 0.815)
    box(0.34, 0.775, 0.46, 0.085, "Adults with diffuse glioma and\noverall-survival follow-up", GREY)
    # exclusions to the right
    arrow(0.57, 0.775, 0.70, 0.775)
    box(0.85, 0.775, 0.28, 0.115,
        f"Excluded (n = {n_excl})\nmissing survival data,\nage < 18, or non-primary", "#F3E5E5")
    arrow(0.34, 0.732, 0.34, 0.670)
    box(0.34, 0.628, 0.46, 0.085, "TCGA development cohort\nn = 1033 (one sample per patient)", BLUE)
    arrow(0.34, 0.585, 0.34, 0.523)
    box(0.34, 0.475, 0.50, 0.105,
        "Train five model classes in TCGA only\nCox, penalised Cox, random survival forest,\n"
        "gradient boosting, stacked ensemble", GREY)
    arrow(0.34, 0.422, 0.34, 0.362)
    box(0.34, 0.318, 0.46, 0.075, "Freeze models (CGGA never used in fitting)", "#EDE7F3")
    # split into two distinct downstream analyses
    arrow(0.34, 0.281, 0.20, 0.205)
    arrow(0.34, 0.281, 0.62, 0.205)
    box(0.205, 0.115, 0.36, 0.16,
        "PRIMARY: frozen external validation\nin 4 CGGA releases\n"
        "mRNAseq_693 (397), mRNAseq_325 (223),\nmRNA_array_301 (246), WEseq_286 (169)\n"
        "discrimination + calibration", GREEN)
    box(0.71, 0.115, 0.34, 0.16,
        "DIAGNOSTIC: calibration updating\nwithin CGGA (cross-validated)\n"
        "Stage 0 frozen → Stage 4 refit\n"
        "shows adaptation needed;\nnot an externally validated model", AMBER)
    return savefig(fig, paths.figures / "figure1_study_flow.png")


# ----------------------------------------------------------------------------- Fig 2 Love (top-k) + supp full
def _love_frame(paths):
    frames = {}
    for rel in ["mRNAseq_693", "mRNAseq_325"]:
        d = pd.read_csv(paths.tables / f"cohort_shift_smd_{rel}.csv")
        d = d[(~d.level.isin(["unknown"])) & d.abs_smd.notna()].copy()
        d["label"] = d.variable.str.replace("_harmonised", "", regex=False).str.replace("_", " ") + ": " + d.level.astype(str)
        frames[rel] = d.set_index("label")["abs_smd"]
    return frames


def _love_plot(paths, labels, fname):
    frames = _love_frame(paths)
    yy = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(labels) + 1.6))
    ax.scatter([frames["mRNAseq_693"].get(k, np.nan) for k in labels], yy,
               c=COHORT["cgga693"], s=46, zorder=3, edgecolor=INK, linewidth=0.4)
    ax.scatter([frames["mRNAseq_325"].get(k, np.nan) for k in labels], yy,
               c=COHORT["cgga325"], s=46, marker="s", zorder=3, edgecolor=INK, linewidth=0.4)
    for thr in (0.1, 0.25):
        ax.axvline(thr, ls=":" if thr == 0.1 else "--", c="#9aa3ad", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlabel("Absolute standardised mean difference (TCGA vs CGGA)")
    ax.set_xlim(0, max(0.6, np.nanmax([frames["mRNAseq_693"].reindex(labels).max(),
                                       frames["mRNAseq_325"].reindex(labels).max()]) + 0.05))
    ax.invert_yaxis()
    # threshold labels placed at the top, clear of the data
    ytop = -0.8
    ax.text(0.1, ytop, "0.10", color="#7b828c", fontsize=7.5, ha="center")
    ax.text(0.25, ytop, "0.25", color="#7b828c", fontsize=7.5, ha="center")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COHORT["cgga693"],
                      markeredgecolor=INK, markersize=8, label="CGGA mRNAseq_693"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor=COHORT["cgga325"],
                      markeredgecolor=INK, markersize=8, label="CGGA mRNAseq_325")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.6 / len(labels)),
              ncol=2, borderaxespad=0)
    fig.subplots_adjust(bottom=0.16)
    return savefig(fig, paths.figures / fname)


def fig_love(paths):
    frames = _love_frame(paths)
    allk = sorted(set(frames["mRNAseq_693"].index) | set(frames["mRNAseq_325"].index),
                  key=lambda k: np.nanmax([frames["mRNAseq_693"].get(k, 0), frames["mRNAseq_325"].get(k, 0)]),
                  reverse=True)
    _love_plot(paths, allk[:12], "figure2_cohort_shift_love.png")          # main: top 12
    _love_plot(paths, allk, "figureS3_cohort_shift_love_full.png")          # supplement: all


# ----------------------------------------------------------------------------- Fig 3 slopegraph
def fig_internal_external(paths):
    internal = pd.read_csv(paths.tables / f"table4_internal_validation_{FS}.csv").set_index("model")
    ext = {rel: pd.read_csv(paths.tables / f"table5_external_{rel}_{FS}.csv").set_index("model")
           for rel in ["mRNAseq_693", "mRNAseq_325"]}
    rows = []
    for m in MODEL_ORDER:
        row = dict(model=m, internal_C=internal.loc[m, "harrell_c"])
        for rel in ["mRNAseq_693", "mRNAseq_325"]:
            row[f"{rel}_C"] = ext[rel].loc[m, "harrell_c"]
            row[f"{rel}_dC"] = ext[rel].loc[m, "harrell_c"] - internal.loc[m, "harrell_c"]
            row[f"{rel}_slope"] = ext[rel].loc[m, "calibration_slope"]
            row[f"{rel}_citl24"] = ext[rel].loc[m, "citl_24m"]
            row[f"{rel}_ibs"] = ext[rel].loc[m, "ibs"]
        rows.append(row)
    main = pd.DataFrame(rows)
    main.round(3).to_csv(paths.tables / "table3_main_results.csv", index=False)

    def _slope(ylim, fname):
        xs = [0, 1, 2]; xlabels = ["TCGA\ninternal (CV)", "CGGA\nmRNAseq_693", "CGGA\nmRNAseq_325"]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for m in MODEL_ORDER:
            r = main[main.model == m].iloc[0]
            ys = [r.internal_C, r["mRNAseq_693_C"], r["mRNAseq_325_C"]]
            ax.plot(xs, ys, "-", color=PALETTE[m], lw=2, zorder=2)
            ax.scatter(xs, ys, color=PALETTE[m], s=42, edgecolor=INK, linewidth=0.5, zorder=3)
        for j, rel in zip([1, 2], ["mRNAseq_693", "mRNAseq_325"]):
            e = ext[rel]
            for m in MODEL_ORDER:
                ax.plot([xs[j], xs[j]], [e.loc[m, "harrell_c_lo"], e.loc[m, "harrell_c_hi"]],
                        color=PALETTE[m], lw=1, alpha=0.5, zorder=1)
        if ylim[0] >= 0.55:
            ax.axhline(0.5, ls=":", c="#c0c0c0", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.3, 2.4); ax.set_ylim(*ylim)
        ax.set_ylabel("Harrell C-index")
        handles = [Line2D([0], [0], color=PALETTE[m], lw=2.4, marker="o",
                          markeredgecolor=INK, label=MODEL_LABEL[m]) for m in MODEL_ORDER]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        fig.subplots_adjust(right=0.74)
        savefig(fig, paths.figures / fname)

    _slope((0.70, 0.82), "figure3_internal_vs_external.png")            # main (truncated)
    _slope((0.50, 1.00), "figureS6_internal_vs_external_fullscale.png")  # supplement (full scale)
    return main


# ----------------------------------------------------------------------------- Fig 4 calibration
def _internal_oof_surv(paths, tcga):
    state = pickle.load(open(paths.data_processed / f"internal_cv_state_{FS}.pkl", "rb"))
    ev = tcga.os_event.to_numpy().astype(bool); tm = tcga.os_time_days.to_numpy().astype(float)
    surv, risk = {}, {}
    for m in ["cox", "coxnet", "rsf", "gbsa"]:
        cnt = np.maximum(state["base"][m]["cnt"], 1)
        surv[m] = (state["base"][m]["surv"] / cnt[:, None])[:, PRIM_IDX]
        risk[m] = state["base"][m]["risk"] / cnt
    Z = np.column_stack([risk[m] for m in ["cox", "coxnet", "rsf", "gbsa"]])
    mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1; Zs = (Z - mu) / sd
    s = np.zeros(len(tcga))
    for tr, va in KFold(5, shuffle=True, random_state=GLOBAL_SEED).split(Zs):
        meta = CoxPHSurvivalAnalysis(alpha=1.0).fit(Zs[tr], Surv.from_arrays(ev[tr], tm[tr]))
        for k, fn in zip(va, meta.predict_survival_function(Zs[va])):
            s[k] = fn(np.clip(PRIMARY_HORIZON_DAYS, fn.x.min(), fn.x.max()))
    surv["stacked"] = s
    return surv


def _calibration_figure(paths, models_to_show, fname, annotate=False):
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    cgga = _load_cohort(paths, "mRNAseq_693")
    oof = _internal_oof_surv(paths, tcga)
    ev_t = tcga.os_event.to_numpy().astype(bool); tm_t = tcga.os_time_days.to_numpy().astype(float)
    ev_c = cgga.os_event.to_numpy().astype(bool); tm_c = cgga.os_time_days.to_numpy().astype(float)
    mdir = paths.models / FS
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9))
    for ax, src in zip(axes, ["int", "ext"]):
        for m in models_to_show:
            if src == "int":
                px, ox = _calib_points(ev_t, tm_t, oof[m], PRIMARY_HORIZON_DAYS)
            else:
                model = pickle.load(open(mdir / f"{m}.pkl", "rb"))
                ps = model.survival_at(cgga, [PRIMARY_HORIZON_DAYS])[:, 0]
                px, ox = _calib_points(ev_c, tm_c, ps, PRIMARY_HORIZON_DAYS)
            if len(px):
                ax.plot(px, ox, "-o", color=PALETTE[m], ms=5, lw=1.6,
                        markeredgecolor=INK, markeredgewidth=0.4, label=MODEL_LABEL[m])
        ax.plot([0, 1], [0, 1], ls="--", c="#9aa3ad", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted 24-month survival")
    axes[0].set_ylabel("Observed (Kaplan-Meier) 24-month survival")
    if annotate:
        axes[1].annotate("points above the diagonal:\nobserved survival exceeds predicted\n(model slightly pessimistic)",
                         xy=(0.55, 0.66), xytext=(0.16, 0.90), fontsize=8.2, color=INK,
                         arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(models_to_show), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    return savefig(fig, paths.figures / fname)


def fig_calibration(paths):
    _calibration_figure(paths, ["coxnet", "stacked"], "figure4_calibration_24m.png", annotate=True)
    _calibration_figure(paths, MODEL_ORDER, "figureS4_calibration_all_models.png", annotate=False)


# ----------------------------------------------------------------------------- Fig 5 updating
def fig_updating(paths):
    d = pd.read_csv(paths.tables / f"table6_updating_mRNAseq_693_{FS}.csv")
    stages = ["0_frozen", "1_in_large", "2_slope", "3_revision", "4_refit"]
    labels = ["frozen", "recalib.\nin-the-large", "slope\nupdate", "limited\nrevision", "full\nrefit"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    xt = list(range(len(stages)))
    for m in MODEL_ORDER:
        sub = d[d.model == m].set_index("stage").reindex(stages)
        axes[0].plot(xt, sub.citl_primary.abs(), "-o", color=PALETTE[m], ms=5,
                     markeredgecolor=INK, markeredgewidth=0.4, label=MODEL_LABEL[m])
        axes[1].plot(xt, sub.ibs, "-o", color=PALETTE[m], ms=5, markeredgecolor=INK, markeredgewidth=0.4)
    for ax in axes:
        ax.set_xticks(xt); ax.set_xticklabels(labels, fontsize=8.4)
    axes[0].set_ylabel("|Calibration-in-the-large| at 24 months")
    axes[1].set_ylabel("Integrated Brier score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.26, wspace=0.28)
    return savefig(fig, paths.figures / "figure5_model_updating.png")


# ----------------------------------------------------------------------------- Fig 6 KM
def fig_risk_groups(paths):
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    cgga = _load_cohort(paths, "mRNAseq_693")
    model = pickle.load(open(paths.models / FS / "coxnet.pkl", "rb"))
    cuts = np.quantile(model.risk_score(tcga), [1 / 3, 2 / 3])
    grp = np.digitize(model.risk_score(cgga), cuts)
    labels = {0: "Low risk (TCGA tertile 1)", 1: "Intermediate", 2: "High risk (TCGA tertile 3)"}
    colors = {0: "#8FBF9F", 1: "#E8CE8A", 2: "#E8A48E"}
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    kmf = KaplanMeierFitter()
    for g in [0, 1, 2]:
        m = grp == g
        if m.sum() == 0:
            continue
        kmf.fit(cgga.os_time_days[m] / 30.4375, cgga.os_event[m], label=f"{labels[g]} (n = {int(m.sum())})")
        kmf.plot_survival_function(ax=ax, color=colors[g], ci_show=True, lw=1.9)
    ax.set_xlabel("Months from diagnosis"); ax.set_ylabel("Overall survival")
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    return savefig(fig, paths.figures / "figure6_risk_groups_km.png")


# ----------------------------------------------------------------------------- Supp S1/S2
def fig_supp(paths):
    recs = []
    for f in sorted(glob.glob(str(paths.tables / f"table5_external_*_{FS}.csv"))):
        d = pd.read_csv(f); rel = f.split("table5_external_")[1].replace(f"_{FS}.csv", "")
        for _, r in d.iterrows():
            recs.append(dict(release=rel, model=r.model, C=r.harrell_c))
    piv = pd.DataFrame(recs).pivot(index="release", columns="model", values="C").reindex(columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    im = ax.imshow(piv.values, cmap="YlGnBu", vmin=0.5, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER))); ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=18, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center",
                    color="w" if piv.values[i, j] > 0.72 else INK, fontsize=8.5)
    fig.colorbar(im, ax=ax, label="Harrell C", shrink=0.85)
    savefig(fig, paths.figures / "figureS1_release_heatmap.png")

    cohorts = {"TCGA": pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet")}
    for rel in CGGA_ALL_RELEASES:
        cohorts[rel] = _load_cohort(paths, rel)
    vars = ["idh_status", "codeletion_1p19q", "mgmt_promoter_methylated", "grade_harmonised", "histology_harmonised"]
    M = np.array([[(cohorts[c][v].astype(str) == "unknown").mean() for v in vars] for c in cohorts])
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    im = ax.imshow(M, cmap="OrRd", vmin=0, vmax=max(0.3, M.max()), aspect="auto")
    ax.set_xticks(range(len(vars))); ax.set_xticklabels([v.replace("_harmonised", "").replace("_", " ") for v in vars], rotation=18, ha="right")
    ax.set_yticks(range(len(cohorts))); ax.set_yticklabels(list(cohorts.keys()))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]*100:.0f}%", ha="center", va="center",
                    color="w" if M[i, j] > 0.4 else INK, fontsize=8.5)
    fig.colorbar(im, ax=ax, label="Proportion unknown", shrink=0.85)
    return savefig(fig, paths.figures / "figureS2_missingness.png")


def main() -> int:
    set_style()
    paths = get_paths()
    fig_flow(paths)
    fig_love(paths)
    main = fig_internal_external(paths)
    fig_calibration(paths)
    fig_updating(paths)
    fig_risk_groups(paths)
    fig_supp(paths)
    log.info("figures written to %s", paths.figures)
    log.info("main results:\n%s", main.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
