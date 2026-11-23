#!/usr/bin/env python
"""12_extensions.py - additional analyses requested during revision.

Steps (run all, or name steps on the command line):

std_diff     Bootstrap intervals for TCGA minus CGGA differences in overall,
             within-class and composition-standardised C (independent cohorts).
propensity   Propensity-based standardisation of discrimination (membership
             model TCGA vs pooled CGGA; CGGA pairs weighted by the odds of TCGA
             membership of both patients).
paired_within  Paired bootstrap differences in within-class C between full
             and reference models.
benchmark_ci Interval for observed minus expected C in the case-mix benchmark;
             benchmark repeated with the baseline hazard re-estimated in CGGA.
uno          IPCW (Uno-type) decomposition truncated at 36 months.
meta         Inverse-variance (fixed and random effects) and event-weighted
             summaries of class-specific C on the logit scale.
treatment    Treatment, MGMT and survival within IDH-wildtype tumours in CGGA;
             Karnofsky performance score within IDH-wildtype grade 4 in TCGA.
duplicates   Record linkage across CGGA releases on clinical fields.
samplesize   Events needed for a 95% interval width of 0.10 for within-class C.
dca          Decision curves for 12-month mortality within IDH-wildtype tumours.
tuning       Alternative penalty-selection rules for the elastic net.
who2021      Refined WHO 2021 classification in TCGA (TERT, +7/-10, EGFR,
             CDKN2A/B) and its effect on within-class discrimination.
"""
from __future__ import annotations

import pathlib
import sys
from importlib import import_module

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.calibration import Recalibrator, prognostic_index  # noqa: E402
from src.preprocessing import make_encoder, survival_y  # noqa: E402
from src.registry import load_models  # noqa: E402
from src.utils import (DAYS_PER_MONTH, GLOBAL_SEED, GRID_MONTHS, PRIMARY_GRID_IDX, PRIMARY_HORIZON_DAYS,  # noqa: E402
                       get_logger, get_paths, set_global_seed)
from src.validation import EvalContext, harrell_c  # noqa: E402

log = get_logger("12_ext")
ic = import_module("11_interpretation_checks")
ext = import_module("08_external_validation")
CLASSES = ic.CLASSES
GBM, AST, OLI, NOS = CLASSES
COHORTS = ["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]
DAY36 = 36 * DAYS_PER_MONTH
DAY12 = 12 * DAYS_PER_MONTH


def pct(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, q)) if len(x) else (np.nan, np.nan)


def arrays(df):
    return (df.os_event.to_numpy().astype(bool), df.os_time_days.to_numpy().astype(float),
            df.integrated_class_harmonised.astype(str).to_numpy())


# ============================================================================ std_diff
def run_std_difference(paths, n_boot=1000):
    tdf, tpr = ic.cohort_data(paths, "TCGA_internal")
    cdf, cpr = ic.cohort_data(paths, "pooled_seq")
    te, tt, tc = arrays(tdf)
    ce, ct, cc = arrays(cdf)
    ref = ic.cell_weights(te, tt, tc)
    rows = []
    for k in ["B:coxnet", "B:cox", "RAG:cox"]:
        res = {}
        for lab, (e, t, c, r, seed) in {"TCGA": (te, tt, tc, tpr[k][0], GLOBAL_SEED + 11),
                                        "CGGA": (ce, ct, cc, cpr[k][0], GLOBAL_SEED + 12)}.items():
            pt = ic.decompose(EvalContext(e, t), r, c, ref)
            rng = np.random.default_rng(seed)
            bs = []
            for _ in range(n_boot):
                i = rng.integers(0, len(e), len(e))
                bs.append(ic.decompose(EvalContext(e[i], t[i]), r[i], c[i], ref))
            res[lab] = (pt, pd.DataFrame(bs))
        for m in ["c_all", "c_within", "c_between", "c_std"]:
            d_pt = res["TCGA"][0][m] - res["CGGA"][0][m]
            d_bs = res["TCGA"][1][m].to_numpy() - res["CGGA"][1][m].to_numpy()
            lo, hi = pct(d_bs)
            rows.append(dict(model=k, metric=m, tcga=res["TCGA"][0][m], cgga=res["CGGA"][0][m],
                             difference=d_pt, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "tcga_cgga_differences.csv", index=False)
    log.info("\n%s", out.round(3).to_string(index=False))


# ============================================================================ propensity
MEM_CAT = ["integrated_class_harmonised", "grade_harmonised", "sex", "histology_harmonised",
           "codeletion_1p19q", "mgmt_promoter_methylated"]


def _membership_design(df, enc=None, spl=None):
    age = df[["age_years"]].to_numpy(float)
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[MEM_CAT].astype(str))
        spl = SplineTransformer(n_knots=4, degree=3).fit(age)
    A = spl.transform(age)
    C = enc.transform(df[MEM_CAT].astype(str))
    cls = pd.get_dummies(df.integrated_class_harmonised.astype(str)).reindex(columns=CLASSES, fill_value=0).to_numpy(float)
    inter = np.hstack([A * cls[:, [j]] for j in range(cls.shape[1])])
    return np.hstack([A, C, inter]), enc, spl


def membership_weights(tdf, cdf, trunc=0.99):
    both = pd.concat([tdf, cdf], ignore_index=True)
    y = np.r_[np.ones(len(tdf)), np.zeros(len(cdf))]
    X, enc, spl = _membership_design(both)
    lr = LogisticRegression(C=1.0, max_iter=5000)
    lr.fit(X, y)
    p = lr.predict_proba(X[len(tdf):])[:, 1]
    w = p / (1 - p)
    w = np.minimum(w, np.quantile(w, trunc))
    return w / w.mean(), lr, X, y


def weighted_decompose(e, t, risk, cls, w):
    ctx = EvalContext(e, t)
    r = np.asarray(risk, float)
    d = r[:, None] - r[None, :]
    s = np.where(np.abs(d) <= 1e-8, 0.5, (d > 0).astype(float))
    W = np.outer(w, w) * ctx.comp
    same = cls[:, None] == cls[None, :]
    out = {"c_all": (W * s).sum() / W.sum(),
           "c_within": (W * s * same).sum() / (W * same).sum(),
           "c_between": (W * s * ~same).sum() / (W * ~same).sum(),
           "share_between": (W * ~same).sum() / W.sum()}
    return out


def run_propensity(paths, n_boot=500):
    tdf, _ = ic.cohort_data(paths, "TCGA_internal")
    cdf, cpr = ic.cohort_data(paths, "pooled_seq")
    ce, ct, cc = arrays(cdf)
    w, lr, X, y = membership_weights(tdf, cdf)
    p_all = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED), method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p_all)
    ess = w.sum() ** 2 / (w ** 2).sum()
    rows = []
    rng = np.random.default_rng(GLOBAL_SEED + 21)
    idx_t = [rng.integers(0, len(tdf), len(tdf)) for _ in range(n_boot)]
    idx_c = [rng.integers(0, len(cdf), len(cdf)) for _ in range(n_boot)]
    for k in ["B:coxnet", "B:cox", "RAG:cox"]:
        r = cpr[k][0]
        unw = weighted_decompose(ce, ct, r, cc, np.ones(len(ce)))
        pt = weighted_decompose(ce, ct, r, cc, w)
        bs = []
        for it, i in zip(idx_t, idx_c):
            wb, *_ = membership_weights(tdf.iloc[it].reset_index(drop=True), cdf.iloc[i].reset_index(drop=True))
            bs.append(weighted_decompose(ce[i], ct[i], r[i], cc[i], wb))
        bs = pd.DataFrame(bs)
        for m in pt:
            lo, hi = pct(bs[m])
            rows.append(dict(model=k, metric=m, unweighted=unw[m], weighted=pt[m], lo=lo, hi=hi,
                             membership_auc_cv=auc, effective_n=ess, n=len(ce)))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "propensity_standardised_c.csv", index=False)
    log.info("membership AUC %.3f, effective n %.1f of %d\n%s", auc, ess, len(ce), out.round(3).to_string(index=False))


# ============================================================================ paired_within
def _subgroups(df):
    cls = df.integrated_class_harmonised.astype(str).to_numpy()
    return {"glioblastoma_IDHwt_all": cls == GBM,
            "glioblastoma_IDHwt_grade4": (cls == GBM) & (df.grade_harmonised.astype(str) == "4").to_numpy(),
            "astrocytoma_IDHmut": cls == AST}


def run_paired_within(paths, n_boot=1000):
    pairs = [("B:cox", "RAG:cox"), ("B:coxnet", "RAG:cox"), ("B:cox", "RA:cox"), ("RAG:cox", "RA:cox")]
    rows = []
    for coh in ["TCGA_internal", "pooled_seq"]:
        df, pr = ic.cohort_data(paths, coh)
        e, t, c = arrays(df)
        rng = np.random.default_rng(GLOBAL_SEED + 31)
        # all within-class pairs (whole-cohort resampling)
        idx = [rng.integers(0, len(e), len(e)) for _ in range(n_boot)]
        for a, b in pairs:
            pa = ic.decompose(EvalContext(e, t), pr[a][0], c)["c_within"]
            pb = ic.decompose(EvalContext(e, t), pr[b][0], c)["c_within"]
            d = []
            for i in idx:
                ctx = EvalContext(e[i], t[i])
                d.append(ic.decompose(ctx, pr[a][0][i], c[i])["c_within"] - ic.decompose(ctx, pr[b][0][i], c[i])["c_within"])
            lo, hi = pct(d)
            rows.append(dict(cohort=coh, group="all_within_class_pairs", model=a, reference=b, c_model=pa,
                             c_reference=pb, difference=pa - pb, lo=lo, hi=hi, n=len(e), n_events=int(e.sum())))
        for g, m in _subgroups(df).items():
            em, tm_ = e[m], t[m]
            rng = np.random.default_rng(GLOBAL_SEED + 32)
            idx = [rng.integers(0, m.sum(), m.sum()) for _ in range(n_boot)]
            for a, b in pairs:
                ra, rb = pr[a][0][m], pr[b][0][m]
                pa, pb = harrell_c(em, tm_, ra), harrell_c(em, tm_, rb)
                d = [harrell_c(em[i], tm_[i], ra[i]) - harrell_c(em[i], tm_[i], rb[i]) for i in idx]
                lo, hi = pct(d)
                rows.append(dict(cohort=coh, group=g, model=a, reference=b, c_model=pa, c_reference=pb,
                                 difference=pa - pb, lo=lo, hi=hi, n=int(m.sum()), n_events=int(em.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "paired_within_class_differences.csv", index=False)
    log.info("\n%s", out.round(3).to_string(index=False))


# ============================================================================ benchmark_ci
GRID = np.arange(0.25, 240.0 + 0.125, 0.25) * DAYS_PER_MONTH


def _reverse_km(e, t):
    ut = np.unique(t)
    n_at = np.array([(t >= u).sum() for u in ut])
    d_c = np.array([((t == u) & ~e).sum() for u in ut])
    d_e = np.array([((t == u) & e).sum() for u in ut])
    return ut, np.cumprod(1 - d_c / np.maximum(n_at - d_e, 1))


def _simulate(S, risk, cls, e, t, rng, n_sim):
    ut, Gc = _reverse_km(e, t)
    res = []
    n = len(risk)
    for _ in range(n_sim):
        u = rng.uniform(size=n)
        below = S <= u[:, None]
        has = below.any(axis=1)
        t_ev = np.where(has, GRID[below.argmax(axis=1)], np.inf)
        j = np.searchsorted(-Gc, -rng.uniform(size=n), side="left")
        t_c = ut[np.minimum(j, len(ut) - 1)]
        res.append(ic.decompose(EvalContext(t_ev <= t_c, np.minimum(t_ev, t_c)), risk, cls))
    r = pd.DataFrame(res)
    return {m: r[m].mean() for m in ["c_all", "c_within", "c_between"]}


def run_benchmark_ci(paths, n_boot=200, n_sim=20):
    models = load_models(paths, ["B:cox", "B:coxnet", "RAG:cox"])
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    cohorts = {"TCGA_apparent": tcga, **{c: ext.cohorts(paths)[c] for c in ["mRNAseq_693", "mRNAseq_325", "pooled_seq"]}}
    rows = []
    for coh, df in cohorts.items():
        e, t, c = arrays(df)
        for k, m in models.items():
            risk = np.asarray(m.risk_score(df)).ravel()
            S_tcga = m.survival_at(df, list(GRID))
            variants = {"TCGA baseline": S_tcga}
            if coh != "TCGA_apparent":
                pi = prognostic_index(m.survival_at(df, [PRIMARY_HORIZON_DAYS])[:, 0])
                rc = Recalibrator("in_large").fit(e, t, pi)
                variants["baseline re-estimated in cohort"] = rc.survival_at(pi, GRID)
            for vname, S in variants.items():
                rng = np.random.default_rng(GLOBAL_SEED + 41)
                obs = ic.decompose(EvalContext(e, t), risk, c)
                exp_pt = _simulate(S, risk, c, e, t, rng, 200)
                diffs = {mm: [] for mm in ["c_all", "c_within", "c_between"]}
                for _ in range(n_boot):
                    i = rng.integers(0, len(e), len(e))
                    ob = ic.decompose(EvalContext(e[i], t[i]), risk[i], c[i])
                    ex = _simulate(S[i], risk[i], c[i], e[i], t[i], rng, n_sim)
                    for mm in diffs:
                        diffs[mm].append(ob[mm] - ex[mm])
                for mm in diffs:
                    lo, hi = pct(diffs[mm])
                    rows.append(dict(cohort=coh, model=k, baseline=vname, metric=mm, observed=obs[mm],
                                     expected=exp_pt[mm], difference=obs[mm] - exp_pt[mm], lo=lo, hi=hi))
                log.info("benchmark %s %s %s done", coh, k, vname)
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "casemix_benchmark_intervals.csv", index=False)
    log.info("\n%s", out.round(3).to_string(index=False))


# ============================================================================ uno
def uno_decompose(e, t, risk, cls, tau=DAY36, ctx=None):
    ctx = ctx or EvalContext(e, t)
    G = ctx.G(t)
    w = np.where(e & (t < tau) & (G > 0), 1.0 / np.maximum(G, 1e-12) ** 2, 0.0)
    comp = (t[:, None] < t[None, :]) & (w[:, None] > 0)
    r = np.asarray(risk, float)
    d = r[:, None] - r[None, :]
    s = np.where(np.abs(d) <= 1e-8, 0.5, (d > 0).astype(float))
    W = w[:, None] * comp
    same = cls[:, None] == cls[None, :]
    num_w, den_w = (W * s * same).sum(), (W * same).sum()
    num_b, den_b = (W * s * ~same).sum(), (W * ~same).sum()
    c_all = (num_w + num_b) / (den_w + den_b)
    return {"c_all": c_all, "c_within": num_w / den_w, "c_between": num_b / den_b,
            "share_between": den_b / (den_w + den_b),
            "attrib_between": (num_b - 0.5 * den_b) / (num_w + num_b - 0.5 * (den_w + den_b))}


def run_uno(paths, n_boot=500):
    keys = ["R:cox", "RA:cox", "RAG:cox", "B:cox", "B:coxnet"]
    rows = []
    for coh in COHORTS:
        df, pr = ic.cohort_data(paths, coh)
        e, t, c = arrays(df)
        # implementation check against scikit-survival
        y = Surv.from_arrays(e, t)
        sk = concordance_index_ipcw(y, y, pr["B:coxnet"][0], tau=DAY36)[0]
        mine = uno_decompose(e, t, pr["B:coxnet"][0], c)["c_all"]
        log.info("%s Uno C check: sksurv %.6f, decomposition %.6f", coh, sk, mine)
        rng = np.random.default_rng(GLOBAL_SEED + 51)
        idx = [rng.integers(0, len(e), len(e)) for _ in range(n_boot)]
        for k in keys:
            pt = uno_decompose(e, t, pr[k][0], c)
            bs = pd.DataFrame([uno_decompose(e[i], t[i], pr[k][0][i], c[i]) for i in idx])
            for m in pt:
                lo, hi = pct(bs[m])
                rows.append(dict(cohort=coh, model=k, metric=m, point=pt[m], lo=lo, hi=hi, sksurv_check=sk if k == "B:coxnet" and m == "c_all" else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "uno_decomposition.csv", index=False)
    v = out[out.model.isin(["B:coxnet", "B:cox"])].pivot_table(index=["model", "metric"], columns="cohort", values="point")
    log.info("\n%s", v.round(3).to_string())


# ============================================================================ meta
def run_meta(paths, n_boot=1000):
    rows = []
    for coh in ["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]:
        df, pr = ic.cohort_data(paths, coh)
        e, t, c = arrays(df)
        for k in ["B:coxnet", "B:cox", "RAG:cox"]:
            est = []
            for cl in [GBM, AST, OLI]:
                m = c == cl
                if e[m].sum() < 10:
                    continue
                em, tm_, r = e[m], t[m], pr[k][0][m]
                cc = harrell_c(em, tm_, r)
                rng = np.random.default_rng(GLOBAL_SEED + 61)
                bs = np.array([harrell_c(em[i], tm_[i], r[i]) for i in
                               (rng.integers(0, m.sum(), m.sum()) for _ in range(n_boot))])
                bs = np.clip(bs, 0.01, 0.99)  # replicates with C of 0 or 1 (small strata) bounded for the logit
                lg = np.log(bs / (1 - bs))
                est.append(dict(cls=cl, c=cc, logit=np.log(cc / (1 - cc)), se=np.std(lg, ddof=1),
                                events=int(em.sum()), n=int(m.sum())))
            E = pd.DataFrame(est)
            wv = 1 / E.se ** 2
            fe = (wv * E.logit).sum() / wv.sum()
            se_fe = np.sqrt(1 / wv.sum())
            q = (wv * (E.logit - fe) ** 2).sum()
            dfree = len(E) - 1
            tau2 = max(0.0, (q - dfree) / (wv.sum() - (wv ** 2).sum() / wv.sum())) if dfree > 0 else 0.0
            wr = 1 / (E.se ** 2 + tau2)
            re = (wr * E.logit).sum() / wr.sum()
            se_re = np.sqrt(1 / wr.sum())
            i2 = max(0.0, (q - dfree) / q) if q > 0 else 0.0
            inv = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
            ev = (E.events * E.c).sum() / E.events.sum()
            pw = ic.decompose(EvalContext(e, t), pr[k][0], c)["c_within"]
            for lab, val, lo, hi in [("fixed effect, inverse variance", inv(fe), inv(fe - 1.96 * se_fe), inv(fe + 1.96 * se_fe)),
                                     ("random effects (DerSimonian-Laird)", inv(re), inv(re - 1.96 * se_re), inv(re + 1.96 * se_re)),
                                     ("event-weighted mean", ev, np.nan, np.nan),
                                     ("pair-weighted (decomposition)", pw, np.nan, np.nan)]:
                rows.append(dict(cohort=coh, model=k, summary=lab, c=val, lo=lo, hi=hi, tau2=tau2, i2=i2,
                                 classes=";".join(f"{r.cls}:{r.c:.3f}({r.events})" for r in E.itertuples())))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "within_class_meta.csv", index=False)
    log.info("\n%s", out.drop(columns="classes").round(3).to_string(index=False))


# ============================================================================ treatment
def _cox_table(d, cols, label, cohort):
    X = d[["T", "E"] + cols].dropna()
    X = X.loc[:, (X.nunique() > 1) | X.columns.isin(["T", "E"])]
    f = CoxPHFitter(penalizer=0.001).fit(X, "T", "E")
    return [dict(cohort=cohort, model=label, term=v, n=len(X), n_events=int(X.E.sum()),
                 hr=float(np.exp(f.params_[v])), lo=float(np.exp(f.confidence_intervals_.loc[v].iloc[0])),
                 hi=float(np.exp(f.confidence_intervals_.loc[v].iloc[1]))) for v in f.params_.index]


def run_treatment(paths):
    rows, desc = [], []
    pooled = ext.cohorts(paths)["pooled_seq"]
    _, pr = ic.cohort_data(paths, "pooled_seq")
    S24 = pr["B:coxnet"][1][:, PRIMARY_GRID_IDX]
    for g, m in [("IDH-wildtype, all grades", pooled.integrated_class_harmonised == GBM),
                 ("IDH-wildtype, grade 4", (pooled.integrated_class_harmonised == GBM) & (pooled.grade_harmonised.astype(str) == "4"))]:
        d = pooled[m.to_numpy()].copy()
        s24 = S24[m.to_numpy()]
        d["T"] = d.os_time_days / DAYS_PER_MONTH; d["E"] = d.os_event.astype(int)
        d["age_per_10y"] = d.age_years / 10
        d["mgmt_methylated"] = (d.mgmt_promoter_methylated == "methylated").astype(int)
        d["mgmt_unknown"] = (d.mgmt_promoter_methylated == "unknown").astype(int)
        d["radiotherapy"] = d.radio_status; d["chemotherapy"] = d.chemo_status
        rows += _cox_table(d, ["age_per_10y", "mgmt_methylated", "mgmt_unknown"], "age + MGMT", f"CGGA pooled, {g}")
        rows += _cox_table(d, ["age_per_10y", "mgmt_methylated", "mgmt_unknown", "radiotherapy", "chemotherapy"],
                           "age + MGMT + treatment", f"CGGA pooled, {g}")
        dc = d[d.chemotherapy == 1]
        rows += _cox_table(dc, ["age_per_10y", "mgmt_methylated", "mgmt_unknown"], "age + MGMT, chemotherapy recipients",
                           f"CGGA pooled, {g}")
        for lab, mm in [("radiotherapy and chemotherapy", (d.radiotherapy == 1) & (d.chemotherapy == 1)),
                        ("other or unknown", ~((d.radiotherapy == 1) & (d.chemotherapy == 1)))]:
            mm = mm.to_numpy()
            from src.validation import km_survival_at
            desc.append(dict(group=g, treatment=lab, n=int(mm.sum()), deaths=int(d.E[mm].sum()),
                             km_24m=km_survival_at(d.E[mm], d.os_time_days[mm], PRIMARY_HORIZON_DAYS),
                             pred_24m=float(s24[mm].mean()),
                             citl_24m=km_survival_at(d.E[mm], d.os_time_days[mm], PRIMARY_HORIZON_DAYS) - float(s24[mm].mean())))
    # TCGA: Karnofsky performance score within IDH-wildtype grade 4
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    pat = pd.read_csv(paths.tcga_raw / "data_clinical_patient.txt", sep="\t", comment="#")[["PATIENT_ID", "KARNOFSKY_PERFORMANCE_SCORE"]]
    tcga = tcga.merge(pat.rename(columns={"PATIENT_ID": "patient_id", "KARNOFSKY_PERFORMANCE_SCORE": "kps"}), on="patient_id", how="left")
    d = tcga[(tcga.integrated_class_harmonised == GBM) & (tcga.grade_harmonised.astype(str) == "4")].copy()
    d["T"] = d.os_time_days / DAYS_PER_MONTH; d["E"] = d.os_event.astype(int); d["age_per_10y"] = d.age_years / 10
    d["kps_per_10"] = d.kps / 10
    d["mgmt_methylated"] = (d.mgmt_promoter_methylated == "methylated").astype(int)
    d["mgmt_unknown"] = (d.mgmt_promoter_methylated == "unknown").astype(int)
    rows += _cox_table(d, ["age_per_10y", "kps_per_10", "mgmt_methylated", "mgmt_unknown"], "age + KPS + MGMT",
                       "TCGA, IDH-wildtype grade 4, KPS recorded")
    dk = d.dropna(subset=["kps"]).reset_index(drop=True)
    cv = []
    for rep in range(5):
        kf = KFold(5, shuffle=True, random_state=GLOBAL_SEED + rep)
        oof = {k: np.zeros(len(dk)) for k in ["age", "age + MGMT", "age + KPS", "age + KPS + MGMT"]}
        cols = {"age": ["age_per_10y"], "age + MGMT": ["age_per_10y", "mgmt_methylated", "mgmt_unknown"],
                "age + KPS": ["age_per_10y", "kps_per_10"], "age + KPS + MGMT": ["age_per_10y", "kps_per_10", "mgmt_methylated", "mgmt_unknown"]}
        for tr, va in kf.split(dk):
            for k, cl in cols.items():
                f = CoxPHFitter(penalizer=0.001).fit(dk.loc[tr, ["T", "E"] + cl], "T", "E")
                oof[k][va] = f.predict_partial_hazard(dk.loc[va, cl]).to_numpy().ravel()
        cv.append({k: harrell_c(dk.E.astype(bool), dk["T"], v) for k, v in oof.items()})
    cv = pd.DataFrame(cv).mean()
    kps_c = pd.DataFrame(dict(model=cv.index, cv_harrell_c=cv.values, n=len(dk), n_events=int(dk.E.sum()),
                              n_grade4_idhwt=len(d)))
    pd.DataFrame(rows).to_csv(paths.tables / "treatment_mgmt_kps_associations.csv", index=False)
    pd.DataFrame(desc).to_csv(paths.tables / "treatment_calibration_idhwt.csv", index=False)
    kps_c.to_csv(paths.tables / "kps_within_glioblastoma_cv_c.csv", index=False)
    log.info("\n%s\n%s\n%s", pd.DataFrame(rows).round(3).to_string(index=False), pd.DataFrame(desc).round(3).to_string(index=False),
             kps_c.round(3).to_string(index=False))


# ============================================================================ duplicates
def run_duplicates(paths):
    """Record linkage across CGGA releases.

    Shared identifiers between releases agree on survival time, vital status,
    sex and grade but not on age (age differs in all shared records), so the
    linkage rule requires identical survival time and vital status, identical
    sex, grade and IDH status, and age within 3 years. Its sensitivity is
    estimated on the shared identifiers; matches between different
    identifiers are candidate duplicates.
    """
    rels = ["mRNAseq_693", "mRNAseq_325", "mRNA_array_301", "WEseq_286"]
    C = ext.cohorts(paths)
    cols = ["patient_id", "age_years", "sex", "grade_harmonised", "idh_status", "os_event", "os_time_days"]
    key = ["sex", "grade_harmonised", "idh_status", "os_event", "os_time_days"]
    rows, cands = [], []
    for a_i, a in enumerate(rels):
        for b in rels[a_i + 1:]:
            A = C[a][cols].copy(); B = C[b][cols].copy()
            for X in (A, B):
                for k in ["sex", "grade_harmonised", "idh_status"]:
                    X[k] = X[k].astype(str)
            m = A.merge(B, on=key, suffixes=("_a", "_b"))
            m = m[(m.age_years_a - m.age_years_b).abs() <= 3]
            shared = set(A.patient_id) & set(B.patient_id)
            same = m[m.patient_id_a == m.patient_id_b]
            diff = m[m.patient_id_a != m.patient_id_b]
            rows.append(dict(release_a=a, release_b=b, shared_identifiers=len(shared),
                             shared_identifiers_linked=int(same.patient_id_a.nunique()),
                             candidate_pairs_different_identifiers=len(diff),
                             candidate_patients_a=int(diff.patient_id_a.nunique())))
            if len(diff):
                cands.append(diff.assign(release_a=a, release_b=b))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "cgga_record_linkage.csv", index=False)
    cd = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame()
    cd.to_csv(paths.tables / "cgga_record_linkage_candidates.csv", index=False)
    log.info("\n%s\n%s", out.to_string(index=False), cd.to_string(index=False))
    # sensitivity: pooled primary analysis without candidate duplicates from mRNAseq_325
    sub = cd[(cd.release_a == "mRNAseq_693") & (cd.release_b == "mRNAseq_325")] if len(cd) else cd
    if len(sub):
        df, pr = ic.cohort_data(paths, "pooled_seq")
        keep = ~df.patient_id.isin(set(sub.patient_id_b)).to_numpy()
        e, t, c = arrays(df)
        res = []
        for k in ["B:coxnet", "B:cox"]:
            full = ic.decompose(EvalContext(e, t), pr[k][0], c)
            red = ic.decompose(EvalContext(e[keep], t[keep]), pr[k][0][keep], c[keep])
            for m_ in ["c_all", "c_within", "c_between"]:
                res.append(dict(model=k, metric=m_, full=full[m_], without_candidates=red[m_], n_removed=int((~keep).sum())))
        pd.DataFrame(res).to_csv(paths.tables / "cgga_record_linkage_sensitivity.csv", index=False)
        log.info("\n%s", pd.DataFrame(res).round(4).to_string(index=False))


# ============================================================================ samplesize
def run_samplesize(paths, n_rep=400):
    grid_events = [50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800]
    rows = []
    for coh in ["TCGA_internal", "pooled_seq"]:
        df, pr = ic.cohort_data(paths, coh)
        e, t, c = arrays(df)
        groups = _subgroups(df)
        for g, m in groups.items():
            em, tm_, r = e[m], t[m], pr["B:coxnet"][0][m]
            rate = em.mean()
            rng = np.random.default_rng(GLOBAL_SEED + 71)
            for ev in grid_events:
                n = int(round(ev / rate))
                cs = [harrell_c(em[i], tm_[i], r[i]) for i in (rng.integers(0, m.sum(), n) for _ in range(n_rep))]
                lo, hi = pct(cs)
                rows.append(dict(cohort=coh, group=g, target_events=ev, n=n, c_mean=float(np.mean(cs)), width=hi - lo))
    out = pd.DataFrame(rows)
    summ = []
    for (coh, g), d in out.groupby(["cohort", "group"]):
        d = d.sort_values("target_events")
        ok = d[d.width <= 0.10]
        if len(ok):
            j = d.index.get_loc(ok.index[0])
            if j > 0:
                a, b = d.iloc[j - 1], d.iloc[j]
                ev = a.target_events + (0.10 - a.width) * (b.target_events - a.target_events) / (b.width - a.width)
            else:
                ev = d.iloc[0].target_events
        else:
            ev = np.nan
        summ.append(dict(cohort=coh, group=g, events_for_width_0_10=ev, observed_c=d.c_mean.mean()))
    out.to_csv(paths.tables / "samplesize_within_class.csv", index=False)
    pd.DataFrame(summ).to_csv(paths.tables / "samplesize_within_class_summary.csv", index=False)
    log.info("\n%s\n%s", out.round(3).to_string(index=False), pd.DataFrame(summ).round(1).to_string(index=False))


# ============================================================================ dca
def _km(e, t, h):
    from src.validation import km_survival_at
    return km_survival_at(e, t, h)


def net_benefit(e, t, risk12, pt, h=DAY12):
    hi = risk12 >= pt
    if hi.sum() == 0:
        return 0.0
    s = _km(e[hi], t[hi], h)
    p = hi.mean()
    return (1 - s) * p - s * p * pt / (1 - pt)


def run_dca(paths, n_boot=500):
    thr = np.round(np.arange(0.20, 0.901, 0.05), 2)
    keys = ["B:coxnet", "B:cox", "RAG:cox", "RA:cox"]
    rows = []
    gi = list(GRID_MONTHS).index(12)
    for coh in ["TCGA_internal", "pooled_seq"]:
        df, pr = ic.cohort_data(paths, coh)
        e, t, c = arrays(df)
        m = c == GBM
        em, tm_ = e[m], t[m]
        s_all = _km(em, tm_, DAY12)
        rng = np.random.default_rng(GLOBAL_SEED + 81)
        idx = [rng.integers(0, m.sum(), m.sum()) for _ in range(n_boot)]
        for p_ in thr:
            nb_all = (1 - s_all) - s_all * p_ / (1 - p_)
            rows.append(dict(cohort=coh, model="treat all", threshold=p_, net_benefit=nb_all, lo=np.nan, hi=np.nan,
                             diff_vs_best_default=np.nan, diff_lo=np.nan, diff_hi=np.nan))
            for k in keys:
                r12 = 1 - pr[k][1][m][:, gi]
                nb = net_benefit(em, tm_, r12, p_)
                bs, dd = [], []
                for i in idx:
                    b_nb = net_benefit(em[i], tm_[i], r12[i], p_)
                    s_b = _km(em[i], tm_[i], DAY12)
                    b_all = (1 - s_b) - s_b * p_ / (1 - p_)
                    bs.append(b_nb); dd.append(b_nb - max(b_all, 0.0))
                lo, hi = pct(bs); dlo, dhi = pct(dd)
                rows.append(dict(cohort=coh, model=k, threshold=p_, net_benefit=nb, lo=lo, hi=hi,
                                 diff_vs_best_default=nb - max(nb_all, 0.0), diff_lo=dlo, diff_hi=dhi,
                                 n=int(m.sum()), km_12m=s_all))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "dca_idhwt_12m.csv", index=False)
    log.info("\n%s", out[out.threshold.isin([0.3, 0.5, 0.7])].round(3).to_string(index=False))


# ============================================================================ tuning
def _partial_loglik(X, e, t, beta):
    eta = X @ beta
    o = np.argsort(-t, kind="mergesort")
    eta_o, e_o, t_o = eta[o], e[o], t[o]
    # cumulative log-sum-exp over the risk set (times >= t), Breslow ties
    lse = np.logaddexp.accumulate(eta_o)
    # for tied times use the risk set including all tied subjects
    ut_last = {}
    for k in range(len(t_o)):
        ut_last[t_o[k]] = k
    idx_last = np.array([ut_last[x] for x in t_o])
    return float(np.sum((eta_o - lse[idx_last])[e_o]))


def run_tuning(paths):
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    pooled = ext.cohorts(paths)["pooled_seq"]
    enc = make_encoder("B_clinical_molecular").fit(tcga)
    X = enc.transform(tcga).to_numpy(); e, t = survival_y(tcga); y = Surv.from_arrays(e, t)
    Xc = enc.transform(pooled).to_numpy(); ec, tc = survival_y(pooled)
    cls_c = pooled.integrated_class_harmonised.astype(str).to_numpy()
    base = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100000).fit(X, y)
    alphas = base.alphas_
    strat = tcga.integrated_class_harmonised.astype(str) + "_" + tcga.os_event.astype(str)
    schemes = {"5-fold": list(KFold(5, shuffle=True, random_state=GLOBAL_SEED).split(X)),
               "5-fold stratified by class and event": list(StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED).split(X, strat))}
    res = {}
    for sname, folds in schemes.items():
        C = np.zeros((len(folds), len(alphas))); PL = np.zeros(len(alphas))
        for f, (tr, va) in enumerate(folds):
            m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=alphas, max_iter=100000).fit(X[tr], Surv.from_arrays(e[tr], t[tr]))
            coefs = m.coef_  # (p, n_alphas)
            for j in range(len(alphas)):
                b = coefs[:, j]
                C[f, j] = harrell_c(e[va], t[va], X[va] @ b)
                # Verweij-van Houwelingen cross-validated partial likelihood
                PL[j] += _partial_loglik(X, e, t, b) - _partial_loglik(X[tr], e[tr], t[tr], b)
        res[sname] = (C, PL)
    rules = {}
    C, PL = res["5-fold"]
    mc, sc = C.mean(0), C.std(0, ddof=1) / np.sqrt(C.shape[0])
    jmax = int(np.argmax(mc))
    rules["maximum CV C"] = jmax
    rules["one-standard-error rule, CV C"] = int(np.min(np.where(mc >= mc[jmax] - sc[jmax])[0]))
    rules["maximum CV partial likelihood"] = int(np.argmax(PL))
    Cs, PLs = res["5-fold stratified by class and event"]
    rules["maximum CV C, class-stratified folds"] = int(np.argmax(Cs.mean(0)))
    rules["maximum CV partial likelihood, class-stratified folds"] = int(np.argmax(PLs))
    ctx = EvalContext(ec, tc)
    mut = (pooled.idh_status == "mutant").to_numpy(); gbm = cls_c == GBM
    rows = []
    for rule, j in rules.items():
        a = alphas[j]
        mfit = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[a], max_iter=100000).fit(X, y)
        r = mfit.predict(Xc)
        d = ic.decompose(ctx, r, cls_c)
        rows.append(dict(rule=rule, alpha=a, n_nonzero=int((np.abs(mfit.coef_.ravel()) > 1e-10).sum()),
                         tcga_cv_c=float(C.mean(0)[j]), cgga_c=d["c_all"], cgga_c_within=d["c_within"],
                         cgga_c_idhmut=harrell_c(ec[mut], tc[mut], r[mut]), cgga_c_gbm=harrell_c(ec[gbm], tc[gbm], r[gbm])))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "tuning_rules.csv", index=False)
    log.info("\n%s", out.round(4).to_string(index=False))


# ============================================================================ who2021
def refined_classes(paths, tcga):
    smp = pd.read_csv(paths.tcga_raw / "data_clinical_sample.txt", sep="\t", comment="#")
    smp = smp.drop_duplicates("PATIENT_ID")[["PATIENT_ID", "TERT_PROMOTER_STATUS", "CHR_7_GAIN_CHR_10_LOSS"]]
    cna = pd.read_csv(paths.tcga_raw / "data_cna.txt", sep="\t", index_col=0)
    cna = cna.drop(columns=[c for c in cna.columns if not str(c).startswith("TCGA")])
    cna.columns = [c[:12] for c in cna.columns]
    cna = cna.T.groupby(level=0).first().T
    g = cna.loc[["CDKN2A", "CDKN2B", "EGFR"]].T
    mol = pd.DataFrame({"patient_id": g.index,
                        "cdkn2ab_homdel": ((g.CDKN2A == -2) | (g.CDKN2B == -2)).astype(int),
                        "egfr_amp": (g.EGFR == 2).astype(int)})
    d = tcga.merge(smp.rename(columns={"PATIENT_ID": "patient_id"}), on="patient_id", how="left").merge(mol, on="patient_id", how="left")
    d["tert_mut"] = d.TERT_PROMOTER_STATUS.map({"Mutant": 1, "WT": 0})
    d["chr7_10"] = d.CHR_7_GAIN_CHR_10_LOSS.map({"Gain chr 7 & loss chr 10": 1, "No combined CNA": 0})
    cls = d.integrated_class_harmonised.astype(str).to_numpy().copy()
    g4 = d.grade_harmonised.astype(str) == "4"
    idhwt = cls == GBM
    mol_gbm = (d.tert_mut == 1) | (d.chr7_10 == 1) | (d.egfr_amp == 1)
    all_known_neg = (d.tert_mut == 0) & (d.chr7_10 == 0) & (d.egfr_amp == 0)
    refined = cls.astype(object)
    refined[idhwt & ~g4 & mol_gbm] = GBM
    refined[idhwt & ~g4 & all_known_neg] = "IDHwt_lower_grade_no_GBM_features"
    refined[idhwt & ~g4 & ~mol_gbm & ~all_known_neg] = "IDHwt_lower_grade_unresolved"
    d["refined_class"] = refined
    ast = cls == AST
    d["who2021_grade"] = d.grade_harmonised.astype(str)
    d.loc[ast & (d.cdkn2ab_homdel == 1), "who2021_grade"] = "4"
    return d


def run_who2021(paths):
    tdf, pr = ic.cohort_data(paths, "TCGA_internal")
    d = refined_classes(paths, tdf)
    e, t, _ = arrays(tdf)
    rc = d.refined_class.astype(str).to_numpy()
    counts = d.groupby(["integrated_class_harmonised", "refined_class"]).size().rename("n").reset_index()
    ast = d.integrated_class_harmonised == AST
    counts_ast = pd.crosstab(d.loc[ast, "grade_harmonised"], d.loc[ast, "cdkn2ab_homdel"].fillna(-1)).reset_index()
    rows = []
    for k in ["RA:cox", "RAG:cox", "B:cox", "B:coxnet"]:
        r = pr[k][0]
        dec_old = ic.decompose(EvalContext(e, t), r, tdf.integrated_class_harmonised.astype(str).to_numpy())
        ctx = EvalContext(e, t)
        # refined classes have extra labels; use a generic within/between split
        rr = np.asarray(r, float)
        s = np.where(np.abs(rr[:, None] - rr[None, :]) <= 1e-8, 0.5, (rr[:, None] > rr[None, :]).astype(float)) * ctx.comp
        same = rc[:, None] == rc[None, :]
        cw = s[same].sum() / (ctx.comp & same).sum(); cb = s[~same].sum() / (ctx.comp & ~same).sum()
        rows.append(dict(model=k, group="all, original classes", c_within=dec_old["c_within"], c_between=dec_old["c_between"]))
        rows.append(dict(model=k, group="all, refined classes", c_within=cw, c_between=cb))
        for g, m in [("molecular glioblastoma (refined)", rc == GBM),
                     ("IDH-wildtype grade 4 only", (rc == GBM) & (tdf.grade_harmonised.astype(str) == "4").to_numpy()),
                     ("astrocytoma, IDH-mutant", rc == AST)]:
            rows.append(dict(model=k, group=g, n=int(m.sum()), n_events=int(e[m].sum()), c_within=harrell_c(e[m], t[m], r[m])))
    # does CDKN2A/B add within astrocytoma? repeated 5-fold CV Cox, complete cases
    a = d[ast & d.cdkn2ab_homdel.notna()].reset_index(drop=True).copy()
    a["T"] = a.os_time_days / DAYS_PER_MONTH; a["E"] = a.os_event.astype(int); a["age_per_10y"] = a.age_years / 10
    a["grade3"] = (a.grade_harmonised.astype(str) == "3").astype(int); a["grade4"] = (a.grade_harmonised.astype(str) == "4").astype(int)
    cols = {"age + grade": ["age_per_10y", "grade3", "grade4"],
            "age + grade + CDKN2A/B": ["age_per_10y", "grade3", "grade4", "cdkn2ab_homdel"]}
    cvs = []
    for rep in range(5):
        oof = {k: np.zeros(len(a)) for k in cols}
        for tr, va in KFold(5, shuffle=True, random_state=GLOBAL_SEED + rep).split(a):
            for k, cl in cols.items():
                f = CoxPHFitter(penalizer=0.01).fit(a.loc[tr, ["T", "E"] + cl], "T", "E")
                oof[k][va] = f.predict_partial_hazard(a.loc[va, cl]).to_numpy().ravel()
        cvs.append({k: harrell_c(a.E.astype(bool), a["T"], v) for k, v in oof.items()})
    cvs = pd.DataFrame(cvs).mean()
    f = CoxPHFitter(penalizer=0.01).fit(a[["T", "E"] + cols["age + grade + CDKN2A/B"]], "T", "E")
    hr = f.summary.loc["cdkn2ab_homdel"]
    for k, v in cvs.items():
        rows.append(dict(model=f"Cox within astrocytoma: {k}", group="astrocytoma, IDH-mutant, CDKN2A/B known",
                         n=len(a), n_events=int(a.E.sum()), c_within=v))
    rows.append(dict(model="HR CDKN2A/B homozygous deletion (adjusted for age and grade)", group="astrocytoma, IDH-mutant",
                     n=len(a), n_events=int(a.E.sum()), c_within=np.nan, hr=float(hr["exp(coef)"]),
                     lo=float(hr["exp(coef) lower 95%"]), hi=float(hr["exp(coef) upper 95%"])))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "who2021_refined.csv", index=False)
    counts.to_csv(paths.tables / "who2021_refined_counts.csv", index=False)
    counts_ast.to_csv(paths.tables / "who2021_astrocytoma_cdkn2ab.csv", index=False)
    marker_cov = d[d.integrated_class_harmonised == GBM].assign(lower=lambda x: x.grade_harmonised.astype(str) != "4")
    cov = marker_cov.groupby("lower")[["tert_mut", "chr7_10", "egfr_amp"]].agg(lambda s: f"{int(s.notna().sum())} known, {int((s == 1).sum())} positive")
    cov.to_csv(paths.tables / "who2021_marker_coverage.csv")
    log.info("\n%s\n%s\n%s\n%s", counts.to_string(index=False), counts_ast.to_string(index=False), cov.to_string(),
             out.round(3).to_string(index=False))


STEPS = {"std_diff": run_std_difference, "propensity": run_propensity, "paired_within": run_paired_within,
         "uno": run_uno, "meta": run_meta, "treatment": run_treatment, "duplicates": run_duplicates,
         "samplesize": run_samplesize, "dca": run_dca, "tuning": run_tuning, "who2021": run_who2021,
         "benchmark_ci": run_benchmark_ci}


def main() -> int:
    set_global_seed()
    paths = get_paths()
    for k in (sys.argv[1:] or list(STEPS)):
        log.info("== step %s", k)
        STEPS[k](paths)
    log.info("extensions complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
