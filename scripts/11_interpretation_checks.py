#!/usr/bin/env python
"""11_interpretation_checks.py - analyses that bear on the interpretation of
external discrimination and calibration.

1. Decomposition of Harrell's C into within-class and between-class pairs with
   bootstrap intervals, the share of comparable pairs between classes, and the
   share of discrimination above chance (C - 0.5) attributable to between-class
   pairs. Pooled C is reconstructed exactly as the pair-weighted average of the
   two components. Within-class pairs are further split by class.
2. The same decomposition with follow-up administratively censored at 36
   months in every cohort, to remove differences in follow-up length.
3. Analyses restricted to molecularly unambiguous diagnoses (IDH known;
   1p/19q known for IDH-mutant tumours; IDH-wildtype tumours of grade 4 only).
4. Case-mix benchmark (Vergouwe et al. 2010): outcomes are simulated from the
   frozen model's own predicted survival for the patients of each cohort, with
   censoring drawn from the cohort's reverse Kaplan-Meier distribution; the
   C-index expected if the model were correct in that case mix is compared with
   the observed C-index.
5. Penalty path of the elastic-net model: internal cross-validated C in TCGA
   and external C overall and within classes for each candidate penalty.
6. Calibration within classes across the updating ladder (held-out
   predictions), with bootstrap intervals for overall and class-specific
   calibration-in-the-large.
7. Class-specific associations of age, grade and MGMT status with survival in
   TCGA and pooled CGGA (Cox models stratified by class).
8. Stability of bootstrap interval bounds (different seeds and replicate
   numbers) for the primary model.
"""
from __future__ import annotations

import pathlib
import sys
from importlib import import_module

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.preprocessing import make_encoder, survival_y  # noqa: E402
from src.registry import load_models  # noqa: E402
from src.utils import (DAYS_PER_MONTH, GLOBAL_SEED, HORIZON_GRID_IDX, PRIMARY_GRID_IDX,  # noqa: E402
                       PRIMARY_HORIZON_DAYS, get_logger, get_paths, set_global_seed)
from src.validation import (EvalContext, bootstrap_panel, calibration_slope_at,  # noqa: E402
                            harrell_c, km_survival_at)

log = get_logger("12_checks")
ext = import_module("08_external_validation")
CLASSES = ["glioblastoma_IDHwt", "astrocytoma_IDHmut", "oligodendroglioma_IDHmut_codel", "diffuse_glioma_NOS"]
DECOMP_KEYS = ["B:cox", "B:coxnet", "B:rsf", "B:gbsa", "B:stacked", "R:cox", "RA:cox", "RAG:cox"]
N_BOOT = 1000


# ----------------------------------------------------------------------------- helpers
def cohort_data(paths, coh):
    if coh == "TCGA_internal":
        df = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
        z = np.load(paths.data_processed / "oof_tcga.npz", allow_pickle=True)
    else:
        df = ext.cohorts(paths)[coh]
        z = np.load(paths.data_processed / f"pred_{coh}.npz", allow_pickle=True)
    assert (z["patient_id"] == df.patient_id.to_numpy()).all()
    keys = sorted({k.split("|")[0] for k in z.files if "|" in k})
    return df, {k: (z[f"{k}|risk"], z[f"{k}|surv"]) for k in keys}


def _cell_codes(cls):
    ci = np.array([CLASSES.index(c) for c in cls])
    lo = np.minimum(ci[:, None], ci[None, :]); hi = np.maximum(ci[:, None], ci[None, :])
    return lo * len(CLASSES) + hi  # unordered class pair -> integer cell


def decompose(ctx: EvalContext, risk, cls, ref_weights=None):
    """Pair-level decomposition using the Harrell comparable-pair matrix.

    C_all = w * C_within + (1 - w) * C_between, with w the share of comparable
    pairs within one class. ``attrib_between`` is the share of the
    discrimination above chance, i.e. of sum over comparable pairs of
    (score - 0.5), contributed by between-class pairs. If ``ref_weights``
    (array: cell -> share of comparable pairs in a reference cohort) is given,
    ``c_std`` is the C-index obtained by weighting this cohort's cell-specific
    concordance with the reference cohort's pair composition; cells empty in
    either cohort are dropped and the weights renormalised.
    """
    r = np.asarray(risk, float)
    d = r[:, None] - r[None, :]
    score = np.where(np.abs(d) <= 1e-8, 0.5, (d > 0).astype(float))
    cell = _cell_codes(cls)[ctx.comp]
    sc = score[ctx.comp]
    K = len(CLASSES) ** 2
    n_cell = np.bincount(cell, minlength=K).astype(float)
    s_cell = np.bincount(cell, weights=sc, minlength=K)
    diag = np.array([i * len(CLASSES) + i for i in range(len(CLASSES))])
    is_w = np.zeros(K, bool); is_w[diag] = True
    n_all = n_cell.sum(); nw = n_cell[is_w].sum(); nb = n_all - nw
    sw = s_cell[is_w].sum(); sb = s_cell[~is_w].sum()
    out = {"c_all": (sw + sb) / n_all,
           "c_within": sw / nw if nw else np.nan,
           "c_between": sb / nb if nb else np.nan,
           "share_between": nb / n_all}
    out["c_reconstructed"] = (1 - out["share_between"]) * out["c_within"] + out["share_between"] * out["c_between"]
    out["share_concordance_between"] = sb / (sw + sb)
    out["attrib_between"] = (sb - 0.5 * nb) / (sw + sb - 0.5 * n_all)
    for i, c in enumerate(CLASSES):
        k = n_cell[diag[i]]
        out[f"c_within_{c}"] = s_cell[diag[i]] / k if k else np.nan
        out[f"pairs_within_{c}"] = int(k)
    out["pairs_within"] = int(nw); out["pairs_between"] = int(nb)
    if ref_weights is not None:
        ok = (n_cell > 0) & (ref_weights > 0)
        cc = np.divide(s_cell, n_cell, out=np.zeros(K), where=n_cell > 0)
        out["c_std"] = (ref_weights[ok] * cc[ok]).sum() / ref_weights[ok].sum() if ok.any() else np.nan
    return out


def cell_weights(event, time, cls):
    """Pair composition of the reference cohort over cells of the three defined
    classes. Cells involving the not-otherwise-specified class (unknown IDH)
    get weight zero, because its frequency reflects annotation completeness
    (11% in TCGA, under 1% in mRNAseq_325) and its cells are near-empty in some
    validation cohorts."""
    ctx = EvalContext(np.asarray(event).astype(bool), np.asarray(time, float))
    cell = _cell_codes(np.asarray(cls).astype(str))[ctx.comp]
    n = np.bincount(cell, minlength=len(CLASSES) ** 2).astype(float)
    nos = CLASSES.index("diffuse_glioma_NOS")
    for a in range(len(CLASSES)):
        lo, hi = min(a, nos), max(a, nos)
        n[lo * len(CLASSES) + hi] = 0.0
    return n / n.sum()


def decomposition_table(event, time, cls, risks: dict, n_boot=N_BOOT, seed=GLOBAL_SEED, ref_weights=None):
    rng = np.random.default_rng(seed)
    event = np.asarray(event).astype(bool); time = np.asarray(time, float); cls = np.asarray(cls).astype(str)
    ctx0 = EvalContext(event, time)
    base = {k: decompose(ctx0, r, cls, ref_weights) for k, r in risks.items()}
    boots = {k: [] for k in risks}
    n = len(event)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ctx = EvalContext(event[i], time[i])
        for k, r in risks.items():
            boots[k].append(decompose(ctx, np.asarray(r)[i], cls[i], ref_weights))
    rows = []
    for k in risks:
        b = pd.DataFrame(boots[k])
        for m, v in base[k].items():
            lo, hi = (np.nanpercentile(b[m], [2.5, 97.5]) if m in b and not m.startswith("pairs") else (np.nan, np.nan))
            rows.append(dict(model=k, metric=m, point=v, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def censor_at(event, time, t):
    e = np.asarray(event).astype(bool) & (np.asarray(time, float) <= t)
    return e, np.minimum(np.asarray(time, float), t)


# ----------------------------------------------------------------------------- 1-3 decompositions
def _variant(df, variant):
    ev = df.os_event.to_numpy().astype(bool); tm = df.os_time_days.to_numpy().astype(float)
    if variant == "full_follow_up":
        return ev, tm, np.ones(len(df), bool)
    if variant == "censored_36m":
        e, t = censor_at(ev, tm, 36 * DAYS_PER_MONTH)
        return e, t, np.ones(len(df), bool)
    m = ((df.idh_status != "unknown")
         & ~((df.idh_status == "mutant") & (df.codeletion_1p19q == "unknown"))
         & ~((df.idh_status == "wildtype") & (df.grade_harmonised != "4"))).to_numpy()
    return ev[m], tm[m], m


def run_decompositions(paths):
    rows = []
    variants = ["full_follow_up", "censored_36m", "unambiguous"]
    tdf, _ = cohort_data(paths, "TCGA_internal")
    tcls = tdf.integrated_class_harmonised.astype(str).to_numpy()
    ref = {}
    for v in variants:
        e, t, m = _variant(tdf, v)
        ref[v] = cell_weights(e, t, tcls[m])
    for coh in ["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]:
        df, pr = cohort_data(paths, coh)
        cls = df.integrated_class_harmonised.astype(str).to_numpy()
        risks = {k: pr[k][0] for k in DECOMP_KEYS}
        for variant in variants:
            e, t, m = _variant(df, variant)
            tab = decomposition_table(e, t, cls[m], {k: r[m] for k, r in risks.items()}, ref_weights=ref[variant])
            tab.insert(0, "variant", variant); tab.insert(0, "cohort", coh)
            tab["n"] = int(m.sum()); tab["n_events"] = int(np.asarray(e).sum())
            rows.append(tab)
            log.info("%s %s n=%d done", coh, variant, m.sum())
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(paths.tables / "c_decomposition_inference.csv", index=False)
    v = out[(out.model == "B:coxnet") & out.metric.isin(["c_all", "c_reconstructed", "c_within", "c_between", "share_between", "attrib_between", "c_std"])]
    log.info("\n%s", v.pivot_table(index=["cohort", "variant"], columns="metric", values="point").round(3).to_string())
    return out


# ----------------------------------------------------------------------------- 4 case-mix benchmark
def simulate_c(model, df, n_sim=200, seed=GLOBAL_SEED, max_months=240, step_months=0.25):
    """Expected Harrell C (overall, within, between) if the model were correct."""
    rng = np.random.default_rng(seed)
    grid = np.arange(step_months, max_months + step_months / 2, step_months) * DAYS_PER_MONTH
    S = model.survival_at(df, list(grid))  # (n, G), non-increasing
    risk = np.asarray(model.risk_score(df)).ravel()
    cls = df.integrated_class_harmonised.astype(str).to_numpy()
    ev = df.os_event.to_numpy().astype(bool); tm = df.os_time_days.to_numpy().astype(float)
    # reverse Kaplan-Meier censoring distribution of the cohort
    ut = np.unique(tm)
    n_at = np.array([(tm >= u).sum() for u in ut])
    d_c = np.array([((tm == u) & ~ev).sum() for u in ut])
    d_e = np.array([((tm == u) & ev).sum() for u in ut])
    Gc = np.cumprod(1 - d_c / np.maximum(n_at - d_e, 1))
    res = []
    n = len(df)
    for _ in range(n_sim):
        u = rng.uniform(size=n)
        # event time: first grid time with S <= u; beyond grid -> no event within max_months
        idx = (S <= u[:, None]).argmax(axis=1)
        has = (S <= u[:, None]).any(axis=1)
        t_ev = np.where(has, grid[idx], np.inf)
        v = rng.uniform(size=n)
        # censoring time: inverse of reverse KM; beyond support -> last observed time
        j = np.searchsorted(-Gc, -v, side="left")
        t_c = np.where(j < len(ut), ut[np.minimum(j, len(ut) - 1)], ut[-1])
        e = t_ev <= t_c
        t = np.minimum(t_ev, t_c)
        ctx = EvalContext(e, t)
        res.append(decompose(ctx, risk, cls))
    r = pd.DataFrame(res)
    return {m: (r[m].mean(), *np.nanpercentile(r[m], [2.5, 97.5])) for m in ["c_all", "c_within", "c_between", "share_between"]}


def run_benchmark(paths):
    models = load_models(paths, ["B:coxnet", "B:cox", "RA:cox"])
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    cohorts = {"TCGA_apparent": tcga, **{c: ext.cohorts(paths)[c] for c in ["mRNAseq_693", "mRNAseq_325", "pooled_seq"]}}
    rows = []
    for coh, df in cohorts.items():
        ev = df.os_event.to_numpy().astype(bool); tm = df.os_time_days.to_numpy().astype(float)
        cls = df.integrated_class_harmonised.astype(str).to_numpy()
        for k, m in models.items():
            obs = decompose(EvalContext(ev, tm), m.risk_score(df), cls)
            sim = simulate_c(m, df)
            for met in ["c_all", "c_within", "c_between", "share_between"]:
                rows.append(dict(cohort=coh, model=k, metric=met, observed=obs[met], expected=sim[met][0],
                                 expected_lo=sim[met][1], expected_hi=sim[met][2]))
        log.info("benchmark %s done", coh)
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "casemix_benchmark.csv", index=False)
    log.info("\n%s", out.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- 5 penalty path
def run_penalty_path(paths):
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    pooled = ext.cohorts(paths)["pooled_seq"]
    enc = make_encoder("B_clinical_molecular").fit(tcga)
    X = enc.transform(tcga).to_numpy(); e, t = survival_y(tcga); y = Surv.from_arrays(e, t)
    Xc = enc.transform(pooled).to_numpy(); ec, tc = survival_y(pooled)
    base = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100000).fit(X, y)
    alphas = base.alphas_[:: max(1, len(base.alphas_) // 12)]
    alphas = np.append(alphas, base.alphas_[-1])  # include the smallest penalty on the path
    cls = pooled.integrated_class_harmonised.astype(str).to_numpy()
    ctx = EvalContext(ec, tc)
    rows = []
    kf = KFold(5, shuffle=True, random_state=GLOBAL_SEED)
    for a in alphas:
        cv = []
        for tr, va in kf.split(X):
            m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[a], max_iter=100000).fit(X[tr], Surv.from_arrays(e[tr], t[tr]))
            cv.append(harrell_c(e[va], t[va], m.predict(X[va])))
        m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[a], max_iter=100000).fit(X, y)
        r = m.predict(Xc)
        d = decompose(ctx, r, cls)
        mut = (pooled.idh_status == "mutant").to_numpy()
        gbm = cls == "glioblastoma_IDHwt"
        rows.append(dict(alpha=a, n_nonzero=int((np.abs(m.coef_.ravel()) > 1e-10).sum()), tcga_cv_c=np.mean(cv),
                         cgga_c=d["c_all"], cgga_c_within=d["c_within"], cgga_c_between=d["c_between"],
                         cgga_c_idhmut=harrell_c(ec[mut], tc[mut], r[mut]),
                         cgga_c_gbm=harrell_c(ec[gbm], tc[gbm], r[gbm])))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "penalty_path.csv", index=False)
    log.info("\n%s", out.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- 6 class-specific recalibration
def run_class_calibration(paths, n_boot=500):
    z = np.load(paths.data_processed / "updating_predictions.npz")
    rows = []
    rng = np.random.default_rng(GLOBAL_SEED)
    for coh in ["mRNAseq_693", "mRNAseq_325"]:
        df = ext.load_cohort(paths, coh)
        ev = df.os_event.to_numpy().astype(bool); tm = df.os_time_days.to_numpy().astype(float)
        cls = df.integrated_class_harmonised.astype(str).to_numpy()
        groups = {"all": np.ones(len(df), bool), "glioblastoma_IDHwt": cls == "glioblastoma_IDHwt",
                  "IDH_mutant": (df.idh_status == "mutant").to_numpy()}
        idx_b = [rng.integers(0, len(df), len(df)) for _ in range(n_boot)]
        for model in ["B:coxnet", "B:cox"]:
            for stage in ["0_frozen", "1_in_large", "2_slope", "3_revision"]:
                S = z[f"{coh}|{model}|{stage}"][:, PRIMARY_GRID_IDX]
                for g, m in groups.items():
                    def citl(i):
                        return km_survival_at(ev[i], tm[i], PRIMARY_HORIZON_DAYS) - S[i].mean()
                    pt = citl(np.where(m)[0])
                    bs = []
                    for i in idx_b:
                        ii = i[m[i]]
                        bs.append(citl(ii) if len(ii) > 10 else np.nan)
                    rows.append(dict(cohort=coh, model=model, stage=stage, group=g, n=int(m.sum()),
                                     citl_24m=pt, citl_lo=np.nanpercentile(bs, 2.5), citl_hi=np.nanpercentile(bs, 97.5),
                                     slope_24m=calibration_slope_at(ev[m], tm[m], S[m], PRIMARY_HORIZON_DAYS)))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "class_specific_recalibration.csv", index=False)
    log.info("\n%s", out[out.model == "B:coxnet"].round(3).to_string(index=False))


# ----------------------------------------------------------------------------- 7 class-specific associations
def run_associations(paths):
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet")
    pooled = ext.cohorts(paths)["pooled_seq"]
    rows = []
    for coh, df in [("TCGA", tcga), ("CGGA pooled", pooled)]:
        for c in ["glioblastoma_IDHwt", "astrocytoma_IDHmut", "oligodendroglioma_IDHmut_codel"]:
            d = df[df.integrated_class_harmonised == c].copy()
            X = pd.DataFrame({"T": d.os_time_days / DAYS_PER_MONTH, "E": d.os_event.astype(int),
                              "age_per_10y": d.age_years / 10})
            if c == "glioblastoma_IDHwt":
                X["mgmt_methylated"] = (d.mgmt_promoter_methylated == "methylated").astype(int)
                X["mgmt_unknown"] = (d.mgmt_promoter_methylated == "unknown").astype(int)
            else:
                X["grade_3_or_4"] = d.grade_harmonised.isin(["3", "4"]).astype(int)
            X = X.loc[:, X.nunique() > 1]
            try:
                f = CoxPHFitter(penalizer=0.001).fit(X, "T", "E")
            except Exception as exc:  # pragma: no cover
                log.warning("fit failed %s %s: %s", coh, c, exc); continue
            for v in f.params_.index:
                rows.append(dict(cohort=coh, integrated_class=c, n=len(d), n_events=int(d.os_event.sum()), term=v,
                                 hr=float(np.exp(f.params_[v])), lo=float(np.exp(f.confidence_intervals_.loc[v].iloc[0])),
                                 hi=float(np.exp(f.confidence_intervals_.loc[v].iloc[1]))))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "class_specific_associations.csv", index=False)
    log.info("\n%s", out.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- 8 bootstrap stability
def run_bootstrap_stability(paths):
    rows = []
    for coh in ["mRNAseq_693", "mRNAseq_325"]:
        df, pr = cohort_data(paths, coh)
        ev = df.os_event.to_numpy(); tm = df.os_time_days.to_numpy()
        r, g = pr["B:coxnet"]
        preds = {"B:coxnet": (r, g[:, list(HORIZON_GRID_IDX)], g)}
        for label, nb, seed in [("500, seed A", 500, GLOBAL_SEED), ("500, seed B", 500, 7), ("2000, seed A", 2000, GLOBAL_SEED)]:
            s, _ = bootstrap_panel(ev, tm, preds, n_boot=nb, seed=seed, with_ici="primary")
            for m in ["harrell_c", "citl_24m", "slope_24m", "ici_24m"]:
                x = s[s.metric == m].iloc[0]
                rows.append(dict(cohort=coh, replicates=label, metric=m, point=x.point, lo=x.lo, hi=x.hi))
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "bootstrap_stability.csv", index=False)
    log.info("\n%s", out.round(3).to_string(index=False))


def main() -> int:
    set_global_seed()
    paths = get_paths()
    steps = {"decomposition": run_decompositions, "benchmark": run_benchmark, "penalty": run_penalty_path,
             "class_calibration": run_class_calibration, "associations": run_associations,
             "bootstrap_stability": run_bootstrap_stability}
    todo = sys.argv[1:] or list(steps)
    for k in todo:
        steps[k](paths)
    log.info("interpretation checks complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
