#!/usr/bin/env python
"""13_expression_signature.py - concordance decomposition for gene-expression
signatures developed in TCGA and validated in CGGA.

Design
* Expression: TCGA PanCancer Atlas RNA-seq (LGG and GBM, primary tumour
  samples) and CGGA mRNAseq_693 and mRNAseq_325 RSEM. log2(x + 1); genes
  present in all three sources and with mean log2 expression of at least 2 in
  each; the 2000 genes with the largest variance in TCGA (no outcome used).
  Each gene is standardised within each source (TCGA, 693, 325), which uses
  CGGA expression but not CGGA outcomes.
* Signature: elastic-net Cox model (l1 ratio 0.9) on the 2000 standardised
  genes, penalty chosen by five-fold cross-validated Harrell C in TCGA;
  out-of-fold TCGA scores use nested penalty selection.
* Combined model: Cox model in TCGA with integrated class, age and grade plus
  the (out-of-fold) signature score.
* Random-signature null: 500 random sets of 20 genes from the same pool, each
  fitted by a ridge Cox model in TCGA.
All models are frozen and applied to the pooled CGGA sequencing releases.
"""
from __future__ import annotations

import pathlib
import sys
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.preprocessing import make_encoder  # noqa: E402
from src.utils import GLOBAL_SEED, get_logger, get_paths, set_global_seed  # noqa: E402
from src.validation import EvalContext, harrell_c  # noqa: E402

log = get_logger("13_expr")
ic = import_module("11_interpretation_checks")
ext = import_module("08_external_validation")
GBM, AST, OLI, NOS = ic.CLASSES
N_GENES, MIN_EXPR, N_RANDOM, RANDOM_SIZE = 2000, 2.0, 500, 20


def load_expression(paths):
    raw = paths.data_raw
    tc = []
    for f in ["lgg_pancan_data_mrna_seq_v2_rsem.txt", "gbm_pancan_data_mrna_seq_v2_rsem.txt"]:
        d = pd.read_csv(raw / "tcga" / "expression" / f, sep="\t")
        d = d.dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id"]).groupby("Hugo_Symbol").mean()
        d = d[[c for c in d.columns if c.endswith("-01")]]
        d.columns = [c[:12] for c in d.columns]
        tc.append(d)
    tcga = pd.concat(tc, axis=1)
    tcga = tcga.loc[:, ~tcga.columns.duplicated()]
    cg = {}
    for r in ["mRNAseq_693", "mRNAseq_325"]:
        d = pd.read_csv(raw / "cgga" / "expression" / f"CGGA.{r}.RSEM-genes.20200506.txt", sep="\t")
        cg[r] = d.groupby("Gene_Name").mean()
    return tcga, cg


def prepare(paths):
    tcga_x, cg = load_expression(paths)
    genes = tcga_x.index.intersection(cg["mRNAseq_693"].index).intersection(cg["mRNAseq_325"].index)
    lx = {"TCGA": np.log2(tcga_x.loc[genes].clip(lower=0) + 1)}
    for r, d in cg.items():
        lx[r] = np.log2(d.loc[genes].clip(lower=0) + 1)
    expressed = np.ones(len(genes), bool)
    for d in lx.values():
        expressed &= (d.mean(axis=1) >= MIN_EXPR).to_numpy()
    g2 = genes[expressed]
    var = lx["TCGA"].loc[g2].var(axis=1).sort_values(ascending=False)
    pool = var.index[:N_GENES]
    z = {}
    for k, d in lx.items():
        d = d.loc[pool]
        z[k] = ((d.T - d.mean(axis=1)) / d.std(axis=1).replace(0, 1)).astype(float)
    return z, pool, len(genes), int(expressed.sum())


def _select_alpha(X, e, t, seed):
    base = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.05, n_alphas=30, max_iter=100000).fit(X, Surv.from_arrays(e, t))
    alphas = base.alphas_
    cv = np.zeros(len(alphas))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=alphas, max_iter=100000).fit(X[tr], Surv.from_arrays(e[tr], t[tr]))
        for j in range(len(alphas)):
            cv[j] += harrell_c(e[va], t[va], X[va] @ m.coef_[:, j]) / 5
    return alphas[int(np.argmax(cv))], float(cv.max())


def fit_signature(X, e, t, seed=GLOBAL_SEED):
    a, cvc = _select_alpha(X, e, t, seed)
    m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[a], max_iter=100000).fit(X, Surv.from_arrays(e, t))
    return m, a, cvc


def main() -> int:
    set_global_seed()
    paths = get_paths()
    z, pool, n_common, n_expr = prepare(paths)
    tdf, tpr = ic.cohort_data(paths, "TCGA_internal")
    keep = tdf.patient_id.isin(z["TCGA"].index).to_numpy()
    T = tdf[keep].reset_index(drop=True)
    Xt = z["TCGA"].loc[T.patient_id].to_numpy()
    et, tt = T.os_event.to_numpy().astype(bool), T.os_time_days.to_numpy().astype(float)
    ct = T.integrated_class_harmonised.astype(str).to_numpy()
    rag_t_oof = tpr["RAG:cox"][0][keep]
    log.info("genes common %d, expressed %d, pool %d; TCGA patients with expression %d of %d (%d deaths)",
             n_common, n_expr, len(pool), keep.sum(), len(tdf), et.sum())

    # signature on all TCGA expression patients; out-of-fold scores with nested selection
    sig, alpha, cvc = fit_signature(Xt, et, tt)
    oof = np.zeros(len(T))
    for tr, va in KFold(5, shuffle=True, random_state=GLOBAL_SEED + 1).split(Xt):
        m, *_ = fit_signature(Xt[tr], et[tr], tt[tr], seed=GLOBAL_SEED + 2)
        oof[va] = Xt[va] @ m.coef_.ravel()
    n_sig_genes = int((np.abs(sig.coef_.ravel()) > 1e-10).sum())

    # combined model: class + age + grade + signature score (TCGA, out-of-fold score)
    enc = make_encoder("R_class_age_grade").fit(T)
    Dt = np.c_[enc.transform(T).to_numpy(), (oof - oof.mean()) / oof.std()]
    comb = CoxPHSurvivalAnalysis(alpha=1.0).fit(Dt, Surv.from_arrays(et, tt))
    rag_only = CoxPHSurvivalAnalysis(alpha=1.0).fit(Dt[:, :-1], Surv.from_arrays(et, tt))
    # out-of-fold combined and RAG scores in TCGA (refit within folds)
    comb_oof = np.zeros(len(T)); rag_oof = np.zeros(len(T))
    for tr, va in KFold(5, shuffle=True, random_state=GLOBAL_SEED + 3).split(Dt):
        comb_oof[va] = CoxPHSurvivalAnalysis(alpha=1.0).fit(Dt[tr], Surv.from_arrays(et[tr], tt[tr])).predict(Dt[va])
        rag_oof[va] = CoxPHSurvivalAnalysis(alpha=1.0).fit(Dt[tr, :-1], Surv.from_arrays(et[tr], tt[tr])).predict(Dt[va, :-1])

    # CGGA pooled (693 then 325, as in script 08); release-specific standardisation
    allc = ext.cohorts(paths)
    pooled = allc["pooled_seq"].reset_index(drop=True)
    rel = np.where(pooled.patient_id.isin(set(allc["mRNAseq_693"].patient_id)), "mRNAseq_693", "mRNAseq_325")
    has = np.array([pid in z[r].index for pid, r in zip(pooled.patient_id, rel)])
    P = pooled[has].reset_index(drop=True)
    relP = rel[has]
    Zc = pd.DataFrame(np.vstack([z[r].loc[pid].to_numpy() for pid, r in zip(P.patient_id, relP)]),
                      index=P.patient_id, columns=pool)
    Xc = Zc.loc[P.patient_id].to_numpy()
    ec, tc_ = P.os_event.to_numpy().astype(bool), P.os_time_days.to_numpy().astype(float)
    cc = P.integrated_class_harmonised.astype(str).to_numpy()
    sig_c = Xc @ sig.coef_.ravel()
    sd_t, mu_t = oof.std(), oof.mean()
    Dc = np.c_[enc.transform(P).to_numpy(), (sig_c - mu_t) / sd_t]
    comb_c = comb.predict(Dc); rag_c = rag_only.predict(Dc[:, :-1])
    log.info("CGGA pooled patients with expression %d of %d", len(P), len(pooled))

    scores = {"TCGA": {"signature": oof, "class + age + grade": rag_oof, "class + age + grade + signature": comb_oof},
              "CGGA": {"signature": sig_c, "class + age + grade": rag_c, "class + age + grade + signature": comb_c}}
    data = {"TCGA": (et, tt, ct, T), "CGGA": (ec, tc_, cc, P)}
    rows, prs = [], []
    for coh, (e, t, c, D) in data.items():
        rng = np.random.default_rng(GLOBAL_SEED + 91)
        idx = [rng.integers(0, len(e), len(e)) for _ in range(1000)]
        ctx0 = EvalContext(e, t)
        boots = {k: [] for k in scores[coh]}
        for i in idx:
            ctx = EvalContext(e[i], t[i])
            for k, s in scores[coh].items():
                boots[k].append(ic.decompose(ctx, s[i], c[i]))
        for k, s in scores[coh].items():
            pt = ic.decompose(ctx0, s, c)
            b = pd.DataFrame(boots[k])
            for m in ["c_all", "c_within", "c_between", "share_between", "attrib_between",
                      "c_within_glioblastoma_IDHwt", "c_within_astrocytoma_IDHmut", "c_within_oligodendroglioma_IDHmut_codel"]:
                lo, hi = np.nanpercentile(b[m], [2.5, 97.5])
                rows.append(dict(cohort=coh, model=k, metric=m, point=pt[m], lo=lo, hi=hi, n=len(e), n_events=int(e.sum())))
            g4 = (c == GBM) & (D.grade_harmonised.astype(str) == "4").to_numpy()
            rows.append(dict(cohort=coh, model=k, metric="c_glioblastoma_grade4", point=harrell_c(e[g4], t[g4], s[g4]),
                             lo=np.nan, hi=np.nan, n=int(g4.sum()), n_events=int(e[g4].sum())))
        # paired differences: combined minus class + age + grade, and signature minus class + age + grade
        for a, bref in [("class + age + grade + signature", "class + age + grade"), ("signature", "class + age + grade")]:
            for m in ["c_all", "c_within", "c_within_glioblastoma_IDHwt", "c_within_astrocytoma_IDHmut"]:
                d = np.array(pd.DataFrame(boots[a])[m]) - np.array(pd.DataFrame(boots[bref])[m])
                pa = ic.decompose(ctx0, scores[coh][a], c)[m]; pb = ic.decompose(ctx0, scores[coh][bref], c)[m]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                prs.append(dict(cohort=coh, model=a, reference=bref, metric=m, difference=pa - pb, lo=lo, hi=hi))
        # class recovery by the signature score
        idh = D.idh_status.astype(str).to_numpy()
        mk = idh != "unknown"
        auc_idh = roc_auc_score((idh[mk] == "wildtype").astype(int), scores[coh]["signature"][mk])
        mut = (idh == "mutant") & (D.codeletion_1p19q.astype(str) != "unknown").to_numpy()
        auc_codel = roc_auc_score((D.codeletion_1p19q.astype(str).to_numpy()[mut] == "noncodeleted").astype(int),
                                  scores[coh]["signature"][mut])
        rows.append(dict(cohort=coh, model="signature", metric="auc_idh_wildtype", point=auc_idh, lo=np.nan, hi=np.nan,
                         n=int(mk.sum()), n_events=np.nan))
        rows.append(dict(cohort=coh, model="signature", metric="auc_noncodeleted_within_idh_mutant", point=auc_codel,
                         lo=np.nan, hi=np.nan, n=int(mut.sum()), n_events=np.nan))

    # random-signature null
    rng = np.random.default_rng(GLOBAL_SEED + 92)
    ctx_c = EvalContext(ec, tc_)
    null = []
    for _ in range(N_RANDOM):
        cols = rng.choice(len(pool), RANDOM_SIZE, replace=False)
        m = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xt[:, cols], Surv.from_arrays(et, tt))
        d = ic.decompose(ctx_c, Xc[:, cols] @ m.coef_, cc)
        gb = cc == GBM
        idh = P.idh_status.astype(str).to_numpy(); mk = idh != "unknown"
        null.append(dict(c_all=d["c_all"], c_within=d["c_within"], c_between=d["c_between"],
                         c_within_gbm=d["c_within_glioblastoma_IDHwt"], attrib_between=d["attrib_between"],
                         auc_idh=roc_auc_score((idh[mk] == "wildtype").astype(int), (Xc[:, cols] @ m.coef_)[mk])))
    null = pd.DataFrame(null)
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "expression_signature_decomposition.csv", index=False)
    pd.DataFrame(prs).to_csv(paths.tables / "expression_signature_differences.csv", index=False)
    null.to_csv(paths.tables / "expression_random_signatures.csv", index=False)
    meta = pd.DataFrame([dict(genes_common=n_common, genes_expressed=n_expr, gene_pool=len(pool), tcga_patients=int(keep.sum()),
                              tcga_deaths=int(et.sum()), cgga_patients=len(P), cgga_deaths=int(ec.sum()), alpha=alpha,
                              tcga_cv_c_at_selected=cvc, signature_genes=n_sig_genes)])
    meta.to_csv(paths.tables / "expression_signature_meta.csv", index=False)
    sig_ref = out[(out.cohort == "CGGA") & (out.model == "signature")].set_index("metric").point
    summ = {m: dict(null_median=null[m].median(), null_lo=null[m].quantile(0.025), null_hi=null[m].quantile(0.975),
                    share_null_above_0_75=float((null[m] > 0.75).mean()))
            for m in ["c_all", "c_within", "c_between", "c_within_gbm", "attrib_between", "auc_idh"]}
    pd.DataFrame(summ).T.to_csv(paths.tables / "expression_random_signatures_summary.csv")
    log.info("\n%s\n%s\n%s\n%s", meta.to_string(index=False), out.round(3).to_string(index=False),
             pd.DataFrame(prs).round(3).to_string(index=False), pd.DataFrame(summ).T.round(3).to_string())
    log.info("signature CGGA: %s", sig_ref.round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
