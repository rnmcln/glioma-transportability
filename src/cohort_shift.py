"""Covariate-shift diagnostics between TCGA and CGGA (Section 9).

These are descriptive comparisons of case mix and a covariate-shift classifier,
not causal analyses. A high cohort-separation AUC indicates that baseline
characteristics differ enough to predict cohort membership, which makes
transportability failure unsurprising rather than anomalous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from .preprocessing import FEATURE_SETS, make_encoder
from .utils import DAYS_PER_MONTH

CONT_VARS = ["age_years"]
CAT_VARS = ["sex", "histology_harmonised", "grade_harmonised", "idh_status",
            "codeletion_1p19q", "mgmt_promoter_methylated", "integrated_class_harmonised"]


def _smd_cont(a, b) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s else float("nan")


def _smd_prop(p1, p2) -> float:
    s = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float((p1 - p2) / s) if s else 0.0


def smd_table(tcga: pd.DataFrame, cgga: pd.DataFrame) -> pd.DataFrame:
    """Standardised mean differences for each variable/level (TCGA vs CGGA)."""
    rows = []
    for v in CONT_VARS:
        rows.append(dict(variable=v, level="(mean)",
                         tcga=round(float(pd.to_numeric(tcga[v], errors="coerce").mean()), 2),
                         cgga=round(float(pd.to_numeric(cgga[v], errors="coerce").mean()), 2),
                         smd=_smd_cont(tcga[v], cgga[v])))
    for v in CAT_VARS:
        levels = sorted(set(tcga[v].astype(str)) | set(cgga[v].astype(str)))
        for lv in levels:
            p1 = float((tcga[v].astype(str) == lv).mean())
            p2 = float((cgga[v].astype(str) == lv).mean())
            rows.append(dict(variable=v, level=lv, tcga=round(p1, 3), cgga=round(p2, 3),
                             smd=_smd_prop(p1, p2)))
    # Outcome / follow-up summaries.
    rows.append(dict(variable="os_event_rate", level="(rate)",
                     tcga=round(float(tcga.os_event.mean()), 3),
                     cgga=round(float(cgga.os_event.mean()), 3),
                     smd=_smd_prop(float(tcga.os_event.mean()), float(cgga.os_event.mean()))))
    rows.append(dict(variable="median_os_months", level="(median)",
                     tcga=round(float(tcga.os_time_days.median() / DAYS_PER_MONTH), 1),
                     cgga=round(float(cgga.os_time_days.median() / DAYS_PER_MONTH), 1),
                     smd=float("nan")))
    out = pd.DataFrame(rows)
    out["abs_smd"] = out.smd.abs()
    return out


def cohort_separation_auc(tcga: pd.DataFrame, cgga: pd.DataFrame,
                          feature_set: str = "B_clinical_molecular", seed: int = 0) -> float:
    """Cross-validated AUC of a classifier predicting cohort membership.

    A random forest is used rather than logistic regression because the one-hot
    design with imbalanced cohort sizes produces quasi-complete separation that
    destabilises an unregularised logistic model; the forest gives a stable
    covariate-shift estimate.
    """
    df = pd.concat([tcga.assign(_y=0), cgga.assign(_y=1)], ignore_index=True)
    enc = make_encoder(feature_set).fit(df)
    X = enc.transform(df).to_numpy()
    y = df._y.to_numpy()
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                 random_state=seed, n_jobs=4)
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def domain_overlap(tcga: pd.DataFrame, cgga: pd.DataFrame) -> dict:
    """Proportion of CGGA patients whose integrated-class x grade stratum and
    molecular-status pattern is also present in TCGA."""
    def key(df):
        return (df.integrated_class_harmonised.astype(str) + "|" + df.grade_harmonised.astype(str))
    seen = set(key(tcga))
    cg = key(cgga)
    in_tcga = cg.isin(seen).mean()
    # Molecular-pattern overlap (IDH x codel x MGMT).
    def molkey(df):
        return (df.idh_status.astype(str) + "|" + df.codeletion_1p19q.astype(str)
                + "|" + df.mgmt_promoter_methylated.astype(str))
    seen_m = set(molkey(tcga))
    mol_overlap = molkey(cgga).isin(seen_m).mean()
    return dict(stratum_overlap=float(in_tcga), molecular_pattern_overlap=float(mol_overlap))
