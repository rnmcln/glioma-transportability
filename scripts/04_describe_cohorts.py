#!/usr/bin/env python
"""04_describe_cohorts.py - cohort characteristics, exclusions and release overlap.

Writes the cohort-characteristics table (clinical and molecular, primary
analysis populations), the exclusion counts underlying the study flow diagram,
and the overlap of CGGA identifiers between releases.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils import CGGA_ALL_RELEASES, DAYS_PER_MONTH, get_logger, get_paths  # noqa: E402

log = get_logger("04_describe")


def _frames(paths):
    out = {"TCGA": pd.read_parquet(paths.data_processed / "tcga_harmonised.parquet")}
    for rel in CGGA_ALL_RELEASES:
        key = rel.lower().replace("_", "")
        out[rel] = pd.read_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet")
    return out


def _pct(s, level):
    return 100.0 * float((s.astype(str) == level).mean())


def _summary(df: pd.DataFrame) -> dict:
    inc = df[df.included_primary_analysis]
    age = pd.to_numeric(inc.age_years, errors="coerce")
    t_m = inc.os_time_days / DAYS_PER_MONTH
    km = KaplanMeierFitter().fit(t_m, inc.os_event)
    rkm = KaplanMeierFitter().fit(t_m, 1 - inc.os_event)  # reverse KM for follow-up
    d = {
        "n_total": len(df), "n_included": len(inc), "n_deaths": int(inc.os_event.sum()),
        "age_median": float(age.median()), "age_q1": float(age.quantile(.25)), "age_q3": float(age.quantile(.75)),
        "female_pct": _pct(inc.sex, "Female"),
        "median_os_months": float(km.median_survival_time_),
        "median_followup_months": float(rkm.median_survival_time_),
        "km_os_24m": float(km.predict(24.0)),
    }
    for g in ["2", "3", "4", "unknown"]:
        d[f"grade{g}_pct"] = _pct(inc.grade_harmonised, g)
    for h in ["glioblastoma", "astrocytoma", "oligodendroglioma", "oligoastrocytoma_legacy", "diffuse_glioma_NOS"]:
        d[f"hist_{h}_pct"] = _pct(inc.histology_harmonised, h)
    for c in ["glioblastoma_IDHwt", "astrocytoma_IDHmut", "oligodendroglioma_IDHmut_codel", "diffuse_glioma_NOS"]:
        d[f"class_{c}_pct"] = _pct(inc.integrated_class_harmonised, c)
    for lv in ["mutant", "wildtype", "unknown"]:
        d[f"idh_{lv}_pct"] = _pct(inc.idh_status, lv)
    for lv in ["codeleted", "noncodeleted", "unknown"]:
        d[f"codel_{lv}_pct"] = _pct(inc.codeletion_1p19q, lv)
    for lv in ["methylated", "unmethylated", "unknown"]:
        d[f"mgmt_{lv}_pct"] = _pct(inc.mgmt_promoter_methylated, lv)
    return d


def main() -> int:
    paths = get_paths()
    fr = _frames(paths)
    tab = pd.DataFrame({k: _summary(v) for k, v in fr.items()})
    tab.to_csv(paths.tables / "cohort_characteristics.csv")
    log.info("\n%s", tab.round(1).to_string())

    # Exclusion reasons (non-exclusive) and the first-listed reason per record.
    ex = []
    for k, v in fr.items():
        exc = v[~v.included_primary_analysis]
        reasons = exc.exclusion_reason.str.split(";").explode().value_counts()
        for r, c in reasons.items():
            ex.append(dict(cohort=k, reason=r, n_records=int(c), n_excluded_total=len(exc)))
    pd.DataFrame(ex).to_csv(paths.tables / "exclusions.csv", index=False)

    # Hierarchical exclusion counts (each record counted once, in this order).
    def _cat(reason):
        r = reason.split(";")
        if "non_primary_tumour" in r:
            return "recurrent_or_secondary"
        if any(x in r for x in ["age_missing", "os_time_missing", "os_event_missing"]):
            return "missing_age_or_survival"
        if "os_time_nonpositive" in r:
            return "survival_time_nonpositive"
        return "age_under_18"
    hx = []
    for k, v in fr.items():
        exc = v[~v.included_primary_analysis]
        for c, n in exc.exclusion_reason.map(_cat).value_counts().items():
            hx.append(dict(cohort=k, category=c, n=int(n)))
    pd.DataFrame(hx).to_csv(paths.tables / "exclusions_hierarchical.csv", index=False)

    # Overlap of CGGA identifiers among included patients of each release.
    inc_ids = {r: set(fr[r].loc[fr[r].included_primary_analysis, "patient_id"]) for r in CGGA_ALL_RELEASES}
    all_ids = {r: set(fr[r].patient_id) for r in CGGA_ALL_RELEASES}
    rows = []
    for a, b in itertools.combinations(CGGA_ALL_RELEASES, 2):
        rows.append(dict(release_a=a, release_b=b, shared_ids_all=len(all_ids[a] & all_ids[b]),
                         shared_ids_included=len(inc_ids[a] & inc_ids[b])))
    ov = pd.DataFrame(rows)
    ov.to_csv(paths.tables / "cgga_release_overlap.csv", index=False)
    log.info("\n%s", ov.to_string(index=False))

    # Agreement of overlapping records (age, OS, event) across releases.
    agree = []
    for a, b in itertools.combinations(CGGA_ALL_RELEASES, 2):
        m = fr[a].merge(fr[b], on="patient_id", suffixes=("_a", "_b"))
        if len(m):
            agree.append(dict(release_a=a, release_b=b, n=len(m),
                              age_equal=int((m.age_years_a == m.age_years_b).sum()),
                              os_time_equal=int((m.os_time_days_a == m.os_time_days_b).sum()),
                              os_event_equal=int((m.os_event_a == m.os_event_b).sum()),
                              idh_equal=int((m.idh_status_a == m.idh_status_b).sum())))
    pd.DataFrame(agree).to_csv(paths.tables / "cgga_overlap_agreement.csv", index=False)
    log.info("\n%s", pd.DataFrame(agree).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
