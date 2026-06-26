#!/usr/bin/env python
"""04_describe_cohorts.py - Table 1 cohort characteristics by cohort/release."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils import CGGA_ALL_RELEASES, DAYS_PER_MONTH, get_logger, get_paths  # noqa: E402

log = get_logger("04_describe")
CATS = ["sex", "grade_harmonised", "histology_harmonised", "idh_status",
        "codeletion_1p19q", "mgmt_promoter_methylated", "integrated_class_harmonised",
        "primary_recurrent_status"]


def _summary(df: pd.DataFrame, label: str) -> dict:
    inc = df[df.included_primary_analysis]
    d = {"cohort": label, "N_total": len(df), "N_included": len(inc)}
    age = pd.to_numeric(inc.age_years, errors="coerce")
    d["age_median"] = round(float(age.median()), 1)
    d["age_IQR"] = f"{age.quantile(.25):.0f}-{age.quantile(.75):.0f}"
    d["event_rate"] = round(float(inc.os_event.mean()), 3)
    d["median_OS_months"] = round(float(inc.os_time_days.median() / DAYS_PER_MONTH), 1)
    # median follow-up among censored (reverse KM approx by median of censored times)
    cens = inc.loc[inc.os_event == 0, "os_time_days"]
    d["median_fu_months_censored"] = round(float(cens.median() / DAYS_PER_MONTH), 1) if len(cens) else np.nan
    for c in CATS:
        vc = inc[c].astype(str).value_counts(normalize=True)
        for lv, frac in vc.items():
            d[f"{c}={lv}"] = round(float(frac), 3)
    return d


def main() -> int:
    paths = get_paths()
    rows = []
    tcga = pd.read_parquet(paths.data_processed / "tcga_harmonised.parquet")
    rows.append(_summary(tcga, "TCGA lgggbm_tcga_pub"))
    for rel in CGGA_ALL_RELEASES:
        key = rel.lower().replace("_", "")
        f = paths.data_processed / f"cgga_{key}_harmonised.parquet"
        if f.exists():
            rows.append(_summary(pd.read_parquet(f), f"CGGA {rel}"))
    out = pd.DataFrame(rows).set_index("cohort").T
    out.to_csv(paths.tables / "table1_cohort_characteristics.csv")
    log.info("wrote table1_cohort_characteristics.csv\n%s",
             out.loc[["N_included", "age_median", "event_rate", "median_OS_months"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
