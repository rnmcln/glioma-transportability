#!/usr/bin/env python
"""10_sensitivity_analyses.py - prespecified subgroup and sensitivity analyses.

Frozen external discrimination/calibration (point estimates) on CGGA subgroups:
glioblastoma IDH-wildtype only, IDH-mutant non-glioblastoma only, primary tumours
(the primary-analysis default), and exclusion of legacy oligoastrocytoma/NOS.
Small strata are reported descriptively (Section 16). Also evaluates the pooled
deduplicated CGGA cohort.
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.survival_models import MODEL_NAMES  # noqa: E402
from src.utils import HORIZONS_DAYS, get_logger, get_paths, set_global_seed  # noqa: E402
from src.validation import evaluate_arrays  # noqa: E402

log = get_logger("10_sensitivity")
ALL_MODELS = MODEL_NAMES + ["stacked"]
FS = "B_clinical_molecular"


def _load(paths, release):
    key = release.lower().replace("_", "")
    df = pd.read_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet")
    return df[df.included_primary_analysis].reset_index(drop=True)


SUBGROUPS = {
    "all_primary": lambda d: d,
    "gbm_idhwt": lambda d: d[d.integrated_class_harmonised == "glioblastoma_IDHwt"],
    "idhmut_nonGBM": lambda d: d[d.idh_status == "mutant"],
    "exclude_legacy_NOS": lambda d: d[~d.histology_harmonised.isin(["oligoastrocytoma_legacy", "diffuse_glioma_NOS"])],
}


def _pooled_dedup(paths):
    """Pool CGGA mRNAseq_693 and mRNAseq_325; drop duplicate CGGA_IDs (overlap)."""
    a = _load(paths, "mRNAseq_693"); b = _load(paths, "mRNAseq_325")
    pooled = pd.concat([a, b], ignore_index=True)
    pooled = pooled.drop_duplicates(subset="patient_id", keep="first").reset_index(drop=True)
    return pooled


def _row(name, df, model):
    if len(df) < 25 or df.os_event.sum() < 10:
        return dict(model=name, n=len(df), n_events=int(df.os_event.sum()), note="too_small")
    ev = df.os_event.to_numpy().astype(bool); tm = df.os_time_days.to_numpy().astype(float)
    risk = model.risk_score(df); surv = model.survival_at(df, list(HORIZONS_DAYS))
    m = evaluate_arrays(ev, tm, risk, surv)
    return dict(model=name, n=len(df), n_events=int(ev.sum()),
                harrell_c=round(m["harrell_c"], 3), calibration_slope=round(m["calibration_slope"], 3),
                citl_24m=round(m["citl_24m"], 3), ibs=round(m["ibs"], 3), note="")


def main() -> int:
    set_global_seed()
    paths = get_paths()
    mdir = paths.models / FS
    models = {n: pickle.load(open(mdir / f"{n}.pkl", "rb")) for n in ALL_MODELS}

    rows = []
    cohorts = {"mRNAseq_693": _load(paths, "mRNAseq_693"),
               "mRNAseq_325": _load(paths, "mRNAseq_325"),
               "pooled_dedup": _pooled_dedup(paths)}
    for coh, df in cohorts.items():
        for sg, fn in SUBGROUPS.items():
            sub = fn(df).reset_index(drop=True)
            for name in ALL_MODELS:
                r = _row(name, sub, models[name]); r["cohort"] = coh; r["subgroup"] = sg
                rows.append(r)
    out = pd.DataFrame(rows)
    cols = ["cohort", "subgroup", "model", "n", "n_events", "harrell_c", "calibration_slope", "citl_24m", "ibs", "note"]
    out = out.reindex(columns=cols)
    out.to_csv(paths.tables / "table7_sensitivity_subgroups.csv", index=False)
    log.info("wrote table7_sensitivity_subgroups.csv (%d rows)", len(out))
    # brief view
    v = out[(out.model.isin(["coxnet", "stacked"])) & (out.note != "too_small")]
    log.info("\n%s", v[["cohort", "subgroup", "model", "n", "harrell_c", "citl_24m"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
