#!/usr/bin/env python
"""10_subgroups_decomposition.py - within-class discrimination and calibration.

1. Prespecified subgroups (glioblastoma IDH-wildtype; IDH-mutant; exclusion of
   legacy oligoastrocytoma/NOS histology; female and male patients) evaluated in TCGA (out-of-fold
   predictions) and in the primary CGGA releases and their pooled set (frozen
   predictions), with bootstrap intervals for Harrell C.
2. Decomposition of Harrell C by integrated molecular class: comparable pairs
   are split into within-class and between-class pairs and C is computed in
   each, together with the share of comparable pairs that are between-class.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.registry import PRIMARY_KEYS, REFERENCE_KEYS  # noqa: E402
from src.utils import GLOBAL_SEED, HORIZON_GRID_IDX, PRIMARY_GRID_IDX, PRIMARY_HORIZON_DAYS, get_logger, get_paths  # noqa: E402
from src.validation import (bootstrap_c_ci, calibration_slope_at, harrell_c,  # noqa: E402
                            harrell_c_pairs, km_survival_at)

log = get_logger("10_subgroups")
SUBGROUPS = {
    "all": lambda d: np.ones(len(d), bool),
    "gbm_idhwt": lambda d: (d.integrated_class_harmonised == "glioblastoma_IDHwt").to_numpy(),
    "idh_mutant": lambda d: (d.idh_status == "mutant").to_numpy(),
    "astro_idhmut": lambda d: (d.integrated_class_harmonised == "astrocytoma_IDHmut").to_numpy(),
    "oligo_codel": lambda d: (d.integrated_class_harmonised == "oligodendroglioma_IDHmut_codel").to_numpy(),
    "female": lambda d: (d.sex == "Female").to_numpy(),
    "male": lambda d: (d.sex == "Male").to_numpy(),
    "exclude_legacy_NOS": lambda d: (~d.histology_harmonised.isin(["oligoastrocytoma_legacy", "diffuse_glioma_NOS"])).to_numpy(),
}
COHORTS = ["TCGA_internal", "mRNAseq_693", "mRNAseq_325", "pooled_seq"]
KEYS = PRIMARY_KEYS + REFERENCE_KEYS


def _load(paths, coh):
    if coh == "TCGA_internal":
        df = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
        z = np.load(paths.data_processed / "oof_tcga.npz", allow_pickle=True)
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from importlib import import_module
        df = import_module("08_external_validation").cohorts(paths)[coh]
        z = np.load(paths.data_processed / f"pred_{coh}.npz", allow_pickle=True)
    assert (z["patient_id"] == df.patient_id.to_numpy()).all()
    return df, {k: (z[f"{k}|risk"], z[f"{k}|surv"]) for k in KEYS}


def main() -> int:
    paths = get_paths()
    sub_rows, dec_rows = [], []
    for coh in COHORTS:
        df, preds = _load(paths, coh)
        ev = df.os_event.to_numpy().astype(bool); tm = df.os_time_days.to_numpy().astype(float)
        cls = df.integrated_class_harmonised.astype(str).to_numpy()
        for sg, fn in SUBGROUPS.items():
            mask = fn(df)
            ok = mask.sum() >= 25 and ev[mask].sum() >= 10
            cis = bootstrap_c_ci(ev[mask], tm[mask], {k: preds[k][0][mask] for k in KEYS}, 1000, GLOBAL_SEED) if ok else {}
            for k in KEYS:
                r, s = preds[k]
                row = dict(cohort=coh, subgroup=sg, model=k, n=int(mask.sum()), n_events=int(ev[mask].sum()))
                if ok:
                    row["harrell_c"] = harrell_c(ev[mask], tm[mask], r[mask])
                    row["harrell_c_lo"], row["harrell_c_hi"] = cis[k]
                    sp = s[mask, PRIMARY_GRID_IDX]
                    row["km_obs_24m"] = km_survival_at(ev[mask], tm[mask], PRIMARY_HORIZON_DAYS)
                    row["pred_mean_24m"] = float(sp.mean())
                    row["citl_24m"] = row["km_obs_24m"] - row["pred_mean_24m"]
                    row["slope_24m"] = calibration_slope_at(ev[mask], tm[mask], sp, PRIMARY_HORIZON_DAYS)
                sub_rows.append(row)
        for k in KEYS:
            d = harrell_c_pairs(ev, tm, preds[k][0], cls)
            dec_rows.append(dict(cohort=coh, model=k, **d))
    sub = pd.DataFrame(sub_rows); dec = pd.DataFrame(dec_rows)
    sub.to_csv(paths.tables / "subgroup_performance.csv", index=False)
    dec.to_csv(paths.tables / "c_index_decomposition.csv", index=False)
    log.info("\n%s", sub[sub.model.isin(["B:coxnet", "B:stacked", "R:cox"])].round(3).to_string(index=False))
    log.info("\n%s", dec.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
