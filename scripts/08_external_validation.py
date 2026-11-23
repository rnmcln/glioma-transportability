#!/usr/bin/env python
"""08_external_validation.py - frozen external validation in each CGGA release.

Every TCGA-trained model is applied unchanged to each CGGA release. Per-patient
risk scores and survival predictions are computed once; a patient-level
bootstrap (500 replicates) evaluates all models on the same resamples, giving
percentile intervals for each metric and paired intervals for differences
between models. The two sensitivity releases are additionally evaluated after
removing patients whose CGGA identifier also occurs in a primary release.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.registry import ALL_KEYS, comparison_pairs, load_models  # noqa: E402
from src.survival_models import predict_all  # noqa: E402
from src.utils import (CGGA_ALL_RELEASES, CGGA_PRIMARY_RELEASES, HORIZON_GRID_IDX,  # noqa: E402
                       get_logger, get_paths, set_global_seed)
from src.validation import bootstrap_panel  # noqa: E402

log = get_logger("08_external")
N_BOOT = 500


def load_cohort(paths, release):
    key = release.lower().replace("_", "")
    df = pd.read_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet")
    return df[df.included_primary_analysis].reset_index(drop=True)


def cohorts(paths) -> dict:
    out = {r: load_cohort(paths, r) for r in CGGA_ALL_RELEASES}
    primary_ids = set(out["mRNAseq_693"].patient_id) | set(out["mRNAseq_325"].patient_id)
    for r in ["mRNA_array_301", "WEseq_286"]:
        d = out[r]
        out[f"{r}_unique"] = d[~d.patient_id.isin(primary_ids)].reset_index(drop=True)
    pooled = pd.concat([out["mRNAseq_693"], out["mRNAseq_325"]], ignore_index=True)
    out["pooled_seq"] = pooled.drop_duplicates("patient_id").reset_index(drop=True)
    return out


def main() -> int:
    set_global_seed()
    paths = get_paths()
    models = load_models(paths, ALL_KEYS)
    sums, difs = [], []
    allc = cohorts(paths)
    for coh, df in allc.items():
        t0 = time.time()
        event = df.os_event.to_numpy().astype(bool)
        time_d = df.os_time_days.to_numpy().astype(float)
        raw = {k: predict_all(m, df) for k, m in models.items()}
        np.savez_compressed(paths.data_processed / f"pred_{coh}.npz",
                            **{f"{k}|risk": v[0] for k, v in raw.items()},
                            **{f"{k}|surv": v[1] for k, v in raw.items()},
                            patient_id=df.patient_id.to_numpy())
        preds = {k: (v[0], v[1][:, list(HORIZON_GRID_IDX)], v[1]) for k, v in raw.items()}
        s, d = bootstrap_panel(event, time_d, preds, n_boot=N_BOOT, pairs=comparison_pairs())
        for x in (s, d):
            x.insert(0, "cohort", coh); x.insert(1, "n", len(df)); x.insert(2, "n_events", int(event.sum()))
        sums.append(s); difs.append(d)
        c = s[(s.metric == "harrell_c") & (s.model.str.startswith("B:"))]
        log.info("%s (n=%d, %d deaths; %.0fs): %s", coh, len(df), event.sum(), time.time() - t0,
                 ", ".join(f"{r.model}={r.point:.3f}" for r in c.itertuples()))
    pd.concat(sums).to_csv(paths.tables / "external_validation_metrics.csv", index=False)
    pd.concat(difs).to_csv(paths.tables / "external_validation_differences.csv", index=False)
    log.info("external validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
