#!/usr/bin/env python
"""05_cohort_shift.py - covariate-shift diagnostics (Section 9).

Standardised mean differences (TCGA vs each CGGA release), a cohort-separation
classifier AUC, and domain-overlap proportions. Outputs feed the Love plot
(Figure 2) and the cohort-shift discussion.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.cohort_shift import cohort_separation_auc, domain_overlap, smd_table  # noqa: E402
from src.utils import CGGA_ALL_RELEASES, get_logger, get_paths, set_global_seed  # noqa: E402

log = get_logger("05_cohort_shift")


def main() -> int:
    set_global_seed()
    paths = get_paths()
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet")

    overlap_rows = []
    for rel in CGGA_ALL_RELEASES:
        key = rel.lower().replace("_", "")
        f = paths.data_processed / f"cgga_{key}_harmonised.parquet"
        if not f.exists():
            continue
        cg = pd.read_parquet(f)
        cg = cg[cg.included_primary_analysis].reset_index(drop=True)

        smd = smd_table(tcga, cg)
        smd.insert(0, "cgga_release", rel)
        smd.to_csv(paths.tables / f"cohort_shift_smd_{rel}.csv", index=False)

        auc = cohort_separation_auc(tcga, cg, "B_clinical_molecular")
        ov = domain_overlap(tcga, cg)
        overlap_rows.append(dict(cgga_release=rel, separation_auc=round(auc, 3),
                                 stratum_overlap=round(ov["stratum_overlap"], 3),
                                 molecular_pattern_overlap=round(ov["molecular_pattern_overlap"], 3),
                                 max_abs_smd=round(float(smd.abs_smd.max()), 3)))
        log.info("%s: separation AUC=%.3f, stratum overlap=%.2f, max|SMD|=%.2f",
                 rel, auc, ov["stratum_overlap"], float(smd.abs_smd.max()))

    pd.DataFrame(overlap_rows).to_csv(paths.tables / "cohort_shift_summary.csv", index=False)
    log.info("wrote cohort_shift_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
