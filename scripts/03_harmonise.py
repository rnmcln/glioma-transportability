#!/usr/bin/env python
"""03_harmonise.py - harmonise TCGA and CGGA to the common analytic schema.

Writes per-cohort harmonised parquet tables, the combined analytic tables used
for development (TCGA primary analysis) and external validation (CGGA releases),
and a machine-generated data dictionary.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_io import load_cgga_raw, load_tcga_raw  # noqa: E402
from src.harmonisation import ANALYTIC_COLUMNS, harmonise_cgga, harmonise_tcga  # noqa: E402
from src.utils import CGGA_ALL_RELEASES, get_logger, get_paths, set_global_seed  # noqa: E402

log = get_logger("03_harmonise")


def main() -> int:
    set_global_seed()
    paths = get_paths()

    tcga = harmonise_tcga(load_tcga_raw(paths))
    tcga.to_parquet(paths.data_processed / "tcga_harmonised.parquet", index=False)

    cgga_frames = {}
    for release in CGGA_ALL_RELEASES:
        try:
            h = harmonise_cgga(load_cgga_raw(paths, release), release)
        except FileNotFoundError as exc:
            log.warning("Skipping %s: %s", release, exc)
            continue
        key = release.lower().replace("_", "")
        h.to_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet", index=False)
        cgga_frames[release] = h

    # Primary analytic development set: TCGA included rows.
    analytic_tcga = tcga[tcga.included_primary_analysis].reset_index(drop=True)
    analytic_tcga.to_parquet(paths.data_processed / "analytic_primary_tcga.parquet", index=False)

    # External validation set: all CGGA releases stacked (release retained).
    if cgga_frames:
        analytic_cgga = pd.concat(cgga_frames.values(), ignore_index=True)
        analytic_cgga.to_parquet(paths.data_processed / "analytic_external_cgga.parquet", index=False)

    # Console summary of analytic counts.
    log.info("ANALYTIC COUNTS (included_primary_analysis):")
    log.info("  TCGA lgggbm_tcga_pub: %d / %d", len(analytic_tcga), len(tcga))
    for rel, h in cgga_frames.items():
        log.info("  CGGA %s: %d / %d", rel, int(h.included_primary_analysis.sum()), len(h))

    _write_data_dictionary(paths, ANALYTIC_COLUMNS)
    return 0


def _write_data_dictionary(paths, columns) -> None:
    descriptions = {
        "cohort": "Source cohort (TCGA or CGGA).",
        "source_dataset": "Release/provenance identifier (e.g. lgggbm_tcga_pub, mRNAseq_693).",
        "patient_id": "De-identified patient identifier from the source portal.",
        "sample_id": "De-identified sample identifier (equals patient_id where samples are 1:1).",
        "age_years": "Age at diagnosis/surgery in years (continuous).",
        "sex": "Harmonised sex: Female / Male / unknown.",
        "diagnosis_original": "Verbatim source histological diagnosis label.",
        "histology_harmonised": "Harmonised histology: glioblastoma / astrocytoma / oligodendroglioma / oligoastrocytoma_legacy / diffuse_glioma_NOS.",
        "grade_original": "Verbatim source grade label.",
        "grade_harmonised": "WHO grade harmonised to 2 / 3 / 4 / unknown.",
        "idh_status": "IDH mutation status: mutant / wildtype / unknown.",
        "idh_mutation_detail": "Source IDH/codeletion subtype detail where available.",
        "codeletion_1p19q": "1p/19q codeletion: codeleted / noncodeleted / unknown.",
        "mgmt_promoter_methylated": "MGMT promoter methylation: methylated / unmethylated / unknown.",
        "primary_recurrent_status": "Primary / Recurrent / Secondary / unknown.",
        "radio_status": "Radiotherapy indicator (descriptive only; CGGA).",
        "chemo_status": "Chemotherapy indicator (descriptive only; CGGA).",
        "os_time_days": "Overall survival time in days (TCGA months x 30.4375; CGGA native days).",
        "os_event": "Death indicator: 1 = dead, 0 = censored/alive.",
        "integrated_class_fine": "Granular WHO-like integrated class including grade.",
        "integrated_class_harmonised": "Coarse WHO-like integrated class used as a model predictor.",
        "included_primary_analysis": "Boolean inclusion flag for the primary analysis.",
        "exclusion_reason": "Semicolon-separated reasons for exclusion (empty if included).",
    }
    lines = [
        "# Data dictionary",
        "",
        "Analytic schema shared by the TCGA development cohort and the CGGA external",
        "validation cohorts. Categorical predictors retain a paired `*_original`",
        "source label. Overall survival time is expressed in days.",
        "",
        "| Variable | Description |",
        "| --- | --- |",
    ]
    for c in columns:
        lines.append(f"| `{c}` | {descriptions.get(c, '')} |")
    lines += [
        "",
        "## Time origin and outcome",
        "",
        "Overall survival is measured from initial diagnosis. In TCGA the recorded",
        "`OS_MONTHS` is converted to days using 30.4375 days per month; in CGGA the",
        "recorded `OS` is already in days. Possible differences in the survival-clock",
        "origin between cohorts are treated as a limitation and examined in sensitivity",
        "analyses rather than assumed identical.",
        "",
        "## Integrated class",
        "",
        "`integrated_class_harmonised` is a WHO-like reconstruction from IDH, 1p/19q",
        "and grade. It is not a full WHO 2021 diagnosis because not all defining",
        "molecular criteria are available in both cohorts.",
    ]
    (paths.root / "data_dictionary.md").write_text("\n".join(lines))
    log.info("Wrote data_dictionary.md")


if __name__ == "__main__":
    raise SystemExit(main())
