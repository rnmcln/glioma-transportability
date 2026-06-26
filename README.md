# Glioma survival prediction: TCGA development and CGGA external validation

Reproducible code for a transportability study of overall-survival prediction
models in adult diffuse glioma. Models are developed in TCGA and evaluated,
frozen, in four CGGA releases, with separate assessment of discrimination and
calibration and a staged recalibration analysis.

This repository contains the analysis pipeline only. The manuscript and
supplementary material are distributed separately.

## Summary of findings

- Patient ranking transports well from TCGA to CGGA (internal Harrell C ~0.79;
  external C 0.745–0.799 across releases and model classes).
- Absolute survival predictions are the less stable component; recalibration-in-
  the-large corrects almost all of the average miscalibration.
- Overall discrimination is largely subtype separation: within glioblastoma
  IDH-wildtype the C-index falls to ~0.61.
- A stacked ensemble does not exceed a penalised Cox model on external validation.

## Repository layout

```
src/         analysis modules (data I/O, harmonisation, preprocessing,
             survival models, stacking, validation, calibration, updating,
             cohort-shift diagnostics, plotting, utilities)
scripts/     numbered pipeline stages 00–11 and run_all.py
tests/       unit tests (harmonisation, metrics, no-leakage stacking, smoke)
data_raw/    populated by the download scripts (not tracked)
data_processed/  harmonised analytic tables (not tracked)
outputs/     figures, tables, models, checksums (not tracked)
```

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

`scikit-survival` requires a C/C++ toolchain if no wheel is available for your
platform; `scripts/00_install_sksurv.sh` builds it from source if needed.

## Reproduce

```bash
# 1. Download public data
python scripts/01_download_tcga.py      # TCGA lgggbm_tcga_pub (cBioPortal datahub)
python scripts/02_import_cgga.py        # CGGA clinical releases (CGGA portal)

# 2. Run the full pipeline (harmonise -> train -> validate -> figures)
python scripts/run_all.py
```

`run_all.py` reruns the longer, checkpointed stages (stacked-ensemble training,
internal cross-validation, external validation, updating) until they complete,
so the pipeline tolerates a capped per-process wall-time. Individual stages can
also be run in order (`scripts/03_harmonise.py` … `scripts/11_figures_tables.py`).

Outputs are written to `outputs/figures` and `outputs/tables`. A machine-readable
manifest of source files, sizes, SHA256 checksums and URLs is written to
`outputs/checksums/data_manifest.json`.

## Tests

```bash
pip install pytest
pytest -q
```

## Data sources

- TCGA `lgggbm_tcga_pub` — cBioPortal datahub (merged LGG/GBM cohort).
- CGGA `mRNAseq_693`, `mRNAseq_325`, `mRNA_array_301`, `WEseq_286` —
  https://www.cgga.org.cn.

All data are public and de-identified. This is a secondary analysis; no causal
claims are made. The analysis plan is in `analysis_plan.md` and the variable
schema in `data_dictionary.md`.

## Licence

MIT (see `LICENSE`).
