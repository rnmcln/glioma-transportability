# Discrimination within and across molecular groups in external validation of glioma survival models

**TCGA-CDR reproducibility release v2.3.0.** TCGA-CDR analysis of discrimination within and across molecular groups in external validation in CGGA. Includes the analysis and figure code, manuscript figures, and aggregate supplementary tables. The TCGA development cohort comprises 1,032 adults.

This release contains the code and aggregate results for the manuscript
using overall survival from the [TCGA Pan-Cancer Clinical Data Resource
(TCGA-CDR)](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018).
The original upstream source files are obtained separately under their
respective terms of use.

- **supplementary_data/** contains the exact aggregate CSVs accompanying the manuscript.
- **figures/** contains the manuscript and supplementary figures in PDF format.
- **reproduction/** contains the endpoint preparer and staged runner.
- **scripts/** and **src/** contain the model and evaluation code.

For the endpoint workbook, obtain **Supplemental Table S1** from [Liu et al.,
Cell 2018](https://doi.org/10.1016/j.cell.2018.02.052). The preparer checks the
workbook MD5 (a4591b2dcee39591f59e5e25a6ce75fa). Use its absolute path wherever
/path/to/TCGA-CDR-SupplementalTableS1.xlsx appears below.

Generated intermediate outputs can include exploratory analyses and unfiltered
rows. Use the versioned supplementary_data/ folder for the exact reported
tables. Earlier annotations and analyses are documented at the historical
[source revision](https://github.com/rnmcln/glioma-transportability/tree/3799112).

The MIT licence applies to the software. Cite this version and the associated
article. See CITATION.cff for the software citation.

## Reproduce the analysis

Use separate source-preparation and analysis folders. From a clean copy of
this release, copy the release into a sibling folder before downloading data.
For example, run the following inside an extracted release folder. Keep the
two working copies separate.

Python requirements are in requirements.txt. The endpoint preparation also
requires R with arrow and openxlsx. The reported runs used R 4.3.1,
arrow 20.0.0.2 and openxlsx 4.2.8.1. The Python versions from the
macOS arm64 run are recorded under reproduction/.

```sh
cp -R . ../concordance-analysis
python3 -m venv ../concordance-env
. ../concordance-env/bin/activate
python -m pip install -r requirements.txt
python scripts/01_download_tcga.py
python scripts/01b_download_molecular.py
python scripts/02_import_cgga.py
python scripts/03_harmonise.py
Rscript reproduction/prepare_concordance.R /path/to/TCGA-CDR-SupplementalTableS1.xlsx data_processed ../concordance-analysis ../concordance-analysis/cdr_endpoint_counts.csv
cp -R data_raw/. ../concordance-analysis/data_raw/
python reproduction/run_staged.py concordance ../concordance-analysis
python reproduction/export_cox.py ../concordance-analysis ../concordance-analysis/outputs/tables/cox_coefficients.csv
python reproduction/release_sensitivity.py ../concordance-analysis ../concordance-analysis/outputs/tables/cgga_within_release.csv
mkdir -p ../concordance-analysis/reviewed-figures
python reproduction/redraw_a.py ../concordance-analysis ../concordance-analysis/reviewed-figures
```

See the source revision for details of the cBioPortal and CGGA inputs.
The distributed expression tables omit internal rows for combined models
whose resampling was incompletely nested.
