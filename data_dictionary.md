# Data dictionary

Analytic schema shared by the TCGA development cohort and the CGGA external
validation cohorts. Categorical predictors retain a paired `*_original`
source label. Overall survival time is expressed in days.

| Variable | Description |
| --- | --- |
| `cohort` | Source cohort (TCGA or CGGA). |
| `source_dataset` | Release/provenance identifier (e.g. lgggbm_tcga_pub, mRNAseq_693). |
| `patient_id` | De-identified patient identifier from the source portal. |
| `sample_id` | De-identified sample identifier (equals patient_id where samples are 1:1). |
| `age_years` | Age at diagnosis/surgery in years (continuous). |
| `sex` | Harmonised sex: Female / Male / unknown. |
| `diagnosis_original` | Verbatim source histological diagnosis label. |
| `histology_harmonised` | Harmonised histology: glioblastoma / astrocytoma / oligodendroglioma / oligoastrocytoma_legacy / diffuse_glioma_NOS. |
| `grade_original` | Verbatim source grade label. |
| `grade_harmonised` | WHO grade harmonised to 2 / 3 / 4 / unknown. |
| `idh_status` | IDH mutation status: mutant / wildtype / unknown. |
| `idh_mutation_detail` | Source IDH/codeletion subtype detail where available. |
| `codeletion_1p19q` | 1p/19q codeletion: codeleted / noncodeleted / unknown. |
| `mgmt_promoter_methylated` | MGMT promoter methylation: methylated / unmethylated / unknown. |
| `primary_recurrent_status` | Primary / Recurrent / Secondary / unknown. |
| `radio_status` | Radiotherapy indicator (descriptive only; CGGA). |
| `chemo_status` | Chemotherapy indicator (descriptive only; CGGA). |
| `os_time_days` | Overall survival time in days (TCGA months x 30.4375; CGGA native days). |
| `os_event` | Death indicator: 1 = dead, 0 = censored/alive. |
| `integrated_class_fine` | Granular WHO-like integrated class including grade. |
| `integrated_class_harmonised` | Coarse WHO-like integrated class used as a model predictor. |
| `included_primary_analysis` | Boolean inclusion flag for the primary analysis. |
| `exclusion_reason` | Semicolon-separated reasons for exclusion (empty if included). |

## Time origin and outcome

Overall survival is measured from initial diagnosis. In TCGA the recorded
`OS_MONTHS` is converted to days using 30.4375 days per month; in CGGA the
recorded `OS` is already in days. Possible differences in the survival-clock
origin between cohorts are treated as a limitation and examined in sensitivity
analyses rather than assumed identical.

## Integrated class

`integrated_class_harmonised` is a WHO-like reconstruction from IDH, 1p/19q
and grade. It is not a full WHO 2021 diagnosis because not all defining
molecular criteria are available in both cohorts.