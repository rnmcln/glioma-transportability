# Analysis plan

## Objective

Quantify how overall-survival prediction performance changes when models
developed in TCGA are applied unchanged to CGGA; separate the transport of
discrimination from the transport of calibration; determine how much of the
transported discrimination is carried by the integrated molecular class; and
quantify how much cohort-specific information is needed to restore calibration.

## Design

Secondary analysis of de-identified public data: development of survival
prediction models in one resource, external validation of the frozen models in
another, and cross-validated model updating. No treatment effects are estimated.

## Data

* Development: TCGA `lgggbm_tcga_pub` (cBioPortal DataHub).
* External validation: CGGA `mRNAseq_693` and `mRNAseq_325` (primary);
  `mRNA_array_301` and `WEseq_286` (sensitivity). Each release is analysed
  separately. The two sequencing releases share no CGGA identifiers and are
  also pooled. The sensitivity releases share identifiers with the primary
  releases and are additionally analysed after removing shared patients.

## Population, outcome, predictors

* Adults (>= 18 years) with diffuse glioma, known overall-survival time (> 0)
  and vital status; primary tumours only; one record per patient.
* Outcome: overall survival; landmark horizons 12, 24 and 36 months; primary
  horizon 24 months.
* Feature sets: A, clinical (age, sex, histology, WHO grade); B, clinical plus
  IDH, 1p/19q and MGMT promoter status (primary). Missing marker status is an
  explicit category. Reference models: integrated class only; integrated class
  plus age.

## Modelling

All models are trained in TCGA only: Cox (ridge penalty 1 for numerical
stability), penalised Cox (elastic net, l1 ratio 0.9, penalty selected by
5-fold cross-validated concordance in TCGA), random survival forest, gradient-
boosted survival, and a stacked ensemble (out-of-fold base-learner risk scores
from repeated 5 x 5 cross-validation, ridge Cox meta-learner). CGGA is never
used for fitting, tuning, variable selection or risk-group cut points.

## Validation

* Internal: repeated 5 x 5 cross-validation in TCGA; out-of-fold predictions
  averaged over repeats.
* External: frozen models applied to each CGGA release.
* Metrics: Harrell C; Uno C (truncated at the 90th percentile of event times);
  cumulative/dynamic AUC at 12/24/36 months; integrated Brier score over 1-36
  months; at each horizon, calibration-in-the-large (Kaplan-Meier minus mean
  predicted survival), calibration slope (coefficient of log(-log S_hat(t)) in
  a Cox model with follow-up censored at t) and integrated calibration index
  (ICI, cloglog restricted-cubic-spline Cox model with three knots).
* Uncertainty: 500 patient-level bootstrap replicates, evaluating all models on
  the same resamples so that between-model differences have paired intervals.
* Risk groups: tertiles of the TCGA risk score, transported unchanged.

## Model updating (diagnostic, within CGGA, 5-fold cross-validation)

Stage 0 frozen; 1 recalibration-in-the-large; 2 slope and baseline update;
3 class-specific baselines with a shared slope; 4 full refit in CGGA. Stages
1-3 act on the prognostic index log(-log S_hat(24 months)) of the frozen model;
slopes are estimated with follow-up censored at 24 months.

## Subgroups and sensitivity analyses

Glioblastoma IDH-wildtype; IDH-mutant; astrocytoma IDH-mutant;
oligodendroglioma IDH-mutant 1p/19q-codeleted; exclusion of legacy
oligoastrocytoma and NOS histology; sensitivity releases with and without
shared patients; pooled sequencing releases; 12 and 36 month horizons;
clinical versus clinical-molecular feature sets.

## Decomposition of discrimination

Comparable pairs (Harrell definition) are classified as within-class or
between-class according to the integrated molecular class, and the C-index is
computed separately in each set of pairs. Version 2.1.0 adds, with 1000
bootstrap replicates: the share of pairs between classes and the share of
concordance above 0.5 contributed by these pairs; class-specific within-class
C; a C-index standardised to the TCGA composition of comparable pairs across
class cells (cells involving unknown IDH status excluded); the decomposition
with follow-up censored at 36 months; and the decomposition restricted to
molecularly unambiguous diagnoses (IDH known, 1p/19q known for IDH-mutant
tumours, IDH-wildtype tumours of grade 4 only).

## Interpretation checks (added in 2.1.0, script 11)

* Reference model with integrated class, age and grade.
* Case-mix benchmark: expected overall, within-class and between-class C if
  each frozen model's predicted survival were correct for the patients and
  censoring of each cohort (200 simulations).
* Cox models within each class in TCGA and pooled CGGA (age, MGMT status,
  grade).
* Class-specific calibration-in-the-large and slope at each stage of the
  updating ladder.
* Elastic-net penalty path evaluated in TCGA and pooled CGGA.
* Stability of bootstrap interval bounds (second seed; 2000 replicates).

These analyses were added after the initial plan.

## Further analyses (added in 2.2.0, scripts 12 and 13)

* Differences in C between TCGA and pooled CGGA with independent bootstrap
  intervals; propensity-based standardisation of discrimination; IPCW (Uno)
  decomposition truncated at 36 months; inverse-variance and event-weighted
  summaries of class-specific C.
* Paired within-class differences between full and reference models.
* Case-mix benchmark with bootstrap intervals for observed minus expected C,
  and with the baseline hazard re-estimated in CGGA.
* Treatment and MGMT within IDH-wildtype tumours (CGGA); Karnofsky
  performance score (TCGA); calibration by treatment status.
* Record linkage between CGGA releases on clinical fields.
* Deaths required for a within-class C interval width of 0.10.
* Decision curves for death within 12 months, IDH-wildtype tumours.
* Alternative elastic-net penalty-selection rules.
* Refined WHO 2021 classification in TCGA (TERT, +7/-10, EGFR, CDKN2A/B).
* Expression signature (elastic-net Cox on 2000 genes) trained in TCGA
  RNA-seq and applied to CGGA RSEM data, with a class-age-grade comparator
  and 500 random 20-gene signatures.

## Version history

* 1.0.0: initial pipeline.
* 2.0.0: calibration slope redefined on the log cumulative-hazard scale at the
  landmark horizon (version 1.0.0 standardised the prognostic index within each
  cohort, which measures separation rather than calibration); recalibration
  re-implemented on the unscaled log cumulative-hazard prognostic index;
  integrated Brier score computed on a monthly grid; added ICI, paired bootstrap
  differences, reference models, within/between-class decomposition, subgroup
  intervals and analyses of shared CGGA identifiers.
* 2.1.0: integrated calibration index and E50/E90 require at least 20 events
  before the horizon (smaller subsets gave unstable spline fits); reference
  model with class, age and grade; script 11 with the interpretation checks
  listed above; figures reorganised around the decomposition (script 12).
* 2.2.0: scripts 01b (copy-number and expression downloads), 12
  (extensions) and 13 (expression signature); figure script renumbered 14.
