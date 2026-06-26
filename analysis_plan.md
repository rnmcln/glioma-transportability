# Analysis plan

## Objective

Quantify how much glioma overall-survival prediction performance changes when
models developed in TCGA are transported to CGGA, whether model complexity
improves transportability relative to a penalised Cox model, and whether limited
external recalibration restores performance without full refitting.

The primary inferential object is not the internally cross-validated performance
of a survival model in TCGA, but the degradation, preservation or recoverability
of performance when the fitted model is transported to CGGA. External
discrimination, external calibration, and the effect of limited recalibration
are therefore the main results. The stacked ensemble is included as a stress
test of algorithmic complexity: if it does not materially improve transported
performance over a penalised Cox model, this is evidence against the assumption
that more flexible algorithms solve the principal limitation of public glioma
survival modelling.

## Design

Secondary analysis of de-identified public datasets; a prediction-model
development, external evaluation and updating study. No treatment effects or
causal claims. Reported following TRIPOD+AI.

## Data

* Development: TCGA `lgggbm_tcga_pub` (cBioPortal datahub).
* External validation: CGGA `mRNAseq_693` and `mRNAseq_325` (primary);
  `mRNA_array_301` and `WEseq_286` (sensitivity). Each release is treated as a
  separate external-validation stratum; a deduplicated pool is secondary.

## Population, outcome, predictors

* Adults (>=18 y) with diffuse glioma, known overall-survival time (>0) and
  status; primary tumours in the primary analysis. One index sample per patient.
* Outcome: overall survival from diagnosis; landmark horizons 12/24/36 months;
  primary horizon 24 months.
* Feature sets: A clinical core; B clinical + canonical molecular (primary);
  C integrated WHO-like class. Treatment variables are descriptive only.

## Modelling

All models trained only in TCGA: Cox, penalised Cox (elastic net, primary
benchmark), random survival forest, gradient-boosted survival, and a
leakage-controlled stacked ensemble (out-of-fold base scores -> penalised-Cox
meta-learner). Molecular missingness handled as an explicit category; CGGA is
never used for fitting, tuning, or risk-group cut points.

## Validation and updating

* Internal: repeated 5x5 cross-validation in TCGA (secondary).
* External (primary): frozen models applied to each CGGA release; Uno/Harrell C,
  integrated Brier, time-dependent AUC, calibration slope,
  calibration-in-the-large, observed-vs-predicted survival; patient-clustered
  bootstrap CIs (>=500). Risk groups defined by TCGA-derived tertiles,
  transported unchanged.
* Updating ladder (Stages 0-4): frozen; recalibration-in-the-large; slope
  update; limited revision (per-stratum baseline); full CGGA refit (upper bound).
  Recalibration parameters estimated on training folds, evaluated on held-out
  folds.

## Primary comparisons

Internal-vs-external degradation per model; calibration vs discrimination
failure after transport; complex models vs penalised Cox externally; frozen vs
recalibrated validation; clinical-only vs clinical-molecular transportability.
Uncertainty intervals and clinical-relevance thresholds are preferred to p
values; a C-index difference of 0.005-0.01 is not treated as meaningful.

## Prespecified subgroups / sensitivity

Glioblastoma IDH-wildtype only; IDH-mutant only; exclusion of legacy
oligoastrocytoma/NOS; per-release; pooled-deduplicated; 12/24/36-month horizons;
clinical-only vs clinical-molecular feature sets.
