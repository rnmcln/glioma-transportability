#!/usr/bin/env Rscript
# Stage existing harmonised cohorts in a clean copy of concordance commit 3799112.
suppressPackageStartupMessages({library(arrow); library(openxlsx)})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("Usage: prepare_concordance.R workbook source_parquet_dir staged_repo audit_csv")
workbook <- args[1]
stopifnot(unname(tools::md5sum(workbook)) == "a4591b2dcee39591f59e5e25a6ce75fa")
src <- args[2]
dst <- file.path(args[3], "data_processed")
dir.create(dst, recursive = TRUE, showWarnings = FALSE)

cdr <- read.xlsx(workbook, sheet = "TCGA-CDR")
cdr <- cdr[cdr$type %in% c("GBM", "LGG"), c("bcr_patient_barcode", "OS", "OS.time")]
stopifnot(!anyDuplicated(cdr$bcr_patient_barcode))
tcga <- as.data.frame(read_parquet(file.path(src, "tcga_harmonised.parquet")))
legacy_primary <- as.data.frame(read_parquet(file.path(src, "analytic_primary_tcga.parquet")))
stopifnot(identical(tcga$patient_id[tcga$included_primary_analysis], legacy_primary$patient_id),
          !anyDuplicated(legacy_primary$patient_id), nrow(legacy_primary) == 1033L)
i <- match(tcga$patient_id, cdr$bcr_patient_barcode)
event <- cdr$OS[i]
time <- cdr$OS.time[i]
usable <- !is.na(time) & is.finite(time) & time >= 0 & event %in% 0:1
was_included <- tcga$included_primary_analysis
stopifnot(!anyNA(i[was_included]), sum(was_included & !usable) == 1L)
legacy <- tcga[was_included & usable, ]
tcga$os_event <- ifelse(usable, as.integer(event), NA_integer_)
tcga$os_time_days <- ifelse(usable, time, NA_real_)
tcga$included_primary_analysis <- was_included & usable
tcga$exclusion_reason[was_included & !usable] <- "os_time_missing;os_event_missing"
primary <- tcga[tcga$included_primary_analysis, ]
stopifnot(nrow(primary) == 1032L, sum(primary$os_event) == 588L, sum(legacy$os_event) == 519L,
          identical(primary$patient_id, legacy$patient_id))

external <- c("analytic_external_cgga.parquet", paste0(
  "cgga_", c("mrnaseq693", "mrnaseq325", "mrnaarray301", "weseq286"), "_harmonised.parquet"))
stopifnot(all(file.copy(file.path(src, external), dst, overwrite = TRUE)))
write_parquet(tcga, file.path(dst, "tcga_harmonised.parquet"))
write_parquet(primary, file.path(dst, "analytic_primary_tcga.parquet"))

summarise <- function(df, endpoint) {
  event <- df$os_event
  time <- df$os_time_days
  horizon <- 730.5
  data.frame(
    endpoint = endpoint, n = nrow(df), deaths = sum(event),
    deaths_by_24_months = sum(event == 1 & time <= horizon),
    censored_before_24_months = sum(event == 0 & time < horizon),
    followup_at_least_24_months = sum(time >= horizon & !(event == 1 & time <= horizon))
  )
}
write.csv(rbind(summarise(legacy, "legacy_paired"), summarise(primary, "tcga_cdr")), args[4], row.names = FALSE)
