#!/usr/bin/env Rscript
# Transform a bed6+ log2-normalized CPM matrix into a tidy log2FC table per Series.
# Uses matched DMSO controls per stimulus (control_treatment column in samples file).
#
# Usage: Rscript expression_log2fc_matched.R <samples_fn> <bed_fn> <output_dir>/
#
# Input bed_fn: BED6+ file where:
#   col 1-6 = standard BED6 (chrom, start, end, name, score, strand)
#   col 4 (name) = featureID
#   col 7+ = one column per sample, values are log2-normalized CPM
#
# Required samples columns: sample, Series, Treatment, dose.nM, control_treatment
#
# Output: one TSV per Series in output_dir/{series}_TidyDataForModelling.tsv.gz
#   Columns: featureID, treatment, sample, dose, y (log2FC vs matched dose=0 controls)

if (interactive()) {
    args <- scan(text = paste(
        "config/dose_response_samples.tsv",
        "rna-seq/ExpressionMatrices/GRCh38_GencodeRelease44Comprehensive/log2Filtered_TMM_CPM.sorted.bed.gz",
        "DoseResponseModelling/Data/ExpressionLogFC"
    ), what = "character")
} else {
    args <- commandArgs(trailingOnly = TRUE)
}

samples_fn <- args[1]
bed_fn     <- args[2]
output_dir <- args[3]

library(tidyverse)
library(data.table)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

samples <- read_tsv(samples_fn, show_col_types = FALSE)

# Read bed6+ file
bed <- fread(bed_fn)
col_names   <- names(bed)
feature_col <- col_names[4]    # BED col 4 = name = featureID
sample_cols <- col_names[7:length(col_names)]

# Pivot to long format once — then split by Series
bed_long <- bed %>%
    dplyr::select(featureID = all_of(feature_col), all_of(sample_cols)) %>%
    pivot_longer(
        cols      = all_of(sample_cols),
        names_to  = "sample",
        values_to = "logCPM"
    )

series_list <- unique(samples$Series)

for (s in series_list) {
    series_samples <- samples %>%
        filter(Series == s) %>%
        dplyr::select(sample, Treatment, dose = dose.nM, control_treatment) %>%
        replace_na(list(dose = 0))

    dat <- bed_long %>%
        inner_join(series_samples, by = "sample")

    # Per-feature baseline: mean logCPM of dose==0 samples, grouped by Treatment
    # (each sample's control_treatment matches a Treatment in the dose==0 rows)
    controls_mean <- dat %>%
        filter(dose == 0) %>%
        group_by(featureID, Treatment) %>%
        summarise(baseline = mean(logCPM, na.rm = TRUE), .groups = "drop")

    dat %>%
        left_join(controls_mean, by = c("featureID", "control_treatment" = "Treatment")) %>%
        mutate(y = logCPM - baseline) %>%
        dplyr::select(featureID, treatment = Treatment, sample, dose, y) %>%
        write_tsv(file.path(output_dir, paste0(s, "_TidyDataForModelling.tsv.gz")))

    message("Wrote Series: ", s)
}
